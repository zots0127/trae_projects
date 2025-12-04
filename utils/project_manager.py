#!/usr/bin/env python3
"""
项目管理器
用于管理AutoML项目的完整生命周期
"""

import json
import yaml
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd

from .comparison_table import ComparisonTableGenerator

class ProjectManager:
    """项目管理器"""
    
    def __init__(self, base_dir: str = "."):
        """
        初始化项目管理器
        
        Args:
            base_dir: 基础目录
        """
        self.base_dir = Path(base_dir)
    
    def create_project_metadata(self, project_dir: str) -> Dict:
        """
        创建项目元数据
        
        Args:
            project_dir: 项目目录
        
        Returns:
            元数据字典
        """
        project_path = Path(project_dir)
        if not project_path.exists():
            raise ValueError(f"项目目录不存在: {project_dir}")
        
        metadata = {
            'project_name': project_path.name,
            'created_at': datetime.now().isoformat(),
            'path': str(project_path.absolute()),
            'models_trained': [],
            'targets': [],
            'best_models': {},
            'data_info': {},
            'comparison_tables': [],
            'training_runs': []
        }
        
        # 扫描训练运行
        for run_dir in project_path.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith('.'):
                run_info = self._analyze_run(run_dir)
                if run_info:
                    metadata['training_runs'].append(run_info)
                    
                    # 收集模型类型
                    if run_info['model'] not in metadata['models_trained']:
                        metadata['models_trained'].append(run_info['model'])
                    
                    # 收集目标
                    for target in run_info.get('targets', []):
                        if target not in metadata['targets']:
                            metadata['targets'].append(target)
                    
                    # 更新最佳模型
                    for target, perf in run_info.get('performance', {}).items():
                        if target not in metadata['best_models'] or \
                           perf.get('r2', 0) > metadata['best_models'][target].get('r2', 0):
                            metadata['best_models'][target] = {
                                'model': run_info['model'],
                                'run': run_info['name'],
                                'r2': perf.get('r2', 0),
                                'rmse': perf.get('rmse', 0),
                                'path': str(run_dir / 'models')
                            }
        
        # 查找对比表格
        for table_file in project_path.glob('comparison_table_*'):
            metadata['comparison_tables'].append(table_file.name)
        
        # 保存元数据
        metadata_file = project_path / 'project_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 项目元数据已创建: {metadata_file}")
        
        return metadata
    
    def _analyze_run(self, run_dir: Path) -> Optional[Dict]:
        """分析单个训练运行"""
        run_info = {
            'name': run_dir.name,
            'path': str(run_dir),
            'model': None,
            'targets': [],
            'performance': {}
        }
        
        # 读取配置
        config_file = run_dir / 'config.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                run_info['model'] = config.get('model', {}).get('model_type')
                run_info['targets'] = config.get('data', {}).get('target_columns', [])
        
        # 读取运行信息
        run_info_file = run_dir / 'run_info.json'
        if run_info_file.exists():
            with open(run_info_file, 'r') as f:
                info = json.load(f)
                run_info['timestamp'] = info.get('timestamp')
                run_info['command'] = info.get('command')
        
        # 收集性能指标
        for summary_file in (run_dir / 'exports').glob('*_summary.json'):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                target = summary.get('target')
                if target:
                    run_info['performance'][target] = {
                        'r2': summary.get('mean_r2', 0),
                        'rmse': summary.get('mean_rmse', 0),
                        'mae': summary.get('mean_mae', 0)
                    }
        
        return run_info if run_info['model'] else None
    
    def generate_comparison_table(self, project_name: str, output_dir: Optional[str] = None,
                                  formats: Optional[List[str]] = None,
                                  decimal_places: Optional[Dict[str, int]] = None) -> Dict[str, str]:
        """
        为项目生成模型性能对比表格（自动扫描项目目录中的 *_summary.json）
        
        Args:
            project_name: 项目名称或路径
            output_dir: 输出目录（默认写入项目根目录）
            formats: 输出格式列表（markdown, html, latex, csv）
            decimal_places: 小数位控制，如 {'r2': 4, 'rmse': 4, 'mae': 4}
        
        Returns:
            各格式文件的输出路径字典
        """
        # 解析项目路径
        project_path = Path(project_name)
        if not project_path.exists():
            project_path = self.base_dir / project_name
        if not project_path.exists():
            raise ValueError(f"项目不存在: {project_name}")

        # 创建生成器并导出
        generator = ComparisonTableGenerator(results_dir=str(project_path))
        exported_files = generator.export_all_formats(
            output_dir=output_dir if output_dir else str(project_path),
            formats=formats if formats else ['markdown', 'html', 'latex', 'csv'],
            decimal_places=decimal_places
        )

        # 更新/写入项目元数据中的 comparison_tables 列表
        metadata_file = project_path / 'project_metadata.json'
        metadata = None
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = None
        if not metadata:
            metadata = self.create_project_metadata(str(project_path))

        # 合并新表格文件名（仅保存文件名，避免绝对路径差异）
        new_names = [Path(p).name for p in exported_files.values()]
        existing = set(metadata.get('comparison_tables', []))
        for name in new_names:
            if name not in existing:
                metadata['comparison_tables'].append(name)
                existing.add(name)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ 对比表已生成，文件数: {len(exported_files)}")
        for k, v in exported_files.items():
            print(f"   - {k}: {v}")

        return exported_files

    def list_projects(self) -> List[Dict]:
        """
        列出所有项目
        
        Returns:
            项目列表
        """
        projects = []
        
        # 查找所有包含project_metadata.json的目录
        for metadata_file in self.base_dir.rglob('project_metadata.json'):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    projects.append({
                        'name': metadata['project_name'],
                        'path': metadata_file.parent,
                        'created': metadata.get('created_at', 'Unknown'),
                        'models': len(metadata.get('models_trained', [])),
                        'runs': len(metadata.get('training_runs', []))
                    })
            except Exception as e:
                continue
        
        # 也查找没有元数据但有模型的目录
        for model_file in self.base_dir.rglob('*.joblib'):
            project_dir = model_file.parent
            while project_dir != self.base_dir and project_dir.parent != self.base_dir:
                project_dir = project_dir.parent
            
            if not (project_dir / 'project_metadata.json').exists():
                # 检查是否已经添加
                if not any(p['path'] == project_dir for p in projects):
                    projects.append({
                        'name': project_dir.name,
                        'path': project_dir,
                        'created': 'Unknown',
                        'models': '?',
                        'runs': '?'
                    })
        
        return projects
    
    def get_project_info(self, project_name: str) -> Dict:
        """
        获取项目详细信息
        
        Args:
            project_name: 项目名称或路径
        
        Returns:
            项目信息字典
        """
        # 查找项目
        project_path = Path(project_name)
        if not project_path.exists():
            # 尝试在base_dir中查找
            project_path = self.base_dir / project_name
        
        if not project_path.exists():
            raise ValueError(f"项目不存在: {project_name}")
        
        # 检查元数据
        metadata_file = project_path / 'project_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            # 动态生成元数据
            return self.create_project_metadata(str(project_path))
    
    def export_project(self, project_name: str, output_path: str = None,
                      format: str = 'zip') -> str:
        """
        导出项目
        
        Args:
            project_name: 项目名称或路径
            output_path: 输出路径
            format: 导出格式 ('zip', 'tar')
        
        Returns:
            导出文件路径
        """
        # 查找项目
        project_path = Path(project_name)
        if not project_path.exists():
            project_path = self.base_dir / project_name
        
        if not project_path.exists():
            raise ValueError(f"项目不存在: {project_name}")
        
        # 确保有元数据
        self.create_project_metadata(str(project_path))
        
        # 设置输出路径
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{project_path.name}_{timestamp}.{format}"
        
        # 导出
        if format == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in project_path.rglob('*'):
                    if file.is_file():
                        arcname = file.relative_to(project_path.parent)
                        zipf.write(file, arcname)
        elif format == 'tar':
            import tarfile
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(project_path, arcname=project_path.name)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"✅ 项目已导出: {output_path}")
        return output_path
    
    def clean_project(self, project_name: str, keep_models: bool = True,
                     keep_results: bool = True) -> None:
        """
        清理项目（删除中间文件）
        
        Args:
            project_name: 项目名称或路径
            keep_models: 是否保留模型文件
            keep_results: 是否保留结果文件
        """
        # 查找项目
        project_path = Path(project_name)
        if not project_path.exists():
            project_path = self.base_dir / project_name
        
        if not project_path.exists():
            raise ValueError(f"项目不存在: {project_name}")
        
        # 清理规则
        patterns_to_delete = [
            '**/checkpoints/*',
            '**/predictions/*',
            '**/feature_importance/*' if not keep_results else None,
            '**/plots/*' if not keep_results else None,
            '**/models/*' if not keep_models else None,
        ]
        
        deleted_count = 0
        for pattern in patterns_to_delete:
            if pattern:
                for file in project_path.glob(pattern):
                    if file.is_file():
                        file.unlink()
                        deleted_count += 1
        
        print(f"✅ 清理完成，删除了 {deleted_count} 个文件")
    
    def generate_project_report(self, project_name: str, output_path: str = None) -> str:
        """
        生成项目报告
        
        Args:
            project_name: 项目名称或路径
            output_path: 输出路径
        
        Returns:
            报告文件路径
        """
        # 获取项目信息
        info = self.get_project_info(project_name)
        
        # 生成Markdown报告
        report = f"# 项目报告: {info['project_name']}\n\n"
        report += f"**创建时间**: {info.get('created_at', 'Unknown')}\n\n"
        
        # 模型信息
        report += "## 训练的模型\n\n"
        if info.get('models_trained'):
            for model in info['models_trained']:
                report += f"- {model}\n"
        else:
            report += "无模型信息\n"
        
        report += "\n## 预测目标\n\n"
        if info.get('targets'):
            for target in info['targets']:
                report += f"- {target}\n"
                if target in info.get('best_models', {}):
                    best = info['best_models'][target]
                    report += f"  - 最佳模型: {best['model']} (R²={best.get('r2', 'N/A'):.4f})\n"
        
        report += "\n## 训练运行\n\n"
        if info.get('training_runs'):
            report += "| 运行 | 模型 | 目标数 | 平均R² |\n"
            report += "|------|------|--------|--------|\n"
            for run in info['training_runs']:
                avg_r2 = 0
                if run.get('performance'):
                    r2_values = [p.get('r2', 0) for p in run['performance'].values()]
                    avg_r2 = sum(r2_values) / len(r2_values) if r2_values else 0
                report += f"| {run['name']} | {run.get('model', 'Unknown')} | "
                report += f"{len(run.get('targets', []))} | {avg_r2:.4f} |\n"
        
        # 保存报告
        if output_path is None:
            project_path = Path(info['path'])
            output_path = project_path / f"project_report_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✅ 项目报告已生成: {output_path}")
        return str(output_path)


def main():
    """主函数，用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='项目管理器')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # list命令
    list_parser = subparsers.add_parser('list', help='列出所有项目')
    
    # info命令
    info_parser = subparsers.add_parser('info', help='获取项目信息')
    info_parser.add_argument('project', help='项目名称或路径')
    
    # export命令
    export_parser = subparsers.add_parser('export', help='导出项目')
    export_parser.add_argument('project', help='项目名称或路径')
    export_parser.add_argument('--output', help='输出路径')
    export_parser.add_argument('--format', default='zip', choices=['zip', 'tar'])
    
    # clean命令
    clean_parser = subparsers.add_parser('clean', help='清理项目')
    clean_parser.add_argument('project', help='项目名称或路径')
    clean_parser.add_argument('--keep-models', action='store_true')
    clean_parser.add_argument('--keep-results', action='store_true')
    
    # report命令
    report_parser = subparsers.add_parser('report', help='生成项目报告')
    report_parser.add_argument('project', help='项目名称或路径')
    report_parser.add_argument('--output', help='输出路径')

    # table命令
    table_parser = subparsers.add_parser('table', help='生成模型对比表')
    table_parser.add_argument('project', help='项目名称或路径')
    table_parser.add_argument('--output', help='输出目录（默认项目根目录）')
    table_parser.add_argument('--formats', nargs='+', default=['markdown','html','latex','csv'],
                              help='输出格式列表')
    
    args = parser.parse_args()
    
    manager = ProjectManager()
    
    if args.command == 'list':
        projects = manager.list_projects()
        if projects:
            print("\n📁 项目列表:")
            for p in projects:
                print(f"  - {p['name']} ({p['path']})")
                print(f"    创建: {p['created']}")
                print(f"    模型: {p['models']}, 运行: {p['runs']}")
        else:
            print("没有找到项目")
    
    elif args.command == 'info':
        info = manager.get_project_info(args.project)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    
    elif args.command == 'export':
        manager.export_project(args.project, args.output, args.format)
    
    elif args.command == 'clean':
        manager.clean_project(args.project, args.keep_models, args.keep_results)
    
    elif args.command == 'report':
        manager.generate_project_report(args.project, args.output)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()