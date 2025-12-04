#!/usr/bin/env python3
"""
运行管理器 - 类似YOLO的自动增量目录管理
自动创建 runs/train, runs/train2, runs/train3 等目录
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml
import json


class RunManager:
    """运行管理器 - 管理实验目录"""
    
    def __init__(self, base_dir: str = "runs", task: str = "train"):
        """
        初始化运行管理器
        
        Args:
            base_dir: 基础目录 (默认: runs)
            task: 任务类型 (train/predict/validate等)
        """
        self.base_dir = Path(base_dir)
        self.task = task
        self.base_dir.mkdir(exist_ok=True)
    
    def get_next_run_dir(self, name: Optional[str] = None, project: Optional[str] = None) -> Path:
        """
        获取下一个运行目录
        
        类似YOLO的目录命名:
        - 默认: runs/train, runs/train2, runs/train3, ...
        - 指定name: runs/train/my_experiment
        - 指定project: my_project/train, my_project/train2, ...
        - 同时指定: my_project/my_experiment
        
        Args:
            name: 实验名称 (可选)
            project: 项目名称 (可选)
        
        Returns:
            运行目录路径
        """
        # 确定基础路径
        if project:
            base_path = Path(project)
        else:
            base_path = self.base_dir
        
        # 如果指定了name，直接使用
        if name:
            run_dir = base_path / name
        else:
            run_dir = base_path / self.task
        
        # 创建目录
        run_dir.mkdir(parents=True, exist_ok=True)
        
        return run_dir
    
    def _get_increment_dir(self, base_path: Path, prefix: str) -> Path:
        """
        获取自增目录
        
        Args:
            base_path: 基础路径
            prefix: 前缀 (如 train)
        
        Returns:
            自增目录路径
        """
        # 查找现有的运行目录
        existing_runs = []
        
        # 匹配模式: prefix, prefix2, prefix3, ...
        pattern = re.compile(f"^{re.escape(prefix)}(\\d*)$")
        
        # 扫描目录
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir():
                    match = pattern.match(item.name)
                    if match:
                        num_str = match.group(1)
                        if num_str == "":
                            existing_runs.append(1)
                        else:
                            existing_runs.append(int(num_str))
        
        # 确定下一个编号
        if not existing_runs:
            # 第一次运行，不加数字
            next_dir = base_path / prefix
        else:
            # 找到最大编号并加1
            max_num = max(existing_runs)
            if max_num == 1 and 1 in existing_runs:
                # 如果存在 prefix (相当于 prefix1)，下一个是 prefix2
                next_dir = base_path / f"{prefix}2"
            else:
                next_dir = base_path / f"{prefix}{max_num + 1}"
        
        return next_dir
    
    @staticmethod
    def parse_run_path(path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析运行路径，提取project和name
        
        Args:
            path: 路径字符串
        
        Returns:
            (project, name) 元组
        """
        parts = Path(path).parts
        
        if len(parts) == 0:
            return None, None
        elif len(parts) == 1:
            # 只有name或task
            return None, parts[0]
        else:
            # project/name 格式
            if parts[0] == "runs":
                # runs/train 格式
                return None, parts[-1] if len(parts) > 1 else None
            else:
                # project/name 格式
                return parts[0], parts[-1]
    
    def save_run_info(self, run_dir: Path, config: dict, command: str = None):
        """
        保存运行信息
        
        Args:
            run_dir: 运行目录
            config: 配置字典
            command: 运行命令
        """
        run_info = {
            'run_dir': str(run_dir),
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'config': config
        }
        
        # 保存为YAML
        info_file = run_dir / "run_info.yaml"
        with open(info_file, 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False)
        
        # 同时保存为JSON
        json_file = run_dir / "run_info.json"
        with open(json_file, 'w') as f:
            json.dump(run_info, f, indent=2)
    
    @staticmethod
    def create_symlink(run_dir: Path, link_name: str = "last"):
        """
        创建指向最新运行的符号链接
        
        Args:
            run_dir: 运行目录
            link_name: 链接名称 (默认: last)
        """
        # 在父目录创建链接
        parent = run_dir.parent
        link_path = parent / link_name
        
        # 删除旧链接
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        
        # 创建新链接 (相对路径)
        try:
            link_path.symlink_to(run_dir.name)
        except Exception:
            # Windows可能不支持符号链接
            pass
    
    def get_latest_run(self, project: Optional[str] = None) -> Optional[Path]:
        """
        获取最新的运行目录
        
        Args:
            project: 项目名称 (可选)
        
        Returns:
            最新运行目录路径
        """
        if project:
            base_path = Path(project)
        else:
            base_path = self.base_dir
        
        if not base_path.exists():
            return None
        
        # 查找所有运行目录
        run_dirs = []
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 获取修改时间
                run_dirs.append((item, item.stat().st_mtime))
        
        if not run_dirs:
            return None
        
        # 按时间排序，返回最新的
        run_dirs.sort(key=lambda x: x[1], reverse=True)
        return run_dirs[0][0]
    
    def list_runs(self, project: Optional[str] = None, limit: int = 10):
        """
        列出运行历史
        
        Args:
            project: 项目名称 (可选)
            limit: 显示数量限制
        
        Returns:
            运行目录列表
        """
        if project:
            base_path = Path(project)
        else:
            base_path = self.base_dir
        
        if not base_path.exists():
            return []
        
        # 收集所有运行
        runs = []
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 读取运行信息
                info_file = item / "run_info.yaml"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = yaml.safe_load(f)
                else:
                    info = {
                        'run_dir': str(item),
                        'timestamp': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                
                runs.append({
                    'path': item,
                    'name': item.name,
                    'timestamp': info.get('timestamp', ''),
                    'config': info.get('config', {})
                })
        
        # 按时间排序
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # 限制数量
        if limit:
            runs = runs[:limit]
        
        return runs
    
    def clean_old_runs(self, project: Optional[str] = None, keep: int = 5):
        """
        清理旧的运行目录
        
        Args:
            project: 项目名称 (可选)
            keep: 保留的运行数量
        """
        runs = self.list_runs(project, limit=None)
        
        if len(runs) <= keep:
            return
        
        # 删除旧的运行
        for run in runs[keep:]:
            import shutil
            shutil.rmtree(run['path'])
            print(f"删除旧运行: {run['path']}")


class ExperimentTracker:
    """实验追踪器 - 记录和管理实验"""
    
    def __init__(self, run_dir: Path):
        """
        初始化实验追踪器
        
        Args:
            run_dir: 运行目录
        """
        self.run_dir = run_dir
        self.metrics_file = run_dir / "metrics.json"
        self.log_file = run_dir / "experiment.log"
        
        # 创建子目录
        (run_dir / "weights").mkdir(exist_ok=True)  # 模型权重
        (run_dir / "plots").mkdir(exist_ok=True)    # 图表
        (run_dir / "predictions").mkdir(exist_ok=True)  # 预测结果
        (run_dir / "exports").mkdir(exist_ok=True)  # 导出文件
        
        # 初始化指标记录
        self.metrics = {
            'epochs': [],
            'train': {},
            'val': {},
            'test': {}
        }
    
    def log_metrics(self, epoch: int, metrics: dict, split: str = 'train'):
        """
        记录指标
        
        Args:
            epoch: 轮次
            metrics: 指标字典
            split: 数据集划分 (train/val/test)
        """
        # 更新内存中的指标
        if split not in self.metrics:
            self.metrics[split] = {}
        
        for key, value in metrics.items():
            if key not in self.metrics[split]:
                self.metrics[split][key] = []
            self.metrics[split][key].append(value)
        
        if epoch not in self.metrics['epochs']:
            self.metrics['epochs'].append(epoch)
        
        # 保存到文件
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_model(self, model, name: str = "best", format: str = "joblib"):
        """
        保存模型
        
        Args:
            model: 模型对象
            name: 模型名称
            format: 保存格式
        """
        weights_dir = self.run_dir / "weights"
        
        if format == "joblib":
            import joblib
            model_path = weights_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
        elif format == "pickle":
            import pickle
            model_path = weights_dir / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        return model_path
    
    def log(self, message: str, level: str = "INFO"):
        """
        写入日志
        
        Args:
            message: 日志消息
            level: 日志级别
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # 同时打印到控制台
        print(f"[{level}] {message}")
    
    def get_summary(self) -> dict:
        """获取实验摘要"""
        if not self.metrics_file.exists():
            return {}
        
        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)
        
        summary = {
            'run_dir': str(self.run_dir),
            'n_epochs': len(metrics.get('epochs', [])),
            'metrics': {}
        }
        
        # 计算最终指标
        for split in ['train', 'val', 'test']:
            if split in metrics and metrics[split]:
                summary['metrics'][split] = {}
                for key, values in metrics[split].items():
                    if values:
                        summary['metrics'][split][key] = {
                            'final': values[-1],
                            'best': min(values) if 'loss' in key or 'rmse' in key else max(values),
                            'mean': sum(values) / len(values)
                        }
        
        return summary


# ========================================
#           便捷函数
# ========================================

def get_run_dir(name: Optional[str] = None, 
                project: Optional[str] = None,
                task: str = "train") -> Path:
    """
    获取运行目录的便捷函数
    
    Args:
        name: 实验名称
        project: 项目名称
        task: 任务类型
    
    Returns:
        运行目录路径
    """
    manager = RunManager(task=task)
    return manager.get_next_run_dir(name=name, project=project)


def setup_experiment(name: Optional[str] = None,
                    project: Optional[str] = None,
                    config: dict = None) -> Tuple[Path, ExperimentTracker]:
    """
    设置实验环境
    
    Args:
        name: 实验名称
        project: 项目名称
        config: 配置字典
    
    Returns:
        (运行目录, 实验追踪器)
    """
    # 获取运行目录
    run_dir = get_run_dir(name=name, project=project)
    
    # 创建追踪器
    tracker = ExperimentTracker(run_dir)
    
    # 保存配置
    if config:
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # 创建符号链接
    RunManager.create_symlink(run_dir, "last")
    
    print(f"💾 实验目录: {run_dir}")
    
    return run_dir, tracker


if __name__ == "__main__":
    # 测试代码
    print("运行管理器测试")
    print("=" * 50)
    
    # 测试自动增量
    manager = RunManager()
    
    # 默认运行
    run1 = manager.get_next_run_dir()
    print(f"运行1: {run1}")
    
    run2 = manager.get_next_run_dir()
    print(f"运行2: {run2}")
    
    # 指定名称
    run3 = manager.get_next_run_dir(name="my_experiment")
    print(f"运行3: {run3}")
    
    # 指定项目
    run4 = manager.get_next_run_dir(project="my_project")
    print(f"运行4: {run4}")
    
    # 同时指定
    run5 = manager.get_next_run_dir(name="best_model", project="my_project")
    print(f"运行5: {run5}")
    
    # 列出运行
    print("\n最近的运行:")
    runs = manager.list_runs(limit=5)
    for run in runs:
        print(f"  - {run['name']}: {run['timestamp']}")
    
    print("\n✅ 测试完成")
