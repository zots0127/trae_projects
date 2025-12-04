#!/usr/bin/env python3
"""
演示如何使用paper_comparison配置训练所有模型并生成对比表格
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Paper Comparison Demo')
    parser.add_argument('--data', default='../data/Database_normalized.csv',
                       help='训练数据文件')
    parser.add_argument('--test-data', default=None,
                       help='测试数据文件（可选）')
    parser.add_argument('--project', default='PaperDemo',
                       help='项目名称')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（只训练3个模型）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("       论文级模型对比演示")
    print("="*60)
    print()
    
    # 构建命令
    cmd_parts = [
        'python', 'automl.py', 'train',
        'config=paper_comparison',
        f'data={args.data}',
        f'project={args.project}'
    ]
    
    if args.test_data:
        cmd_parts.append(f'test_data={args.test_data}')
    
    if args.quick:
        # 快速模式：只训练几个关键模型
        cmd_parts.extend([
            'optimization.automl_models=[xgboost,catboost,lightgbm]',
            'training.n_folds=5'
        ])
        print("🚀 快速模式：只训练 XGBoost, CatBoost, LightGBM")
    else:
        print("📊 完整模式：训练所有13个模型")
    
    # 显示命令
    print("\n执行命令:")
    print(" ".join(cmd_parts))
    print()
    
    # 执行命令
    import subprocess
    result = subprocess.run(cmd_parts, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✅ 训练完成！")
        
        # 查找结果目录
        project_dir = Path(args.project)
        if project_dir.exists():
            # 找到最新的训练目录
            train_dirs = sorted(project_dir.glob('train*'), key=lambda x: x.stat().st_mtime)
            if train_dirs:
                latest_dir = train_dirs[-1]
                print(f"\n📁 结果目录: {latest_dir}")
                
                # 生成对比表格
                print("\n生成对比表格...")
                try:
                    sys.path.append('.')
                    from utils.comparison_table import ComparisonTableGenerator
                    
                    generator = ComparisonTableGenerator(str(latest_dir))
                    exported = generator.export_all_formats()
                    
                    # 显示最佳模型
                    print("\n" + "="*60)
                    print("最佳模型总结")
                    print("="*60)
                    best_models = generator.get_best_models()
                    for target, info in best_models.items():
                        print(f"\n{target}:")
                        print(f"  最佳模型: {info['algorithm']}")
                        print(f"  R²: {info['r2']}")
                        print(f"  RMSE: {info['rmse']}")
                    
                    print("\n📊 生成的表格文件:")
                    for fmt, path in exported.items():
                        print(f"  - {fmt.upper()}: {path}")
                    
                except Exception as e:
                    print(f"⚠️ 生成表格时出错: {e}")
    else:
        print("\n❌ 训练失败")
        sys.exit(1)


if __name__ == '__main__':
    main()