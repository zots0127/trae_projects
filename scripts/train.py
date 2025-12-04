#!/usr/bin/env python3
"""
统一训练脚本 - 支持单个模型和批量训练
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import json

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 定义所有标准模型配置
ALL_MODELS = [
    "adaboost",
    "catboost", 
    "decision_tree",
    "elastic_net",
    "extra_trees",
    "gradient_boosting",
    "knn",
    "lasso",
    "lightgbm",
    "random_forest",
    "ridge",
    "xgboost",
    # "svr",  # SVR通常较慢，可选
]

def train_single_model(model_name, project_name, data_file, config_level="standard", **kwargs):
    """训练单个模型"""
    
    # 构建配置名称
    if model_name == "adaboost":
        config_name = f"ada_boost_{config_level}"
    else:
        config_name = f"{model_name}_{config_level}"
    
    print(f"\n训练模型: {model_name} (配置: {config_name})")
    print("-" * 40)
    
    # 构建命令
    cmd = [
        "python", "automl.py", "train",
        f"config={config_name}",
        f"data={data_file}",
        f"project={project_name}",
        f"name={model_name}",
    ]
    
    # 添加额外参数
    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")
    
    try:
        # 执行训练
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {model_name} 训练完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} 训练失败")
        if e.stderr:
            print(f"错误: {e.stderr[:200]}")  # 只显示前200字符
        return False

def train_all_models(project_name=None, data_file=None, config_level="standard", models=None):
    """训练所有模型"""
    
    # 默认参数
    if not data_file:
        data_file = "../data/Database_normalized.csv"
    
    if not project_name:
        project_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not models:
        models = ALL_MODELS
    
    print("=" * 60)
    print(f"批量训练模型")
    print(f"项目: {project_name}")
    print(f"数据: {data_file}")
    print(f"配置级别: {config_level}")
    print(f"模型数量: {len(models)}")
    print("=" * 60)
    
    # 训练参数
    train_params = {
        "multi_target": "independent",
        "nan_handling": "skip",
        "n_folds": 10 if config_level == "standard" else 5,
        "save_final_model": "true",
        "verbose": 0
    }
    
    # 记录结果
    results = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] ", end="")
        success = train_single_model(
            model_name=model,
            project_name=project_name,
            data_file=data_file,
            config_level=config_level,
            **train_params
        )
        
        results.append({
            'model': model,
            'status': 'success' if success else 'failed'
        })
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("训练完成汇总:")
    print("-" * 40)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"成功: {success_count}/{len(models)}")
    
    if success_count < len(models):
        failed = [r['model'] for r in results if r['status'] == 'failed']
        print(f"失败: {', '.join(failed)}")
    
    # 保存训练信息
    info_file = Path(project_name) / "training_info.json"
    if Path(project_name).exists():
        with open(info_file, 'w') as f:
            json.dump({
                'project': project_name,
                'data': data_file,
                'config_level': config_level,
                'models': models,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\n训练信息已保存: {info_file}")
    
    return project_name, results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一训练脚本')
    
    # 模式选择
    parser.add_argument('mode', choices=['single', 'all', 'paper'], 
                       help='训练模式: single-单个模型, all-所有模型, paper-论文表格')
    
    # 通用参数
    parser.add_argument('--model', '-m', help='模型名称(single模式)')
    parser.add_argument('--project', '-p', help='项目名称')
    parser.add_argument('--data', '-d', default='../data/Database_normalized.csv', 
                       help='数据文件路径')
    parser.add_argument('--config', '-c', default='standard',
                       choices=['debug', 'quick', 'standard', 'full'],
                       help='配置级别')
    parser.add_argument('--models', nargs='+', help='指定要训练的模型列表(all模式)')
    
    args = parser.parse_args()
    
    # 根据模式执行
    if args.mode == 'single':
        if not args.model:
            print("❌ single模式需要指定--model参数")
            sys.exit(1)
        
        project = args.project or f"single_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = train_single_model(
            model_name=args.model,
            project_name=project,
            data_file=args.data,
            config_level=args.config,
            multi_target="independent",
            nan_handling="skip",
            n_folds=10 if args.config == "standard" else 5,
            save_final_model="true"
        )
        
        if success:
            print(f"\n✅ 训练完成！项目: {project}")
        else:
            print(f"\n❌ 训练失败")
            sys.exit(1)
    
    elif args.mode == 'all':
        models = args.models if args.models else ALL_MODELS
        project = args.project or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        train_all_models(
            project_name=project,
            data_file=args.data,
            config_level=args.config,
            models=models
        )
    
    elif args.mode == 'paper':
        # 论文表格专用配置
        project = args.project or f"paper_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("🎯 训练论文表格所需的所有模型...")
        train_all_models(
            project_name=project,
            data_file=args.data,
            config_level='standard',  # 论文使用标准配置
            models=ALL_MODELS
        )
        
        print(f"\n✅ 完成！使用以下命令生成表格:")
        print(f"   python scripts/generate_table.py {project}")

if __name__ == "__main__":
    main()