#!/usr/bin/env python3
"""
Unified training script - supports single model and batch training
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
    """Train a single model"""
    
    # 构建配置名称
    if model_name == "adaboost":
        config_name = f"ada_boost_{config_level}"
    else:
        config_name = f"{model_name}_{config_level}"
    
    print(f"\nTraining model: {model_name} (config: {config_name})")
    print("-" * 40)
    
    # 构建命令
    automl_py = str(Path(__file__).parent.parent / "automl.py")
    cmd = [
        "python", automl_py, "train",
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
        print(f"INFO: {model_name} training completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {model_name} training failed")
        if e.stderr:
            print(f"ERROR: {e.stderr[:200]}")
        return False

def train_all_models(project_name=None, data_file=None, config_level="standard", models=None):
    """Train all models"""
    
    # 默认参数
    if not data_file:
        data_file = "data/Database_normalized.csv"
    
    if not project_name:
        project_name = "train"
    
    if not models:
        models = ALL_MODELS
    
    print("=" * 60)
    print(f"Batch model training")
    print(f"Project: {project_name}")
    print(f"Data: {data_file}")
    print(f"Config level: {config_level}")
    print(f"Model count: {len(models)}")
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
    print("Training summary:")
    print("-" * 40)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Success: {success_count}/{len(models)}")
    
    if success_count < len(models):
        failed = [r['model'] for r in results if r['status'] == 'failed']
        print(f"Failed: {', '.join(failed)}")
    
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
            }, f, indent=2, ensure_ascii=True)
        print(f"\nINFO: Training info saved: {info_file}")
    
    return project_name, results

def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description='Unified training script')
    
    # 模式选择
    parser.add_argument('mode', choices=['single', 'all', 'paper'], 
                       help='Training mode: single, all, paper')
    
    # 通用参数
    parser.add_argument('--model', '-m', help='Model name (single mode)')
    parser.add_argument('--project', '-p', help='Project name')
    parser.add_argument('--data', '-d', default='data/Database_normalized.csv', 
                       help='Data file path')
    parser.add_argument('--config', '-c', default='standard',
                       choices=['debug', 'quick', 'standard', 'full'],
                       help='Config level')
    parser.add_argument('--models', nargs='+', help='Models to train (all mode)')
    
    args = parser.parse_args()
    
    # Execute by mode
    if args.mode == 'single':
        if not args.model:
            print("ERROR: --model is required for single mode")
            sys.exit(1)
        
        project = args.project or f"single_{args.model}"
        
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
            print(f"\nINFO: Training completed. Project: {project}")
        else:
            print(f"\nERROR: Training failed")
            sys.exit(1)
    
    elif args.mode == 'all':
        models = args.models if args.models else ALL_MODELS
        project = args.project or "batch"
        
        train_all_models(
            project_name=project,
            data_file=args.data,
            config_level=args.config,
            models=models
        )
    
    elif args.mode == 'paper':
        project = args.project or "paper_table"
        
        print("INFO: Training all models required for paper table...")
        train_all_models(
            project_name=project,
            data_file=args.data,
            config_level='standard',  # 论文使用标准配置
            models=ALL_MODELS
        )
        
        print(f"\nINFO: Completed. Generate table with:")
        print(f"   python scripts/generate_table.py {project}")

if __name__ == "__main__":
    main()
