#!/usr/bin/env python3
"""
为所有模型生成标准化配置文件
按照XGBoost的规格：debug, quick, standard, full
"""

import os
from pathlib import Path

# 模型配置模板
MODELS_CONFIG = {
    'random_forest': {
        'name': 'RandomForest随机森林',
        'debug': {'n_estimators': 10, 'max_depth': 3, 'min_samples_split': 5},
        'quick': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        'standard': {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5},
        'full': {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 5},
        'extra_params': {'min_samples_leaf': 2, 'max_features': 'sqrt', 'bootstrap': True, 'random_state': 42, 'n_jobs': -1}
    },
    'gradient_boosting': {
        'name': 'GradientBoosting梯度提升',
        'debug': {'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.1},
        'quick': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
        'standard': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1},
        'full': {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.05},
        'extra_params': {'subsample': 0.8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 42}
    },
    'ada_boost': {
        'name': 'AdaBoost自适应提升',
        'debug': {'n_estimators': 10, 'learning_rate': 1.0},
        'quick': {'n_estimators': 100, 'learning_rate': 1.0},
        'standard': {'n_estimators': 200, 'learning_rate': 0.5},
        'full': {'n_estimators': 500, 'learning_rate': 0.3},
        'extra_params': {'random_state': 42}
    },
    'extra_trees': {
        'name': 'ExtraTrees极端随机树',
        'debug': {'n_estimators': 10, 'max_depth': 3, 'min_samples_split': 5},
        'quick': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        'standard': {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5},
        'full': {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 5},
        'extra_params': {'min_samples_leaf': 2, 'max_features': 'sqrt', 'bootstrap': False, 'random_state': 42, 'n_jobs': -1}
    },
    'decision_tree': {
        'name': 'DecisionTree决策树',
        'debug': {'max_depth': 3, 'min_samples_split': 5},
        'quick': {'max_depth': 5, 'min_samples_split': 5},
        'standard': {'max_depth': 10, 'min_samples_split': 5},
        'full': {'max_depth': 15, 'min_samples_split': 2},
        'extra_params': {'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 42}
    },
    'svr': {
        'name': 'SVR支持向量回归',
        'debug': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
        'quick': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
        'standard': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
        'full': {'C': 10.0, 'epsilon': 0.01, 'kernel': 'rbf'},
        'extra_params': {'gamma': 'scale'}
    },
    'knn': {
        'name': 'KNN K近邻',
        'debug': {'n_neighbors': 3},
        'quick': {'n_neighbors': 5},
        'standard': {'n_neighbors': 5},
        'full': {'n_neighbors': 10},
        'extra_params': {'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2, 'metric': 'minkowski', 'n_jobs': -1}
    },
    'ridge': {
        'name': 'Ridge岭回归',
        'debug': {'alpha': 1.0},
        'quick': {'alpha': 1.0},
        'standard': {'alpha': 1.0},
        'full': {'alpha': 0.5},
        'extra_params': {'fit_intercept': True, 'normalize': False, 'solver': 'auto', 'random_state': 42}
    },
    'lasso': {
        'name': 'Lasso套索回归',
        'debug': {'alpha': 0.1, 'max_iter': 100},
        'quick': {'alpha': 0.1, 'max_iter': 500},
        'standard': {'alpha': 0.1, 'max_iter': 1000},
        'full': {'alpha': 0.05, 'max_iter': 2000},
        'extra_params': {'fit_intercept': True, 'normalize': False, 'tol': 0.0001, 'random_state': 42}
    },
    'elastic_net': {
        'name': 'ElasticNet弹性网络',
        'debug': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 100},
        'quick': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 500},
        'standard': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000},
        'full': {'alpha': 0.05, 'l1_ratio': 0.5, 'max_iter': 2000},
        'extra_params': {'fit_intercept': True, 'normalize': False, 'tol': 0.0001, 'random_state': 42}
    }
}

# 配置级别说明
LEVEL_DESCRIPTIONS = {
    'debug': '调试模板（快速测试）',
    'quick': '快速训练（5分钟）',
    'standard': '标准训练（15分钟）',
    'full': '完整训练（30分钟）'
}

# 配置级别对应的训练设置（全部使用10折交叉验证）
LEVEL_TRAINING = {
    'debug': {'n_folds': 10, 'morgan_bits': 512},
    'quick': {'n_folds': 10, 'morgan_bits': 1024},
    'standard': {'n_folds': 10, 'morgan_bits': 1024},
    'full': {'n_folds': 10, 'morgan_bits': 2048}
}

def generate_config(model_type, model_info, level):
    """生成配置文件内容"""
    
    # 合并参数
    hyperparameters = {**model_info[level], **model_info['extra_params']}
    
    # 根据级别设置
    training_config = LEVEL_TRAINING[level]
    
    config_content = f"""name: {model_type}_{level}
description: {model_info['name']}{LEVEL_DESCRIPTIONS[level]}

model:
  model_type: {model_type}
  hyperparameters:"""
    
    # 添加超参数
    for key, value in hyperparameters.items():
        if isinstance(value, str):
            config_content += f"\n    {key}: {value}"
        else:
            config_content += f"\n    {key}: {value}"
    
    # 添加训练配置
    config_content += f"""

training:
  n_folds: {training_config['n_folds']}
  save_final_model: {'false' if level == 'debug' else 'true'}
  verbose: true

feature:
  feature_type: combined
  morgan_bits: {training_config['morgan_bits']}
  morgan_radius: 2
  use_cache: true
  combination_method: mean

data:
  multi_target_strategy: independent
  nan_handling: skip"""
    
    return config_content

def main():
    """主函数"""
    base_dir = Path(__file__).parent.parent / 'config'
    
    for model_type, model_info in MODELS_CONFIG.items():
        model_dir = base_dir / model_type
        model_dir.mkdir(exist_ok=True)
        
        for level in ['debug', 'quick', 'standard', 'full']:
            config_file = model_dir / f"{model_type}_{level}.yaml"
            config_content = generate_config(model_type, model_info, level)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            print(f"✅ 创建: {config_file.relative_to(base_dir.parent)}")
    
    print("\n📊 配置生成完成！")

if __name__ == "__main__":
    main()