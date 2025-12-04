"""
AutoML v3 - YOLO风格的通用自动机器学习框架

使用方法:
    from v3 import AutoML
    
    model = AutoML('xgboost')
    model.train('data.csv')
    results = model.predict(test_data)
"""

# 主要接口
from .automl_model import (
    AutoML,
    XGBoost,
    LightGBM, 
    CatBoost,
    RandomForest,
    load_model
)

# 版本信息
__version__ = '3.0.0'
__author__ = 'AutoML Team'

# 导出的公共API
__all__ = [
    'AutoML',
    'XGBoost',
    'LightGBM',
    'CatBoost', 
    'RandomForest',
    'load_model'
]

# 快速使用函数
def quick_train(data_path: str, model_type: str = 'xgboost', **kwargs):
    """快速训练模型"""
    model = AutoML(model_type)
    model.train(data_path, **kwargs)
    return model

def quick_predict(model_path: str, smiles_list: list):
    """快速预测"""
    model = load_model(model_path)
    return model.predict(smiles_list)