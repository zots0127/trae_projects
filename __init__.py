"""
AutoML v3 - YOLO-style general automated machine learning framework

Usage:
    from v3 import AutoML
    
    model = AutoML('xgboost')
    model.train('data.csv')
    results = model.predict(test_data)
"""

# Main interfaces
from .automl_model import (
    AutoML,
    XGBoost,
    LightGBM, 
    CatBoost,
    RandomForest,
    load_model
)

# Version info
__version__ = '3.0.0'
__author__ = 'AutoML Team'

# Exported public API
__all__ = [
    'AutoML',
    'XGBoost',
    'LightGBM',
    'CatBoost', 
    'RandomForest',
    'load_model'
]

# Quick-use functions
def quick_train(data_path: str, model_type: str = 'xgboost', **kwargs):
    """Quickly train a model"""
    model = AutoML(model_type)
    model.train(data_path, **kwargs)
    return model

def quick_predict(model_path: str, smiles_list: list):
    """Quick prediction"""
    model = load_model(model_path)
    return model.predict(smiles_list)
