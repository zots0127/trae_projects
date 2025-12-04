#!/usr/bin/env python3
"""
机器学习模型模块
包含各种ML模型的实现和训练逻辑
"""

import numpy as np
import inspect
from typing import Dict, List, Tuple, Optional, Union
import joblib
from pathlib import Path
from datetime import datetime

# 机器学习相关
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ========================================
#           模型默认参数配置
# ========================================

MODEL_PARAMS = {
    'xgboost': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    },
    'lightgbm': {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    },
    'catboost': {
        'loss_function': 'RMSE',
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'random_state': 42,
        'verbose': False
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    },
    'adaboost': {
        'n_estimators': 300,
        'learning_rate': 0.3,
        'loss': 'square',
        'random_state': 42
    },
    'extra_trees': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    },
    'svr': {
        'kernel': 'rbf',
        'C': 100.0,
        'epsilon': 0.01,
        'gamma': 'scale',
        'cache_size': 1000,
        'max_iter': 5000
    },
    'knn': {
        'n_neighbors': 15,
        'weights': 'distance',
        'algorithm': 'ball_tree',
        'leaf_size': 20,
        'p': 2,
        'metric': 'minkowski',
        'n_jobs': -1
    },
    'decision_tree': {
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'ridge': {
        'alpha': 1.0,
        'random_state': 42
    },
    'lasso': {
        'alpha': 0.01,
        'max_iter': 5000,
        'tol': 0.0001,
        'selection': 'random',
        'random_state': 42
    },
    'elastic_net': {
        'alpha': 0.01,
        'l1_ratio': 0.3,
        'max_iter': 5000,
        'tol': 0.0001,
        'selection': 'random',
        'random_state': 42
    },
    'mlp': {
        'hidden_layer_sizes': (256, 128),  # 优化后的网络结构，平衡性能和过拟合风险
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,  # 降低正则化强度，允许更好拟合
        'batch_size': 128,  # 固定批次大小，提升训练稳定性
        'learning_rate': 'adaptive',  # 自适应学习率策略
        'learning_rate_init': 0.0005,  # 适中的初始学习率
        'max_iter': 2000,  # 增加最大迭代次数
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.2,  # 增加验证集比例
        'n_iter_no_change': 50,  # 增加早停耐心值
        'tol': 0.0001  # 降低收敛容忍度，提升精度
    }
}


# ========================================
#           基础模型类
# ========================================

class BaseModel:
    """基础模型类"""
    
    def __init__(self, model_type: str, params: Dict = None):
        """
        初始化模型
        
        Args:
            model_type: 模型类型
            params: 模型参数
        """
        self.model_type = model_type
        # 获取默认参数
        default_params = MODEL_PARAMS.get(model_type, {}).copy()
        
        # 如果提供了params，只使用对该模型有效的参数
        if params:
            # 过滤出只对该模型有效的参数
            valid_params = {}
            for key, value in params.items():
                # 只添加在默认参数中存在的键
                if key in default_params:
                    valid_params[key] = value
            # 更新默认参数
            default_params.update(valid_params)
        
        self.params = default_params
        self.model = None
        self.is_trained = False
        self.scaler = None
        # SVR、KNN和MLP需要数据标准化
        self.needs_scaling = model_type in ['svr', 'knn', 'mlp']
        # MLP还需要对目标值标准化
        self.needs_target_scaling = model_type in ['mlp']
        self.target_scaler = None
        
    def create_model(self):
        """创建模型实例"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**self.params)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**self.params)
        elif self.model_type == 'catboost':
            self.model = cb.CatBoostRegressor(**self.params)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.params)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self.params)
        elif self.model_type == 'adaboost':
            self.model = AdaBoostRegressor(**self.params)
        elif self.model_type == 'extra_trees':
            self.model = ExtraTreesRegressor(**self.params)
        elif self.model_type == 'svr':
            self.model = SVR(**self.params)
        elif self.model_type == 'knn':
            self.model = KNeighborsRegressor(**self.params)
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeRegressor(**self.params)
        elif self.model_type == 'ridge':
            self.model = Ridge(**self.params)
        elif self.model_type == 'lasso':
            self.model = Lasso(**self.params)
        elif self.model_type == 'elastic_net':
            self.model = ElasticNet(**self.params)
        elif self.model_type == 'mlp':
            self.model = MLPRegressor(**self.params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return self.model
    
    def fit(self, X, y, **kwargs):
        """训练模型"""
        if self.model is None:
            self.create_model()

        # 对SVR、KNN和MLP进行数据标准化
        if self.needs_scaling:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # MLP还需要对目标值进行标准化
        if self.needs_target_scaling:
            self.target_scaler = StandardScaler()
            y = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # 特殊处理某些模型的训练参数
        if self.model_type == 'xgboost' and 'eval_set' in kwargs:
            fit_fn = getattr(self.model, 'fit')
            sig = inspect.signature(fit_fn)
            fit_kwargs = {
                'eval_set': kwargs['eval_set'],
                'verbose': kwargs.get('verbose', False)
            }
            es_rounds = kwargs.get('early_stopping_rounds', None)
            # Prefer callbacks if supported
            if 'callbacks' in sig.parameters and es_rounds:
                try:
                    import xgboost as xgb
                    fit_kwargs['callbacks'] = [xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)]
                except Exception:
                    pass
            # Fallback to early_stopping_rounds if supported
            if 'early_stopping_rounds' in sig.parameters and es_rounds and 'callbacks' not in fit_kwargs:
                fit_kwargs['early_stopping_rounds'] = es_rounds
            # Call fit with supported args only
            self.model.fit(X, y, **fit_kwargs)
        elif self.model_type == 'lightgbm' and 'eval_set' in kwargs:
            fit_fn = getattr(self.model, 'fit')
            sig = inspect.signature(fit_fn)
            fit_kwargs = {
                'eval_set': kwargs['eval_set']
            }
            # Handle verbosity
            if 'verbose' in sig.parameters:
                fit_kwargs['verbose'] = kwargs.get('verbose', False)
            # Early stopping preference: callbacks -> param
            es_rounds = kwargs.get('early_stopping_rounds', None)
            if es_rounds:
                if 'callbacks' in sig.parameters:
                    cb = []
                    try:
                        cb.append(lgb.early_stopping(es_rounds, verbose=False))
                        if not kwargs.get('verbose', False):
                            cb.append(lgb.log_evaluation(0))
                    except Exception:
                        pass
                    if cb:
                        fit_kwargs['callbacks'] = cb
                if 'early_stopping_rounds' in sig.parameters and 'callbacks' not in fit_kwargs:
                    fit_kwargs['early_stopping_rounds'] = es_rounds
            self.model.fit(X, y, **fit_kwargs)
        elif self.model_type == 'catboost':
            self.model.fit(X, y, verbose=kwargs.get('verbose', False))
        else:
            self.model.fit(X, y)
        
        self.is_trained = True
        return self.model
    
    def predict(self, X):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")

        # 如果训练时使用了标准化，预测时也要标准化
        if self.needs_scaling and self.scaler is not None:
            X = self.scaler.transform(X)

        predictions = self.model.predict(X)

        # 如果对目标值进行了标准化，需要反标准化
        if self.needs_target_scaling and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()

        return predictions
    
    def save(self, filepath: Union[str, Path]):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 如果有scaler或target_scaler，一起保存
        if self.scaler is not None or self.target_scaler is not None:
            save_dict = {
                'model': self.model,
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'model_type': self.model_type
            }
            joblib.dump(save_dict, filepath)
        else:
            joblib.dump(self.model, filepath)
    
    def load(self, filepath: Union[str, Path]):
        """加载模型"""
        loaded = joblib.load(filepath)
        
        # 检查是否包含scaler
        if isinstance(loaded, dict) and 'model' in loaded:
            self.model = loaded['model']
            self.scaler = loaded.get('scaler', None)
        else:
            self.model = loaded
            self.scaler = None
        
        self.is_trained = True
        return self.model


# ========================================
#           XGBoost专用训练器
# ========================================

class XGBoostTrainer:
    """XGBoost训练器类"""
    
    def __init__(self, params: Dict = None, n_folds: int = 10):
        """
        初始化训练器
        
        Args:
            params: XGBoost参数
            n_folds: 交叉验证折数
        """
        self.params = params or MODEL_PARAMS['xgboost'].copy()
        self.n_folds = n_folds
        self.models = []
        self.cv_results = []
        self.best_model = None
        
        print(f"\n✅ XGBoost训练器初始化")
        print(f"   交叉验证: {self.n_folds}折")
        print(f"   XGBoost参数:")
        for key, value in self.params.items():
            print(f"     {key}: {value}")
    
    def train_cv(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        执行K折交叉验证训练
        
        Args:
            X: 特征矩阵
            y: 目标值
        
        Returns:
            交叉验证结果
        """
        print(f"\n🚀 开始{self.n_folds}折交叉验证训练...")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.params.get('random_state', 42))
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'mape': []
        }
        
        all_predictions = np.zeros_like(y)
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n  折 {fold}/{self.n_folds}:")
            
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练模型
            model = xgb.XGBRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # 预测
            y_pred = model.predict(X_val)
            all_predictions[val_idx] = y_pred
            
            # 计算指标
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # MAPE (避免除零)
            mask = y_val != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
            else:
                mape = np.nan
            
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)
            cv_scores['mape'].append(mape)
            
            fold_models.append(model)
            
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE:  {mae:.4f}")
            print(f"    R²:   {r2:.4f}")
            if not np.isnan(mape):
                print(f"    MAPE: {mape:.2f}%")
        
        # 保存模型
        self.models = fold_models
        
        # 计算平均得分
        results = {
            'cv_scores': cv_scores,
            'mean_rmse': np.mean(cv_scores['rmse']),
            'std_rmse': np.std(cv_scores['rmse']),
            'mean_mae': np.mean(cv_scores['mae']),
            'std_mae': np.std(cv_scores['mae']),
            'mean_r2': np.mean(cv_scores['r2']),
            'std_r2': np.std(cv_scores['r2']),
            'mean_mape': np.nanmean(cv_scores['mape']),
            'std_mape': np.nanstd(cv_scores['mape']),
            'predictions': all_predictions,
            'true_values': y
        }
        
        self.cv_results = results
        
        print(f"\n📊 交叉验证结果汇总:")
        print(f"   RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
        print(f"   MAE:  {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
        print(f"   R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
        if not np.isnan(results['mean_mape']):
            print(f"   MAPE: {results['mean_mape']:.2f}% ± {results['std_mape']:.2f}%")
        
        return results
    
    def train_full(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
        """
        在全部数据上训练最终模型
        
        Args:
            X: 特征矩阵
            y: 目标值
        
        Returns:
            训练好的模型
        """
        print(f"\n🎯 训练最终模型（全部数据）...")
        
        model = xgb.XGBRegressor(**self.params)
        model.fit(X, y, verbose=False)
        
        self.best_model = model
        
        # 计算训练集指标
        y_pred = model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_r2 = r2_score(y, y_pred)
        
        print(f"   训练RMSE: {train_rmse:.4f}")
        print(f"   训练R²:   {train_r2:.4f}")
        
        return model
    
    def save_model(self, model: xgb.XGBRegressor, filepath: Union[str, Path]):
        """
        保存模型
        
        Args:
            model: 模型对象
            filepath: 保存路径
        """
        joblib.dump(model, filepath)
        print(f"   💾 模型已保存: {filepath}")
        
        return filepath


# ========================================
#           通用模型训练器
# ========================================

class ModelTrainer:
    """通用模型训练器"""
    
    def __init__(self, model_type: str, params: Dict = None, n_folds: int = 10):
        """
        初始化训练器
        
        Args:
            model_type: 模型类型
            params: 模型参数
            n_folds: 交叉验证折数
        """
        self.model_type = model_type
        self.params = params or MODEL_PARAMS.get(model_type, {}).copy()
        self.n_folds = n_folds
        self.models = []
        self.cv_results = []
        self.best_model = None
        
        print(f"\n✅ {model_type.upper()}训练器初始化")
        print(f"   交叉验证: {self.n_folds}折")
    
    def train_cv(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict:
        """
        执行K折交叉验证训练
        
        Args:
            X: 特征矩阵
            y: 目标值
            verbose: 是否显示详细信息
        
        Returns:
            交叉验证结果
        """
        if verbose:
            print(f"\n🚀 开始{self.n_folds}折交叉验证训练...")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'mape': []
        }
        
        all_predictions = np.zeros_like(y)
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            if verbose:
                print(f"\n  折 {fold}/{self.n_folds}:")
            
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建并训练模型
            model = BaseModel(self.model_type, self.params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # 预测
            y_pred = model.predict(X_val)
            all_predictions[val_idx] = y_pred
            
            # 计算指标
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # MAPE (避免除零)
            mask = y_val != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
            else:
                mape = np.nan
            
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)
            cv_scores['mape'].append(mape)
            
            fold_models.append(model)
            
            if verbose:
                print(f"    RMSE: {rmse:.4f}")
                print(f"    MAE:  {mae:.4f}")
                print(f"    R²:   {r2:.4f}")
                if not np.isnan(mape):
                    print(f"    MAPE: {mape:.2f}%")
        
        # 保存模型
        self.models = fold_models
        
        # 计算平均得分
        results = {
            'model_type': self.model_type,
            'cv_scores': cv_scores,
            'mean_rmse': np.mean(cv_scores['rmse']),
            'std_rmse': np.std(cv_scores['rmse']),
            'mean_mae': np.mean(cv_scores['mae']),
            'std_mae': np.std(cv_scores['mae']),
            'mean_r2': np.mean(cv_scores['r2']),
            'std_r2': np.std(cv_scores['r2']),
            'mean_mape': np.nanmean(cv_scores['mape']),
            'std_mape': np.nanstd(cv_scores['mape']),
            'predictions': all_predictions,
            'true_values': y
        }
        
        self.cv_results = results
        
        if verbose:
            print(f"\n📊 交叉验证结果汇总:")
            print(f"   RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
            print(f"   MAE:  {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
            print(f"   R²:   {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
            if not np.isnan(results['mean_mape']):
                print(f"   MAPE: {results['mean_mape']:.2f}% ± {results['std_mape']:.2f}%")
        
        return results
    
    def train_full(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        在全部数据上训练最终模型
        
        Args:
            X: 特征矩阵
            y: 目标值
            verbose: 是否显示详细信息
        
        Returns:
            训练好的模型
        """
        if verbose:
            print(f"\n🎯 训练最终模型（全部数据）...")
        
        model = BaseModel(self.model_type, self.params)
        model.fit(X, y, verbose=False)
        
        self.best_model = model
        
        # 计算训练集指标
        y_pred = model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_r2 = r2_score(y, y_pred)
        
        if verbose:
            print(f"   训练RMSE: {train_rmse:.4f}")
            print(f"   训练R²:   {train_r2:.4f}")
        
        return model
    
    def save_model(self, model, filepath: Union[str, Path]):
        """
        保存模型
        
        Args:
            model: 模型对象
            filepath: 保存路径
        """
        if isinstance(model, BaseModel):
            model.save(filepath)
        else:
            joblib.dump(model, filepath)
        print(f"   💾 模型已保存: {filepath}")
        
        return filepath


# ========================================
#           模型工厂
# ========================================

class ModelFactory:
    """模型工厂类，用于创建各种模型训练器"""
    
    SUPPORTED_MODELS = [
        'xgboost', 'lightgbm', 'catboost',
        'random_forest', 'gradient_boosting', 'adaboost', 'extra_trees',
        'svr', 'knn', 'decision_tree',
        'ridge', 'lasso', 'elastic_net', 'mlp'
    ]
    
    @classmethod
    def create_trainer(cls, model_type: str, params: Dict = None, n_folds: int = 10):
        """
        创建模型训练器
        
        Args:
            model_type: 模型类型
            params: 模型参数
            n_folds: 交叉验证折数
        
        Returns:
            训练器实例
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型类型: {model_type}. 支持的模型: {cls.SUPPORTED_MODELS}")
        
        if model_type == 'xgboost':
            return XGBoostTrainer(params, n_folds)
        else:
            return ModelTrainer(model_type, params, n_folds)
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """获取支持的模型列表"""
        return cls.SUPPORTED_MODELS.copy()
    
    @classmethod
    def get_model_params(cls, model_type: str) -> Dict:
        """获取模型默认参数"""
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型类型: {model_type}")
        return MODEL_PARAMS.get(model_type, {}).copy()


# ========================================
#           辅助函数
# ========================================

def generate_model_filename(model_type: str, target_col: str, suffix: str = "") -> str:
    """
    生成模型文件名
    
    Args:
        model_type: 模型类型
        target_col: 目标列名
        suffix: 文件名后缀
    
    Returns:
        文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更全面的特殊字符替换，生成shell友好的文件名
    clean_target = (target_col
                   .replace('(', '_')
                   .replace(')', '')
                   .replace('/', '_')
                   .replace('*', 'x')
                   .replace('^', '')
                   .replace(' ', '_'))
    
    # 移除可能的重复下划线
    while '__' in clean_target:
        clean_target = clean_target.replace('__', '_')
    clean_target = clean_target.strip('_')
    
    filename = f"{model_type}_{clean_target}{suffix}_{timestamp}.joblib"
    return filename


def load_model(filepath: Union[str, Path]):
    """
    加载模型
    
    Args:
        filepath: 模型文件路径
    
    Returns:
        加载的模型
    """
    return joblib.load(filepath)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    评估模型性能
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        评估指标字典
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (避免除零)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }