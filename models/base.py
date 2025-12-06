#!/usr/bin/env python3
"""
Machine learning model modules
Includes implementations and training logic for various ML models
"""
from __future__ import annotations

import numpy as np
import inspect
from typing import Dict, List, Tuple, Optional, Union
import joblib
from pathlib import Path

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ========================================
#           Default model parameter configuration
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
        'hidden_layer_sizes': (256, 128),  # Optimized network structure, balances performance and overfitting risk
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,  # Lower regularization strength to allow better fit
        'batch_size': 128,  # Fixed batch size for training stability
        'learning_rate': 'adaptive',  # Adaptive learning rate strategy
        'learning_rate_init': 0.0005,  # Moderate initial learning rate
        'max_iter': 2000,  # Increase maximum iterations
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.2,  # Increase validation split
        'n_iter_no_change': 50,  # Increase early stopping patience
        'tol': 0.0001  # Lower tolerance for higher precision
    }
}


# ========================================
#           Base model class
# ========================================

class BaseModel:
    """Base model class"""
    
    def __init__(self, model_type: str, params: Dict = None):
        """
        Initialize model
        
        Args:
            model_type: model type
            params: model parameters
        """
        self.model_type = model_type
        # Get default parameters
        default_params = MODEL_PARAMS.get(model_type, {}).copy()
        
        # If params are provided, only use parameters valid for this model
        if params:
            # Filter parameters valid for this model
            valid_params = {}
            for key, value in params.items():
                # Only add keys present in default parameters
                if key in default_params:
                    valid_params[key] = value
            # Update default parameters
            default_params.update(valid_params)
        
        self.params = default_params
        self.model = None
        self.is_trained = False
        self.scaler = None
        # SVR, KNN and MLP require feature scaling
        self.needs_scaling = model_type in ['svr', 'knn', 'mlp']
        # MLP also needs target scaling
        self.needs_target_scaling = model_type in ['mlp']
        self.target_scaler = None
        
    def create_model(self):
        """Create model instance"""
        if self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**self.params)
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**self.params)
        elif self.model_type == 'catboost':
            import catboost as cb
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
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return self.model
    
    def fit(self, X, y, **kwargs):
        """Train model"""
        if self.model is None:
            self.create_model()

        # Scale features for SVR, KNN, and MLP
        if self.needs_scaling:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Scale target values for MLP
        if self.needs_target_scaling:
            self.target_scaler = StandardScaler()
            y = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Special handling for certain model training parameters
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
                        import lightgbm as lgb
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
        """Predict"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # If scaling was used during training, scale features for prediction
        if self.needs_scaling and self.scaler is not None:
            X = self.scaler.transform(X)

        predictions = self.model.predict(X)

        # If target values were scaled, inverse transform predictions
        if self.needs_target_scaling and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()

        return predictions
    
    def save(self, filepath: Union[str, Path]):
        """Save model"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Save scaler and target_scaler together if present
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
        """Load model"""
        loaded = joblib.load(filepath)
        
        # Check if scaler is included
        if isinstance(loaded, dict) and 'model' in loaded:
            self.model = loaded['model']
            self.scaler = loaded.get('scaler', None)
        else:
            self.model = loaded
            self.scaler = None
        
        self.is_trained = True
        return self.model


# ========================================
#           XGBoost-specific trainer
# ========================================

class XGBoostTrainer:
    """XGBoost trainer class"""
    
    def __init__(self, params: Dict = None, n_folds: int = 10):
        """
        Initialize trainer
        
        Args:
            params: XGBoost parameters
            n_folds: number of cross-validation folds
        """
        self.params = params or MODEL_PARAMS['xgboost'].copy()
        self.n_folds = n_folds
        self.models = []
        self.cv_results = []
        self.best_model = None
        
        print(f"\nXGBoost trainer initialized")
        print(f"   Cross-validation: {self.n_folds} folds")
        print(f"   XGBoost parameters:")
        for key, value in self.params.items():
            print(f"     {key}: {value}")
    
    def train_cv(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform K-fold cross-validation training
        
        Args:
            X: feature matrix
            y: target values
        
        Returns:
            Cross-validation results
        """
        print(f"\nStarting {self.n_folds}-fold cross-validation training...")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.params.get('random_state', 42))
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'mape': []
        }
        
        all_predictions = np.zeros_like(y)
        fold_models = []
        
        # Lazy import xgboost to avoid import overhead and dependency issues
        import xgboost as xgb

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n  Fold {fold}/{self.n_folds}:")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = xgb.XGBRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predict
            y_pred = model.predict(X_val)
            all_predictions[val_idx] = y_pred
            
            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # MAPE (avoid division by zero)
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
            print(f"    R^2:   {r2:.4f}")
            if not np.isnan(mape):
                print(f"    MAPE: {mape:.2f}%")
        
        # Save models
        self.models = fold_models
        
        # Compute average scores
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
        
        print(f"\nCross-validation summary:")
        print(f"   RMSE: {results['mean_rmse']:.4f} +/- {results['std_rmse']:.4f}")
        print(f"   MAE:  {results['mean_mae']:.4f} +/- {results['std_mae']:.4f}")
        print(f"   R^2:   {results['mean_r2']:.4f} +/- {results['std_r2']:.4f}")
        if not np.isnan(results['mean_mape']):
            print(f"   MAPE: {results['mean_mape']:.2f}% +/- {results['std_mape']:.2f}%")
        
        return results
    
    def train_full(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
        """
        Train the final model on all data
        
        Args:
            X: feature matrix
            y: target values
        
        Returns:
            Trained model
        """
        print(f"\nTraining final model (full data)...")
        
        # Lazy import xgboost
        import xgboost as xgb
        model = xgb.XGBRegressor(**self.params)
        model.fit(X, y, verbose=False)
        
        self.best_model = model
        
        # Compute training metrics
        y_pred = model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_r2 = r2_score(y, y_pred)
        
        print(f"   Train RMSE: {train_rmse:.4f}")
        print(f"   Train R^2:   {train_r2:.4f}")
        
        return model
    
    def save_model(self, model: xgb.XGBRegressor, filepath: Union[str, Path]):
        """
        Save model
        
        Args:
            model: model object
            filepath: save path
        """
        joblib.dump(model, filepath)
        print(f"   Model saved: {filepath}")
        
        return filepath


# ========================================
#           General model trainer
# ========================================

class ModelTrainer:
    """General model trainer"""
    
    def __init__(self, model_type: str, params: Dict = None, n_folds: int = 10):
        """
        Initialize trainer
        
        Args:
            model_type: model type
            params: model parameters
            n_folds: number of folds
        """
        self.model_type = model_type
        self.params = params or MODEL_PARAMS.get(model_type, {}).copy()
        self.n_folds = n_folds
        self.models = []
        self.cv_results = []
        self.best_model = None
        
        print(f"\n{model_type.upper()} trainer initialized")
        print(f"   Cross-validation: {self.n_folds} folds")
    
    def train_cv(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict:
        """
        Perform K-fold cross-validation training
        
        Args:
            X: feature matrix
            y: target values
            verbose: whether to show details
        
        Returns:
            Cross-validation results
        """
        if verbose:
            print(f"\nStarting {self.n_folds}-fold cross-validation training...")
        
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
                print(f"\n  Fold {fold}/{self.n_folds}:")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = BaseModel(self.model_type, self.params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Predict
            y_pred = model.predict(X_val)
            all_predictions[val_idx] = y_pred
            
            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # MAPE (avoid division by zero)
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
                print(f"    R^2:   {r2:.4f}")
                if not np.isnan(mape):
                    print(f"    MAPE: {mape:.2f}%")
        
        # Save models
        self.models = fold_models
        
        # Compute average scores
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
            print(f"\nCross-validation summary:")
            print(f"   RMSE: {results['mean_rmse']:.4f} +/- {results['std_rmse']:.4f}")
            print(f"   MAE:  {results['mean_mae']:.4f} +/- {results['std_mae']:.4f}")
            print(f"   R^2:   {results['mean_r2']:.4f} +/- {results['std_r2']:.4f}")
            if not np.isnan(results['mean_mape']):
                print(f"   MAPE: {results['mean_mape']:.2f}% +/- {results['std_mape']:.2f}%")
        
        return results
    
    def train_full(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Train the final model on all data
        
        Args:
            X: feature matrix
            y: target values
            verbose: whether to show details
        
        Returns:
            Trained model
        """
        if verbose:
            print(f"\nTraining final model (full data)...")
        
        model = BaseModel(self.model_type, self.params)
        model.fit(X, y, verbose=False)
        
        self.best_model = model
        
        # Compute training metrics
        y_pred = model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_r2 = r2_score(y, y_pred)
        
        if verbose:
            print(f"   Train RMSE: {train_rmse:.4f}")
            print(f"   Train R^2:   {train_r2:.4f}")
        
        return model
    
    def save_model(self, model, filepath: Union[str, Path]):
        """
        Save model
        
        Args:
            model: model object
            filepath: save path
        """
        if isinstance(model, BaseModel):
            model.save(filepath)
        else:
            joblib.dump(model, filepath)
        print(f"   Model saved: {filepath}")
        
        return filepath


# ========================================
#           Model factory
# ========================================

class ModelFactory:
    """Model factory to create various model trainers"""
    
    SUPPORTED_MODELS = [
        'xgboost', 'lightgbm', 'catboost',
        'random_forest', 'gradient_boosting', 'adaboost', 'extra_trees',
        'svr', 'knn', 'decision_tree',
        'ridge', 'lasso', 'elastic_net', 'mlp'
    ]
    
    @classmethod
    def create_trainer(cls, model_type: str, params: Dict = None, n_folds: int = 10):
        """
        Create a model trainer
        
        Args:
            model_type: model type
            params: model parameters
            n_folds: number of folds
        
        Returns:
            Trainer instance
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: {cls.SUPPORTED_MODELS}")
        
        if model_type == 'xgboost':
            return XGBoostTrainer(params, n_folds)
        else:
            return ModelTrainer(model_type, params, n_folds)
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get supported model list"""
        return cls.SUPPORTED_MODELS.copy()
    
    @classmethod
    def get_model_params(cls, model_type: str) -> Dict:
        """Get default model parameters"""
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        return MODEL_PARAMS.get(model_type, {}).copy()


# ========================================
#           Helper functions
# ========================================

def generate_model_filename(model_type: str, target_col: str, suffix: str = "") -> str:
    """
    Generate model filename
    
    Args:
        model_type: model type
        target_col: target column name
        suffix: filename suffix
    
    Returns:
        Filename
    """
    # Replace special characters thoroughly to generate shell-friendly filename
    clean_target = (target_col
                   .replace('(', '_')
                   .replace(')', '')
                   .replace('/', '_')
                   .replace('*', 'x')
                   .replace('^', '')
                   .replace(' ', '_'))
    
    # Remove possible duplicate underscores
    while '__' in clean_target:
        clean_target = clean_target.replace('__', '_')
    clean_target = clean_target.strip('_')
    
    filename = f"{model_type}_{clean_target}{suffix}.joblib"
    return filename


def load_model(filepath: Union[str, Path]):
    """
    Load model
    
    Args:
        filepath: model file path
    
    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Evaluate model performance
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
    
    Returns:
        Metrics dictionary
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
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
