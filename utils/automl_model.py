#!/usr/bin/env python3
"""
AutoML Model - YOLO风格的简洁API（编程式接口）
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, List, Dict, Optional, Any, Tuple
from datetime import datetime

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_extractor import FeatureExtractor
from models.base import BaseModel, evaluate_model, MODEL_PARAMS
# from models.optimizer import OptunaOptimizer  # Optimization removed

warnings.filterwarnings('ignore')


class AutoML:
    """轻量 AutoML 编程式接口（train/predict/optimize）。"""

    def __init__(self, model: str = 'xgboost', device: str = 'cpu', verbose: bool = True):
        self.model_type = model
        self.device = device
        self.verbose = verbose
        self.model = None
        self.feature_extractor = FeatureExtractor(use_cache=True)
        self.is_trained = False
        self.feature_type = 'combined'
        self.target_columns = None
        self.trained_models: Dict[str, Any] = {}
        if self.verbose:
            print(f"✅ AutoML初始化: {model}")

    def train(self,
              data: Union[str, pd.DataFrame],
              epochs: Optional[int] = None,
              val_split: float = 0.2,
              save_dir: str = 'runs',
              optimize: bool = False,
              **kwargs) -> Dict[str, Dict[str, float]]:
        if self.verbose:
            print(f"\n{'='*60}\n开始训练 {self.model_type}\n{'='*60}")
        df = pd.read_csv(data) if isinstance(data, str) else data
        possible_targets = ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)']
        self.target_columns = [c for c in possible_targets if c in df.columns]
        if not self.target_columns:
            raise ValueError(f"未找到目标列，需要: {possible_targets}")
        X, _ = self._prepare_features(df)
        results: Dict[str, Dict[str, float]] = {}
        for target in self.target_columns:
            if self.verbose:
                print(f"\n🎯 训练目标: {target}")
            y = df[target].values
            valid = ~np.isnan(y)
            X_t, y_t = X[valid], y[valid]
            if target == 'PLQY' and y_t.max() > 1.5:
                y_t = y_t / 100
            if optimize:
                model = self._train_with_optimization(X_t, y_t, target)
            else:
                params = kwargs.copy()
                if epochs:
                    params['n_estimators'] = epochs
                model = BaseModel(self.model_type, params)
                model.fit(X_t, y_t)
            self.trained_models[target] = model
            y_pred = model.predict(X_t)
            metrics = evaluate_model(y_t, y_pred)
            results[target] = metrics
            if self.verbose:
                print(f"   RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
        self.is_trained = True
        if save_dir:
            self.save(save_dir)
        if self.verbose:
            print("\n✅ 训练完成!")
        return results

    def predict(self, inputs: Union[List[str], pd.DataFrame, np.ndarray], target: Optional[str] = None) -> Dict[str, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        if isinstance(inputs, list):
            X = self._extract_features_from_smiles(inputs)
        elif isinstance(inputs, pd.DataFrame):
            X, _ = self._prepare_features(inputs)
        else:
            X = inputs
        outputs: Dict[str, np.ndarray] = {}
        for tgt in ([target] if target else self.trained_models.keys()):
            if tgt not in self.trained_models:
                continue
            pred = self.trained_models[tgt].predict(X)
            if tgt == 'PLQY' and pred.max() <= 1.0:
                pred = pred * 100
            outputs[tgt] = pred
        return outputs

    def optimize(self, data: Union[str, pd.DataFrame], n_trials: int = 50, target: Optional[str] = None, **kwargs) -> Dict[str, Dict]:
        """优化功能已移除 - 返回默认参数"""
        # Optimization functionality has been removed
        # Return default parameters instead
        from models.base import MODEL_PARAMS
        default_params = MODEL_PARAMS.get(self.model_type, {})
        targets = [target] if target else self.target_columns
        return {tgt: default_params for tgt in targets}

    def save(self, path: str = 'model.pkl') -> str:
        p = Path(path)
        if p.suffix == '':
            p.mkdir(parents=True, exist_ok=True)
            model_file = p / f"{self.model_type}_automl.pkl"
        else:
            model_file = p
            p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model_type': self.model_type,
            'trained_models': self.trained_models,
            'target_columns': self.target_columns,
            'feature_type': self.feature_type,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }, model_file)
        if self.verbose:
            print(f"💾 模型已保存: {model_file}")
        return str(model_file)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        data = joblib.load(path)
        self.model_type = data['model_type']
        self.trained_models = data['trained_models']
        self.target_columns = data['target_columns']
        self.feature_type = data['feature_type']
        self.is_trained = data['is_trained']
        if self.verbose:
            print(f"✅ 模型已加载: {path}")
        return self

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List]:
        smiles_cols = ['L1', 'L2', 'L3']
        available = [c for c in smiles_cols if c in df.columns]
        if not available:
            raise ValueError("未找到SMILES列 (L1, L2, L3)")
        feats, smiles_list = [], []
        for _, row in df.iterrows():
            smiles = [row[c] for c in available if pd.notna(row[c])]
            if smiles:
                feat = self.feature_extractor.extract_combination(smiles, feature_type=self.feature_type, combination_method='mean')
                feats.append(feat)
                smiles_list.append(smiles)
        return np.array(feats), smiles_list

    def _extract_features_from_smiles(self, smiles_list: List[str]) -> np.ndarray:
        feats = []
        for s in smiles_list:
            parts = s.split(';') if ';' in s else [s]
            feat = self.feature_extractor.extract_combination(parts, feature_type=self.feature_type, combination_method='mean')
            feats.append(feat)
        return np.array(feats)

    def _train_with_optimization(self, X: np.ndarray, y: np.ndarray, target: str):
        # Optimization has been removed - train with default parameters
        from models.base import MODEL_PARAMS
        params = MODEL_PARAMS.get(self.model_type, {})
        model = BaseModel(self.model_type, params)
        model.fit(X, y)
        return model


def load_model(path: str) -> AutoML:
    m = AutoML(verbose=False)
    m.load(path)
    return m


