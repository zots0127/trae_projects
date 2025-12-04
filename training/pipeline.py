#!/usr/bin/env python3
"""
基于配置的训练管道
类似YOLO的一键训练系统
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import traceback
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入自定义模块
from config.system import (
    ExperimentConfig, ConfigManager, ConfigValidator,
    BatchExperimentConfig, load_config
)
from core.feature_extractor import FeatureExtractor
from models.base import ModelFactory, evaluate_model, generate_model_filename
from training.logger import TrainingLogger
from sklearn.model_selection import KFold
import joblib
from utils.timing import TimingTracker
from utils.file_feature_cache import FileFeatureCache

warnings.filterwarnings('ignore')


# ========================================
#           训练管道
# ========================================

class TrainingPipeline:
    """基于配置的训练管道"""
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化训练管道
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.logger = None
        self.data = None
        self.features = None
        self.targets = None
        
        # 验证配置
        if not ConfigValidator.validate_all(config):
            raise ValueError("配置验证失败")
        
        print("\n" + "="*60)
        print(f"🚀 训练管道初始化: {config.name}")
        print("="*60)
        print(f"模型: {config.model.model_type}")
        print(f"特征: {config.feature.feature_type}")
        print(f"交叉验证: {config.training.n_folds}折")
        # 初始化细粒度计时器
        self.timing = TimingTracker()
        
    def load_data(self, target_col: str = None) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            target_col: 如果指定，只为该目标列过滤数据；否则加载所有数据
        """
        if target_col is None:
            # 兼容性：如果没有指定目标，加载所有数据（用于初始检查）
            print(f"\n📊 加载数据: {self.config.data.data_path}")
            with self.timing.measure('data_load_train'):
                df = pd.read_csv(self.config.data.data_path)
            print(f"   原始数据: {len(df)} 行, {len(df.columns)} 列")
            
            # 检查可用的目标列
            available_targets = []
            target_stats = {}
            for target in self.config.data.target_columns:
                if target in df.columns:
                    available_targets.append(target)
                    n_valid = df[target].notna().sum()
                    target_stats[target] = n_valid
                    print(f"   {target}: {n_valid} 个有效值")
            
            if not available_targets:
                raise ValueError(f"没有找到任何目标列: {self.config.data.target_columns}")
            
            # 根据多目标策略显示数据选择信息
            if not hasattr(self.config.data, 'multi_target_strategy'):
                self.config.data.multi_target_strategy = 'independent'
            
            if self.config.data.multi_target_strategy == 'intersection':
                # 计算交集数据量
                valid_mask = pd.Series([True] * len(df))
                for target in available_targets:
                    valid_mask &= df[target].notna()
                n_intersection = valid_mask.sum()
                print(f"\n   📊 多目标策略: 交集模式")
                print(f"      所有目标都有值的数据: {n_intersection} 行")
                print(f"      数据利用率: {n_intersection/len(df)*100:.1f}%")
            elif self.config.data.multi_target_strategy == 'independent':
                print(f"\n   📊 多目标策略: 独立模式")
                print(f"      每个目标独立使用其有效数据")
            elif self.config.data.multi_target_strategy == 'union':
                print(f"\n   📊 多目标策略: 并集模式")
                print(f"      使用所有数据，缺失值将被填充")
            
            self.available_targets = available_targets
            self.target_stats = target_stats
            self.raw_data = df  # 保存原始数据
            return df
        else:
            # 为特定目标加载和过滤数据
            if not hasattr(self, 'raw_data'):
                with self.timing.measure('data_load_train'):
                    df = pd.read_csv(self.config.data.data_path)
                self.raw_data = df
            else:
                df = self.raw_data.copy()
            
            # 处理缺失值
            if not hasattr(self.config.data, 'nan_handling'):
                self.config.data.nan_handling = 'skip'  # 默认值
            if not hasattr(self.config.data, 'multi_target_strategy'):
                self.config.data.multi_target_strategy = 'independent'  # 默认值
            
            # 根据多目标策略和缺失值处理策略处理数据
            if self.config.data.multi_target_strategy == 'intersection':
                # 交集模式：只使用所有目标都有值的数据
                print(f"\n   📌 使用交集模式处理 {target_col}")
                valid_mask = pd.Series([True] * len(df))
                for target in self.config.data.target_columns:
                    if target in df.columns:
                        valid_mask &= df[target].notna()
                
                # 检查SMILES列
                if self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                    for col in self.config.data.smiles_columns:
                        if col in df.columns:
                            valid_mask &= df[col].notna()
                
                df_valid = df[valid_mask].copy()
                n_dropped = len(df) - len(df_valid)
                print(f"   {target_col} 的有效数据: {len(df_valid)} 行 (交集模式)")
                # 直接设置数据并返回
                self.data = df_valid
                return df_valid
                
            elif self.config.data.multi_target_strategy == 'union':
                # 并集模式：使用所有数据，配合nan_handling策略
                print(f"\n   📌 使用并集模式处理 {target_col}")
                df_valid = df.copy()
                
                # 根据nan_handling策略填充缺失值
                if self.config.data.nan_handling != 'skip':
                    # 处理目标列缺失值
                    if target_col in df_valid.columns:
                        n_missing = df_valid[target_col].isna().sum()
                        if n_missing > 0:
                            if self.config.data.nan_handling == 'mean':
                                mean_val = df_valid[target_col].mean()
                                df_valid[target_col].fillna(mean_val, inplace=True)
                                print(f"   ✅ 使用均值 {mean_val:.4f} 填充了 {n_missing} 个缺失值")
                            elif self.config.data.nan_handling == 'median':
                                median_val = df_valid[target_col].median()
                                df_valid[target_col].fillna(median_val, inplace=True)
                                print(f"   ✅ 使用中位数 {median_val:.4f} 填充了 {n_missing} 个缺失值")
                            elif self.config.data.nan_handling == 'zero':
                                df_valid[target_col].fillna(0, inplace=True)
                                print(f"   ✅ 使用0填充了 {n_missing} 个缺失值")
                
                # SMILES缺失仍需跳过
                if self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                    for col in self.config.data.smiles_columns:
                        if col in df_valid.columns:
                            mask = df_valid[col].notna()
                            n_missing = (~mask).sum()
                            if n_missing > 0:
                                df_valid = df_valid[mask]
                                print(f"   ⚠️ 跳过了 {n_missing} 行SMILES缺失的数据")
                
                print(f"   {target_col} 的有效数据: {len(df_valid)} 行 (并集模式)")
                # 直接设置数据并返回
                self.data = df_valid
                return df_valid
                
            elif self.config.data.multi_target_strategy == 'independent':
                # 独立模式：每个目标独立处理（原有逻辑）
                pass
                
            # 根据不同策略处理缺失值（独立模式的原有逻辑）
            if self.config.data.multi_target_strategy == 'independent' and self.config.data.nan_handling == 'skip':
                # 筛选有效数据：只检查当前目标列和SMILES列
                valid_mask = pd.Series([True] * len(df))
                
                # 检查目标列
                if target_col in df.columns:
                    valid_mask &= df[target_col].notna()
                else:
                    raise ValueError(f"目标列不存在: {target_col}")
                
                # 检查SMILES列（仅当使用分子特征时）
                if self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                    for col in self.config.data.smiles_columns:
                        if col in df.columns:
                            valid_mask &= df[col].notna()
                        else:
                            print(f"   ⚠️ SMILES列不存在: {col}")
                elif self.config.feature.feature_type == 'tabular':
                    # 对于表格数据，不需要SMILES列
                    pass
                
                df_valid = df[valid_mask].copy()
                n_dropped = len(df) - len(df_valid)
                if n_dropped > 0:
                    print(f"   {target_col} 的有效数据: {len(df_valid)} 行 (跳过了 {n_dropped} 行含缺失值的数据)")
                else:
                    print(f"   {target_col} 的有效数据: {len(df_valid)} 行")
                    
            else:
                # 其他缺失值处理策略
                df_valid = df.copy()
                
                # 处理目标列缺失值
                if target_col in df_valid.columns:
                    n_missing = df_valid[target_col].isna().sum()
                    if n_missing > 0:
                        if self.config.data.nan_handling == 'mean':
                            mean_val = df_valid[target_col].mean()
                            df_valid[target_col].fillna(mean_val, inplace=True)
                            print(f"   ✅ 使用均值 {mean_val:.4f} 填充了 {n_missing} 个目标缺失值")
                        elif self.config.data.nan_handling == 'median':
                            median_val = df_valid[target_col].median()
                            df_valid[target_col].fillna(median_val, inplace=True)
                            print(f"   ✅ 使用中位数 {median_val:.4f} 填充了 {n_missing} 个目标缺失值")
                        elif self.config.data.nan_handling == 'zero':
                            df_valid[target_col].fillna(0, inplace=True)
                            print(f"   ✅ 使用0填充了 {n_missing} 个目标缺失值")
                        elif self.config.data.nan_handling == 'forward':
                            df_valid[target_col].fillna(method='ffill', inplace=True)
                            df_valid[target_col].fillna(method='bfill', inplace=True)
                            print(f"   ✅ 使用前向填充处理了 {n_missing} 个目标缺失值")
                        elif self.config.data.nan_handling == 'interpolate':
                            df_valid[target_col] = df_valid[target_col].interpolate()
                            df_valid[target_col].fillna(method='bfill', inplace=True)
                            df_valid[target_col].fillna(method='ffill', inplace=True)
                            print(f"   ✅ 使用插值处理了 {n_missing} 个目标缺失值")
                
                # SMILES列缺失必须跳过
                if self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                    for col in self.config.data.smiles_columns:
                        if col in df_valid.columns:
                            mask = df_valid[col].notna()
                            n_missing = (~mask).sum()
                            if n_missing > 0:
                                df_valid = df_valid[mask]
                                print(f"   ⚠️ 跳过了 {n_missing} 行SMILES缺失的数据 (列: {col})")
                
                print(f"   {target_col} 的有效数据: {len(df_valid)} 行")
            
            self.data = df_valid
            return df_valid
    
    def extract_features(self) -> np.ndarray:
        """提取特征"""
        if self.data is None:
            self.load_data()
        
        print(f"\n🔧 提取{self.config.feature.feature_type}特征...")
        
        # 开始特征提取计时
        with self.timing.measure('feature_extraction', {'type': self.config.feature.feature_type}):
            self._extract_features_internal()
        
        # 计算吞吐量
        if self.features is not None:
            self.timing.calculate_throughput('feature_extraction', len(self.features))
            
        return self.features
    
    def _extract_features_internal(self) -> np.ndarray:
        """内部特征提取实现"""
        # 初始化特征提取器
        feature_extractor = FeatureExtractor(
            use_cache=self.config.feature.use_cache,
            cache_dir=self.config.feature.cache_dir,
            morgan_bits=self.config.feature.morgan_bits if hasattr(self.config.feature, 'morgan_bits') else None,
            morgan_radius=self.config.feature.morgan_radius if hasattr(self.config.feature, 'morgan_radius') else None,
            descriptor_count=getattr(self.config.feature, 'descriptor_count', 85)
        )
        
        # 检查是否为分子数据（有SMILES列）
        has_smiles = any(col in self.data.columns for col in self.config.data.smiles_columns)
        
        if has_smiles and self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
            # 分子特征提取，优先使用文件级缓存并按子集索引切片
            features = None
            try:
                file_cache = FileFeatureCache(cache_dir='file_feature_cache')
                X_full = file_cache.load_features(
                    file_path=str(self.config.data.data_path),
                    feature_type=self.config.feature.feature_type,
                    morgan_bits=getattr(self.config.feature, 'morgan_bits', 1024),
                    morgan_radius=getattr(self.config.feature, 'morgan_radius', 2),
                    smiles_columns=self.config.data.smiles_columns,
                    combination_method=getattr(self.config.feature, 'combination_method', 'mean'),
                    descriptor_count=getattr(self.config.feature, 'descriptor_count', 85)
                )
                if X_full is not None:
                    # 使用原始索引选择当前目标的数据子集
                    subset_index = self.data.index.to_numpy()
                    features = X_full[subset_index]
                    print("   ✅ 训练特征使用文件级缓存 (已切片至当前子集)")
            except Exception:
                features = None

            if features is None:
                # 如果没有缓存，尝试一次性为整个训练文件计算并写入缓存
                try:
                    raw_df = getattr(self, 'raw_data', None)
                    if raw_df is None:
                        raw_df = pd.read_csv(self.config.data.data_path)
                        self.raw_data = raw_df

                    print("   ⏳ 未命中文件级缓存，正在为整个训练文件提取一次特征...")
                    feats_full = []
                    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="提取分子特征(全文件)"):
                        smiles_list = []
                        for col in self.config.data.smiles_columns:
                            if col in row and pd.notna(row[col]):
                                smiles_list.append(row[col])
                            else:
                                smiles_list.append(None)
                        f = feature_extractor.extract_combination(
                            smiles_list,
                            feature_type=self.config.feature.feature_type,
                            combination_method=self.config.feature.combination_method
                        )
                        feats_full.append(f)
                    X_full = np.array(feats_full)

                    # 写入缓存供后续目标/阶段复用
                    try:
                        file_cache.save_features(
                            features=X_full,
                            file_path=str(self.config.data.data_path),
                            feature_type=self.config.feature.feature_type,
                            morgan_bits=getattr(self.config.feature, 'morgan_bits', 1024),
                            morgan_radius=getattr(self.config.feature, 'morgan_radius', 2),
                            smiles_columns=self.config.data.smiles_columns,
                            combination_method=getattr(self.config.feature, 'combination_method', 'mean'),
                            descriptor_count=getattr(self.config.feature, 'descriptor_count', 85),
                            row_count=len(raw_df),
                            failed_indices=[]
                        )
                        print("   💾 已缓存训练特征(全文件)")
                    except Exception:
                        pass

                    # 切片到当前子集
                    subset_index = self.data.index.to_numpy()
                    features = X_full[subset_index]
                except Exception:
                    # 回退到原逐行提取逻辑
                    features = []
                    for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="提取分子特征"):
                        smiles_list = []
                        for col in self.config.data.smiles_columns:
                            if col in row and pd.notna(row[col]):
                                smiles_list.append(row[col])
                            else:
                                smiles_list.append(None)
                        feat = feature_extractor.extract_combination(
                            smiles_list,
                            feature_type=self.config.feature.feature_type,
                            combination_method=self.config.feature.combination_method
                        )
                        features.append(feat)
                    features = np.array(features)
        else:
            # 表格数据特征提取（新功能）
            print("   检测到表格数据，使用通用特征提取...")
            
            # 获取目标列以排除
            target_cols = self.config.data.target_columns if hasattr(self.config.data, 'target_columns') else []
            
            # 使用新的DataFrame提取方法
            if hasattr(feature_extractor, 'extract_from_dataframe'):
                features = feature_extractor.extract_from_dataframe(
                    self.data,
                    smiles_columns=self.config.data.smiles_columns if has_smiles else None,
                    target_columns=target_cols,
                    feature_type='tabular' if not has_smiles else 'auto'
                )
            else:
                # 后备方案：使用所有非目标列作为特征
                feature_cols = [col for col in self.data.columns if col not in target_cols]
                features = self.data[feature_cols].values
        
        print(f"   特征维度: {features.shape}")
        
        # 处理NaN和Inf
        n_nan = np.isnan(features).sum()
        n_inf = np.isinf(features).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"   ⚠️ 发现 {n_nan} 个NaN值, {n_inf} 个Inf值，正在处理...")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.features = features
        return features
    
    def prepare_target(self, target_col: str) -> np.ndarray:
        """准备目标变量"""
        if self.data is None:
            self.load_data()
        
        y = self.data[target_col].values
        
        # 单位转换
        if target_col == 'PLQY' and y.max() > 1.5:
            print(f"   转换PLQY: 百分比 → 小数")
            y = y / 100
        
        return y
    
    def train_single_target(self, target_col: str) -> Dict:
        """
        训练单个目标
        
        Args:
            target_col: 目标列名
        
        Returns:
            训练结果
        """
        print(f"\n{'='*60}")
        print(f"训练目标: {target_col}")
        print(f"{'='*60}")
        
        # 为该目标独立加载和过滤数据
        self.load_data(target_col=target_col)
        
        # 提取特征（每个目标独立提取）
        self.features = None  # 重置特征
        self.extract_features()
        
        X = self.features
        y = self.prepare_target(target_col)
        
        print(f"   样本数: {len(X)}")
        print(f"   特征数: {X.shape[1]}")
        print(f"   目标范围: [{y.min():.2f}, {y.max():.2f}]")
        
        # 优化功能已被移除，直接使用默认参数训练
        
        # 创建训练器
        trainer = ModelFactory.create_trainer(
            self.config.model.model_type,
            self.config.model.hyperparameters,
            self.config.training.n_folds
        )
        
        # 初始化记录器
        if self.config.logging.auto_save:
            if self.logger is None:
                self.logger = TrainingLogger(
                    project_name=self.config.logging.project_name,
                    base_dir=self.config.logging.base_dir,
                    auto_save=self.config.logging.auto_save,
                    save_plots=self.config.logging.save_plots
                )
            
            # 开始实验
            experiment_id = self.logger.start_experiment(
                model_type=self.config.model.model_type,
                target=target_col,
                feature_type=self.config.feature.feature_type,
                hyperparameters=self.config.model.hyperparameters,
                n_folds=self.config.training.n_folds,
                n_samples=len(X),
                n_features=X.shape[1],
                config=self.config.to_dict()
            )
        
        # 执行交叉验证
        kf = KFold(
            n_splits=self.config.training.n_folds,
            shuffle=True,
            random_state=self.config.data.random_seed
        )
        
        all_predictions = np.zeros_like(y)
        fold_models = []
        fold_metrics = []
        
        # 初始化特征重要性记录器（如果启用）
        feature_importance_recorder = None
        if self.config.training.save_feature_importance:
            from utils import FeatureImportanceRecorder
            feature_importance_recorder = FeatureImportanceRecorder(
                save_dir=Path(self.config.logging.base_dir) / self.config.logging.project_name,
                model_name=self.config.model.model_type,
                target=target_col
            )
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            # 记录折开始
            if self.logger:
                self.logger.log_fold_start(fold_idx, train_idx.tolist(), val_idx.tolist())
            
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建并训练模型
            from models import BaseModel
            model = BaseModel(self.config.model.model_type, self.config.model.hyperparameters)
            early_rounds = self.config.training.early_stopping_rounds if self.config.training.early_stopping else None
            
            # 记录每折的训练时间
            with self.timing.measure(f'fold_{fold_idx}_training', {'fold': fold_idx, 'samples': len(train_idx)}):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=early_rounds
                )
            
            # 预测
            with self.timing.measure(f'fold_{fold_idx}_prediction', {'fold': fold_idx, 'samples': len(val_idx)}):
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                all_predictions[val_idx] = y_val_pred
            
            # 评估
            train_metrics = evaluate_model(y_train, y_train_pred)
            val_metrics = evaluate_model(y_val, y_val_pred)
            
            fold_metrics.append(val_metrics)
            fold_models.append(model)
            
            # 提取并记录特征重要性（如果启用且模型支持）
            if feature_importance_recorder:
                try:
                    # 尝试从模型中提取特征重要性
                    importance = FeatureImportanceRecorder.extract_importance_from_model(model.model)
                    if importance is not None:
                        # 生成特征名称（如果需要）
                        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                        feature_importance_recorder.add_fold_importance(
                            fold_idx, importance, feature_names
                        )
                except Exception as e:
                    if self.config.training.verbose > 1:
                        print(f"    ⚠️ 无法提取特征重要性: {e}")
            
            # 记录折结束
            if self.logger:
                self.logger.log_fold_end(
                    y_train=y_train,
                    y_train_pred=y_train_pred,
                    y_val=y_val,
                    y_val_pred=y_val_pred,
                    metrics={**val_metrics, 'train_rmse': train_metrics['rmse'], 'train_r2': train_metrics['r2']}
                )
            
            # 显示进度
            if self.config.training.verbose > 0:
                print(f"\n  折 {fold_idx}/{self.config.training.n_folds}:")
                print(f"    训练 - RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
                print(f"    验证 - RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        
        # 计算总体指标
        final_metrics = evaluate_model(y, all_predictions)
        
        # 计算平均指标
        avg_metrics = {}
        for metric in self.config.training.metrics:
            values = [fold[metric] for fold in fold_metrics if metric in fold]
            if values:
                avg_metrics[f"{metric}_mean"] = np.mean(values)
                avg_metrics[f"{metric}_std"] = np.std(values)
        
        print(f"\n📊 交叉验证结果:")
        for metric in self.config.training.metrics:
            if f"{metric}_mean" in avg_metrics:
                print(f"   {metric.upper()}: {avg_metrics[f'{metric}_mean']:.4f} ± {avg_metrics[f'{metric}_std']:.4f}")
        
        # 保存特征重要性（如果启用）
        if feature_importance_recorder:
            try:
                feature_importance_recorder.save_importance()
            except Exception as e:
                if self.config.training.verbose > 0:
                    print(f"   ⚠️ 保存特征重要性失败: {e}")
        
        # 训练最终模型
        final_model = None
        if self.config.training.save_final_model:
            print(f"\n🎯 训练最终模型（全部数据）...")
            final_model = BaseModel(self.config.model.model_type, self.config.model.hyperparameters)
            with self.timing.measure('final_model_training'):
                final_model.fit(X, y, verbose=False)
            
            # 保存模型
            model_dir = Path(self.config.logging.base_dir) / self.config.logging.project_name / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_filename = generate_model_filename(
                self.config.model.model_type,
                target_col,
                "_final"
            )
            model_path = model_dir / model_filename
            final_model.save(model_path)
            print(f"   💾 模型已保存: {model_path}")
            
            # 保存最终模型的特征重要性（如果启用）
            if self.config.training.save_feature_importance:
                try:
                    from utils import FeatureImportanceRecorder
                    importance = FeatureImportanceRecorder.extract_importance_from_model(final_model.model)
                    if importance is not None:
                        # 创建一个新的记录器用于最终模型
                        final_importance_recorder = FeatureImportanceRecorder(
                            save_dir=Path(self.config.logging.base_dir) / self.config.logging.project_name,
                            model_name=f"{self.config.model.model_type}_final",
                            target=target_col
                        )
                        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                        final_importance_recorder.add_fold_importance(0, importance, feature_names)
                        final_importance_recorder.save_importance()
                except Exception as e:
                    if self.config.training.verbose > 1:
                        print(f"   ⚠️ 保存最终模型特征重要性失败: {e}")

        # 若提供测试集，进行测试评估（仅使用完整数据训练的最终模型）
        test_evaluation = None
        test_predictions = None
        if getattr(self.config.data, 'test_data_path', None):
            try:
                test_path = Path(self.config.data.test_data_path)
                print(f"\n" + "="*50)
                print(f"🧪 测试集评估 (Test Evaluation)")
                print("="*50)
                print(f"文件: {test_path.name}")
                if test_path.exists():
                    print(f"状态: ✅ 文件存在")
                    print(f"路径: {test_path.resolve()}")
                    with self.timing.measure('data_load_test'):
                        df_test = pd.read_csv(test_path)
                    # 准备测试特征：与训练相同的流程
                    feature_extractor = FeatureExtractor(
                        use_cache=self.config.feature.use_cache,
                        cache_dir=self.config.feature.cache_dir,
                        morgan_bits=self.config.feature.morgan_bits if hasattr(self.config.feature, 'morgan_bits') else None,
                        morgan_radius=self.config.feature.morgan_radius if hasattr(self.config.feature, 'morgan_radius') else None,
                        descriptor_count=getattr(self.config.feature, 'descriptor_count', 85)
                    )
                    has_smiles = any(col in df_test.columns for col in self.config.data.smiles_columns)
                    if has_smiles and self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                        # 优先尝试文件级缓存
                        X_test = None
                        try:
                            file_cache = FileFeatureCache(cache_dir='file_feature_cache')
                            X_test = file_cache.load_features(
                                file_path=str(test_path),
                                feature_type=self.config.feature.feature_type,
                                morgan_bits=getattr(self.config.feature, 'morgan_bits', 1024),
                                morgan_radius=getattr(self.config.feature, 'morgan_radius', 2),
                                smiles_columns=self.config.data.smiles_columns,
                                combination_method=getattr(self.config.feature, 'combination_method', 'mean'),
                                descriptor_count=getattr(self.config.feature, 'descriptor_count', 85)
                            )
                            if X_test is not None:
                                print("\n✅ 从文件级缓存加载测试特征，跳过提取")
                                print(f"   形状: {X_test.shape}")
                                print("   开始选择推理模型与预测")
                        except Exception as _e:
                            # 缓存失败时静默回退到正常提取
                            X_test = None

                        if X_test is None:
                            feats = []
                            for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="提取分子特征(测试)"):
                                smiles_list = []
                                for col in self.config.data.smiles_columns:
                                    if col in row and pd.notna(row[col]):
                                        smiles_list.append(row[col])
                                    else:
                                        smiles_list.append(None)
                                with self.timing.measure('feature_extract_test_single'):
                                    f = feature_extractor.extract_combination(
                                        smiles_list,
                                        feature_type=self.config.feature.feature_type,
                                        combination_method=self.config.feature.combination_method
                                    )
                                feats.append(f)
                            X_test = np.array(feats)

                            # 写入文件级缓存，供其它目标复用
                            try:
                                file_cache.save_features(
                                    features=X_test,
                                    file_path=str(test_path),
                                    feature_type=self.config.feature.feature_type,
                                    morgan_bits=getattr(self.config.feature, 'morgan_bits', 1024),
                                    morgan_radius=getattr(self.config.feature, 'morgan_radius', 2),
                                    smiles_columns=self.config.data.smiles_columns,
                                    combination_method=getattr(self.config.feature, 'combination_method', 'mean'),
                                    descriptor_count=getattr(self.config.feature, 'descriptor_count', 85),
                                    row_count=len(df_test),
                                    failed_indices=[]
                                )
                                print("💾 已缓存测试特征，后续目标将复用")
                            except Exception:
                                pass
                    else:
                        target_cols = self.config.data.target_columns if hasattr(self.config.data, 'target_columns') else []
                        with self.timing.measure('feature_extract_test_tabular'):
                            X_test = feature_extractor.extract_from_dataframe(
                                df_test,
                                smiles_columns=self.config.data.smiles_columns if has_smiles else None,
                                target_columns=target_cols,
                                feature_type='tabular' if not has_smiles else 'auto'
                            )
                    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

                    # 选择用于预测的模型（如果保存了最终模型则用最终模型，否则用简单平均折模型）
                    model_for_inference = final_model
                    if model_for_inference is None and len(fold_models) > 0:
                        print(f"   使用折模型集成预测, 数量: {len(fold_models)}")
                        with self.timing.measure('test_predict_oof_ensemble'):
                            preds_list = []
                            for j, m in enumerate(fold_models, 1):
                                print(f"   折 {j}/{len(fold_models)} 预测开始")
                                p = m.predict(X_test)
                                preds_list.append(p)
                                print(f"   折 {j} 预测完成")
                        test_predictions = np.mean(np.vstack(preds_list), axis=0)
                        print(f"   集成预测完成，输出形状: {np.array(test_predictions).shape}")
                    else:
                        print("   使用最终模型进行预测")
                        with self.timing.measure('test_predict_final_model'):
                            test_predictions = model_for_inference.predict(X_test)
                        print(f"   最终模型预测完成，输出形状: {np.array(test_predictions).shape}")

                    # 若测试集中包含当前目标列，计算指标
                    if target_col in df_test.columns:
                        y_test = df_test[target_col].values
                        if target_col == 'PLQY' and y_test.max() > 1.5:
                            y_test = y_test / 100
                        test_evaluation = evaluate_model(y_test, test_predictions)
                        
                        # 详细的测试结果输出
                        print(f"\n📊 测试结果 ({target_col}):")
                        print(f"   样本数: {len(y_test)}")
                        print(f"   ├─ RMSE: {test_evaluation['rmse']:.4f}")
                        print(f"   ├─ MAE:  {test_evaluation['mae']:.4f}")
                        print(f"   ├─ R²:   {test_evaluation['r2']:.4f}")
                        print(f"   └─ MAPE: {test_evaluation.get('mape', 0):.2f}%")

                    # 保存测试预测
                    if self.logger:
                        exp_dir = Path(self.config.logging.base_dir) / self.config.logging.project_name
                        exports_dir = exp_dir / 'exports'
                        exports_dir.mkdir(parents=True, exist_ok=True)
                        out_csv = exports_dir / f"test_predictions_{self.config.model.model_type}_{target_col}.csv"
                        df_out = df_test.copy()
                        df_out['prediction'] = test_predictions
                        df_out.to_csv(out_csv, index=False)
                        
                        # 保存测试指标（若有）
                        if test_evaluation is not None:
                            out_json = exports_dir / f"test_metrics_{self.config.model.model_type}_{target_col}.json"
                            import json as _json
                            with open(out_json, 'w') as f:
                                _json.dump(test_evaluation, f, indent=2)
                        
                        # 输出保存信息
                        print(f"\n💾 测试结果已保存:")
                        print(f"   预测文件: {out_csv.name}")
                        if test_evaluation is not None:
                            print(f"   指标文件: {out_json.name}")
                        print(f"   保存目录: {exports_dir}")
                        print("="*50)
                else:
                    print(f"   ⚠️ 测试集路径不存在: {test_path}")
                    print(f"      当前工作目录: {Path.cwd()}")
                    # 尝试其他可能的路径
                    alternative_paths = [
                        Path(test_path.name),  # 当前目录
                        Path("../data") / test_path.name,  # ../data目录
                        Path("data") / test_path.name,  # data目录
                    ]
                    for alt_path in alternative_paths:
                        if alt_path.exists():
                            print(f"      💡 文件可能在: {alt_path}")
            except Exception as e:
                print(f"   ⚠️ 测试集评估失败: {e}")
                import traceback
                if self.config.training.verbose > 1:
                    traceback.print_exc()
        
        # 结束实验
        if self.logger:
            self.logger.end_experiment(final_metrics)
            try:
                timing_summary = self.timing.get_summary()
                for k, v in timing_summary.get('records', {}).items():
                    self.logger.add_timing(k, v.get('duration', 0))
            except Exception:
                pass
        
        # 打印和保存时间统计
        if self.config.training.verbose > 0:
            print("\n" + "="*50)
            print("⏱️ 时间统计")
            print("="*50)
            self.timing.print_summary()
        
        # 保存时间报告
        if self.logger:
            try:
                exp_dir = Path(self.config.logging.base_dir) / self.config.logging.project_name
                timing_dir = exp_dir / 'timing'
                timing_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存JSON格式
                self.timing.save_report(
                    timing_dir / f"timing_{self.config.model.model_type}_{target_col}.json",
                    format='json'
                )
                
                # 保存文本格式
                self.timing.save_report(
                    timing_dir / f"timing_{self.config.model.model_type}_{target_col}.txt",
                    format='txt'
                )
                
                if self.config.training.verbose > 0:
                    print(f"\n💾 时间报告已保存到: {timing_dir}")
            except Exception as e:
                if self.config.training.verbose > 1:
                    print(f"⚠️ 保存时间报告失败: {e}")
            
            # 导出论文数据
            if self.config.logging.export_for_paper:
                self.logger.export_for_paper(experiment_id)
        
        return {
            'target': target_col,
            'final_metrics': final_metrics,
            'avg_metrics': avg_metrics,
            'fold_metrics': fold_metrics,
            'predictions': all_predictions,
            'true_values': y,
            'test_metrics': test_evaluation,
            'test_predictions_saved': self.config.data.test_data_path is not None
        }
    
    def train_all_targets(self) -> Dict:
        """训练所有目标"""
        results = {}
        
        for target in self.available_targets:
            try:
                print(f"\n训练目标: {target}")
                result = self.train_single_target(target)
                results[target] = result
            except Exception as e:
                print(f"训练 {target} 失败: {e}")
                results[target] = {'error': str(e)}
        
        return results
    
    def run(self, targets: Optional[List[str]] = None) -> Dict:
        """
        运行训练管道
        
        Args:
            targets: 要训练的目标列表，None表示训练所有
        
        Returns:
            训练结果字典
        """
        print(f"\n🚀 开始训练: {self.config.name}")
        
        # 初始加载数据以检查可用目标
        self.load_data()
        
        # 确定目标
        if targets:
            targets_to_train = [t for t in targets if t in self.available_targets]
        else:
            targets_to_train = self.available_targets
        
        if not targets_to_train:
            raise ValueError("没有可训练的目标")
        
        print(f"\n将训练 {len(targets_to_train)} 个目标: {targets_to_train}")
        
        # 训练所有目标
        results = {}
        for target in targets_to_train:
            try:
                print(f"\n训练目标: {target}")
                result = self.train_single_target(target)
                results[target] = result
            except Exception as e:
                print(f"训练 {target} 失败: {e}")
                results[target] = {'error': str(e)}
        
        # 打印汇总
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """打印训练汇总"""
        print("\n" + "="*60)
        print("训练汇总")
        print("="*60)
        
        for target, result in results.items():
            if 'error' in result:
                print(f"\n❌ {target}: 失败 - {result['error']}")
            else:
                print(f"\n✅ {target}:")
                if 'final_metrics' in result:
                    metrics = result['final_metrics']
                    print(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}" if isinstance(metrics.get('rmse'), (int, float)) else f"   RMSE: N/A")
                    print(f"   MAE:  {metrics.get('mae', 'N/A'):.4f}" if isinstance(metrics.get('mae'), (int, float)) else f"   MAE: N/A")
                    print(f"   R²:   {metrics.get('r2', 'N/A'):.4f}" if isinstance(metrics.get('r2'), (int, float)) else f"   R²: N/A")
                if 'avg_metrics' in result:
                    avg = result['avg_metrics']
                    print(f"   CV平均: RMSE={avg.get('rmse', 'N/A'):.4f}" if isinstance(avg.get('rmse'), (int, float)) else f"   CV平均: N/A")


# ========================================
#           批量训练管道
# ========================================

class BatchTrainingPipeline:
    """批量训练管道"""
    
    def __init__(self, batch_config: BatchExperimentConfig):
        """
        初始化批量训练管道
        
        Args:
            batch_config: 批量实验配置
        """
        self.batch_config = batch_config
        self.results = {}
    
    def run(self) -> Dict:
        """运行批量训练"""
        configs = self.batch_config.generate_configs()
        
        print(f"\n🚀 批量训练: {len(configs)} 个实验")
        print("="*60)
        
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] 实验: {config.name}")
            
            try:
                pipeline = TrainingPipeline(config)
                result = pipeline.run()
                self.results[config.name] = {
                    'config': config,
                    'results': result,
                    'status': 'success'
                }
            except Exception as e:
                print(f"❌ 实验失败: {e}")
                self.results[config.name] = {
                    'config': config,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # 汇总结果
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """打印批量训练汇总"""
        print("\n" + "="*60)
        print("批量训练汇总")
        print("="*60)
        
        success_count = sum(1 for r in self.results.values() if r['status'] == 'success')
        print(f"\n成功: {success_count}/{len(self.results)}")
        
        # 找出最佳模型
        best_models = {}
        for name, result in self.results.items():
            if result['status'] == 'success':
                for target, target_result in result['results'].items():
                    if 'final_metrics' in target_result:
                        key = f"{target}_rmse"
                        rmse = target_result['final_metrics']['rmse']
                        if key not in best_models or rmse < best_models[key]['value']:
                            best_models[key] = {
                                'experiment': name,
                                'value': rmse
                            }
        
        if best_models:
            print("\n🏆 最佳模型:")
            for key, info in best_models.items():
                target = key.replace('_rmse', '')
                print(f"   {target}: {info['experiment']} (RMSE: {info['value']:.4f})")


# ========================================
#           命令行接口
# ========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='基于配置的机器学习训练管道',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 配置相关参数
    parser.add_argument('config', nargs='?', help='配置文件路径或模板名称')
    parser.add_argument('--template', '-t', help='使用预定义模板')
    parser.add_argument('--list-templates', action='store_true', help='列出所有可用模板')
    parser.add_argument('--wizard', action='store_true', help='使用配置向导')
    parser.add_argument('--save-config', help='保存配置到文件')
    
    # 训练相关参数
    parser.add_argument('--target', help='指定训练目标（逗号分隔）')
    parser.add_argument('--dry-run', action='store_true', help='只验证配置，不执行训练')
    parser.add_argument('--test-data', dest='test_data', help='可选：指定测试集CSV路径，用于完整训练后评估')
    
    # 覆盖配置参数
    parser.add_argument('--model', help='模型类型')
    parser.add_argument('--feature', help='特征类型')
    parser.add_argument('--folds', type=int, help='交叉验证折数')
    parser.add_argument('--project', help='项目名称')
    
    args = parser.parse_args()
    
    # 配置管理器
    manager = ConfigManager()
    
    # 列出模板
    if args.list_templates:
        print("\n可用模板:")
        for template in manager.list_templates():
            desc = manager.templates[template].description
            print(f"  - {template}: {desc}")
        return
    
    # 配置向导
    if args.wizard:
        config = manager.create_from_wizard()
    
    # 加载配置
    elif args.config:
        # 尝试作为模板
        if args.config in manager.list_templates():
            config = manager.get_template(args.config)
        # 作为文件路径
        else:
            config = load_config(args.config)
    
    # 使用模板
    elif args.template:
        config = manager.get_template(args.template)
    
    # 默认配置
    else:
        print("使用默认配置 (xgboost_quick)")
        config = manager.get_template('xgboost_quick')
    
    # 覆盖配置
    if args.model:
        config.model.model_type = args.model
    if args.feature:
        config.feature.feature_type = args.feature
    if args.folds:
        config.training.n_folds = args.folds
    if args.project:
        config.logging.project_name = args.project
    # 测试集参数
    if args.test_data:
        config.data.test_data_path = args.test_data
    
    # 保存配置
    if args.save_config:
        path = manager.save_config(config, args.save_config, 'yaml')
        print(f"配置已保存: {path}")
    
    # 验证配置
    if not ConfigValidator.validate_all(config):
        print("配置验证失败")
        return
    
    # 干运行
    if args.dry_run:
        print("\n配置信息:")
        print(config.to_yaml())
        print("\n✅ 配置验证通过（干运行模式）")
        return
    
    # 运行训练
    try:
        pipeline = TrainingPipeline(config)
        
        # 确定目标
        targets = None
        if args.target:
            targets = [t.strip() for t in args.target.split(',')]
        
        # 执行训练
        results = pipeline.run(targets)
        
        print("\n✨ 训练完成!")
        
        # 保存最终配置
        if config.logging.auto_save:
            final_config_path = (
                Path(config.logging.base_dir) / 
                config.logging.project_name / 
                "experiment_config.yaml"
            )
            config.to_yaml(str(final_config_path))
            print(f"实验配置已保存: {final_config_path}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
