#!/usr/bin/env python3
"""
XGBoost训练器模块
支持数据加载、模型训练、10折交叉验证
"""

# ========================================
#           全局配置参数
# ========================================

# 数据配置
DEFAULT_DATA_PATH = "data/Database_normalized.csv"  # 默认数据路径
SMILES_COLUMNS = ['L1', 'L2', 'L3']       # SMILES列名
TARGET_COLUMNS = ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)']  # 目标列名

# 特征配置
FEATURE_TYPE = 'combined'  # 特征类型: 'morgan', 'descriptors', 'combined'
USE_CACHE = True           # 是否使用特征缓存

# 训练配置
N_FOLDS = 10              # 交叉验证折数
RANDOM_STATE = 42         # 随机种子
TEST_SIZE = 0.2           # 测试集比例（仅用于train_test_split模式）

# XGBoost默认参数
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 1
}

# 输出配置
SAVE_MODELS = True        # 是否保存模型
MODEL_DIR = "models"      # 模型保存目录
RESULTS_DIR = "results"   # 结果保存目录
SAVE_PREDICTIONS = True   # 是否保存预测结果

# ========================================
#           导入依赖库
# ========================================

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入特征提取模块
from core.feature_extractor import FeatureExtractor, MORGAN_BITS, DESCRIPTOR_NAMES

# 导入模型模块
from models.base import XGBoostTrainer, ModelFactory, generate_model_filename, evaluate_model

# 忽略警告
warnings.filterwarnings('ignore')

# ========================================
#           数据加载和预处理
# ========================================

class DataLoader:
    """数据加载器类"""
    
    def __init__(self, data_path: str = None, feature_type: str = None, use_cache: bool = None):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
            feature_type: 特征类型
            use_cache: 是否使用缓存
        """
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.feature_type = feature_type or FEATURE_TYPE
        self.use_cache = use_cache if use_cache is not None else USE_CACHE
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(use_cache=self.use_cache)
        
        print(f"✅ 数据加载器初始化")
        print(f"   数据路径: {self.data_path}")
        print(f"   特征类型: {self.feature_type}")
        print(f"   使用缓存: {self.use_cache}")
    
    def load_data(self) -> pd.DataFrame:
        """加载CSV数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"\n📊 加载数据: {len(df)} 行, {len(df.columns)} 列")
        
        # 检查必需的列
        missing_cols = []
        for col in SMILES_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"缺少SMILES列: {missing_cols}")
        
        # 检查目标列
        available_targets = [col for col in TARGET_COLUMNS if col in df.columns]
        if not available_targets:
            raise ValueError(f"没有找到任何目标列: {TARGET_COLUMNS}")
        
        print(f"   SMILES列: {SMILES_COLUMNS}")
        print(f"   目标列: {available_targets}")
        
        return df
    
    def extract_features(self, df: pd.DataFrame, show_progress: bool = True) -> np.ndarray:
        """
        提取特征
        
        Args:
            df: 数据框
            show_progress: 是否显示进度条
        
        Returns:
            特征矩阵
        """
        print(f"\n🔧 提取{self.feature_type}特征...")
        
        features = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="提取特征") if show_progress else df.iterrows()
        
        for idx, row in iterator:
            # 获取三个SMILES
            smiles_list = [row[col] if pd.notna(row[col]) else None for col in SMILES_COLUMNS]
            
            # 提取组合特征
            feat = self.feature_extractor.extract_combination(
                smiles_list, 
                feature_type=self.feature_type,
                combination_method='mean'
            )
            
            features.append(feat)
        
        features = np.array(features)
        print(f"   特征维度: {features.shape}")
        
        # 处理NaN和Inf
        n_nan = np.isnan(features).sum()
        n_inf = np.isinf(features).sum()
        
        if n_nan > 0 or n_inf > 0:
            print(f"   ⚠️ 发现 {n_nan} 个NaN值, {n_inf} 个Inf值")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def prepare_data(self, target_col: str) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """
        准备训练数据
        
        Args:
            target_col: 目标列名
        
        Returns:
            (特征矩阵, 目标值, 有效索引)
        """
        # 加载数据
        df = self.load_data()
        
        # 筛选有效数据
        valid_mask = df[target_col].notna()
        for col in SMILES_COLUMNS:
            valid_mask &= df[col].notna()
        
        df_valid = df[valid_mask].copy()
        print(f"\n📊 目标变量: {target_col}")
        print(f"   有效样本: {len(df_valid)}/{len(df)}")
        
        if len(df_valid) == 0:
            raise ValueError(f"没有有效的训练数据")
        
        # 提取特征
        X = self.extract_features(df_valid)
        
        # 获取目标值
        y = df_valid[target_col].values
        
        # 单位转换（如果PLQY是百分比）
        if target_col == 'PLQY' and y.max() > 1.5:
            print("   转换PLQY: 百分比 → 小数")
            y = y / 100
        
        return X, y, df_valid.index

# ========================================
#           结果保存辅助函数
# ========================================

def save_training_results(results: Dict, target_col: str, model_type: str, n_folds: int):
    """
    保存训练结果
    
    Args:
        results: 结果字典
        target_col: 目标列名
        model_type: 模型类型
        n_folds: 交叉验证折数
    """
    if not SAVE_PREDICTIONS:
        return
    
    # 创建输出目录
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存CV预测结果
    cv_df = pd.DataFrame({
        'true': results['true_values'],
        'predicted': results['predictions'],
        'error': results['true_values'] - results['predictions']
    })
    
    csv_file = Path(RESULTS_DIR) / f"cv_predictions_{model_type}_{target_col}_{timestamp}.csv"
    cv_df.to_csv(csv_file, index=False)
    print(f"   💾 预测结果已保存: {csv_file}")
    
    # 保存评估指标
    metrics = {
        'target': target_col,
        'model_type': model_type,
        'n_samples': len(results['true_values']),
        'n_folds': n_folds,
        'mean_rmse': float(results['mean_rmse']),
        'std_rmse': float(results['std_rmse']),
        'mean_mae': float(results['mean_mae']),
        'std_mae': float(results['std_mae']),
        'mean_r2': float(results['mean_r2']),
        'std_r2': float(results['std_r2']),
        'mean_mape': float(results['mean_mape']) if not np.isnan(results['mean_mape']) else None,
        'std_mape': float(results['std_mape']) if not np.isnan(results['std_mape']) else None,
        'timestamp': timestamp
    }
    
    json_file = Path(RESULTS_DIR) / f"cv_metrics_{model_type}_{target_col}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   💾 评估指标已保存: {json_file}")

# ========================================
#           主训练流程
# ========================================

class Trainer:
    """主训练器类"""
    
    def __init__(self, data_path: str = None, feature_type: str = None):
        """
        初始化主训练器
        
        Args:
            data_path: 数据路径
            feature_type: 特征类型
        """
        self.data_loader = DataLoader(data_path, feature_type)
        self.xgb_trainer = XGBoostTrainer()
        self.results = {}
    
    def train_target(self, target_col: str):
        """
        训练单个目标
        
        Args:
            target_col: 目标列名
        """
        print(f"\n{'='*60}")
        print(f"训练目标: {target_col}")
        print(f"{'='*60}")
        
        try:
            # 准备数据
            X, y, idx = self.data_loader.prepare_data(target_col)
            
            # 交叉验证
            cv_results = self.xgb_trainer.train_cv(X, y)
            
            # 训练最终模型
            final_model = self.xgb_trainer.train_full(X, y)
            
            # 保存模型和结果
            self.xgb_trainer.save_model(final_model, target_col, "_final")
            self.xgb_trainer.save_results(cv_results, target_col)
            
            # 保存结果
            self.results[target_col] = {
                'cv_results': cv_results,
                'final_model': final_model,
                'n_samples': len(y)
            }
            
            print(f"\n✅ {target_col} 训练完成!")
            
        except Exception as e:
            print(f"\n❌ {target_col} 训练失败: {e}")
            self.results[target_col] = {'error': str(e)}
    
    def train_all(self):
        """训练所有目标"""
        # 加载数据以确定可用的目标
        df = self.data_loader.load_data()
        available_targets = [col for col in TARGET_COLUMNS if col in df.columns]
        
        print(f"\n🎯 将训练 {len(available_targets)} 个目标: {available_targets}")
        
        for target in available_targets:
            self.train_target(target)
        
        # 汇总结果
        self.print_summary()
    
    def print_summary(self):
        """打印训练汇总"""
        print(f"\n{'='*60}")
        print("训练汇总")
        print(f"{'='*60}")
        
        for target, result in self.results.items():
            if 'error' in result:
                print(f"\n❌ {target}: 失败 - {result['error']}")
            else:
                cv_res = result['cv_results']
                print(f"\n✅ {target}:")
                print(f"   样本数: {result['n_samples']}")
                print(f"   RMSE: {cv_res['mean_rmse']:.4f} ± {cv_res['std_rmse']:.4f}")
                print(f"   R²:   {cv_res['mean_r2']:.4f} ± {cv_res['std_r2']:.4f}")

# ========================================
#           命令行接口
# ========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='XGBoost训练器 - 支持10折交叉验证',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data', '-d', type=str, default=DEFAULT_DATA_PATH,
                       help=f'数据文件路径 (默认: {DEFAULT_DATA_PATH})')
    parser.add_argument('--target', '-t', type=str,
                       help='指定目标列（不指定则训练所有）')
    parser.add_argument('--feature', '-f', type=str, default=FEATURE_TYPE,
                       choices=['morgan', 'descriptors', 'combined'],
                       help=f'特征类型 (默认: {FEATURE_TYPE})')
    parser.add_argument('--folds', '-k', type=int,
                       help=f'交叉验证折数 (默认: {N_FOLDS})')
    parser.add_argument('--no-cache', action='store_true',
                       help='不使用特征缓存')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存模型和结果')
    
    args = parser.parse_args()
    
    # 设置参数
    use_cache = not args.no_cache
    save_models = not args.no_save
    save_predictions = not args.no_save
    n_folds = args.folds if args.folds else N_FOLDS
    
    print("="*60)
    print("XGBoost 分子性质预测训练器")
    print("="*60)
    print(f"\n配置:")
    print(f"  数据文件: {args.data}")
    print(f"  特征类型: {args.feature}")
    print(f"  交叉验证: {n_folds}折")
    print(f"  使用缓存: {use_cache}")
    print(f"  保存结果: {save_models}")
    
    # 创建数据加载器（传入use_cache参数）
    data_loader = DataLoader(args.data, args.feature, use_cache)
    
    # 创建XGBoost训练器（传入n_folds参数）
    xgb_trainer = XGBoostTrainer(n_folds=n_folds)
    
    # 根据save参数设置
    if not save_models:
        xgb_trainer.save_model = lambda *args, **kwargs: None
        xgb_trainer.save_results = lambda *args, **kwargs: None
    
    # 创建主训练器
    trainer = Trainer(args.data, args.feature)
    trainer.data_loader = data_loader
    trainer.xgb_trainer = xgb_trainer
    
    # 训练
    if args.target:
        trainer.train_target(args.target)
    else:
        trainer.train_all()
    
    print("\n✨ 训练完成!")

if __name__ == "__main__":
    main()