#!/usr/bin/env python3
"""
XGBoost trainer module
Supports data loading, model training, and 10-fold cross-validation
"""

# ========================================
#           Global configuration
# ========================================

# Data configuration
DEFAULT_DATA_PATH = "data/PhosIrDB.csv"  # default data path
SMILES_COLUMNS = ['L1', 'L2', 'L3']       # SMILES columns
TARGET_COLUMNS = ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)']  # target columns

# Feature configuration
FEATURE_TYPE = 'combined'  # Feature type: 'morgan', 'descriptors', 'combined'
USE_CACHE = True           # Use feature cache

# Training configuration
N_FOLDS = 10              # number of cross-validation folds
RANDOM_STATE = 42         # random seed
TEST_SIZE = 0.2           # test split ratio (only for train_test_split mode)

# Default XGBoost parameters
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

# Output configuration
SAVE_MODELS = True        # save models
MODEL_DIR = "models"      # model save directory
RESULTS_DIR = "results"   # results directory
SAVE_PREDICTIONS = True   # save predictions

# ========================================
#           Imports
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import feature extractor
from core.feature_extractor import FeatureExtractor, MORGAN_BITS, DESCRIPTOR_NAMES

# Import model module
from models.base import XGBoostTrainer, ModelFactory, generate_model_filename, evaluate_model

# Suppress warnings
warnings.filterwarnings('ignore')

# ========================================
#           Data loading and preprocessing
# ========================================

class DataLoader:
    """Data loader class"""
    
    def __init__(self, data_path: str = None, feature_type: str = None, use_cache: bool = None):
        """
        Initialize data loader
        
        Args:
            data_path: data file path
            feature_type: feature type
            use_cache: whether to use cache
        """
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.feature_type = feature_type or FEATURE_TYPE
        self.use_cache = use_cache if use_cache is not None else USE_CACHE
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(use_cache=self.use_cache)
        
        print("DataLoader initialized")
        print(f"   Data path: {self.data_path}")
        print(f"   Feature type: {self.feature_type}")
        print(f"   Use cache: {self.use_cache}")
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"\nLoaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        missing_cols = []
        for col in SMILES_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Missing SMILES columns: {missing_cols}")
        
        # Check target columns
        available_targets = [col for col in TARGET_COLUMNS if col in df.columns]
        if not available_targets:
            raise ValueError(f"No target columns found: {TARGET_COLUMNS}")
        
        print(f"   SMILES columns: {SMILES_COLUMNS}")
        print(f"   Target columns: {available_targets}")
        
        return df
    
    def extract_features(self, df: pd.DataFrame, show_progress: bool = True) -> np.ndarray:
        """
        Extract features
        
        Args:
            df: DataFrame
            show_progress: whether to show progress bar
        
        Returns:
            Feature matrix
        """
        print(f"\nExtracting {self.feature_type} features...")
        
        features = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="Extract features") if show_progress else df.iterrows()
        
        for idx, row in iterator:
            # Get three SMILES
            smiles_list = [row[col] if pd.notna(row[col]) else None for col in SMILES_COLUMNS]
            
            # Extract combined features
            feat = self.feature_extractor.extract_combination(
                smiles_list, 
                feature_type=self.feature_type,
                combination_method='mean'
            )
            
            features.append(feat)
        
        features = np.array(features)
        print(f"   Feature shape: {features.shape}")
        
        # Handle NaN and Inf
        n_nan = np.isnan(features).sum()
        n_inf = np.isinf(features).sum()
        
        if n_nan > 0 or n_inf > 0:
            print(f"   Found {n_nan} NaN values, {n_inf} Inf values")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def prepare_data(self, target_col: str) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """
        Prepare training data
        
        Args:
            target_col: target column name
        
        Returns:
            (feature matrix, target values, valid index)
        """
        # Load data
        df = self.load_data()
        
        # Filter valid data
        valid_mask = df[target_col].notna()
        for col in SMILES_COLUMNS:
            valid_mask &= df[col].notna()
        
        df_valid = df[valid_mask].copy()
        print(f"\nTarget variable: {target_col}")
        print(f"   Valid samples: {len(df_valid)}/{len(df)}")
        
        if len(df_valid) == 0:
            raise ValueError("No valid training data")
        
        # Extract features
        X = self.extract_features(df_valid)
        
        # Get target values
        y = df_valid[target_col].values
        
        # Unit conversion (if PLQY is percentage)
        if target_col == 'PLQY' and y.max() > 1.5:
            print("   Convert PLQY: percentage -> decimal")
            y = y / 100
        
        return X, y, df_valid.index

# ========================================
#           Result saving helpers
# ========================================

def save_training_results(results: Dict, target_col: str, model_type: str, n_folds: int):
    """
    Save training results
    
    Args:
        results: results dict
        target_col: target column name
        model_type: model type
        n_folds: number of folds
    """
    if not SAVE_PREDICTIONS:
        return
    
    # Create output directory
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_target = (target_col
                    .replace('(', '_')
                    .replace(')', '')
                    .replace('/', '_')
                    .replace('*', 'x')
                    .replace('^', '')
                    .replace(' ', '_'))
    while '__' in clean_target:
        clean_target = clean_target.replace('__', '_')
    clean_target = clean_target.strip('_')
    
    # Save CV predictions
    cv_df = pd.DataFrame({
        'true': results['true_values'],
        'predicted': results['predictions'],
        'error': results['true_values'] - results['predictions']
    })
    
    csv_file = Path(RESULTS_DIR) / f"cv_predictions_{model_type}_{clean_target}.csv"
    cv_df.to_csv(csv_file, index=False)
    print(f"   Saved predictions: {csv_file}")
    
    # Save evaluation metrics
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
    
    json_file = Path(RESULTS_DIR) / f"cv_metrics_{model_type}_{clean_target}.json"
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics: {json_file}")

# ========================================
#           Main training flow
# ========================================

class Trainer:
    """Main trainer class"""
    
    def __init__(self, data_path: str = None, feature_type: str = None):
        """
        Initialize main trainer
        
        Args:
            data_path: data path
            feature_type: feature type
        """
        self.data_loader = DataLoader(data_path, feature_type)
        self.xgb_trainer = XGBoostTrainer()
        self.results = {}
    
    def train_target(self, target_col: str):
        """
        Train a single target
        
        Args:
            target_col: target column name
        """
        print(f"\n{'='*60}")
        print(f"Training target: {target_col}")
        print(f"{'='*60}")
        
        try:
            # Prepare data
            X, y, idx = self.data_loader.prepare_data(target_col)
            
            # Cross-validation
            cv_results = self.xgb_trainer.train_cv(X, y)
            
            # Train final model
            final_model = self.xgb_trainer.train_full(X, y)
            
            # Save model and results
            self.xgb_trainer.save_model(final_model, target_col, "_final")
            self.xgb_trainer.save_results(cv_results, target_col)
            
            # Save results
            self.results[target_col] = {
                'cv_results': cv_results,
                'final_model': final_model,
                'n_samples': len(y)
            }
            
            print(f"\n{target_col} training complete!")
            
        except Exception as e:
            print(f"\n{target_col} training failed: {e}")
            self.results[target_col] = {'error': str(e)}
    
    def train_all(self):
        """Train all targets"""
        # Load data to determine available targets
        df = self.data_loader.load_data()
        available_targets = [col for col in TARGET_COLUMNS if col in df.columns]
        
        print(f"\nWill train {len(available_targets)} targets: {available_targets}")
        
        for target in available_targets:
            self.train_target(target)
        
        # Summarize results
        self.print_summary()
    
    def print_summary(self):
        """Print training summary"""
        print(f"\n{'='*60}")
        print("Training summary")
        print(f"{'='*60}")
        
        for target, result in self.results.items():
            if 'error' in result:
                print(f"\n{target}: failed - {result['error']}")
            else:
                cv_res = result['cv_results']
                print(f"\n{target}:")
                print(f"   Samples: {result['n_samples']}")
                print(f"   RMSE: {cv_res['mean_rmse']:.4f} +/- {cv_res['std_rmse']:.4f}")
                print(f"   R^2:   {cv_res['mean_r2']:.4f} +/- {cv_res['std_r2']:.4f}")

# ========================================
#           Command-line interface
# ========================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='XGBoost trainer - supports 10-fold cross-validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data', '-d', type=str, default=DEFAULT_DATA_PATH,
                       help=f'Data file path (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--target', '-t', type=str,
                       help='Specify target column (train all if not set)')
    parser.add_argument('--feature', '-f', type=str, default=FEATURE_TYPE,
                       choices=['morgan', 'descriptors', 'combined'],
                       help=f'Feature type (default: {FEATURE_TYPE})')
    parser.add_argument('--folds', '-k', type=int,
                       help=f'Number of folds (default: {N_FOLDS})')
    parser.add_argument('--no-cache', action='store_true',
                       help='Do not use feature cache')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save models and results')
    
    args = parser.parse_args()
    
    # Set parameters
    use_cache = not args.no_cache
    save_models = not args.no_save
    save_predictions = not args.no_save
    n_folds = args.folds if args.folds else N_FOLDS
    
    print("="*60)
    print("XGBoost molecular property prediction trainer")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Data file: {args.data}")
    print(f"  Feature type: {args.feature}")
    print(f"  Cross-validation: {n_folds} folds")
    print(f"  Use cache: {use_cache}")
    print(f"  Save results: {save_models}")
    
    # Create data loader (pass use_cache)
    data_loader = DataLoader(args.data, args.feature, use_cache)
    
    # Create XGBoost trainer (pass n_folds)
    xgb_trainer = XGBoostTrainer(n_folds=n_folds)
    
    # Configure by save parameter
    if not save_models:
        xgb_trainer.save_model = lambda *args, **kwargs: None
        xgb_trainer.save_results = lambda *args, **kwargs: None
    
    # Create main trainer
    trainer = Trainer(args.data, args.feature)
    trainer.data_loader = data_loader
    trainer.xgb_trainer = xgb_trainer
    
    # Train
    if args.target:
        trainer.train_target(args.target)
    else:
        trainer.train_all()
    
    print("\nTraining finished!")

if __name__ == "__main__":
    main()
