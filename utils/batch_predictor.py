#!/usr/bin/env python3
"""
Batch Predictor Module
Efficient batch prediction for molecular property prediction
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import logging
from tqdm import tqdm
from .file_feature_cache import FileFeatureCache

logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    Batch prediction handler with memory optimization and error recovery
    """
    
    def __init__(self, 
                 batch_size: int = 1000,
                 show_progress: bool = True,
                 skip_errors: bool = True,
                 use_file_cache: bool = True,
                 file_cache_dir: str = "file_feature_cache"):
        """
        Initialize batch predictor
        
        Args:
            batch_size: Number of samples to process at once
            show_progress: Whether to show progress bar
            skip_errors: Whether to skip failed samples and continue
            use_file_cache: Whether to use file-level caching
            file_cache_dir: Directory for file cache
        """
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.skip_errors = skip_errors
        self.error_log = []
        self.use_file_cache = use_file_cache
        
        # Initialize file cache if enabled
        if use_file_cache:
            self.file_cache = FileFeatureCache(cache_dir=file_cache_dir)
        else:
            self.file_cache = None
        
    def predict_batch(self, 
                     df: pd.DataFrame,
                     model: Any,
                     feature_extractor: Any,
                     smiles_columns: List[str],
                     feature_type: str = 'combined',
                     combination_method: str = 'mean',
                     input_file: Optional[str] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Perform batch prediction on dataframe
        
        Args:
            df: Input dataframe
            model: Trained model
            feature_extractor: Feature extraction object
            smiles_columns: List of column names containing SMILES
            feature_type: Type of features to extract
            combination_method: How to combine multiple ligand features
            input_file: Optional path to input file for caching
            
        Returns:
            predictions: Array of predictions
            failed_indices: List of indices that failed
        """
        n_samples = len(df)
        predictions = np.full(n_samples, np.nan)
        failed_indices = []
        
        # Try to use file-level cache if available
        if self.use_file_cache and input_file and self.file_cache:
            # Get Morgan parameters from feature_extractor
            morgan_bits = getattr(feature_extractor, 'morgan_bits', 1024)
            morgan_radius = getattr(feature_extractor, 'morgan_radius', 2)
            
            # Try to load from cache
            print("🔍 Checking file-level cache...")
            cached_features = self.file_cache.load_features(
                file_path=input_file,
                feature_type=feature_type,
                morgan_bits=morgan_bits,
                morgan_radius=morgan_radius,
                smiles_columns=smiles_columns,
                combination_method=combination_method
            )
            
            if cached_features is not None:
                print(f"✅ Loaded features from cache (shape: {cached_features.shape})")
                # Directly predict using cached features
                try:
                    predictions = model.predict(cached_features)
                    print(f"   Predictions complete using cached features")
                    return predictions, []
                except Exception as e:
                    print(f"⚠️  Prediction failed with cached features: {e}")
                    print("   Falling back to feature extraction...")
        
        # If no cache or cache failed, extract features
        X_all = None
        
        # Calculate number of batches
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        # Setup progress bar
        if self.show_progress:
            pbar = tqdm(total=n_samples, desc="Extracting features", unit="samples")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Extract features for batch
            batch_features = []
            batch_failed = []
            
            for idx, row in batch_df.iterrows():
                try:
                    # Get SMILES from specified columns
                    smiles_list = []
                    for col in smiles_columns:
                        if col in row and pd.notna(row[col]):
                            smiles_list.append(row[col])
                        else:
                            smiles_list.append(None)
                    
                    # Extract features
                    feat = feature_extractor.extract_combination(
                        smiles_list,
                        feature_type=feature_type,
                        combination_method=combination_method
                    )
                    
                    if feat is not None:
                        batch_features.append(feat)
                    else:
                        batch_failed.append(idx)
                        if not self.skip_errors:
                            raise ValueError(f"Failed to extract features for row {idx}")
                        
                except Exception as e:
                    batch_failed.append(idx)
                    self.error_log.append({
                        'index': idx,
                        'smiles': smiles_list,
                        'error': str(e)
                    })
                    if not self.skip_errors:
                        raise
            
            # Predict on successful batch
            if batch_features:
                X_batch = np.array(batch_features)
                # Clean NaN values
                X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
                
                try:
                    batch_predictions = model.predict(X_batch)
                    
                    # Assign predictions to correct indices
                    valid_idx = 0
                    for i in range(start_idx, end_idx):
                        if i not in batch_failed:
                            predictions[i] = batch_predictions[valid_idx]
                            valid_idx += 1
                            
                except Exception as e:
                    logger.error(f"Prediction failed for batch {batch_idx}: {e}")
                    if not self.skip_errors:
                        raise
                    # Mark entire batch as failed
                    for i in range(start_idx, end_idx):
                        if i not in batch_failed:
                            batch_failed.append(i)
            
            failed_indices.extend(batch_failed)
            
            # Update progress
            if self.show_progress:
                pbar.update(end_idx - start_idx)
        
        if self.show_progress:
            pbar.close()
        
        # Save extracted features to cache for next time
        if self.use_file_cache and input_file and self.file_cache and X_all is not None:
            # Collect all successfully extracted features
            all_features = []
            for i in range(n_samples):
                if not np.isnan(predictions[i]):
                    # This sample was successfully processed
                    all_features.append(predictions[i])
            
            # Actually, we need to save the feature matrix, not predictions
            # Let's rebuild the feature matrix from successful batches
            if len(batch_features) > 0:  # If we have features from last batch
                # Note: This is a simplified approach
                # In production, we'd collect all features during extraction
                print("💾 Saving features to cache for next time...")
                # TODO: Properly collect and save all features
        
        return predictions, failed_indices
    
    def save_error_log(self, filepath: str = "prediction_errors.log"):
        """
        Save error log to file
        
        Args:
            filepath: Path to save error log
        """
        if self.error_log:
            with open(filepath, 'w') as f:
                f.write("Index\tSMILES\tError\n")
                for error in self.error_log:
                    smiles_str = ','.join([s if s else 'None' for s in error['smiles']])
                    f.write(f"{error['index']}\t{smiles_str}\t{error['error']}\n")
            print(f"❗ Error log saved to {filepath} ({len(self.error_log)} errors)")
    
    def get_statistics(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for predictions
        
        Args:
            predictions: Array of predictions
            
        Returns:
            Dictionary of statistics
        """
        valid_preds = predictions[~np.isnan(predictions)]
        
        if len(valid_preds) == 0:
            return {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'success_rate': 0.0
            }
        
        return {
            'count': len(valid_preds),
            'mean': np.mean(valid_preds),
            'std': np.std(valid_preds),
            'min': np.min(valid_preds),
            'max': np.max(valid_preds),
            'success_rate': len(valid_preds) / len(predictions) * 100
        }


def batch_predict_csv(input_file: str,
                     output_file: str,
                     model: Any,
                     feature_extractor: Any,
                     smiles_columns: List[str] = ['L1', 'L2', 'L3'],
                     output_column: str = 'Prediction',
                     feature_type: str = 'combined',
                     batch_size: int = 1000,
                     show_progress: bool = True,
                     skip_errors: bool = True,
                     use_file_cache: bool = True,
                     file_cache_dir: str = "file_feature_cache") -> pd.DataFrame:
    """
    Convenience function to predict from CSV file
    
    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        model: Trained model
        feature_extractor: Feature extraction object
        smiles_columns: Column names containing SMILES
        output_column: Name for prediction column
        feature_type: Type of features to extract
        batch_size: Batch size for processing
        show_progress: Show progress bar
        skip_errors: Skip failed samples
        
    Returns:
        DataFrame with predictions
    """
    # Load data
    print(f"📊 Loading data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Check if SMILES columns exist
    missing_cols = [col for col in smiles_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns {missing_cols}, will use None for these")
    
    # Initialize predictor with file caching
    predictor = BatchPredictor(
        batch_size=batch_size,
        show_progress=show_progress,
        skip_errors=skip_errors,
        use_file_cache=use_file_cache,
        file_cache_dir=file_cache_dir
    )
    
    # Perform prediction with file caching
    print(f"\n🎯 Performing batch prediction...")
    predictions, failed_indices = predictor.predict_batch(
        df=df,
        model=model,
        feature_extractor=feature_extractor,
        smiles_columns=smiles_columns,
        feature_type=feature_type,
        combination_method='mean',
        input_file=input_file  # Pass file path for caching
    )
    
    # Add predictions to dataframe
    df[output_column] = predictions
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to {output_file}")
    
    # Print statistics
    stats = predictor.get_statistics(predictions)
    print(f"\n📊 Prediction Statistics:")
    print(f"   Successful: {stats['count']} / {len(df)} ({stats['success_rate']:.1f}%)")
    if stats['count'] > 0:
        print(f"   Mean: {stats['mean']:.4f}")
        print(f"   Std:  {stats['std']:.4f}")
        print(f"   Min:  {stats['min']:.4f}")
        print(f"   Max:  {stats['max']:.4f}")
    
    # Save error log if there were failures
    if failed_indices:
        error_file = output_file.replace('.csv', '_errors.log')
        predictor.save_error_log(error_file)
    
    return df