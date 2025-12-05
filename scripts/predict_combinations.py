#!/usr/bin/env python3
"""
Predict properties for combinations using trained models
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_extractor import FeatureExtractor

def load_models(project_dir, model_name='xgboost'):
    """Load trained models (supports AutoML paths and Paper_* autodiscovery)"""
    print("Loading models...")

    models = {}

    project_path = Path(project_dir)
    possible_dirs = [
        project_path / 'all_models' / 'automl_train' / model_name / 'models',
        project_path / model_name / 'models',
        project_path / 'models' / model_name,
    ]

    model_dir = None
    for d in possible_dirs:
        if d.exists():
            model_dir = d
            break

    if model_dir is None:
        root = project_path.parent if project_path.name == 'paper_table' else project_path
        candidates = []
        try:
            for d in root.glob('Paper_*'):
                mdir = d / 'all_models' / 'automl_train' / model_name / 'models'
                if mdir.exists():
                    candidates.append(mdir)
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_dir = candidates[0]
                print(f"INFO: Auto-switched to latest model directory: {model_dir}")
        except Exception:
            pass

    if model_dir is None or not model_dir.exists():
        print(f"ERROR: Model directory not found: {project_path}/{model_name}/models")
        return models

    print(f"INFO: Model directory: {model_dir}")

    for model_file in model_dir.glob("*.joblib"):
        filename = model_file.stem
        if 'wavelength' in filename.lower():
            models['wavelength'] = joblib.load(model_file)
            print(f"INFO: Wavelength model: {model_file.name}")
        elif 'plqy' in filename.lower():
            models['PLQY'] = joblib.load(model_file)
            print(f"INFO: PLQY model: {model_file.name}")

    print(f"INFO: Loaded {len(models)} models")
    return models

def extract_features_batch(df, feature_type='combined', batch_size=1000):
    """Extract features in batches"""
    print(f"\nINFO: Extracting features (batch size: {batch_size})...")
    
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True
    )
    
    n_samples = len(df)
    features_list = []
    valid_indices = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_df = df.iloc[i:batch_end]
        
        for idx, row in batch_df.iterrows():
            try:
                smiles_list = [row['L1'], row['L2'], row['L3']]
                features = extractor.extract_combination(smiles_list)
                
                if features is not None:
                    features_list.append(features)
                    valid_indices.append(idx)
            except Exception:
                continue
        
        if (i + batch_size) % 10000 == 0 or batch_end == n_samples:
            print(f"  Progress: {batch_end:,}/{n_samples:,} ({100*batch_end/n_samples:.1f}%)")
    
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"INFO: Extracted features: {len(X):,}")
        return X, df_valid
    else:
        return None, None

def predict_batch(models, X, df_valid, batch_size=10000):
    """Predict in batches"""
    print("\nINFO: Predicting properties...")
    
    predictions = {}
    
    for target, model in models.items():
        print(f"  Predict {target}...")
        
        n_samples = len(X)
        preds = []
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            batch_pred = model.predict(batch_X)
            preds.append(batch_pred)
        
        predictions[target] = np.concatenate(preds)
        
        print(f"    Range: [{predictions[target].min():.3f}, {predictions[target].max():.3f}]")
        print(f"    Mean: {predictions[target].mean():.3f}")
    
    # Add predictions to DataFrame
    if 'wavelength' in predictions:
        df_valid['Predicted_wavelength'] = predictions['wavelength']
    if 'PLQY' in predictions:
        df_valid['Predicted_PLQY'] = predictions['PLQY']
    
    return df_valid

def analyze_results(df):
    """Analyze prediction results"""
    print("\n" + "=" * 60)
    print("Prediction result analysis")
    print("-" * 60)
    
    if 'Predicted_PLQY' in df.columns:
        # Top 10 PLQY
        top10 = df.nlargest(10, 'Predicted_PLQY')
        print("\nTop 10 PLQY combinations:")
        for i, row in enumerate(top10.iterrows(), 1):
            idx, data = row
            print(f"\n  #{i}: PLQY = {data['Predicted_PLQY']:.4f}")
            if i <= 3:
                print(f"      L1/L2: {data['L1'][:50]}...")
                print(f"      L3: {data['L3'][:50]}...")
                if 'Predicted_wavelength' in df.columns:
                    print(f"      Wavelength = {data['Predicted_wavelength']:.1f} nm")
    
    if 'Predicted_wavelength' in df.columns:
        print(f"\nWavelength distribution:")
        print(f"  Min: {df['Predicted_wavelength'].min():.1f} nm")
        print(f"  Max: {df['Predicted_wavelength'].max():.1f} nm")
        print(f"  Mean: {df['Predicted_wavelength'].mean():.1f} nm")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Predict properties for combinations')
    parser.add_argument('--project', '-p',
                       default='paper_table',
                       help='Model project directory')
    parser.add_argument('--input', '-i', 
                       default=None,
                       help='Combination file (default: PROJECT/ir_assemble.csv)')
    parser.add_argument('--output', '-o',
                       default=None,
                       help='Output file (default: PROJECT/ir_assemble_predicted_all.csv)')
    parser.add_argument('--top', '-t', type=int, default=1000,
                       help='Save Top-N candidates')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch size')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Predict combination properties")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Resolve defaults for input/output based on project
    if args.input is None:
        args.input = f"{args.project}/ir_assemble.csv"
    if args.output is None:
        args.output = f"{args.project}/ir_assemble_predicted_all.csv"
    
    print(f"\nLoading combination file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"INFO: Loaded {len(df):,} combinations")
    
    # 2. Load models
    models = load_models(args.project)
    if not models:
        print("ERROR: No models found")
        return
    
    # 3. Extract features
    X, df_valid = extract_features_batch(df, batch_size=args.batch_size)
    if X is None:
        print("ERROR: Feature extraction failed")
        return
    
    # 4. Predict
    df_predicted = predict_batch(models, X, df_valid)
    
    # 5. Save results
    print(f"\nSaving predictions...")
    df_predicted.to_csv(args.output, index=False)
    print(f"INFO: Full results: {args.output}")
    
    # Save Top candidates
    if 'Predicted_PLQY' in df_predicted.columns:
        top_file = args.output.replace('.csv', f'_top{args.top}.csv')
        top_df = df_predicted.nlargest(args.top, 'Predicted_PLQY')
        top_df.to_csv(top_file, index=False)
        print(f"INFO: Top {args.top}: {top_file}")
    
    # 6. Analyze results
    analyze_results(df_predicted)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
