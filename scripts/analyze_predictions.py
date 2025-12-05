#!/usr/bin/env python3
"""
Prediction analysis script - generate PLQY range accuracy heatmap and other plots
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
import json
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error
import glob

# Add parent directory to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_predictions(project_dir, model_name=None):
    """
    Load prediction results
    
    Args:
        project_dir: project directory
        model_name: model name, use best if None
    
    Returns:
        Dict containing actual and predicted arrays
    """
    project_path = Path(project_dir)
    
    if not project_path.exists():
        print(f"ERROR: Project directory not found: {project_dir}")
        return None
    
    # If no model specified, default to xgboost
    if model_name is None:
        model_name = 'xgboost'
    
    model_dir = project_path / model_name
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        # Try other possible directories
        for possible_name in ['xgboost', 'lightgbm', 'catboost', 'gradient_boosting']:
            model_dir = project_path / possible_name
            if model_dir.exists():
                print(f"INFO: Using model: {possible_name}")
                break
        else:
            print("ERROR: No model directories found")
            return None
    
    # Find prediction files
    predictions_dir = model_dir / 'predictions'
    if not predictions_dir.exists():
        print(f"ERROR: Predictions directory not found: {predictions_dir}")
        return None

    # Collect prediction results for all targets
    all_predictions = {}

    # Collect data by target type across folds
    target_types = {'wavelength': [], 'PLQY': [], 'tau': []}

    # Find CSV files
    csv_files = list(predictions_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        # Extract target type from filename
        filename = csv_file.stem
        
        # Determine target type
        target_type = None
        if 'wavelength' in filename.lower():
            target_type = 'wavelength'
        elif 'plqy' in filename.lower():
            target_type = 'PLQY'
        elif 'tau' in filename.lower():
            target_type = 'tau'
        else:
            continue
        
        # Read prediction data
        try:
            df = pd.read_csv(csv_file)
            
            # Find actual and predicted columns
            actual_col = None
            pred_col = None
            
            for col in df.columns:
                if 'actual' in col.lower() or 'true' in col.lower() or 'experimental' in col.lower():
                    actual_col = col
                elif 'predict' in col.lower() or 'pred' in col.lower():
                    pred_col = col
            
            if actual_col and pred_col:
                # Use subset if split column exists (prefer test, then val)
                if 'split' in df.columns:
                    if 'test' in df['split'].values:
                        test_df = df[df['split'] == 'test']
                    elif 'val' in df['split'].values:
                        test_df = df[df['split'] == 'val']
                    else:
                        # Otherwise use all data
                        test_df = df
                else:
                    test_df = df
                
                if len(test_df) > 0:
                    target_types[target_type].append({
                        'actual': test_df[actual_col].values,
                        'predicted': test_df[pred_col].values
                    })
        except Exception as e:
            print(f"WARNING: Failed to read {csv_file}: {e}")
    
    # Merge data across folds
    for target_type in ['wavelength', 'PLQY', 'tau']:
        if target_types[target_type]:
            actual_all = np.concatenate([d['actual'] for d in target_types[target_type]])
            predicted_all = np.concatenate([d['predicted'] for d in target_types[target_type]])
            
            all_predictions[target_type] = {
                'actual': actual_all,
                'predicted': predicted_all
            }
            print(f"INFO: Loaded {target_type} predictions: {len(actual_all)} samples")
    
    return all_predictions

def plot_plqy_range_accuracy(predictions, output_dir):
    """
    Plot PLQY-range prediction accuracy heatmap
    
    Args:
        predictions: dict containing actual and predicted
        output_dir: output directory
    """
    if 'PLQY' not in predictions:
        print("WARNING: No PLQY prediction data")
        return
    
    actual = predictions['PLQY']['actual']
    predicted = predictions['PLQY']['predicted']
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Define PLQY ranges
    bins = [0, 0.1, 0.5, 1.0]
    labels = ['0-0.1', '0.1-0.5', '0.5-1.0']
    
    # Bin actual and predicted values
    actual_binned = pd.cut(actual, bins=bins, labels=labels, include_lowest=True)
    predicted_binned = pd.cut(predicted, bins=bins, labels=labels, include_lowest=True)
    
    # Remove NaNs after binning (out-of-range)
    mask2 = ~(actual_binned.isna() | predicted_binned.isna())
    actual_binned = actual_binned[mask2]
    predicted_binned = predicted_binned[mask2]
    
    # Create confusion matrix
    cm = confusion_matrix(actual_binned, predicted_binned, labels=labels)
    
    # Normalize to percentages per row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Use blue palette
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Draw heatmap
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f',
                cmap=cmap,
                vmin=0, 
                vmax=1,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Accuracy'},
                ax=ax)
    
    ax.set_xlabel('Predicted PLQY Range', fontsize=12)
    ax.set_ylabel('Actual PLQY Range', fontsize=12)
    ax.set_title('PLQY Prediction Accuracy by Range', fontsize=14)
    
    # Layout
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / 'plqy_range_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"INFO: Saved PLQY range accuracy plot: {save_path}")
    
    # Print statistics
    print("\nPLQY range prediction stats:")
    print("-" * 40)
    for i, actual_label in enumerate(labels):
        total = cm[i].sum()
        if total > 0:
            accuracy = cm[i, i] / total
            print(f"{actual_label}: {accuracy:.2%} accuracy ({cm[i, i]}/{total} samples)")
    
    # Overall accuracy
    overall_accuracy = np.trace(cm) / cm.sum()
    print(f"\nOverall accuracy: {overall_accuracy:.2%}")

def plot_prediction_scatter_all(predictions, output_dir):
    """
    Plot prediction scatter for all targets
    """
    n_targets = len(predictions)
    if n_targets == 0:
        print("WARNING: No prediction data")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))
    
    if n_targets == 1:
        axes = [axes]
    
    target_names = {
        'wavelength': 'Wavelength (nm)',
        'PLQY': 'PLQY', 
        'tau': 'Lifetime (us)'
    }
    
    for idx, (target, data) in enumerate(predictions.items()):
        ax = axes[idx]
        
        actual = data['actual']
        predicted = data['predicted']
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Compute metrics
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        
        # Scatter plot
        ax.scatter(actual, predicted, alpha=0.5, s=20, c='#1f77b4')
        
        # Diagonal line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', lw=1, alpha=0.7, label='Perfect prediction')
        
        # Labels
        display_name = target_names.get(target, target)
        ax.set_xlabel(f'Actual {display_name}', fontsize=11)
        ax.set_ylabel(f'Predicted {display_name}', fontsize=11)
        ax.set_title(f'{display_name} Prediction', fontsize=12)
        
        # Add metric text
        ax.text(0.05, 0.95, f'R^2 = {r2:.3f}\nMAE = {mae:.2f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    plt.suptitle('Model Prediction Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / 'prediction_scatter_all.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"INFO: Saved prediction scatter: {save_path}")

def plot_residual_analysis(predictions, output_dir):
    """
    Plot residual analysis figures
    """
    for target, data in predictions.items():
        actual = data['actual']
        predicted = data['predicted']
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Residuals
        residuals = predicted - actual
        
        # Create figures
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs predicted
        ax = axes[0, 0]
        ax.scatter(predicted, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # 2. Residual histogram
        ax = axes[0, 1]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        ax = axes[1, 0]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # 4. Residuals vs actual
        ax = axes[1, 1]
        ax.scatter(actual, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Actual')
        ax.grid(True, alpha=0.3)
        
        target_names = {
            'wavelength': 'Wavelength',
            'PLQY': 'PLQY',
            'tau': 'Lifetime'
        }
        display_name = target_names.get(target, target)
        plt.suptitle(f'Residual Analysis - {display_name}', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / f'residual_analysis_{target}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"INFO: Saved residual analysis: {save_path}")

def generate_prediction_report(predictions, output_dir):
    """
    Generate prediction analysis report
    """
    report = {}
    
    for target, data in predictions.items():
        actual = data['actual']
        predicted = data['predicted']
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Compute metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        
        metrics = {
            'n_samples': len(actual),
            'r2_score': r2_score(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mae': mean_absolute_error(actual, predicted),
            'mape': mean_absolute_percentage_error(actual, predicted) * 100,
            'residual_mean': np.mean(predicted - actual),
            'residual_std': np.std(predicted - actual),
            'actual_range': [float(actual.min()), float(actual.max())],
            'predicted_range': [float(predicted.min()), float(predicted.max())]
        }
        
        report[target] = metrics
    
    # Save report
    report_file = output_dir / 'prediction_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    
    print(f"INFO: Saved prediction report: {report_file}")
    
    # Print report summary
    print("\n" + "=" * 60)
    print("Prediction performance report")
    print("=" * 60)
    
    target_names = {
        'wavelength': 'Wavelength (nm)',
        'PLQY': 'PLQY',
        'tau': 'Lifetime (us)'
    }
    
    for target, metrics in report.items():
        display_name = target_names.get(target, target)
        print(f"\n{display_name}:")
        print("-" * 40)
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  R^2 Score: {metrics['r2_score']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  Actual range: [{metrics['actual_range'][0]:.2f}, {metrics['actual_range'][1]:.2f}]")
        print(f"  Predicted range: [{metrics['predicted_range'][0]:.2f}, {metrics['predicted_range'][1]:.2f}]")

def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description='Prediction analysis')
    
    parser.add_argument('project', help='Project directory')
    parser.add_argument('--model', '-m', help='Model name (default best)')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--plots', nargs='+',
                       choices=['range', 'scatter', 'residual', 'all'],
                       default=['all'],
                       help='Plot types to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.project) / 'analysis'
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Prediction analysis")
    print("=" * 60)
    print(f"Project: {args.project}")
    print(f"Output dir: {output_dir}")
    
    # Load predictions
    predictions = load_predictions(args.project, args.model)
    
    if not predictions:
        print("ERROR: Failed to load predictions")
        return
    
    # Generate plots
    if 'all' in args.plots or 'range' in args.plots:
        plot_plqy_range_accuracy(predictions, output_dir)
    
    if 'all' in args.plots or 'scatter' in args.plots:
        plot_prediction_scatter_all(predictions, output_dir)
    
    if 'all' in args.plots or 'residual' in args.plots:
        plot_residual_analysis(predictions, output_dir)
    
    # Generate report
    generate_prediction_report(predictions, output_dir)
    
    print("\n" + "=" * 60)
    print("INFO: Analysis completed!")
    print(f"Saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
