#!/usr/bin/env python3
"""
Generate submission-ready project figures (C-G)
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
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set plotting style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

def load_data(data_file):
    """Load original dataset"""
    df = pd.read_csv(data_file)
    return df

def load_predictions(project_dir, model_name='xgboost'):
    """Load prediction results.
    Preferred search order:
    1) `project_dir/<model>/predictions`
    2) `project_dir/all_models/automl_train/<model>/exports/csv/`
    3) `project_dir/automl_train/<model>/exports/csv/` (if `project_dir` already contains `all_models`)
    4) Fallback to `project_dir/predictions`
    """
    project_path = Path(project_dir)
    model_dir = project_path / model_name
    
    # Default path
    predictions_dir = model_dir / 'predictions'
    
    # If default path does not exist, try AutoML path
    if not model_dir.exists() or not predictions_dir.exists():
        # Check if project_path already contains all_models
        if project_path.name == 'all_models' or 'all_models' in project_path.parts:
            # Already under all_models: look under automl_train directly
            automl_dir = project_path / 'automl_train' / model_name / 'exports' / 'csv'
        else:
            # Otherwise, prepend all_models in the path
            automl_dir = project_path / 'all_models' / 'automl_train' / model_name / 'exports' / 'csv'
        
        if automl_dir.exists():
            predictions_dir = automl_dir
            print(f"INFO: Using AutoML predictions directory: {predictions_dir}")
        else:
            # Fallback: use unified predictions directory
            predictions_dir = project_path / 'predictions'
            if not predictions_dir.exists():
                print(f"WARNING: Prediction directory not found: {model_dir/'predictions'} or {predictions_dir} or {automl_dir}")
                return None
    all_predictions = {}
    target_types = {'wavelength': [], 'PLQY': [], 'tau': []}
    
    csv_files = list(predictions_dir.glob("*.csv"))
    
    # If AutoML directory, only select files containing 'all_predictions'
    if 'automl_train' in str(predictions_dir):
        csv_files = [f for f in csv_files if 'all_predictions' in f.name]
    
    for csv_file in csv_files:
        filename = csv_file.stem
        
        target_type = None
        if 'wavelength' in filename.lower() or 'Max_wavelength' in filename:
            target_type = 'wavelength'
        elif 'plqy' in filename.lower() or 'PLQY' in filename:
            target_type = 'PLQY'
        elif 'tau' in filename.lower():
            target_type = 'tau'
        else:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            
            actual_col = None
            pred_col = None
            
            # Prefer 'true' and 'predicted' columns (AutoML format)
            if 'true' in df.columns and 'predicted' in df.columns:
                actual_col = 'true'
                pred_col = 'predicted'
            else:
                for col in df.columns:
                    if 'true' in col.lower() or 'actual' in col.lower():
                        actual_col = col
                    elif 'predict' in col.lower():
                        pred_col = col
            
            if actual_col and pred_col:
                if 'split' in df.columns:
                    if 'test' in df['split'].values:
                        test_df = df[df['split'] == 'test']
                    elif 'val' in df['split'].values:
                        test_df = df[df['split'] == 'val']
                    else:
                        test_df = df
                else:
                    test_df = df
                
                if len(test_df) > 0:
                    target_types[target_type].append({
                        'actual': test_df[actual_col].values,
                        'predicted': test_df[pred_col].values
                    })
        except Exception as e:
            print(f"WARNING: Failed to read file {csv_file}: {e}")
    
    for target_type in ['wavelength', 'PLQY', 'tau']:
        if target_types[target_type]:
            actual_all = np.concatenate([d['actual'] for d in target_types[target_type]])
            predicted_all = np.concatenate([d['predicted'] for d in target_types[target_type]])
            
            all_predictions[target_type] = {
                'actual': actual_all,
                'predicted': predicted_all
            }
    
    # Extra: try loading test-set predictions (exports/test_predictions_*.csv) to override/supplement
    try:
        exports_dir = project_path / 'exports'
        if exports_dir.exists():
            test_files = list(exports_dir.glob('test_predictions_*.csv'))
            for tf in test_files:
                name = tf.stem.lower()
                target_type = None
                if 'wavelength' in name or 'max_wavelength' in name:
                    target_type = 'wavelength'
                elif 'plqy' in name:
                    target_type = 'PLQY'
                elif 'tau' in name:
                    target_type = 'tau'
                if target_type is None:
                    continue
                try:
                    df = pd.read_csv(tf)
                    # Prediction column
                    pred_col = 'prediction' if 'prediction' in df.columns else None
                    if pred_col is None:
                        continue
                    # Ground-truth column (if present)
                    candidate_actual_cols = [
                        'Max_wavelength(nm)', 'Max_wavelengthnm', 'wavelength',
                        'PLQY', 'tau(s*10^-6)', 'tausx10^-6', 'tau'
                    ]
                    actual_col = next((c for c in candidate_actual_cols if c in df.columns), None)
                    if actual_col is None:
                        # If no ground-truth column, skip this target (cannot plot scatter)
                        continue
                    actual = df[actual_col].values
                    predicted = df[pred_col].values
                    mask = ~(pd.isna(actual) | pd.isna(predicted))
                    actual = actual[mask]
                    predicted = predicted[mask]
                    all_predictions[target_type] = {
                        'actual': actual,
                        'predicted': predicted
                    }
                except Exception as e:
                    print(f"WARNING: Failed to read test predictions {tf}: {e}")
    except Exception:
        pass

    return all_predictions

def plot_figure_c(df, output_dir):
    """
    Figure C: Wavelength-PLQY scatter plot colored by solvent
    """
    print("Generating Figure C: Wavelength-PLQY scatter plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Define solvent types and colors
    solvent_colors = {
        'CH2Cl2': '#2E75B6',    # dark blue
        'CH3CN': '#70AD47',     # green
        'Toluene': '#FFC000',   # orange
        'Others': '#7030A0'     # purple
    }
    
    # Find wavelength and PLQY columns
    wavelength_col = None
    plqy_col = None
    
    for col in df.columns:
        if 'wavelength' in col.lower() and 'max' in col.lower():
            wavelength_col = col
        if 'plqy' in col.lower():
            plqy_col = col
    
    if wavelength_col and plqy_col:
        # Create scatter plot
        if 'Solvent' in df.columns:
            # With solvent information
            for solvent, color in solvent_colors.items():
                mask = df['Solvent'] == solvent
                if mask.sum() > 0:
                    ax.scatter(df.loc[mask, wavelength_col], 
                              df.loc[mask, plqy_col],
                              c=color, label=solvent, alpha=0.6, s=30, marker='s')
        else:
            # No solvent information; use default color
            ax.scatter(df[wavelength_col], df[plqy_col], 
                      alpha=0.6, s=30, c='#2E75B6', marker='s')
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PLQY', fontsize=12, fontweight='bold')
        ax.set_xlim(440, 880)
        ax.set_ylim(0, 1.0)
        
        # Set x-axis ticks
        ax.set_xticks([440, 550, 660, 770, 880])
        ax.set_xticklabels(['440 nm', '550 nm', '660 nm', '770 nm', '880 nm'])
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add label c
        ax.text(0.02, 0.98, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        plt.tight_layout()
        save_path = output_dir / 'figure_c_wavelength_plqy.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved: {save_path}")

        # Export data used for plotting
        try:
            data_out = df[[wavelength_col, plqy_col]].copy()
            if 'Solvent' in df.columns:
                data_out['Solvent'] = df['Solvent']
            data_out.to_csv(output_dir / 'figure_c_data.csv', index=False)
        except Exception:
            pass

def plot_figure_d(df, output_dir):
    """
    Figure D: PLQY distribution histogram (stacked bar)
    """
    print("Generating Figure D: PLQY distribution histogram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    plqy_col = None
    for col in df.columns:
        if 'plqy' in col.lower():
            plqy_col = col
            break
    
    if plqy_col:
        # Define PLQY ranges
        bins = [-0.001, 0.1, 0.5, 1.001]
        labels = ['<=0.1', '0.1-0.5', '>0.5']
        
        # Count entries in each range
        df['PLQY_range'] = pd.cut(df[plqy_col], bins=bins, labels=labels)
        
        if 'Solvent' in df.columns:
            # Define solvent colors
            solvent_colors = {
                'CH2Cl2': '#2E75B6',
                'CH3CN': '#70AD47',
                'Toluene': '#FFC000',
                'Others': '#7030A0'
            }
            
            # Build stacked data
            data_matrix = []
            solvents = ['CH2Cl2', 'CH3CN', 'Toluene', 'Others']
            
            for label in labels:
                row = []
                for solvent in solvents:
                    count = df[(df['PLQY_range'] == label) & (df['Solvent'] == solvent)].shape[0]
                    row.append(count)
                data_matrix.append(row)
            
            # Plot stacked bar chart
            x = np.arange(len(labels))
            width = 0.6
            bottom = np.zeros(len(labels))
            
            for i, solvent in enumerate(solvents):
                values = [data_matrix[j][i] for j in range(len(labels))]
                ax.bar(x, values, width, bottom=bottom, 
                       label=solvent, color=solvent_colors[solvent])
                bottom += values
        else:
            # Simple histogram
            counts = df['PLQY_range'].value_counts()[labels].fillna(0)
            ax.bar(range(len(labels)), counts.values, color='#2E75B6')
        
        ax.set_xlabel('PLQY Range', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of entries', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 800)
        
        if 'Solvent' in df.columns:
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Add label d
        ax.text(0.02, 0.98, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        plt.tight_layout()
        save_path = output_dir / 'figure_d_plqy_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved: {save_path}")

        # Export data used for plotting
        try:
            out_df = df[['PLQY_range']].copy()
            if 'Solvent' in df.columns:
                out_df['Solvent'] = df['Solvent']
            out_df.to_csv(output_dir / 'figure_d_data.csv', index=False)
        except Exception:
            pass

def plot_figure_e_f(predictions, output_dir):
    """
    Figures E and F: Predicted vs Experimental scatter plots
    """
    print("Generating Figures E and F: Predicted vs Experimental scatter plots...")
    
    if not predictions:
        print("WARNING: No prediction data")
        return
    
    # Create two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Figure e: wavelength prediction
    if 'wavelength' in predictions:
        ax = axes[0]
        actual = predictions['wavelength']['actual']
        predicted = predictions['wavelength']['predicted']
        
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        
        ax.scatter(actual, predicted, alpha=0.5, s=20, c='#2E75B6')
        
        # Add diagonal line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, alpha=0.7)
        
        ax.set_xlabel('Experimental Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Wavelength (nm)', fontsize=12, fontweight='bold')
        
        # Add metric text
        ax.text(0.05, 0.95, f'MAE = {mae:.1f}\nR^2 = {r2:.2f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.text(0.02, 0.98, 'e', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        ax.grid(True, alpha=0.3, linestyle='--')

        # Export data
        try:
            pd.DataFrame({'actual': actual, 'predicted': predicted}).to_csv(
                output_dir / 'figure_e_wavelength_data.csv', index=False
            )
        except Exception:
            pass
    
    # Figure f: PLQY prediction
    if 'PLQY' in predictions:
        ax = axes[1]
        actual = predictions['PLQY']['actual']
        predicted = predictions['PLQY']['predicted']
        
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        
        ax.scatter(actual, predicted, alpha=0.5, s=20, c='#FFC000')
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'r--', lw=1.5, alpha=0.7)
        
        ax.set_xlabel('Experimental PLQY', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted PLQY', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Add metric text
        ax.text(0.05, 0.95, f'MAE = {mae:.2f}\nR^2 = {r2:.2f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.text(0.02, 0.98, 'f', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        ax.grid(True, alpha=0.3, linestyle='--')

        # Export data
        try:
            pd.DataFrame({'actual': actual, 'predicted': predicted}).to_csv(
                output_dir / 'figure_f_plqy_data.csv', index=False
            )
        except Exception:
            pass
    
    plt.tight_layout()
    save_path = output_dir / 'figure_e_f_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved: {save_path}")

def plot_figure_g(predictions, output_dir):
    """
    Figure G: PLQY-range prediction accuracy heatmap
    """
    print("Generating Figure G: PLQY-range accuracy heatmap...")
    
    if 'PLQY' not in predictions:
        print("WARNING: No PLQY prediction data")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
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
    
    # Remove NaNs after binning
    mask2 = ~(actual_binned.isna() | predicted_binned.isna())
    actual_binned = actual_binned[mask2]
    predicted_binned = predicted_binned[mask2]
    
    # Create confusion matrix
    cm = confusion_matrix(actual_binned, predicted_binned, labels=labels)
    
    # Normalize to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Use blue color palette
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f',
                cmap=cmap,
                vmin=0, 
                vmax=1,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Accuracy'},
                ax=ax,
                square=True,
                linewidths=1,
                linecolor='white')
    
    ax.set_xlabel('Predicted PLQY Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual PLQY Range', fontsize=12, fontweight='bold')
    
    # Add label g
    ax.text(-0.15, 1.05, 'g', transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top')
    
    plt.tight_layout()
    save_path = output_dir / 'figure_g_plqy_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved: {save_path}")

    # Export confusion matrix data
    try:
        cm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
        cm_df.to_csv(output_dir / 'figure_g_cm_data.csv')
    except Exception:
        pass

def generate_all_figures(project_dir, data_file, output_dir):
    """Generate all figures"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Generate project figures")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = load_data(data_file)
    print(f"INFO: Loaded {len(df)} samples")
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = load_predictions(project_dir)
    if predictions:
        for key, value in predictions.items():
            print(f"INFO: {key}: {len(value['actual'])} predictions")
    
    # Generate figures
    print("\nGenerating figures...")
    print("-" * 40)
    
    # Figure c: Wavelength-PLQY scatter
    plot_figure_c(df, output_path)
    
    # Figure d: PLQY distribution
    plot_figure_d(df, output_path)
    
    # Figures e & f: prediction scatter
    if predictions:
        plot_figure_e_f(predictions, output_path)
        
        # Figure g: PLQY range accuracy
        plot_figure_g(predictions, output_path)
    
    print("\n" + "=" * 60)
    print("INFO: All figures generated")
    print(f"Saved to: {output_path}")
    print("=" * 60)
    
    # Return generated files
    files = list(output_path.glob("figure_*.png"))
    return files

def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description='Generate project figures')
    
    parser.add_argument('--project', '-p', default='.',
                       help='Project directory')
    parser.add_argument('--data', '-d', default='data/PhosIrDB.csv',
                       help='Data file')
    parser.add_argument('--output', '-o', help='Output directory')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path(args.project) / 'figures'
    
    # Generate all figures
    files = generate_all_figures(args.project, args.data, output_dir)
    
    # Show generated files
    print("\nGenerated figure files:")
    print("-" * 40)
    for f in sorted(files):
        print(f"  {f}")

if __name__ == "__main__":
    main()
