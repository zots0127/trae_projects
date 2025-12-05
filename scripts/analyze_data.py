#!/usr/bin/env python3
"""
Data analysis and visualization script - generate paper figures
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
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

# Add parent directory to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(data_file):
    """Load data"""
    df = pd.read_csv(data_file)
    print(f"INFO: Loaded data: {data_file}")
    print(f"   Samples: {len(df)}")
    print(f"   Features: {len(df.columns)}")
    return df

def plot_wavelength_plqy_scatter(df, output_dir):
    """
    Plot Wavelength-PLQY scatter (Figure c-like), color by solvent
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Solvent types and colors
    solvent_colors = {
        'CH2Cl2': '#1f77b4',  # blue
        'CH3CN': '#2ca02c',   # green
        'Toluene': '#ff7f0e',  # orange
        'Others': '#9467bd'    # purple
    }
    
    # Extract wavelength and PLQY columns
    wavelength_col = None
    plqy_col = None
    
    for col in df.columns:
        if 'wavelength' in col.lower() or 'max_wavelength' in col.lower():
            wavelength_col = col
        if 'plqy' in col.lower():
            plqy_col = col
    
    if wavelength_col and plqy_col:
        # Group by solvent if available
        if 'Solvent' in df.columns:
            for solvent, color in solvent_colors.items():
                mask = df['Solvent'] == solvent
                ax.scatter(df.loc[mask, wavelength_col], 
                          df.loc[mask, plqy_col],
                          c=color, label=solvent, alpha=0.6, s=20)
        else:
            # No solvent info, use single color
            ax.scatter(df[wavelength_col], df[plqy_col], 
                      alpha=0.6, s=20, c='#1f77b4')
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('PLQY', fontsize=12)
        ax.set_xlim(440, 880)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / 'wavelength_plqy_scatter.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved scatter: {save_path}")
    else:
        print("WARNING: Wavelength or PLQY column not found")

def plot_plqy_distribution(df, output_dir):
    """
    Plot PLQY distribution histogram (Figure d-like)
    Group by solvent and PLQY range
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    plqy_col = None
    for col in df.columns:
        if 'plqy' in col.lower():
            plqy_col = col
            break
    
    if plqy_col:
        # Define PLQY ranges
        bins = [0, 0.1, 0.5, 1.0]
        labels = ['0-0.1', '0.1-0.5', '0.5-1.0']
        
        # If solvent information available
        if 'Solvent' in df.columns:
            solvent_types = ['CH2Cl2', 'CH3CN', 'Toluene', 'Others']
            colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
            
            # Create grouped data
            data_by_range = []
            for i in range(len(bins)-1):
                range_data = []
                for solvent in solvent_types:
                    mask = (df['Solvent'] == solvent) & \
                           (df[plqy_col] > bins[i]) & \
                           (df[plqy_col] <= bins[i+1])
                    range_data.append(mask.sum())
                data_by_range.append(range_data)
            
            # Draw stacked bar chart
            x = np.arange(len(labels))
            width = 0.6
            bottom = np.zeros(len(labels))
            
            for j, (solvent, color) in enumerate(zip(solvent_types, colors)):
                values = [data_by_range[i][j] for i in range(len(labels))]
                ax.bar(x, values, width, bottom=bottom, label=solvent, color=color)
                bottom += values
        else:
            # Simple histogram
            df[plqy_col].hist(bins=bins, ax=ax, edgecolor='black')
        
        ax.set_xlabel('PLQY Range', fontsize=12)
        ax.set_ylabel('Number of entries', fontsize=12)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        save_path = output_dir / 'plqy_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved distribution: {save_path}")

def plot_prediction_scatter(df, predictions_file, output_dir):
    """
    Plot predicted vs experimental scatter (Figure e/f-like)
    """
    if not predictions_file or not Path(predictions_file).exists():
        print("WARNING: Prediction file not provided or does not exist")
        return
    
    # Load predictions
    pred_df = pd.read_csv(predictions_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Find wavelength and PLQY prediction columns
    targets = ['wavelength', 'plqy']
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        # Find relevant columns
        exp_col = None
        pred_col = None
        
        for col in pred_df.columns:
            if target in col.lower() and 'experimental' in col.lower():
                exp_col = col
            elif target in col.lower() and 'predicted' in col.lower():
                pred_col = col
        
        if exp_col and pred_col:
            x = pred_df[exp_col].values
            y = pred_df[pred_col].values
            
            # Remove NaNs
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            # Metrics
            r2 = r2_score(x, y)
            mae = mean_absolute_error(x, y)
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.5, s=10, c='#1f77b4')
            
            # Diagonal
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=1, alpha=0.7)
            
            # Labels
            if 'wavelength' in target:
                ax.set_xlabel('Experimental Wavelength (nm)', fontsize=12)
                ax.set_ylabel('Predicted Wavelength (nm)', fontsize=12)
                title = f'Wavelength Prediction'
            else:
                ax.set_xlabel('Experimental PLQY', fontsize=12)
                ax.set_ylabel('Predicted PLQY', fontsize=12)
                title = f'PLQY Prediction'
            
            # Metrics text
            ax.text(0.05, 0.95, f'MAE = {mae:.2f}\nR^2 = {r2:.2f}',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(title, fontsize=13)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'prediction_scatter.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved prediction scatter: {save_path}")

def plot_correlation_matrix(df, output_dir):
    """
    Plot correlation matrix (Figure g-like)
    """
    # Select PLQY-related numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Select key columns (if available)
    key_cols = []
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in ['plqy', 'wavelength', 'tau', 'lifetime']):
            key_cols.append(col)
    
    if len(key_cols) < 2:
        key_cols = numeric_cols[:min(10, len(numeric_cols))]  # select first 10 numeric columns
    
    if len(key_cols) >= 2:
        # Compute correlation matrix
        corr_matrix = df[key_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Use custom colormap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Draw heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                   cmap=cmap, center=0,
                   square=True, linewidths=1,
                   cbar_kws={"shrink": .8},
                   ax=ax)
        
        ax.set_title('Correlation Matrix', fontsize=14)
        
        plt.tight_layout()
        save_path = output_dir / 'correlation_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved correlation matrix: {save_path}")

def generate_summary_stats(df, output_dir):
    """Generate summary statistics"""
    stats = {}
    
    # Basic stats
    stats['total_samples'] = len(df)
    stats['total_features'] = len(df.columns)
    
    # PLQY stats
    plqy_col = None
    for col in df.columns:
        if 'plqy' in col.lower():
            plqy_col = col
            break
    
    if plqy_col:
        stats['plqy'] = {
            'mean': df[plqy_col].mean(),
            'std': df[plqy_col].std(),
            'min': df[plqy_col].min(),
            'max': df[plqy_col].max(),
            'median': df[plqy_col].median()
        }
    
    # Wavelength stats
    wavelength_col = None
    for col in df.columns:
        if 'wavelength' in col.lower():
            wavelength_col = col
            break
    
    if wavelength_col:
        stats['wavelength'] = {
            'mean': df[wavelength_col].mean(),
            'std': df[wavelength_col].std(),
            'min': df[wavelength_col].min(),
            'max': df[wavelength_col].max(),
            'median': df[wavelength_col].median()
        }
    
    # Save stats
    stats_file = output_dir / 'summary_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=True)
    
    print(f"INFO: Saved summary stats: {stats_file}")
    
    # Print statistics summary
    print("\n" + "=" * 60)
    print("Data statistics summary")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    
    if 'plqy' in stats:
        print(f"\nPLQY stats:")
        print(f"  Mean: {stats['plqy']['mean']:.3f}")
        print(f"  Std: {stats['plqy']['std']:.3f}")
        print(f"  Range: [{stats['plqy']['min']:.3f}, {stats['plqy']['max']:.3f}]")
    
    if 'wavelength' in stats:
        print(f"\nWavelength stats:")
        print(f"  Mean: {stats['wavelength']['mean']:.1f} nm")
        print(f"  Std: {stats['wavelength']['std']:.1f} nm")
        print(f"  Range: [{stats['wavelength']['min']:.1f}, {stats['wavelength']['max']:.1f}] nm")

def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description='Data analysis and visualization')
    
    parser.add_argument('--data', '-d', required=True,
                       help='Data file path')
    parser.add_argument('--predictions', '-p',
                       help='Predictions file (optional)')
    parser.add_argument('--output', '-o',
                       help='Output directory')
    parser.add_argument('--plots', nargs='+',
                       choices=['scatter', 'distribution', 'prediction', 'correlation', 'all'],
                       default=['all'],
                       help='Types of plots to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Data analysis and visualization")
    print("=" * 60)
    print(f"Data file: {args.data}")
    print(f"Output dir: {output_dir}")
    
    # Load data
    df = load_data(args.data)
    
    # Generate plots
    if 'all' in args.plots or 'scatter' in args.plots:
        plot_wavelength_plqy_scatter(df, output_dir)
    
    if 'all' in args.plots or 'distribution' in args.plots:
        plot_plqy_distribution(df, output_dir)
    
    if 'all' in args.plots or 'prediction' in args.plots:
        if args.predictions:
            plot_prediction_scatter(df, args.predictions, output_dir)
    
    if 'all' in args.plots or 'correlation' in args.plots:
        plot_correlation_matrix(df, output_dir)
    
    # Generate summary stats
    generate_summary_stats(df, output_dir)
    
    print("\n" + "=" * 60)
    print("INFO: Analysis completed!")
    print(f"Saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
