#!/usr/bin/env python3
"""
Fix generation of AutoML model comparison tables
Adapt to optuna_results directory structure
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime

def collect_automl_results(output_dir):
    """Collect all AutoML training model results"""
    output_dir = Path(output_dir)
    results = []
    
    # Locate automl_train directory
    automl_dir = output_dir / 'automl_train'
    if not automl_dir.exists():
        print(f"ERROR: automl_train directory not found: {automl_dir}")
        return pd.DataFrame()
    
    # Scan all models' optuna_results
    for model_dir in automl_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Locate optuna_results directory
        optuna_dir = model_dir / 'optuna_results'
        if optuna_dir.exists():
            # Find automl_summary files
            for summary_file in optuna_dir.glob('automl_summary_*.json'):
                target_name = summary_file.stem.replace('automl_summary_', '')
                
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                # Iterate over all models' results
                for model_name, model_data in summary_data.get('all_models', {}).items():
                    if 'fold_results' in model_data:
                        fold_results = model_data['fold_results']
                        
                        # Extract metrics per fold
                        r2_scores = [fold['r2'] for fold in fold_results]
                        rmse_scores = [fold['rmse'] for fold in fold_results]
                        mae_scores = [fold['mae'] for fold in fold_results]
                        
                        results.append({
                            'Model': model_name.upper(),
                            'Target': target_name.replace('_', ' '),
                            'R2_mean': np.mean(r2_scores),
                            'R2_std': np.std(r2_scores),
                            'RMSE_mean': np.mean(rmse_scores),
                            'RMSE_std': np.std(rmse_scores),
                            'MAE_mean': np.mean(mae_scores),
                            'MAE_std': np.std(mae_scores),
                            'N_folds': len(fold_results),
                            'Best_R2': model_data.get('best_r2', np.mean(r2_scores))
                        })
    
    # Deduplicate (there may be multiple identical results)
    df = pd.DataFrame(results)
    if not df.empty:
        # Deduplicate by Model and Target, keep first
        df = df.drop_duplicates(subset=['Model', 'Target'], keep='first')
    return df

def generate_comparison_tables(df, output_dir):
    """Generate comparison tables in multiple formats"""
    if df.empty:
        print("ERROR: No data to generate tables")
        return
    
    output_dir = Path(output_dir)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    # 1. Generate full CSV
    df.to_csv(tables_dir / 'model_comparison_full.csv', index=False)
    print(f"INFO: Full comparison CSV: {tables_dir}/model_comparison_full.csv")
    
    # 2. Generate Markdown table
    markdown_content = "# Model Performance Comparison (Cross-Validation)\n\n"
    
    for target in df['Target'].unique():
        markdown_content += f"\n## {target}\n\n"
        target_df = df[df['Target'] == target].copy()
        target_df = target_df.sort_values('R2_mean', ascending=False)
        
        markdown_content += "| Model | R^2 | RMSE | MAE |\n"
        markdown_content += "|-------|-----|------|-----|\n"
        
        for _, row in target_df.iterrows():
            r2 = f"{row['R2_mean']:.4f} +/- {row['R2_std']:.4f}"
            rmse = f"{row['RMSE_mean']:.2f} +/- {row['RMSE_std']:.2f}"
            mae = f"{row['MAE_mean']:.2f} +/- {row['MAE_std']:.2f}"
            
            # Mark best model
            if row['R2_mean'] == target_df['R2_mean'].max():
                model_name = f"**{row['Model']}**"
            else:
                model_name = row['Model']
            
            markdown_content += f"| {model_name} | {r2} | {rmse} | {mae} |\n"
    
    with open(tables_dir / 'model_comparison.md', 'w') as f:
        f.write(markdown_content)
    print(f"INFO: Markdown table: {tables_dir}/model_comparison.md")
    
    # 3. Generate LaTeX table
    latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\begin{document}

\begin{table}[htbp]
\centering
\caption{Model Performance Comparison}
\label{tab:model_comparison}
\begin{tabular}{llccc}
\toprule
Target & Model & R$^2$ & RMSE & MAE \\
\midrule
"""
    
    for target in df['Target'].unique():
        target_df = df[df['Target'] == target].copy()
        target_df = target_df.sort_values('R2_mean', ascending=False)
        
        for idx, row in target_df.iterrows():
            if idx == target_df.index[0]:
                target_str = target.replace('_', r'\_')
            else:
                target_str = ""
            
            r2 = f"{row['R2_mean']:.4f} $\\pm$ {row['R2_std']:.4f}"
            rmse = f"{row['RMSE_mean']:.2f} $\\pm$ {row['RMSE_std']:.2f}"
            mae = f"{row['MAE_mean']:.2f} $\\pm$ {row['MAE_std']:.2f}"
            
            latex_content += f"{target_str} & {row['Model']} & {r2} & {rmse} & {mae} \\\\\n"
        
        if target != df['Target'].unique()[-1]:
            latex_content += r"\midrule" + "\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

\end{document}"""
    
    with open(tables_dir / 'model_comparison.tex', 'w') as f:
        f.write(latex_content)
    print(f"INFO: LaTeX table: {tables_dir}/model_comparison.tex")
    
    # 4. Generate best model summary
    best_models = []
    for target in df['Target'].unique():
        target_df = df[df['Target'] == target]
        best_idx = target_df['R2_mean'].idxmax()
        best = target_df.loc[best_idx]
        
        best_models.append({
            'Target': target,
            'Best Model': best['Model'],
            'R^2': f"{best['R2_mean']:.4f} +/- {best['R2_std']:.4f}",
            'RMSE': f"{best['RMSE_mean']:.2f} +/- {best['RMSE_std']:.2f}",
            'MAE': f"{best['MAE_mean']:.2f} +/- {best['MAE_std']:.2f}"
        })
    
    best_df = pd.DataFrame(best_models)
    best_df.to_csv(tables_dir / 'best_models_summary.csv', index=False)
    print(f"INFO: Best models summary: {tables_dir}/best_models_summary.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("Best model summary")
    print("="*60)
    print(best_df.to_string(index=False))

def main():
    # Parse CLI args
    import argparse
    parser = argparse.ArgumentParser(description='Generate AutoML model comparison tables')
    parser.add_argument('--project', default='runs/train', help='Project directory')
    args = parser.parse_args()
    
    print("="*60)
    print("Generate AutoML model comparison tables")
    print("="*60)
    
    # Collect results
    print("\nCollecting model results...")
    df = collect_automl_results(args.project)
    
    if df.empty:
        print("ERROR: No model results found")
        return 1
    
    print(f"INFO: Found {len(df)} model results")
    print(f"   Models: {df['Model'].nunique()}")
    print(f"   Targets: {df['Target'].nunique()}")
    
    # Generate tables
    print("\nGenerating comparison tables...")
    generate_comparison_tables(df, args.project)
    
    print("\nINFO: Completed!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
