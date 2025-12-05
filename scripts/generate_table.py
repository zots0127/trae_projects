#!/usr/bin/env python3
"""
Unified table generation script - build performance comparison tables from training results
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 模型名称映射（用于显示）
MODEL_DISPLAY_NAMES = {
    'adaboost': 'AdaBoost',
    'ada_boost': 'AdaBoost',
    'catboost': 'CatBoost',
    'decision_tree': 'Decision Tree',
    'elastic_net': 'Elastic Net',
    'extra_trees': 'Extra Trees',
    'gradient_boosting': 'Gradient Boosting',
    'knn': 'K-Nearest Neighbors',
    'lasso': 'Lasso',
    'lightgbm': 'LightGBM',
    'random_forest': 'Random Forest',
    'ridge': 'Ridge',
    'xgboost': 'XGBoost',
    'svr': 'Support Vector Machine'
}

def load_model_results(project_path, model_name):
    """Load results for a single model"""
    results = {}
    
    # 查找模型目录
    model_dirs = [
        project_path / model_name,
        project_path / f"{model_name}_standard",
        project_path / model_name.replace('_', '')
    ]
    
    model_dir = None
    for dir_path in model_dirs:
        if dir_path.exists():
            model_dir = dir_path
            break
    
    if not model_dir:
        print(f"  WARNING: Model directory not found: {model_name}")
        return results
    
    # 首先尝试从exports目录读取summary.json文件
    exports_dir = model_dir / 'exports'
    if exports_dir.exists():
        summary_files = list(exports_dir.glob("*_summary.json"))
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                target = data.get('target', '')
                
                # 获取指标
                if 'mean_r2' in data:
                    results[target] = {
                        'r2_mean': data['mean_r2'],
                        'r2_std': data.get('std_r2', 0),
                        'rmse_mean': data.get('mean_rmse', 0),
                        'rmse_std': data.get('std_rmse', 0),
                        'mae_mean': data.get('mean_mae', 0),
                        'mae_std': data.get('std_mae', 0),
                        'n_folds': data.get('n_folds', 10)
                    }
            except Exception as e:
                print(f"  WARNING: Failed to read summary {summary_file}: {e}")
    
    # 如果没有找到summary文件，尝试旧格式
    if not results:
        # 遍历目标目录
        for target_dir in model_dir.iterdir():
            if not target_dir.is_dir():
                continue
            
            # 查找结果文件
            result_files = list(target_dir.glob("**/experiment_results.json"))
            if not result_files:
                continue
            
            # 读取结果
            try:
                with open(result_files[0], 'r') as f:
                    data = json.load(f)
                
                target = data.get('target_column', target_dir.name)
                cv_results = data.get('cv_results', {})
                
                # 提取指标
                r2_scores = []
                rmse_scores = []
                mae_scores = []
                
                for fold_result in cv_results.values():
                    if isinstance(fold_result, dict):
                        if 'r2' in fold_result:
                            r2_scores.append(fold_result['r2'])
                        if 'rmse' in fold_result:
                            rmse_scores.append(fold_result['rmse'])
                        if 'mae' in fold_result:
                            mae_scores.append(fold_result['mae'])
                
                if r2_scores:
                    results[target] = {
                        'r2_mean': np.mean(r2_scores),
                        'r2_std': np.std(r2_scores),
                        'rmse_mean': np.mean(rmse_scores),
                        'rmse_std': np.std(rmse_scores),
                        'mae_mean': np.mean(mae_scores),
                        'mae_std': np.std(mae_scores),
                        'n_folds': len(r2_scores)
                    }
                    
            except Exception as e:
                print(f"  WARNING: Failed to read results {result_files[0]}: {e}")
    
    return results

def generate_performance_table(project_name, output_format='all', target_names=None):
    """Generate performance comparison table"""
    
    project_path = Path(project_name)
    if not project_path.exists():
        print(f"ERROR: Project directory not found: {project_name}")
        return None
    
    print("=" * 60)
    print(f"Generate performance table")
    print(f"Project: {project_name}")
    print("=" * 60)
    
    # Default target name mapping
    if not target_names:
        target_names = {
            'Max_wavelength(nm)': 'Wavelength (nm)',
            'Max_wavelengthnm': 'Wavelength (nm)',
            'PLQY': 'PLQY',
            'tau(s*10^-6)': 'Lifetime (us)',
            'tausx10^-6': 'Lifetime (us)'
        }
    
    # 收集所有结果
    all_results = []
    
    # 遍历每个模型
    print("\nCollecting model results...")
    for model_dir in project_path.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
        
        # Skip non-model directories
        if model_dir.name in ['tables', 'logs', 'last', 'checkpoints', 'tmp']:
            continue
        
        model_name = model_dir.name.replace('_standard', '')
        print(f"  Processing: {model_name}")
        
        results = load_model_results(project_path, model_dir.name)
        
        for target, metrics in results.items():
            # Determine display target name
            display_target = target
            for key, value in target_names.items():
                if key in target or key.replace('_', '').replace('(', '').replace(')', '') in target.replace('_', '').replace('(', '').replace(')', ''):
                    display_target = value
                    break
            
            # Display model name
            display_model = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            
            # 添加结果
            all_results.append({
                'Target': display_target,
                'Model': display_model,
                'R^2': f"{metrics['r2_mean']:.4f} +/- {metrics['r2_std']:.4f}",
                'RMSE': f"{metrics['rmse_mean']:.2f} +/- {metrics['rmse_std']:.2f}",
                'MAE': f"{metrics['mae_mean']:.2f} +/- {metrics['mae_std']:.2f}",
                'Folds': metrics['n_folds']
            })
    
    if not all_results:
        print("WARNING: No results found")
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 按目标和模型排序
    df = df.sort_values(['Target', 'Model'])
    
    # 输出目录
    output_dir = project_path / 'tables'
    output_dir.mkdir(exist_ok=True)
    
    # 生成不同格式的表格
    outputs = []
    
    if output_format in ['all', 'console']:
        print("\n" + "=" * 80)
        print("Performance comparison table")
        print("=" * 80)
        
        for target in df['Target'].unique():
            target_df = df[df['Target'] == target]
            print(f"\n{target}:")
            print("-" * 60)
            print(target_df[['Model', 'R^2', 'RMSE', 'MAE']].to_string(index=False))
            
            # Best model by R^2
            best_r2 = -1
            best_model = None
            for _, row in target_df.iterrows():
                r2_value = float(row['R^2'].split(' +/- ')[0])
                if r2_value > best_r2:
                    best_r2 = r2_value
                    best_model = row['Model']
            
            if best_model:
                print(f"\nBest Model: {best_model} (R^2 = {best_r2:.4f})")
    
    if output_format in ['all', 'csv']:
        csv_file = output_dir / 'performance_table.csv'
        df.to_csv(csv_file, index=False)
        outputs.append(f"CSV: {csv_file}")
        print(f"\nINFO: Saved CSV table: {csv_file}")
    
    if output_format in ['all', 'excel']:
        try:
            excel_file = output_dir / 'performance_table.xlsx'
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 总表
                df.to_excel(writer, sheet_name='All Results', index=False)
                
                # 每个目标单独一个sheet
                for target in df['Target'].unique():
                    target_df = df[df['Target'] == target]
                    sheet_name = target.replace('/', '_').replace('(', '').replace(')', '')[:31]
                    target_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            outputs.append(f"Excel: {excel_file}")
            print(f"INFO: Saved Excel table: {excel_file}")
        except ImportError:
            print("WARNING: Need openpyxl: pip install openpyxl")
    
    if output_format in ['all', 'markdown', 'md']:
        md_file = output_dir / 'performance_table.md'
        with open(md_file, 'w') as f:
            f.write("# Model Performance Comparison\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for target in df['Target'].unique():
                f.write(f"\n## {target}\n\n")
                target_df = df[df['Target'] == target]
                
                f.write("| Model | R^2 | RMSE | MAE |\n")
                f.write("|-------|-----|------|-----|\n")
                
                for _, row in target_df.iterrows():
                    f.write(f"| {row['Model']} | {row['R^2']} | {row['RMSE']} | {row['MAE']} |\n")
                
                # Best model
                best_r2 = -1
                best_model = None
                for _, row in target_df.iterrows():
                    r2_value = float(row['R^2'].split(' +/- ')[0])
                    if r2_value > best_r2:
                        best_r2 = r2_value
                        best_model = row['Model']
                
                if best_model:
                    f.write(f"\n**Best Model**: {best_model} (R^2 = {best_r2:.4f})\n")
        
        outputs.append(f"Markdown: {md_file}")
        print(f"INFO: Saved Markdown table: {md_file}")
    
    if output_format in ['all', 'latex', 'tex']:
        latex_file = output_dir / 'performance_table.tex'
        with open(latex_file, 'w') as f:
            # LaTeX header
            f.write("% Performance Comparison Table\n")
            f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            
            for target in df['Target'].unique():
                f.write(f"% {target}\n")
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{Model Performance for {target}}}\n")
                f.write("\\begin{tabular}{lccc}\n")
                f.write("\\hline\n")
                f.write("Model & R^2 & RMSE & MAE \\\n")
                f.write("\\hline\n")
                
                target_df = df[df['Target'] == target]
                for _, row in target_df.iterrows():
                    # 转义LaTeX特殊字符
                    model = row['Model'].replace('_', '\\_')
                    f.write(f"{model} & {row['R^2']} & {row['RMSE']} & {row['MAE']} \\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
        
        outputs.append(f"LaTeX: {latex_file}")
        print(f"INFO: Saved LaTeX table: {latex_file}")
    
    print("\n" + "=" * 60)
    print("INFO: Table generation completed!")
    if outputs:
        print("\nOutputs:")
        for output in outputs:
            print(f"  - {output}")
    print("=" * 60)
    
    return df

def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description='Generate performance comparison tables')
    
    parser.add_argument('project', help='Project directory name')
    parser.add_argument('--format', '-f', default='all',
                       choices=['all', 'console', 'csv', 'excel', 'markdown', 'md', 'latex', 'tex'],
                       help='Output format')
    parser.add_argument('--targets', '-t', nargs='+',
                       help='Custom target name mapping (format: original=display)')
    
    args = parser.parse_args()
    
    # 解析自定义目标名称
    target_names = None
    if args.targets:
        target_names = {}
        for mapping in args.targets:
            if '=' in mapping:
                orig, display = mapping.split('=', 1)
                target_names[orig] = display
    
    # 生成表格
    df = generate_performance_table(args.project, args.format, target_names)
    
    if df is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
