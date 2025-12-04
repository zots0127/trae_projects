#!/usr/bin/env python3
"""
修复AutoML训练结果的模型对比表格生成
适配optuna_results目录结构
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime

def collect_automl_results(output_dir):
    """收集AutoML训练的所有模型结果"""
    output_dir = Path(output_dir)
    results = []
    
    # 查找automl_train目录
    automl_dir = output_dir / 'automl_train'
    if not automl_dir.exists():
        print(f"❌ 未找到automl_train目录: {automl_dir}")
        return pd.DataFrame()
    
    # 扫描所有模型的optuna_results
    for model_dir in automl_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # 查找optuna_results目录
        optuna_dir = model_dir / 'optuna_results'
        if optuna_dir.exists():
            # 查找automl_summary文件
            for summary_file in optuna_dir.glob('automl_summary_*.json'):
                target_name = summary_file.stem.replace('automl_summary_', '')
                
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                # 遍历所有模型的结果
                for model_name, model_data in summary_data.get('all_models', {}).items():
                    if 'fold_results' in model_data:
                        fold_results = model_data['fold_results']
                        
                        # 提取各折的指标
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
    
    # 去重（可能有多个相同的结果）
    df = pd.DataFrame(results)
    if not df.empty:
        # 根据Model和Target去重，保留第一个
        df = df.drop_duplicates(subset=['Model', 'Target'], keep='first')
    return df

def generate_comparison_tables(df, output_dir):
    """生成多种格式的对比表格"""
    if df.empty:
        print("❌ 没有数据生成表格")
        return
    
    output_dir = Path(output_dir)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    # 1. 生成完整CSV
    df.to_csv(tables_dir / 'model_comparison_full.csv', index=False)
    print(f"✅ 完整对比表: {tables_dir}/model_comparison_full.csv")
    
    # 2. 生成Markdown表格
    markdown_content = "# Model Performance Comparison (Cross-Validation)\n\n"
    
    for target in df['Target'].unique():
        markdown_content += f"\n## {target}\n\n"
        target_df = df[df['Target'] == target].copy()
        target_df = target_df.sort_values('R2_mean', ascending=False)
        
        markdown_content += "| Model | R² | RMSE | MAE |\n"
        markdown_content += "|-------|-----|------|-----|\n"
        
        for _, row in target_df.iterrows():
            r2 = f"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}"
            rmse = f"{row['RMSE_mean']:.2f} ± {row['RMSE_std']:.2f}"
            mae = f"{row['MAE_mean']:.2f} ± {row['MAE_std']:.2f}"
            
            # 标记最佳模型
            if row['R2_mean'] == target_df['R2_mean'].max():
                model_name = f"**{row['Model']}** 🏆"
            else:
                model_name = row['Model']
            
            markdown_content += f"| {model_name} | {r2} | {rmse} | {mae} |\n"
    
    with open(tables_dir / 'model_comparison.md', 'w') as f:
        f.write(markdown_content)
    print(f"✅ Markdown表格: {tables_dir}/model_comparison.md")
    
    # 3. 生成LaTeX表格
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
    print(f"✅ LaTeX表格: {tables_dir}/model_comparison.tex")
    
    # 4. 生成最佳模型总结
    best_models = []
    for target in df['Target'].unique():
        target_df = df[df['Target'] == target]
        best_idx = target_df['R2_mean'].idxmax()
        best = target_df.loc[best_idx]
        
        best_models.append({
            'Target': target,
            'Best Model': best['Model'],
            'R²': f"{best['R2_mean']:.4f} ± {best['R2_std']:.4f}",
            'RMSE': f"{best['RMSE_mean']:.2f} ± {best['RMSE_std']:.2f}",
            'MAE': f"{best['MAE_mean']:.2f} ± {best['MAE_std']:.2f}"
        })
    
    best_df = pd.DataFrame(best_models)
    best_df.to_csv(tables_dir / 'best_models_summary.csv', index=False)
    print(f"✅ 最佳模型总结: {tables_dir}/best_models_summary.csv")
    
    # 打印总结
    print("\n" + "="*60)
    print("最佳模型总结")
    print("="*60)
    print(best_df.to_string(index=False))

def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='生成AutoML模型对比表格')
    parser.add_argument('--project', default='runs/train', help='项目目录')
    args = parser.parse_args()
    
    print("="*60)
    print("生成AutoML模型对比表格")
    print("="*60)
    
    # 收集结果
    print("\n收集模型结果...")
    df = collect_automl_results(args.project)
    
    if df.empty:
        print("❌ 未找到任何模型结果")
        return 1
    
    print(f"✅ 找到 {len(df)} 个模型结果")
    print(f"   模型: {df['Model'].nunique()} 个")
    print(f"   目标: {df['Target'].nunique()} 个")
    
    # 生成表格
    print("\n生成对比表格...")
    generate_comparison_tables(df, args.project)
    
    print("\n✅ 完成！")
    return 0

if __name__ == '__main__':
    sys.exit(main())