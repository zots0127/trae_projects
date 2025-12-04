#!/usr/bin/env python3
"""
生成论文所需的所有图表
包括图c-g的所有分析图
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

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置绘图风格
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

def load_data(data_file):
    """加载原始数据"""
    df = pd.read_csv(data_file)
    return df

def load_predictions(project_dir, model_name='xgboost'):
    """加载预测结果
    优先从 project_dir/<model>/predictions 读取；
    若不存在，尝试 project_dir/all_models/automl_train/<model>/exports/csv/；
    或 project_dir/automl_train/<model>/exports/csv/ (如果project_dir已经包含all_models)；
    最后回退到 project_dir/predictions
    """
    project_path = Path(project_dir)
    model_dir = project_path / model_name
    
    # 默认路径
    predictions_dir = model_dir / 'predictions'
    
    # 如果默认路径不存在，尝试AutoML路径
    if not model_dir.exists() or not predictions_dir.exists():
        # 检查project_path是否已经包含all_models
        if project_path.name == 'all_models' or 'all_models' in project_path.parts:
            # 如果已经在all_models目录中，直接查找automl_train
            automl_dir = project_path / 'automl_train' / model_name / 'exports' / 'csv'
        else:
            # 否则添加all_models路径
            automl_dir = project_path / 'all_models' / 'automl_train' / model_name / 'exports' / 'csv'
        
        if automl_dir.exists():
            predictions_dir = automl_dir
            print(f"  使用AutoML预测目录: {predictions_dir}")
        else:
            # 回退：使用统一预测目录
            predictions_dir = project_path / 'predictions'
            if not predictions_dir.exists():
                print(f"⚠️ 未找到预测目录: {model_dir/'predictions'} 或 {predictions_dir} 或 {automl_dir}")
                return None
    all_predictions = {}
    target_types = {'wavelength': [], 'PLQY': [], 'tau': []}
    
    csv_files = list(predictions_dir.glob("*.csv"))
    
    # 如果是AutoML目录，只选择all_predictions文件
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
            
            # 优先查找'true'和'predicted'列（AutoML格式）
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
            print(f"⚠️ 读取文件失败 {csv_file}: {e}")
    
    for target_type in ['wavelength', 'PLQY', 'tau']:
        if target_types[target_type]:
            actual_all = np.concatenate([d['actual'] for d in target_types[target_type]])
            predicted_all = np.concatenate([d['predicted'] for d in target_types[target_type]])
            
            all_predictions[target_type] = {
                'actual': actual_all,
                'predicted': predicted_all
            }
    
    # 额外：尝试加载测试集预测（exports/test_predictions_*.csv）以覆盖/补充
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
                    # 预测列
                    pred_col = 'prediction' if 'prediction' in df.columns else None
                    if pred_col is None:
                        continue
                    # 真值列（若存在）
                    candidate_actual_cols = [
                        'Max_wavelength(nm)', 'Max_wavelengthnm', 'wavelength',
                        'PLQY', 'tau(s*10^-6)', 'tausx10^-6', 'tau'
                    ]
                    actual_col = next((c for c in candidate_actual_cols if c in df.columns), None)
                    if actual_col is None:
                        # 如果没有真值列，跳过该目标（无法画散点）
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
                    print(f"⚠️ 无法读取测试预测 {tf}: {e}")
    except Exception:
        pass

    return all_predictions

def plot_figure_c(df, output_dir):
    """
    图c: 波长-PLQY散点图，按溶剂类型着色
    """
    print("生成图c: 波长-PLQY散点图...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 定义溶剂类型和颜色
    solvent_colors = {
        'CH2Cl2': '#2E75B6',    # 深蓝色
        'CH3CN': '#70AD47',     # 绿色
        'Toluene': '#FFC000',   # 橙色
        'Others': '#7030A0'     # 紫色
    }
    
    # 查找波长和PLQY列
    wavelength_col = None
    plqy_col = None
    
    for col in df.columns:
        if 'wavelength' in col.lower() and 'max' in col.lower():
            wavelength_col = col
        if 'plqy' in col.lower():
            plqy_col = col
    
    if wavelength_col and plqy_col:
        # 创建散点图
        if 'Solvent' in df.columns:
            # 如果有溶剂信息
            for solvent, color in solvent_colors.items():
                mask = df['Solvent'] == solvent
                if mask.sum() > 0:
                    ax.scatter(df.loc[mask, wavelength_col], 
                              df.loc[mask, plqy_col],
                              c=color, label=solvent, alpha=0.6, s=30, marker='s')
        else:
            # 没有溶剂信息，使用默认颜色
            ax.scatter(df[wavelength_col], df[plqy_col], 
                      alpha=0.6, s=30, c='#2E75B6', marker='s')
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PLQY', fontsize=12, fontweight='bold')
        ax.set_xlim(440, 880)
        ax.set_ylim(0, 1.0)
        
        # 设置x轴刻度
        ax.set_xticks([440, 550, 660, 770, 880])
        ax.set_xticklabels(['440 nm', '550 nm', '660 nm', '770 nm', '880 nm'])
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加标签c
        ax.text(0.02, 0.98, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        plt.tight_layout()
        save_path = output_dir / 'figure_c_wavelength_plqy.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_path}")

        # 导出用于绘图的数据
        try:
            data_out = df[[wavelength_col, plqy_col]].copy()
            if 'Solvent' in df.columns:
                data_out['Solvent'] = df['Solvent']
            data_out.to_csv(output_dir / 'figure_c_data.csv', index=False)
        except Exception:
            pass

def plot_figure_d(df, output_dir):
    """
    图d: PLQY分布直方图（堆叠柱状图）
    """
    print("生成图d: PLQY分布直方图...")
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    plqy_col = None
    for col in df.columns:
        if 'plqy' in col.lower():
            plqy_col = col
            break
    
    if plqy_col:
        # 定义PLQY范围
        bins = [-0.001, 0.1, 0.5, 1.001]
        labels = ['≤0.1', '0.1-0.5', '>0.5']
        
        # 计算每个范围的数量
        df['PLQY_range'] = pd.cut(df[plqy_col], bins=bins, labels=labels)
        
        if 'Solvent' in df.columns:
            # 定义溶剂颜色
            solvent_colors = {
                'CH2Cl2': '#2E75B6',
                'CH3CN': '#70AD47',
                'Toluene': '#FFC000',
                'Others': '#7030A0'
            }
            
            # 创建堆叠数据
            data_matrix = []
            solvents = ['CH2Cl2', 'CH3CN', 'Toluene', 'Others']
            
            for label in labels:
                row = []
                for solvent in solvents:
                    count = df[(df['PLQY_range'] == label) & (df['Solvent'] == solvent)].shape[0]
                    row.append(count)
                data_matrix.append(row)
            
            # 绘制堆叠柱状图
            x = np.arange(len(labels))
            width = 0.6
            bottom = np.zeros(len(labels))
            
            for i, solvent in enumerate(solvents):
                values = [data_matrix[j][i] for j in range(len(labels))]
                ax.bar(x, values, width, bottom=bottom, 
                       label=solvent, color=solvent_colors[solvent])
                bottom += values
        else:
            # 简单直方图
            counts = df['PLQY_range'].value_counts()[labels].fillna(0)
            ax.bar(range(len(labels)), counts.values, color='#2E75B6')
        
        ax.set_xlabel('PLQY Range', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of entries', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 800)
        
        if 'Solvent' in df.columns:
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # 添加标签d
        ax.text(0.02, 0.98, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        plt.tight_layout()
        save_path = output_dir / 'figure_d_plqy_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_path}")

        # 导出用于绘图的数据
        try:
            out_df = df[['PLQY_range']].copy()
            if 'Solvent' in df.columns:
                out_df['Solvent'] = df['Solvent']
            out_df.to_csv(output_dir / 'figure_d_data.csv', index=False)
        except Exception:
            pass

def plot_figure_e_f(predictions, output_dir):
    """
    图e和f: 预测vs实验散点图
    """
    print("生成图e和f: 预测vs实验散点图...")
    
    if not predictions:
        print("⚠️ 没有预测数据")
        return
    
    # 创建两个子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图e: 波长预测
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
        
        # 添加对角线
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, alpha=0.7)
        
        ax.set_xlabel('Experimental λem (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted λem (nm)', fontsize=12, fontweight='bold')
        
        # 添加指标文本
        ax.text(0.05, 0.95, f'MAE = {mae:.1f}\nR² = {r2:.2f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.text(0.02, 0.98, 'e', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        ax.grid(True, alpha=0.3, linestyle='--')

        # 导出数据
        try:
            pd.DataFrame({'actual': actual, 'predicted': predicted}).to_csv(
                output_dir / 'figure_e_wavelength_data.csv', index=False
            )
        except Exception:
            pass
    
    # 图f: PLQY预测
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
        
        # 添加对角线
        ax.plot([0, 1], [0, 1], 'r--', lw=1.5, alpha=0.7)
        
        ax.set_xlabel('Experimental PLQY', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted PLQY', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # 添加指标文本
        ax.text(0.05, 0.95, f'MAE = {mae:.2f}\nR² = {r2:.2f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.text(0.02, 0.98, 'f', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        ax.grid(True, alpha=0.3, linestyle='--')

        # 导出数据
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
    print(f"✅ 保存: {save_path}")

def plot_figure_g(predictions, output_dir):
    """
    图g: PLQY范围预测准确率热图
    """
    print("生成图g: PLQY范围准确率热图...")
    
    if 'PLQY' not in predictions:
        print("⚠️ 没有PLQY预测数据")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    actual = predictions['PLQY']['actual']
    predicted = predictions['PLQY']['predicted']
    
    # 移除NaN值
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    # 定义PLQY范围
    bins = [0, 0.1, 0.5, 1.0]
    labels = ['0-0.1', '0.1-0.5', '0.5-1.0']
    
    # 将实际值和预测值分组
    actual_binned = pd.cut(actual, bins=bins, labels=labels, include_lowest=True)
    predicted_binned = pd.cut(predicted, bins=bins, labels=labels, include_lowest=True)
    
    # 移除分组后的NaN值
    mask2 = ~(actual_binned.isna() | predicted_binned.isna())
    actual_binned = actual_binned[mask2]
    predicted_binned = predicted_binned[mask2]
    
    # 创建混淆矩阵
    cm = confusion_matrix(actual_binned, predicted_binned, labels=labels)
    
    # 归一化为百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 使用蓝色调色板
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # 绘制热图
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
    
    # 添加标签g
    ax.text(-0.15, 1.05, 'g', transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top')
    
    plt.tight_layout()
    save_path = output_dir / 'figure_g_plqy_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {save_path}")

    # 导出混淆矩阵数据
    try:
        cm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
        cm_df.to_csv(output_dir / 'figure_g_cm_data.csv')
    except Exception:
        pass

def generate_all_figures(project_dir, data_file, output_dir):
    """生成所有图表"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("生成论文图表")
    print("=" * 60)
    
    # 加载数据
    print("\n加载数据...")
    df = load_data(data_file)
    print(f"✅ 加载 {len(df)} 个样本")
    
    # 加载预测结果
    print("\n加载预测结果...")
    predictions = load_predictions(project_dir)
    if predictions:
        for key, value in predictions.items():
            print(f"✅ {key}: {len(value['actual'])} 个预测")
    
    # 生成各个图表
    print("\n生成图表...")
    print("-" * 40)
    
    # 图c: 波长-PLQY散点图
    plot_figure_c(df, output_path)
    
    # 图d: PLQY分布
    plot_figure_d(df, output_path)
    
    # 图e和f: 预测散点图
    if predictions:
        plot_figure_e_f(predictions, output_path)
        
        # 图g: PLQY范围准确率
        plot_figure_g(predictions, output_path)
    
    print("\n" + "=" * 60)
    print("✅ 所有图表生成完成！")
    print(f"保存位置: {output_path}")
    print("=" * 60)
    
    # 返回文件列表
    files = list(output_path.glob("figure_*.png"))
    return files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成论文图表')
    
    parser.add_argument('--project', '-p', default='paper_table',
                       help='项目目录')
    parser.add_argument('--data', '-d', default='../data/Database_normalized.csv',
                       help='数据文件')
    parser.add_argument('--output', '-o', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path(args.project) / 'paper_figures'
    
    # 生成所有图表
    files = generate_all_figures(args.project, args.data, output_dir)
    
    # 显示生成的文件
    print("\n📁 生成的图表文件：")
    print("-" * 40)
    for f in sorted(files):
        print(f"  {f}")

if __name__ == "__main__":
    main()
