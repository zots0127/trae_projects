#!/usr/bin/env python3
"""
数据分析和可视化脚本 - 生成论文图表
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

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(data_file):
    """加载数据"""
    df = pd.read_csv(data_file)
    print(f"✅ 加载数据: {data_file}")
    print(f"   样本数: {len(df)}")
    print(f"   特征数: {len(df.columns)}")
    return df

def plot_wavelength_plqy_scatter(df, output_dir):
    """
    绘制波长-PLQY散点图（类似图c）
    按溶剂类型着色
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 定义溶剂类型和颜色
    solvent_colors = {
        'CH2Cl2': '#1f77b4',  # 蓝色
        'CH3CN': '#2ca02c',   # 绿色
        'Toluene': '#ff7f0e',  # 橙色
        'Others': '#9467bd'    # 紫色
    }
    
    # 提取波长和PLQY数据
    wavelength_col = None
    plqy_col = None
    
    for col in df.columns:
        if 'wavelength' in col.lower() or 'max_wavelength' in col.lower():
            wavelength_col = col
        if 'plqy' in col.lower():
            plqy_col = col
    
    if wavelength_col and plqy_col:
        # 如果有溶剂列，按溶剂分组
        if 'Solvent' in df.columns:
            for solvent, color in solvent_colors.items():
                mask = df['Solvent'] == solvent
                ax.scatter(df.loc[mask, wavelength_col], 
                          df.loc[mask, plqy_col],
                          c=color, label=solvent, alpha=0.6, s=20)
        else:
            # 没有溶剂信息，使用单一颜色
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
        print(f"✅ 保存散点图: {save_path}")
    else:
        print("⚠️ 未找到波长或PLQY列")

def plot_plqy_distribution(df, output_dir):
    """
    绘制PLQY分布直方图（类似图d）
    按溶剂和PLQY范围分组
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    plqy_col = None
    for col in df.columns:
        if 'plqy' in col.lower():
            plqy_col = col
            break
    
    if plqy_col:
        # 定义PLQY范围
        bins = [0, 0.1, 0.5, 1.0]
        labels = ['≤0.1', '0.1-0.5', '>0.5']
        
        # 如果有溶剂信息
        if 'Solvent' in df.columns:
            solvent_types = ['CH2Cl2', 'CH3CN', 'Toluene', 'Others']
            colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
            
            # 创建分组数据
            data_by_range = []
            for i in range(len(bins)-1):
                range_data = []
                for solvent in solvent_types:
                    mask = (df['Solvent'] == solvent) & \
                           (df[plqy_col] > bins[i]) & \
                           (df[plqy_col] <= bins[i+1])
                    range_data.append(mask.sum())
                data_by_range.append(range_data)
            
            # 绘制堆叠柱状图
            x = np.arange(len(labels))
            width = 0.6
            bottom = np.zeros(len(labels))
            
            for j, (solvent, color) in enumerate(zip(solvent_types, colors)):
                values = [data_by_range[i][j] for i in range(len(labels))]
                ax.bar(x, values, width, bottom=bottom, label=solvent, color=color)
                bottom += values
        else:
            # 简单直方图
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
        print(f"✅ 保存分布图: {save_path}")

def plot_prediction_scatter(df, predictions_file, output_dir):
    """
    绘制预测vs实验散点图（类似图e和f）
    """
    if not predictions_file or not Path(predictions_file).exists():
        print("⚠️ 未提供预测文件或文件不存在")
        return
    
    # 加载预测结果
    pred_df = pd.read_csv(predictions_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 查找波长和PLQY的预测列
    targets = ['wavelength', 'plqy']
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        # 查找相关列
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
            
            # 移除NaN值
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            # 计算指标
            r2 = r2_score(x, y)
            mae = mean_absolute_error(x, y)
            
            # 绘制散点图
            ax.scatter(x, y, alpha=0.5, s=10, c='#1f77b4')
            
            # 添加对角线
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=1, alpha=0.7)
            
            # 设置标签
            if 'wavelength' in target:
                ax.set_xlabel('Experimental λem (nm)', fontsize=12)
                ax.set_ylabel('Predicted λem (nm)', fontsize=12)
                title = f'Wavelength Prediction'
            else:
                ax.set_xlabel('Experimental PLQY', fontsize=12)
                ax.set_ylabel('Predicted PLQY', fontsize=12)
                title = f'PLQY Prediction'
            
            # 添加指标文本
            ax.text(0.05, 0.95, f'MAE = {mae:.2f}\nR² = {r2:.2f}',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(title, fontsize=13)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'prediction_scatter.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存预测散点图: {save_path}")

def plot_correlation_matrix(df, output_dir):
    """
    绘制相关性矩阵（类似图g）
    """
    # 选择PLQY相关的数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 选择关键列（如果存在）
    key_cols = []
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in ['plqy', 'wavelength', 'tau', 'lifetime']):
            key_cols.append(col)
    
    if len(key_cols) < 2:
        key_cols = numeric_cols[:min(10, len(numeric_cols))]  # 选择前10个数值列
    
    if len(key_cols) >= 2:
        # 计算相关性矩阵
        corr_matrix = df[key_cols].corr()
        
        # 创建热图
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 使用自定义颜色图
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # 绘制热图
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
        print(f"✅ 保存相关性矩阵: {save_path}")

def generate_summary_stats(df, output_dir):
    """生成汇总统计"""
    stats = {}
    
    # 基本统计
    stats['total_samples'] = len(df)
    stats['total_features'] = len(df.columns)
    
    # PLQY统计
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
    
    # 波长统计
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
    
    # 保存统计
    stats_file = output_dir / 'summary_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ 保存统计信息: {stats_file}")
    
    # 打印统计摘要
    print("\n" + "=" * 60)
    print("📊 数据统计摘要")
    print("=" * 60)
    print(f"样本总数: {stats['total_samples']}")
    
    if 'plqy' in stats:
        print(f"\nPLQY统计:")
        print(f"  均值: {stats['plqy']['mean']:.3f}")
        print(f"  标准差: {stats['plqy']['std']:.3f}")
        print(f"  范围: [{stats['plqy']['min']:.3f}, {stats['plqy']['max']:.3f}]")
    
    if 'wavelength' in stats:
        print(f"\n波长统计:")
        print(f"  均值: {stats['wavelength']['mean']:.1f} nm")
        print(f"  标准差: {stats['wavelength']['std']:.1f} nm")
        print(f"  范围: [{stats['wavelength']['min']:.1f}, {stats['wavelength']['max']:.1f}] nm")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据分析和可视化')
    
    parser.add_argument('--data', '-d', required=True,
                       help='数据文件路径')
    parser.add_argument('--predictions', '-p',
                       help='预测结果文件（可选）')
    parser.add_argument('--output', '-o',
                       help='输出目录')
    parser.add_argument('--plots', nargs='+',
                       choices=['scatter', 'distribution', 'prediction', 'correlation', 'all'],
                       default=['all'],
                       help='要生成的图表类型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("数据分析和可视化")
    print("=" * 60)
    print(f"数据文件: {args.data}")
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    df = load_data(args.data)
    
    # 生成图表
    if 'all' in args.plots or 'scatter' in args.plots:
        plot_wavelength_plqy_scatter(df, output_dir)
    
    if 'all' in args.plots or 'distribution' in args.plots:
        plot_plqy_distribution(df, output_dir)
    
    if 'all' in args.plots or 'prediction' in args.plots:
        if args.predictions:
            plot_prediction_scatter(df, args.predictions, output_dir)
    
    if 'all' in args.plots or 'correlation' in args.plots:
        plot_correlation_matrix(df, output_dir)
    
    # 生成统计摘要
    generate_summary_stats(df, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()