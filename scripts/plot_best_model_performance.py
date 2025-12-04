#!/usr/bin/env python3
"""
生成最佳模型的性能图表
包括预测散点图和PLQY混淆矩阵
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error
import json

def find_best_models(comparison_file):
    """从对比表中找出最佳模型"""
    df = pd.read_csv(comparison_file)
    
    best_models = {}
    for target in df['Target'].unique():
        target_df = df[df['Target'] == target]
        best_idx = target_df['R2_mean'].idxmax()
        best_model = target_df.loc[best_idx]
        
        # 标准化目标名称
        if 'wavelength' in target.lower():
            key = 'wavelength'
        elif 'plqy' in target.lower():
            key = 'plqy'
        else:
            continue
            
        best_models[key] = {
            'model': best_model['Model'].lower(),
            'r2': best_model['R2_mean'],
            'mae': best_model['MAE_mean'],
            'rmse': best_model['RMSE_mean']
        }
        
    return best_models

def load_predictions(project_dir, model_name, target):
    """加载模型的交叉验证预测结果"""
    # 查找预测文件
    pattern = f"all_models/automl_train/{model_name}/exports/csv/{model_name}_{target}_*_all_predictions.csv"
    files = list(Path(project_dir).glob(pattern))
    
    if not files:
        print(f"⚠️ 未找到预测文件: {pattern}")
        return None
        
    # 读取预测数据
    df = pd.read_csv(files[0])
    return df

def plot_scatter(true_values, pred_values, target_name, metrics, output_dir):
    """绘制预测散点图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 设置颜色
    if 'wavelength' in target_name.lower():
        color = '#2E86AB'  # 蓝色
        unit = ' (nm)'
    else:
        color = '#F24236'  # 橙红色
        unit = ''
    
    # 绘制散点
    ax.scatter(true_values, pred_values, 
              alpha=0.6, s=20, color=color, edgecolors='none')
    
    # 添加对角线
    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', lw=2, alpha=0.8, label='Perfect prediction')
    
    # 添加指标文本
    text = f"MAE = {metrics['mae']:.2f}{unit}\n"
    text += f"R² = {metrics['r2']:.2f}"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           fontsize=16, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 设置标签
    if 'wavelength' in target_name.lower():
        ax.set_xlabel('Experimental λₑₘ (nm)', fontsize=14)
        ax.set_ylabel('Predicted λₑₘ (nm)', fontsize=14)
        title = 'Wavelength Prediction Performance'
    else:
        ax.set_xlabel('Experimental PLQY', fontsize=14)
        ax.set_ylabel('Predicted PLQY', fontsize=14)
        title = 'PLQY Prediction Performance'
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围相同
    ax.set_aspect('equal', adjustable='box')
    
    # 保存图片
    output_file = output_dir / f'{target_name}_scatter.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 保存散点图: {output_file}")
    return output_file

def plot_confusion_matrix_with_bins(true_values, pred_values, bins, labels, title_suffix, output_dir, filename_suffix):
    """绘制指定区间的混淆矩阵"""
    
    # 将连续值转换为分类
    true_cats = pd.cut(true_values, bins=bins, labels=labels, include_lowest=True)
    pred_cats = pd.cut(pred_values, bins=bins, labels=labels, include_lowest=True)
    
    # 移除NaN值
    valid_mask = ~(true_cats.isna() | pred_cats.isna())
    true_cats_clean = true_cats[valid_mask]
    pred_cats_clean = pred_cats[valid_mask]
    
    # 检查是否有足够的数据
    if len(true_cats_clean) == 0:
        print(f"  ⚠️ 没有有效数据用于生成混淆矩阵 ({title_suffix})")
        return None, None
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_cats_clean, pred_cats_clean, labels=labels)
    
    # 归一化（按行）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 处理NaN (当某行全为0时)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # 确定图片大小
    n_bins = len(labels)
    if n_bins <= 5:
        figsize = (10, 8)
        annot_size = 10
    elif n_bins <= 10:
        figsize = (14, 12)
        annot_size = 8
    else:
        figsize = (16, 14)
        annot_size = 6
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用不同的颜色方案
    if '10x10' in title_suffix:
        cmap = 'YlOrRd'
    elif '5x5' in title_suffix:
        cmap = 'Blues'
    else:
        cmap = 'viridis'
    
    # 创建注释 - 显示实际数量和百分比
    annot_data = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            if cm[i, j] > 0:
                row.append(f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})')
            else:
                row.append('')
        annot_data.append(row)
    
    sns.heatmap(cm_normalized, annot=annot_data, fmt='', 
                cmap=cmap, vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion'},
                annot_kws={'size': annot_size},
                ax=ax)
    
    # 旋转标签以提高可读性
    if n_bins > 5:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    ax.set_xlabel('Predicted PLQY Range', fontsize=12)
    ax.set_ylabel('Experimental PLQY Range', fontsize=12)
    ax.set_title(f'PLQY Confusion Matrix - {title_suffix}', fontsize=14, fontweight='bold')
    
    # 添加对角线准确率
    diagonal_acc = np.trace(cm_normalized) / len(labels)
    ax.text(0.02, 0.98, f'Diagonal Accuracy: {diagonal_acc:.2%}', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图片
    output_file = output_dir / f'plqy_confusion_matrix_{filename_suffix}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 保存混淆矩阵: {output_file}")
    
    # 保存混淆矩阵数据
    cm_df = pd.DataFrame(cm, 
                        index=[f'True_{l}' for l in labels],
                        columns=[f'Pred_{l}' for l in labels])
    cm_file = output_dir / f'plqy_confusion_matrix_{filename_suffix}.csv'
    cm_df.to_csv(cm_file)
    
    # 保存归一化版本
    cm_norm_df = pd.DataFrame(cm_normalized, 
                             index=[f'True_{l}' for l in labels],
                             columns=[f'Pred_{l}' for l in labels])
    cm_norm_file = output_dir / f'plqy_confusion_matrix_{filename_suffix}_normalized.csv'
    cm_norm_df.to_csv(cm_norm_file)
    
    return output_file, cm_file

def plot_plqy_confusion_matrix(true_values, pred_values, output_dir):
    """绘制多种配置的PLQY混淆矩阵"""
    
    results = {}
    
    # 创建confusion_matrices子目录
    cm_dir = output_dir / 'confusion_matrices'
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n  生成多种混淆矩阵配置...")
    
    # 1. 10x10 矩阵 (0.1间隔)
    print("    - 10x10矩阵 (0.1间隔)...")
    bins_10x10 = [i/10 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    labels_10x10 = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    
    img_10x10, data_10x10 = plot_confusion_matrix_with_bins(
        true_values, pred_values, 
        bins_10x10, labels_10x10,
        '10x10 (0.1 intervals)',
        cm_dir, '10x10'
    )
    results['10x10'] = {'image': str(img_10x10), 'data': str(data_10x10)}
    
    # 2. 5x5 矩阵 (0.2间隔)
    print("    - 5x5矩阵 (0.2间隔)...")
    bins_5x5 = [i/5 for i in range(6)]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_5x5 = [f'{i/5:.1f}-{(i+1)/5:.1f}' for i in range(5)]
    
    img_5x5, data_5x5 = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_5x5, labels_5x5,
        '5x5 (0.2 intervals)',
        cm_dir, '5x5'
    )
    results['5x5'] = {'image': str(img_5x5), 'data': str(data_5x5)}
    
    # 3. 自定义范围 - 低中高 (3x3)
    print("    - 3x3矩阵 (低中高)...")
    bins_3x3 = [0, 0.3, 0.7, 1.0]
    labels_3x3 = ['Low (0-0.3)', 'Medium (0.3-0.7)', 'High (0.7-1.0)']
    
    img_3x3, data_3x3 = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_3x3, labels_3x3,
        'Low-Medium-High',
        cm_dir, '3x3_custom'
    )
    results['3x3_custom'] = {'image': str(img_3x3), 'data': str(data_3x3)}
    
    # 4. 精细分类 - 重点关注高PLQY区域
    print("    - 高PLQY精细矩阵...")
    bins_high = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    labels_high = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.7', 
                   '0.7-0.8', '0.8-0.85', '0.85-0.9', '0.9-0.95', '0.95-1.0']
    
    img_high, data_high = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_high, labels_high,
        'High PLQY Focus',
        cm_dir, 'high_plqy_focus'
    )
    results['high_plqy_focus'] = {'image': str(img_high), 'data': str(data_high)}
    
    # 5. 自定义4x4矩阵 (0-0.1, 0.1-0.4, 0.4-0.7, 0.7-1.0)
    print("    - 4x4矩阵 (0-0.1, 0.1-0.4, 0.4-0.7, 0.7-1.0)...")
    bins_4x4 = [0, 0.1, 0.4, 0.7, 1.0]
    labels_4x4 = ['0-0.1', '0.1-0.4', '0.4-0.7', '0.7-1.0']

    img_4x4, data_4x4 = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_4x4, labels_4x4,
        '4x4 (0-0.1, 0.1-0.4, 0.4-0.7, 0.7-1.0)',
        cm_dir, '4x4_custom'
    )
    results['4x4_custom'] = {'image': str(img_4x4), 'data': str(data_4x4)}
    
    # 保留原始的简单混淆矩阵（向后兼容）
    bins_simple = [0, 0.1, 0.5, 1.0]
    labels_simple = ['0-0.1', '0.1-0.5', '0.5-1.0']
    img_simple, data_simple = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_simple, labels_simple,
        'Simple (Original)',
        output_dir, 'simple'
    )
    
    # 汇总统计
    print("\n  混淆矩阵生成完成:")
    print(f"    • 10x10 (精细): {cm_dir}/plqy_confusion_matrix_10x10.png")
    print(f"    • 5x5 (中等): {cm_dir}/plqy_confusion_matrix_5x5.png")
    print(f"    • 3x3 (简单): {cm_dir}/plqy_confusion_matrix_3x3_custom.png")
    print(f"    • 高PLQY精细: {cm_dir}/plqy_confusion_matrix_high_plqy_focus.png")
    print(f"    • 4x4 (自定义): {cm_dir}/plqy_confusion_matrix_4x4_custom.png")
    
    return results

def save_prediction_data(true_values, pred_values, target_name, output_dir):
    """保存预测数据到CSV"""
    df = pd.DataFrame({
        'true': true_values,
        'predicted': pred_values,
        'error': pred_values - true_values,
        'absolute_error': np.abs(pred_values - true_values),
        'percentage_error': 100 * np.abs(pred_values - true_values) / np.abs(true_values)
    })
    
    output_file = output_dir / f'best_model_{target_name}_predictions.csv'
    df.to_csv(output_file, index=False)
    print(f"  ✅ 保存预测数据: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='生成最佳模型性能图表')
    parser.add_argument('--project', '-p', required=True,
                       help='项目目录')
    parser.add_argument('--output', '-o', 
                       help='输出目录（默认: project/figures/model_performance）')
    
    args = parser.parse_args()
    
    project_dir = Path(args.project)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_dir / 'figures' / 'model_performance'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("生成最佳模型性能图表")
    print("="*80)
    print(f"项目目录: {project_dir}")
    print(f"输出目录: {output_dir}")
    
    # 1. 找出最佳模型
    comparison_file = project_dir / 'model_comparison_detailed.csv'
    if not comparison_file.exists():
        print(f"❌ 未找到模型对比文件: {comparison_file}")
        return
        
    print("\n查找最佳模型...")
    best_models = find_best_models(comparison_file)
    
    for target, info in best_models.items():
        print(f"  {target.upper()}: {info['model'].upper()} (R²={info['r2']:.4f}, MAE={info['mae']:.4f})")
    
    # 2. 处理每个目标
    results = {}
    
    for target_key, model_info in best_models.items():
        print(f"\n处理 {target_key.upper()} 预测...")
        
        # 确定目标列名
        if target_key == 'wavelength':
            target_col = 'Max_wavelength(nm)'
        else:
            target_col = 'PLQY'
            
        # 加载预测数据
        pred_df = load_predictions(project_dir, model_info['model'], target_col)
        if pred_df is None:
            continue
            
        true_values = pred_df['true'].values
        pred_values = pred_df['predicted'].values
        
        # 计算指标
        metrics = {
            'r2': r2_score(true_values, pred_values),
            'mae': mean_absolute_error(true_values, pred_values),
            'rmse': np.sqrt(np.mean((true_values - pred_values)**2))
        }
        
        print(f"  实际指标: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
        
        # 绘制散点图
        plot_scatter(true_values, pred_values, target_key, metrics, output_dir)
        
        # 保存数据
        save_prediction_data(true_values, pred_values, target_key, output_dir)
        
        results[target_key] = {
            'model': model_info['model'],
            'metrics': metrics,
            'n_samples': len(true_values)
        }
        
        # 如果是PLQY，额外生成混淆矩阵
        if target_key == 'plqy':
            plot_plqy_confusion_matrix(true_values, pred_values, output_dir)
    
    # 3. 保存汇总信息
    summary = {
        'project': str(project_dir),
        'best_models': best_models,
        'actual_performance': results
    }
    
    summary_file = output_dir / 'performance_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ 保存性能汇总: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ 最佳模型性能图表生成完成!")
    print("="*80)
    print(f"输出目录: {output_dir}")
    print("生成的文件:")
    for f in output_dir.glob('*'):
        if f.is_file():
            print(f"  • {f.name}")

if __name__ == "__main__":
    main()