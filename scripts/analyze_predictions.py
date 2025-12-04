#!/usr/bin/env python3
"""
预测结果分析脚本 - 生成PLQY范围准确率热图和其他分析图表
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

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_predictions(project_dir, model_name=None):
    """
    加载预测结果
    
    Args:
        project_dir: 项目目录
        model_name: 指定模型名称，如果为None则使用最佳模型
    
    Returns:
        DataFrame包含实际值和预测值
    """
    project_path = Path(project_dir)
    
    if not project_path.exists():
        print(f"❌ 项目目录不存在: {project_dir}")
        return None
    
    # 如果没有指定模型，找最佳模型（这里默认用xgboost）
    if model_name is None:
        model_name = 'xgboost'
    
    model_dir = project_path / model_name
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        # 尝试查找其他可能的目录
        for possible_name in ['xgboost', 'lightgbm', 'catboost', 'gradient_boosting']:
            model_dir = project_path / possible_name
            if model_dir.exists():
                print(f"✅ 使用模型: {possible_name}")
                break
        else:
            print("❌ 找不到任何模型目录")
            return None
    
    # 查找预测文件
    predictions_dir = model_dir / 'predictions'
    if not predictions_dir.exists():
        print(f"❌ 预测目录不存在: {predictions_dir}")
        return None
    
    # 收集所有目标的预测结果
    all_predictions = {}
    
    # 按目标类型收集所有fold的数据
    target_types = {'wavelength': [], 'PLQY': [], 'tau': []}
    
    # 查找CSV文件
    csv_files = list(predictions_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        # 从文件名提取目标类型
        filename = csv_file.stem
        
        # 判断目标类型
        target_type = None
        if 'wavelength' in filename.lower():
            target_type = 'wavelength'
        elif 'plqy' in filename.lower():
            target_type = 'PLQY'
        elif 'tau' in filename.lower():
            target_type = 'tau'
        else:
            continue
        
        # 读取预测数据
        try:
            df = pd.read_csv(csv_file)
            
            # 查找实际值和预测值列
            actual_col = None
            pred_col = None
            
            for col in df.columns:
                if 'actual' in col.lower() or 'true' in col.lower() or 'experimental' in col.lower():
                    actual_col = col
                elif 'predict' in col.lower() or 'pred' in col.lower():
                    pred_col = col
            
            if actual_col and pred_col:
                # 使用验证集数据（如果有split列）
                if 'split' in df.columns:
                    # 优先使用test，如果没有则使用val
                    if 'test' in df['split'].values:
                        test_df = df[df['split'] == 'test']
                    elif 'val' in df['split'].values:
                        test_df = df[df['split'] == 'val']
                    else:
                        # 如果都没有，使用所有数据
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
    
    # 合并所有fold的数据
    for target_type in ['wavelength', 'PLQY', 'tau']:
        if target_types[target_type]:
            actual_all = np.concatenate([d['actual'] for d in target_types[target_type]])
            predicted_all = np.concatenate([d['predicted'] for d in target_types[target_type]])
            
            all_predictions[target_type] = {
                'actual': actual_all,
                'predicted': predicted_all
            }
            print(f"✅ 加载 {target_type} 预测数据: {len(actual_all)} 个样本")
    
    return all_predictions

def plot_plqy_range_accuracy(predictions, output_dir):
    """
    绘制PLQY范围预测准确率热图（类似图g）
    
    Args:
        predictions: 包含actual和predicted的字典
        output_dir: 输出目录
    """
    if 'PLQY' not in predictions:
        print("⚠️ 没有PLQY预测数据")
        return
    
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
    
    # 移除分组后的NaN值（可能因为超出范围）
    mask2 = ~(actual_binned.isna() | predicted_binned.isna())
    actual_binned = actual_binned[mask2]
    predicted_binned = predicted_binned[mask2]
    
    # 创建混淆矩阵
    cm = confusion_matrix(actual_binned, predicted_binned, labels=labels)
    
    # 归一化为百分比（按行归一化，即每个实际范围内的预测分布）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
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
                ax=ax)
    
    ax.set_xlabel('Predicted PLQY Range', fontsize=12)
    ax.set_ylabel('Actual PLQY Range', fontsize=12)
    ax.set_title('PLQY Prediction Accuracy by Range', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    save_path = output_dir / 'plqy_range_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存PLQY范围准确率图: {save_path}")
    
    # 打印统计信息
    print("\n📊 PLQY范围预测统计:")
    print("-" * 40)
    for i, actual_label in enumerate(labels):
        total = cm[i].sum()
        if total > 0:
            accuracy = cm[i, i] / total
            print(f"{actual_label}: {accuracy:.2%} 准确率 ({cm[i, i]}/{total} 样本)")
    
    # 计算整体准确率
    overall_accuracy = np.trace(cm) / cm.sum()
    print(f"\n整体准确率: {overall_accuracy:.2%}")

def plot_prediction_scatter_all(predictions, output_dir):
    """
    绘制所有目标的预测散点图
    """
    n_targets = len(predictions)
    if n_targets == 0:
        print("⚠️ 没有预测数据")
        return
    
    # 创建子图
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))
    
    if n_targets == 1:
        axes = [axes]
    
    target_names = {
        'wavelength': 'λem (nm)',
        'PLQY': 'PLQY', 
        'tau': 'τ (μs)'
    }
    
    for idx, (target, data) in enumerate(predictions.items()):
        ax = axes[idx]
        
        actual = data['actual']
        predicted = data['predicted']
        
        # 移除NaN值
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        # 计算指标
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        
        # 绘制散点图
        ax.scatter(actual, predicted, alpha=0.5, s=20, c='#1f77b4')
        
        # 添加对角线
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', lw=1, alpha=0.7, label='Perfect prediction')
        
        # 设置标签
        display_name = target_names.get(target, target)
        ax.set_xlabel(f'Actual {display_name}', fontsize=11)
        ax.set_ylabel(f'Predicted {display_name}', fontsize=11)
        ax.set_title(f'{display_name} Prediction', fontsize=12)
        
        # 添加指标文本
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    plt.suptitle('Model Prediction Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # 保存图形
    save_path = output_dir / 'prediction_scatter_all.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存预测散点图: {save_path}")

def plot_residual_analysis(predictions, output_dir):
    """
    绘制残差分析图
    """
    for target, data in predictions.items():
        actual = data['actual']
        predicted = data['predicted']
        
        # 移除NaN值
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        # 计算残差
        residuals = predicted - actual
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 残差vs预测值
        ax = axes[0, 0]
        ax.scatter(predicted, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # 2. 残差直方图
        ax = axes[0, 1]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Q图
        ax = axes[1, 0]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # 4. 残差vs实际值
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
        
        # 保存图形
        save_path = output_dir / f'residual_analysis_{target}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 保存残差分析图: {save_path}")

def generate_prediction_report(predictions, output_dir):
    """
    生成预测分析报告
    """
    report = {}
    
    for target, data in predictions.items():
        actual = data['actual']
        predicted = data['predicted']
        
        # 移除NaN值
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        # 计算各种指标
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
    
    # 保存报告
    report_file = output_dir / 'prediction_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ 保存预测报告: {report_file}")
    
    # 打印报告摘要
    print("\n" + "=" * 60)
    print("📊 预测性能报告")
    print("=" * 60)
    
    target_names = {
        'wavelength': 'Wavelength (nm)',
        'PLQY': 'PLQY',
        'tau': 'Lifetime (μs)'
    }
    
    for target, metrics in report.items():
        display_name = target_names.get(target, target)
        print(f"\n{display_name}:")
        print("-" * 40)
        print(f"  样本数: {metrics['n_samples']}")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  实际范围: [{metrics['actual_range'][0]:.2f}, {metrics['actual_range'][1]:.2f}]")
        print(f"  预测范围: [{metrics['predicted_range'][0]:.2f}, {metrics['predicted_range'][1]:.2f}]")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预测结果分析')
    
    parser.add_argument('project', help='项目目录')
    parser.add_argument('--model', '-m', help='模型名称（默认使用最佳模型）')
    parser.add_argument('--output', '-o', help='输出目录')
    parser.add_argument('--plots', nargs='+',
                       choices=['range', 'scatter', 'residual', 'all'],
                       default=['all'],
                       help='要生成的图表类型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.project) / 'analysis'
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("预测结果分析")
    print("=" * 60)
    print(f"项目: {args.project}")
    print(f"输出目录: {output_dir}")
    
    # 加载预测数据
    predictions = load_predictions(args.project, args.model)
    
    if not predictions:
        print("❌ 无法加载预测数据")
        return
    
    # 生成图表
    if 'all' in args.plots or 'range' in args.plots:
        plot_plqy_range_accuracy(predictions, output_dir)
    
    if 'all' in args.plots or 'scatter' in args.plots:
        plot_prediction_scatter_all(predictions, output_dir)
    
    if 'all' in args.plots or 'residual' in args.plots:
        plot_residual_analysis(predictions, output_dir)
    
    # 生成报告
    generate_prediction_report(predictions, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()