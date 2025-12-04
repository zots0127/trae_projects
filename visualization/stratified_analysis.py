#!/usr/bin/env python3
"""
分段性能分析模块
生成模型在不同数值范围内的性能分析图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def plot_plqy_confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str = "PLQY Prediction Accuracy by Range",
    bins: List[float] = None,
    labels: List[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Dict:
    """
    生成PLQY预测准确性的混淆矩阵热图
    
    Args:
        actual: 实际值数组
        predicted: 预测值数组
        output_path: 输出路径
        title: 图表标题
        bins: 分段边界，默认[0, 0.1, 0.5, 1.0]
        labels: 分段标签，默认['0-0.1', '0.1-0.5', '0.5-1.0']
        figsize: 图表大小
    
    Returns:
        包含混淆矩阵和统计信息的字典
    """
    # 设置默认分段
    if bins is None:
        bins = [0, 0.1, 0.5, 1.0]
    if labels is None:
        labels = ['0-0.1', '0.1-0.5', '0.5-1.0']
    
    # 移除NaN值
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        print("⚠️ 没有有效数据用于生成混淆矩阵")
        return {}
    
    # 将实际值和预测值分组
    actual_binned = pd.cut(actual, bins=bins, labels=labels, include_lowest=True)
    predicted_binned = pd.cut(predicted, bins=bins, labels=labels, include_lowest=True)
    
    # 移除分组后的NaN值
    mask2 = ~(actual_binned.isna() | predicted_binned.isna())
    actual_binned = actual_binned[mask2]
    predicted_binned = predicted_binned[mask2]
    # 同时过滤原始数组
    actual_filtered = actual[mask2]
    predicted_filtered = predicted[mask2]
    
    # 创建混淆矩阵
    cm = confusion_matrix(actual_binned, predicted_binned, labels=labels)
    
    # 归一化为百分比（按行归一化）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
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
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_path / 'plqy_confusion_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算每个范围的统计信息
    range_stats = {}
    for i, label in enumerate(labels):
        mask = actual_binned == label
        if mask.sum() > 0:
            range_actual = actual_filtered[actual_binned == label]
            range_predicted = predicted_filtered[actual_binned == label]
            
            range_stats[label] = {
                'count': len(range_actual),
                'accuracy': cm_normalized[i, i],  # 对角线值
                'r2': r2_score(range_actual, range_predicted) if len(range_actual) > 1 else 0,
                'rmse': np.sqrt(mean_squared_error(range_actual, range_predicted)),
                'mae': mean_absolute_error(range_actual, range_predicted)
            }
    
    # 保存混淆矩阵数据
    cm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    cm_df.to_csv(output_path / 'plqy_confusion_matrix.csv')
    
    # 保存范围统计
    stats_df = pd.DataFrame(range_stats).T
    stats_df.to_csv(output_path / 'plqy_range_statistics.csv')
    
    print(f"✅ PLQY混淆矩阵已保存: {output_file}")
    
    return {
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'range_statistics': range_stats,
        'total_samples': len(actual)
    }


def plot_performance_by_range(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    target_name: str = "PLQY",
    bins: List[float] = None,
    labels: List[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> Dict:
    """
    生成不同数值范围内的性能对比图
    
    Args:
        actual: 实际值数组
        predicted: 预测值数组
        output_path: 输出路径
        target_name: 目标变量名称
        bins: 分段边界
        labels: 分段标签
        figsize: 图表大小
    
    Returns:
        包含性能统计的字典
    """
    # 根据目标设置默认分段
    if bins is None:
        if "PLQY" in target_name:
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            labels = ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
        elif "wavelength" in target_name.lower():
            bins = [400, 450, 500, 550, 600, 650, 700]
            labels = ['400-450', '450-500', '500-550', '550-600', '600-650', '650-700']
        else:
            # 自动分段
            bins = np.percentile(actual[~np.isnan(actual)], [0, 20, 40, 60, 80, 100])
            labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    
    # 移除NaN值
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        print("⚠️ 没有有效数据用于生成性能分析")
        return {}
    
    # 分组数据
    actual_binned = pd.cut(actual, bins=bins, labels=labels, include_lowest=True)
    
    # 计算每个范围的性能指标
    performance_data = []
    for label in labels:
        mask = actual_binned == label
        if mask.sum() > 1:  # 至少需要2个样本计算R²
            range_actual = actual[mask]
            range_predicted = predicted[mask]
            
            performance_data.append({
                'Range': label,
                'Count': len(range_actual),
                'R²': r2_score(range_actual, range_predicted),
                'RMSE': np.sqrt(mean_squared_error(range_actual, range_predicted)),
                'MAE': mean_absolute_error(range_actual, range_predicted)
            })
    
    if not performance_data:
        print("⚠️ 没有足够的数据生成性能分析")
        return {}
    
    perf_df = pd.DataFrame(performance_data)
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # R²柱状图
    ax = axes[0]
    bars = ax.bar(perf_df['Range'], perf_df['R²'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Range', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('R² Score by Range', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(perf_df['Range'], rotation=45, ha='right')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, perf_df['R²']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE柱状图
    ax = axes[1]
    bars = ax.bar(perf_df['Range'], perf_df['RMSE'], color='coral', alpha=0.8)
    ax.set_xlabel('Range', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('RMSE by Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(perf_df['Range'], rotation=45, ha='right')
    
    # 样本数量柱状图
    ax = axes[2]
    bars = ax.bar(perf_df['Range'], perf_df['Count'], color='seagreen', alpha=0.8)
    ax.set_xlabel('Range', fontsize=11)
    ax.set_ylabel('Sample Count', fontsize=11)
    ax.set_title('Sample Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(perf_df['Range'], rotation=45, ha='right')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, perf_df['Count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'{target_name} Performance by Range', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图表
    output_file = output_path / f'{target_name.lower().replace(" ", "_")}_performance_by_range.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据
    perf_df.to_csv(output_path / f'{target_name.lower().replace(" ", "_")}_performance_by_range.csv', index=False)
    
    print(f"✅ 性能分析图已保存: {output_file}")
    
    return {
        'performance_df': perf_df,
        'total_samples': len(actual),
        'num_ranges': len(perf_df)
    }


def generate_stratified_analysis(
    predictions: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    targets: List[str] = None
) -> Dict:
    """
    生成完整的分段性能分析
    
    Args:
        predictions: 预测结果字典，格式为 {target: {'actual': array, 'predicted': array}}
        output_dir: 输出目录
        targets: 要分析的目标列表，None表示分析所有
    
    Returns:
        分析结果字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建分段分析子目录
    analysis_dir = output_dir / 'stratified_analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # 确定要分析的目标
    if targets is None:
        targets = list(predictions.keys())
    
    for target in targets:
        if target not in predictions:
            print(f"⚠️ 未找到目标 {target} 的预测数据")
            continue
        
        print(f"\n分析目标: {target}")
        print("-" * 40)
        
        actual = predictions[target].get('actual')
        predicted = predictions[target].get('predicted')
        
        if actual is None or predicted is None:
            print(f"⚠️ 目标 {target} 缺少实际值或预测值")
            continue
        
        # 创建目标子目录
        target_dir = analysis_dir / target.replace('(', '').replace(')', '').replace('*', 'x')
        target_dir.mkdir(exist_ok=True)
        
        # 对PLQY生成混淆矩阵
        if "PLQY" in target:
            cm_result = plot_plqy_confusion_matrix(
                actual, predicted, target_dir,
                title=f"{target} Prediction Accuracy by Range"
            )
            results[f"{target}_confusion_matrix"] = cm_result
        
        # 生成性能分析图
        perf_result = plot_performance_by_range(
            actual, predicted, target_dir,
            target_name=target
        )
        results[f"{target}_performance"] = perf_result
    
    # 生成汇总报告
    generate_summary_report(results, analysis_dir)
    
    print(f"\n✅ 分段性能分析完成")
    print(f"   输出目录: {analysis_dir}")
    
    return results


def generate_summary_report(results: Dict, output_dir: Path):
    """
    生成分段分析的汇总报告
    
    Args:
        results: 分析结果字典
        output_dir: 输出目录
    """
    report_lines = []
    report_lines.append("# 分段性能分析报告\n")
    report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("\n## 分析摘要\n")
    
    for key, value in results.items():
        if 'confusion_matrix' in key:
            target = key.replace('_confusion_matrix', '')
            report_lines.append(f"\n### {target} - 混淆矩阵分析\n")
            
            if 'range_statistics' in value:
                report_lines.append("| 范围 | 样本数 | 准确率 | R² | RMSE | MAE |\n")
                report_lines.append("|------|--------|--------|-----|------|-----|\n")
                
                for range_name, stats in value['range_statistics'].items():
                    report_lines.append(
                        f"| {range_name} | {stats['count']} | "
                        f"{stats['accuracy']:.3f} | {stats['r2']:.3f} | "
                        f"{stats['rmse']:.3f} | {stats['mae']:.3f} |\n"
                    )
        
        elif 'performance' in key:
            target = key.replace('_performance', '')
            report_lines.append(f"\n### {target} - 性能分析\n")
            
            if 'performance_df' in value and value['performance_df'] is not None:
                df = value['performance_df']
                report_lines.append(df.to_markdown(index=False))
                report_lines.append("\n")
    
    # 保存报告
    report_path = output_dir / 'stratified_analysis_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    print(f"✅ 汇总报告已保存: {report_path}")


if __name__ == "__main__":
    # 测试代码
    print("分段性能分析模块测试")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # PLQY测试数据
    actual_plqy = np.random.beta(2, 5, n_samples)  # 偏向低值的分布
    predicted_plqy = actual_plqy + np.random.normal(0, 0.1, n_samples)
    predicted_plqy = np.clip(predicted_plqy, 0, 1)
    
    # 波长测试数据
    actual_wavelength = np.random.normal(550, 50, n_samples)
    predicted_wavelength = actual_wavelength + np.random.normal(0, 20, n_samples)
    
    # 创建预测字典
    predictions = {
        'PLQY': {
            'actual': actual_plqy,
            'predicted': predicted_plqy
        },
        'Max_wavelength(nm)': {
            'actual': actual_wavelength,
            'predicted': predicted_wavelength
        }
    }
    
    # 生成分析
    output_dir = Path('test_stratified_analysis')
    results = generate_stratified_analysis(predictions, output_dir)
    
    print("\n✅ 测试完成")