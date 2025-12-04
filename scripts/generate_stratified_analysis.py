#!/usr/bin/env python3
"""
生成分段性能分析的独立脚本
用于从训练结果生成PLQY混淆矩阵等分段分析图表
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from visualization.stratified_analysis import generate_stratified_analysis


def load_predictions_from_project(project_dir: Path) -> dict:
    """
    从项目目录加载预测结果
    
    Args:
        project_dir: 项目目录路径
    
    Returns:
        预测结果字典
    """
    predictions = {}
    
    print(f"从项目目录加载预测数据: {project_dir}")
    
    # 方法1: 查找CV预测结果文件
    for model_dir in project_dir.glob('*/'):
        # 查找cv_predictions文件
        for result_file in model_dir.glob('results/cv_predictions_*.csv'):
            target = result_file.stem.replace('cv_predictions_', '')
            
            # 读取预测结果
            df = pd.read_csv(result_file)
            if 'actual' in df.columns and 'predicted' in df.columns:
                predictions[target] = {
                    'actual': df['actual'].values,
                    'predicted': df['predicted'].values
                }
                print(f"  ✅ 加载 {target}: {len(df)} 个预测")
    
    # 方法2: 从training_results.json提取
    if not predictions:
        print("  未找到cv_predictions文件，尝试从training_results.json提取...")
        
        for model_dir in project_dir.glob('*/'):
            results_file = model_dir / 'results' / 'training_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # 提取各个模型的fold预测结果
                for model_name, model_data in results.items():
                    if isinstance(model_data, dict) and 'targets' in model_data:
                        for target, target_data in model_data['targets'].items():
                            if 'fold_results' in target_data:
                                all_actual = []
                                all_predicted = []
                                
                                for fold in target_data['fold_results']:
                                    if 'predictions' in fold:
                                        pred_data = fold['predictions']
                                        if isinstance(pred_data, dict):
                                            all_actual.extend(pred_data.get('actual', []))
                                            all_predicted.extend(pred_data.get('predicted', []))
                                
                                if all_actual:
                                    # 清理目标名称
                                    clean_target = target.replace('(', '').replace(')', '').replace('*', 'x')
                                    key = f'{clean_target}_{model_name}'
                                    predictions[key] = {
                                        'actual': np.array(all_actual),
                                        'predicted': np.array(all_predicted)
                                    }
                                    print(f"  ✅ 加载 {key}: {len(all_actual)} 个预测")
    
    # 方法3: 查找AutoML训练结果中的预测文件
    if not predictions:
        print("  尝试从AutoML训练结果加载...")
        
        # 查找all_models/automl_train目录
        automl_dir = project_dir / 'all_models' / 'automl_train'
        if automl_dir.exists():
            for model_dir in automl_dir.glob('*/'):
                model_name = model_dir.name
                
                # 查找CSV预测文件
                for pred_file in model_dir.glob('exports/csv/*_all_predictions.csv'):
                    try:
                        df = pd.read_csv(pred_file)
                        
                        # 从文件名解析目标
                        filename = pred_file.stem  # 例如: xgboost_PLQY_20250913_221305_all_predictions
                        parts = filename.split('_')
                        
                        # 找到目标名称（在模型名称之后，日期之前）
                        target = None
                        for i, part in enumerate(parts):
                            if part == model_name and i < len(parts) - 1:
                                # 目标名称可能包含多个部分
                                target_parts = []
                                for j in range(i+1, len(parts)):
                                    if parts[j].isdigit() and len(parts[j]) == 8:  # 日期部分
                                        break
                                    target_parts.append(parts[j])
                                if target_parts:
                                    target = '_'.join(target_parts)
                                    break
                        
                        if not target:
                            # 备用方法：检查列名
                            if 'Max_wavelength(nm)' in str(pred_file):
                                target = 'wavelength'
                            elif 'PLQY' in str(pred_file):
                                target = 'PLQY'
                            elif 'tau' in str(pred_file):
                                target = 'tau'
                        
                        if target and 'true' in df.columns and 'predicted' in df.columns:
                            key = f'{target}_{model_name}'
                            predictions[key] = {
                                'actual': df['true'].values,
                                'predicted': df['predicted'].values
                            }
                            print(f"  ✅ 加载 {key}: {len(df)} 个预测")
                    except Exception as e:
                        print(f"  ⚠️ 读取 {pred_file.name} 失败: {e}")
    
    # 方法4: 查找exports目录中的预测文件（保留原有方法）
    if not predictions:
        print("  尝试从exports目录加载...")
        
        for export_file in project_dir.glob('*/exports/json/*_complete.json'):
            try:
                with open(export_file, 'r') as f:
                    data = json.load(f)
                
                if 'cv_results' in data:
                    cv_data = data['cv_results']
                    if 'all_predictions' in cv_data:
                        target = data.get('target_column', 'unknown')
                        model = data.get('model_type', 'unknown')
                        
                        all_pred = cv_data['all_predictions']
                        if all_pred and len(all_pred) > 0:
                            # 提取实际值和预测值
                            actuals = []
                            predicteds = []
                            
                            for pred in all_pred:
                                if isinstance(pred, dict) and 'actual' in pred and 'predicted' in pred:
                                    actuals.append(pred['actual'])
                                    predicteds.append(pred['predicted'])
                            
                            if actuals:
                                clean_target = target.replace('(', '').replace(')', '').replace('*', 'x')
                                key = f'{clean_target}_{model}'
                                predictions[key] = {
                                    'actual': np.array(actuals),
                                    'predicted': np.array(predicteds)
                                }
                                print(f"  ✅ 加载 {key}: {len(actuals)} 个预测")
            except Exception as e:
                print(f"  ⚠️ 读取 {export_file.name} 失败: {e}")
    
    return predictions


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成分段性能分析图表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_stratified_analysis.py --project Paper_0913_123456 --output Paper_0913_123456/figures
  python generate_stratified_analysis.py --project results/training_001
        """
    )
    
    parser.add_argument(
        '--project', '-p',
        type=str,
        required=True,
        help='项目目录路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录路径（默认为项目目录/figures）'
    )
    
    parser.add_argument(
        '--targets', '-t',
        type=str,
        nargs='+',
        default=None,
        help='要分析的目标列表（默认分析所有）'
    )
    
    args = parser.parse_args()
    
    # 设置路径
    project_dir = Path(args.project)
    if not project_dir.exists():
        print(f"❌ 项目目录不存在: {project_dir}")
        return 1
    
    output_dir = Path(args.output) if args.output else project_dir / 'figures'
    
    print("="*60)
    print("分段性能分析")
    print("="*60)
    print(f"项目目录: {project_dir}")
    print(f"输出目录: {output_dir}")
    
    # 加载预测数据
    print("\n加载预测数据...")
    predictions = load_predictions_from_project(project_dir)
    
    if not predictions:
        print("❌ 未找到任何预测数据")
        return 1
    
    print(f"\n找到 {len(predictions)} 个预测数据集")
    
    # 生成分段分析
    print("\n生成分析图表...")
    results = generate_stratified_analysis(
        predictions=predictions,
        output_dir=output_dir,
        targets=args.targets
    )
    
    print("\n" + "="*60)
    print("✅ 分段性能分析完成")
    print("="*60)
    print(f"输出目录: {output_dir / 'stratified_analysis'}")
    
    # 列出生成的文件
    analysis_dir = output_dir / 'stratified_analysis'
    if analysis_dir.exists():
        print("\n生成的文件:")
        for file in sorted(analysis_dir.rglob('*')):
            if file.is_file():
                print(f"  - {file.relative_to(output_dir)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())