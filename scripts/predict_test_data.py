#!/usr/bin/env python3
"""
使用训练好的XGBoost模型预测测试数据集
输入: Database_ours_0903update_normalized.csv
输出: test_predict.csv
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
from datetime import datetime
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_extractor import FeatureExtractor

def load_models(project_dir, model_name='xgboost', use_intersection=False):
    """加载训练好的模型
    
    Args:
        project_dir: 项目目录
        model_name: 模型名称
        use_intersection: 是否使用交集训练的模型
    """
    print("\n" + "="*80)
    print("加载预训练模型")
    print("-"*80)
    
    models = {}
    
    # 根据是否使用交集选择模型目录
    if use_intersection:
        # 交集训练的模型通常在 intersection 子目录
        model_dir = Path(project_dir) / model_name / 'intersection' / f'{model_name}_intersection' / 'models'
        if not model_dir.exists():
            # 尝试其他可能的路径
            model_dir = Path(project_dir) / model_name / 'intersection' / 'models'
            if not model_dir.exists():
                model_dir = Path(project_dir) / f'{model_name}_intersection' / 'models'
        print(f"  📌 使用交集训练模型")
    else:
        # 尝试多种可能的模型路径
        model_dir = Path(project_dir) / 'all_models' / 'automl_train' / model_name / 'models'
        if not model_dir.exists():
            # 尝试旧版路径
            model_dir = Path(project_dir) / model_name / 'models'
        print(f"  📌 使用完整数据训练模型")
    
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return models
    
    print(f"  📁 模型目录: {model_dir}")
    
    # 查找模型文件（只加载wavelength和PLQY）
    for model_file in model_dir.glob("*.joblib"):
        filename = model_file.stem
        if 'wavelength' in filename.lower():
            models['wavelength'] = joblib.load(model_file)
            print(f"  ✅ 波长模型: {model_file.name}")
        elif 'plqy' in filename.lower():
            models['PLQY'] = joblib.load(model_file)
            print(f"  ✅ PLQY模型: {model_file.name}")
    
    print(f"\n成功加载 {len(models)} 个模型")
    return models

def extract_features_for_test(df, feature_type='combined', combination_method='mean', descriptor_count=115):
    """提取测试数据特征"""
    print("\n" + "="*80)
    print("特征提取")
    print("-"*80)
    print(f"  • 样本数: {len(df):,}")
    print(f"  • 特征类型: {feature_type}")
    
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True,
        descriptor_count=descriptor_count
    )
    
    features_list = []
    valid_indices = []
    failed_count = 0
    
    start_time = time.time()
    
    # 提取特征
    for idx, row in df.iterrows():
        try:
            # 提取组合特征
            smiles_list = [row['L1'], row['L2'], row['L3']]
            features = extractor.extract_combination(
                smiles_list,
                feature_type=feature_type,
                combination_method=combination_method
            )
            
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
            else:
                failed_count += 1
                print(f"  ⚠️ 行 {idx}: 特征提取失败")
        except Exception as e:
            failed_count += 1
            print(f"  ⚠️ 行 {idx}: {str(e)}")
    
    elapsed = time.time() - start_time
    
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].copy()
        print(f"\n✅ 特征提取完成:")
        print(f"  • 成功: {len(X):,} 个样本")
        print(f"  • 失败: {failed_count:,} 个样本")
        print(f"  • 用时: {elapsed:.2f} 秒")
        return X, df_valid
    else:
        print(f"\n❌ 特征提取失败")
        return None, None

def predict_test(models, X, df_valid):
    """预测测试数据"""
    print("\n" + "="*80)
    print("批量预测")
    print("-"*80)
    print(f"  • 样本数: {len(X):,}")
    
    predictions = {}
    
    # 预测每个目标
    for target, model in models.items():
        print(f"\n预测 {target}...")
        start_time = time.time()
        
        preds = model.predict(X)
        predictions[target] = preds
        
        elapsed = time.time() - start_time
        print(f"  ✅ 完成:")
        print(f"    • 最小值: {preds.min():.6f}")
        print(f"    • 最大值: {preds.max():.6f}")
        print(f"    • 平均值: {preds.mean():.6f}")
        print(f"    • 标准差: {preds.std():.6f}")
        print(f"    • 用时: {elapsed:.3f} 秒")
    
    # 添加预测到DataFrame
    if 'wavelength' in predictions:
        df_valid['Predicted_wavelength'] = predictions['wavelength']
    if 'PLQY' in predictions:
        df_valid['Predicted_PLQY'] = predictions['PLQY']
    
    return df_valid

def compare_with_actual(df):
    """如果存在实际值，进行比较"""
    print("\n" + "="*80)
    print("预测结果分析")
    print("-"*80)
    
    # 检查是否有实际值
    has_actual_wavelength = 'Max_wavelength(nm)' in df.columns and not df['Max_wavelength(nm)'].isna().all()
    has_actual_plqy = 'PLQY' in df.columns and not df['PLQY'].isna().all()
    
    if has_actual_wavelength and 'Predicted_wavelength' in df.columns:
        actual = df['Max_wavelength(nm)'].dropna()
        pred = df.loc[actual.index, 'Predicted_wavelength']
        
        if len(actual) > 0:
            mae = np.abs(actual - pred).mean()
            rmse = np.sqrt(((actual - pred) ** 2).mean())
            r2 = 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
            
            print(f"\n📊 波长预测性能:")
            print(f"  • 样本数: {len(actual)}")
            print(f"  • MAE: {mae:.2f} nm")
            print(f"  • RMSE: {rmse:.2f} nm")
            print(f"  • R²: {r2:.4f}")
    
    if has_actual_plqy and 'Predicted_PLQY' in df.columns:
        # 处理PLQY单位（如果是百分比转换为小数）
        actual = df['PLQY'].dropna()
        if actual.max() > 1.5:  # 可能是百分比
            actual = actual / 100
        
        pred = df.loc[actual.index, 'Predicted_PLQY']
        
        if len(actual) > 0:
            mae = np.abs(actual - pred).mean()
            rmse = np.sqrt(((actual - pred) ** 2).mean())
            r2 = 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
            
            print(f"\n📊 PLQY预测性能:")
            print(f"  • 样本数: {len(actual)}")
            print(f"  • MAE: {mae:.4f}")
            print(f"  • RMSE: {rmse:.4f}")
            print(f"  • R²: {r2:.4f}")
    
    # PLQY分布
    if 'Predicted_PLQY' in df.columns:
        plqy = df['Predicted_PLQY']
        print(f"\n📊 PLQY预测分布:")
        print(f"  • 最小值: {plqy.min():.4f}")
        print(f"  • 25分位: {plqy.quantile(0.25):.4f}")
        print(f"  • 中位数: {plqy.median():.4f}")
        print(f"  • 75分位: {plqy.quantile(0.75):.4f}")
        print(f"  • 最大值: {plqy.max():.4f}")
        print(f"  • 平均值: {plqy.mean():.4f}")
        
        # 范围分布
        ranges = [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
        print(f"\n  PLQY范围分布:")
        for min_val, max_val in ranges:
            count = ((plqy >= min_val) & (plqy < max_val)).sum()
            pct = 100 * count / len(plqy)
            print(f"    [{min_val:.1f}, {max_val:.1f}): {count:4d} ({pct:5.1f}%)")
    
    # Top 5高PLQY样本
    if 'Predicted_PLQY' in df.columns:
        print(f"\n🏆 Top 5 高PLQY预测:")
        top5 = df.nlargest(5, 'Predicted_PLQY')
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"\n  #{i} (行号: {idx}):")
            print(f"    PLQY预测: {row['Predicted_PLQY']:.4f}")
            if 'Predicted_wavelength' in df.columns:
                print(f"    波长预测: {row['Predicted_wavelength']:.1f} nm")
            if 'PLQY' in row and pd.notna(row['PLQY']):
                actual_plqy = row['PLQY']
                if actual_plqy > 1.5:
                    actual_plqy = actual_plqy / 100
                print(f"    PLQY实际: {actual_plqy:.4f}")
            if 'Max_wavelength(nm)' in row and pd.notna(row['Max_wavelength(nm)']):
                print(f"    波长实际: {row['Max_wavelength(nm)']:.1f} nm")

def main():
    parser = argparse.ArgumentParser(description='预测测试数据集')
    parser.add_argument('--project', '-p',
                       help='模型项目目录 (默认: paper_table)')
    parser.add_argument('--input', '-i', 
                       default='Database_ours_0903update_normalized.csv',
                       help='测试数据文件')
    parser.add_argument('--output', '-o',
                       help='输出文件 (默认: PROJECT/test_predict.csv)')
    parser.add_argument('--model', '-m',
                       default='xgboost',
                       help='模型类型')
    parser.add_argument('--intersection', action='store_true',
                       help='使用交集训练的模型（只用三个目标都有值的数据训练）')
    parser.add_argument('--combination-method', default='mean',
                       choices=['mean', 'sum', 'concat'],
                       help='多个配体特征的合并方式')
    parser.add_argument('--descriptor-count', type=int, default=115,
                       help='分子描述符数量')
    
    args = parser.parse_args()
    
    # 设置默认项目目录为固定名称
    if not args.project:
        args.project = 'paper_table'
    
    # 设置默认输出路径
    if not args.output:
        if args.intersection:
            args.output = f"{args.project}/test_predict_intersection.csv"
        else:
            args.output = f"{args.project}/test_predict.csv"
    
    print("="*80)
    print("测试数据预测")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n配置:")
    print(f"  • 输入文件: {args.input}")
    print(f"  • 模型目录: {args.project}/{args.model}")
    print(f"  • 模型类型: {'交集训练模型' if args.intersection else '完整数据训练模型'}")
    print(f"  • 输出文件: {args.output}")
    print(f"  • 特征合并: {args.combination_method}")
    print(f"  • 描述符数量: {args.descriptor_count}")
    
    start_time = time.time()
    
    # 1. 加载测试数据
    print(f"\n加载测试数据...")
    df = pd.read_csv(args.input)
    print(f"  ✅ 加载 {len(df):,} 个样本")
    print(f"  ✅ 列: {', '.join(df.columns[:10])}")
    
    # 2. 加载模型
    models = load_models(args.project, args.model, use_intersection=args.intersection)
    if not models:
        print("❌ 没有找到模型")
        return
    
    # 3. 提取特征
    X, df_valid = extract_features_for_test(
        df,
        combination_method=args.combination_method,
        descriptor_count=args.descriptor_count
    )
    if X is None:
        print("❌ 特征提取失败")
        return
    
    # 4. 预测
    df_predicted = predict_test(models, X, df_valid)
    
    # 5. 分析结果
    compare_with_actual(df_predicted)
    
    # 6. 保存结果
    print(f"\n保存预测结果...")
    
    # 保存包含所有原始列和预测列的完整数据
    output_df = df.copy()
    
    # 将预测结果合并回原始数据框
    if 'Predicted_wavelength' in df_predicted.columns:
        output_df.loc[df_predicted.index, 'Predicted_wavelength'] = df_predicted['Predicted_wavelength']
    if 'Predicted_PLQY' in df_predicted.columns:
        output_df.loc[df_predicted.index, 'Predicted_PLQY'] = df_predicted['Predicted_PLQY']
    
    # 保存文件
    output_df.to_csv(args.output, index=False)
    print(f"  ✅ 保存到: {args.output}")
    print(f"  ✅ 行数: {len(output_df):,}")
    print(f"  ✅ 成功预测: {len(df_predicted):,}")
    
    # 保存只包含预测成功的样本
    valid_output = args.output.replace('.csv', '_valid_only.csv')
    df_predicted.to_csv(valid_output, index=False)
    print(f"  ✅ 有效预测: {valid_output}")
    
    # 如果使用交集模型，说明数据特点
    if args.intersection:
        print(f"\n  📌 注意: 使用了交集训练的模型")
        print(f"     这些模型只用三个目标都有值的样本训练")
        print(f"     通常具有更好的一致性但可能泛化能力略低")
    
    # 总结
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("✅ 预测完成!")
    print(f"  • 总用时: {total_time:.2f} 秒")
    print(f"  • 处理速度: {len(df)/total_time:.0f} samples/s")
    print(f"  • 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
