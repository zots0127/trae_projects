#!/usr/bin/env python3
"""
高效生成虚拟数据库 - 使用向量化操作
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import joblib
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_extractor import FeatureExtractor

def load_and_deduplicate_ligands(data_file):
    """加载数据并去重配体"""
    print("加载并去重配体...")
    df = pd.read_csv(data_file)
    
    # 合并L1和L2，去重
    l1_series = df['L1'].dropna()
    l2_series = df['L2'].dropna()
    
    # 使用set快速去重
    l12_set = set(l1_series) | set(l2_series)
    l12_unique = sorted(list(l12_set))  # 排序以保证结果一致性
    
    # L3去重
    l3_unique = sorted(df['L3'].dropna().unique().tolist())
    
    print(f"  L1原始: {len(df['L1'].dropna())} → 唯一: {len(df['L1'].dropna().unique())}")
    print(f"  L2原始: {len(df['L2'].dropna())} → 唯一: {len(df['L2'].dropna().unique())}")
    print(f"  L1∪L2去重后: {len(l12_unique)} 个配体")
    print(f"  L3唯一: {len(l3_unique)} 个配体")
    print(f"  理论组合数: {len(l12_unique) * len(l3_unique):,}")
    
    return l12_unique, l3_unique, df

def generate_combinations_vectorized(l12_unique, l3_unique):
    """使用向量化操作快速生成组合"""
    print("\n生成组合（向量化）...")
    
    # 使用NumPy的meshgrid生成所有组合
    # 这比itertools.product更快
    n_l12 = len(l12_unique)
    n_l3 = len(l3_unique)
    
    # 创建索引网格
    l12_idx, l3_idx = np.meshgrid(range(n_l12), range(n_l3), indexing='ij')
    
    # 展平索引
    l12_idx_flat = l12_idx.flatten()
    l3_idx_flat = l3_idx.flatten()
    
    # 使用索引获取实际的SMILES字符串
    l12_array = np.array(l12_unique)
    l3_array = np.array(l3_unique)
    
    # 向量化获取组合
    l1_values = l12_array[l12_idx_flat]
    l2_values = l1_values.copy()  # L1=L2
    l3_values = l3_array[l3_idx_flat]
    
    # 创建DataFrame（一次性创建，避免逐行添加）
    assembled_df = pd.DataFrame({
        'L1': l1_values,
        'L2': l2_values,
        'L3': l3_values
    })
    
    print(f"  生成组合数: {len(assembled_df):,}")
    
    return assembled_df

def batch_extract_features(df, feature_type='combined', batch_size=1000, combination_method='mean', descriptor_count=85):
    """批量提取特征，提高效率"""
    print("\n批量提取分子特征...")
    
    # 初始化特征提取器
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True,
        descriptor_count=descriptor_count
    )
    
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    features_list = []
    valid_indices = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        batch_df = df.iloc[start_idx:end_idx]
        
        batch_features = []
        batch_valid_idx = []
        
        for idx, row in batch_df.iterrows():
            try:
                smiles_list = [row['L1'], row['L2'], row['L3']]
                # 使用extract_combination方法合并多个配体的特征
                features = extractor.extract_combination(
                    smiles_list,
                    feature_type=feature_type,
                    combination_method=combination_method
                )
                if features is not None:
                    batch_features.append(features)
                    batch_valid_idx.append(idx)
            except Exception:
                continue
        
        if batch_features:
            features_list.extend(batch_features)
            valid_indices.extend(batch_valid_idx)
        
        # 进度显示
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  处理进度: {end_idx}/{n_samples} ({100*end_idx/n_samples:.1f}%)")
    
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"✅ 成功提取特征: {len(X):,} 个组合")
        return X, df_valid
    else:
        print("❌ 没有成功提取任何特征")
        return None, None

def batch_predict(X, model, batch_size=10000):
    """批量预测，避免内存溢出"""
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    predictions = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        batch_X = X[start_idx:end_idx]
        batch_pred = model.predict(batch_X)
        predictions.append(batch_pred)
    
    return np.concatenate(predictions)

def load_trained_models(project_dir, model_name='xgboost'):
    """一次性加载所有模型"""
    print("\n加载训练好的模型...")
    
    project_path = Path(project_dir)
    model_dir = project_path / model_name / 'models'
    
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return {}
    
    models = {}
    targets = {
        'wavelength': ['wavelength', 'Max_wavelength'],
        'PLQY': ['PLQY', 'plqy'],
        'tau': ['tau', 'lifetime']
    }
    
    for target_key, patterns in targets.items():
        for pattern in patterns:
            model_files = list(model_dir.glob(f"*{pattern}*.joblib"))
            if model_files:
                model_file = sorted(model_files)[-1]
                print(f"  ✅ 加载 {target_key} 模型: {model_file.name}")
                models[target_key] = joblib.load(model_file)
                break
    
    return models

def predict_all_properties(X, df_valid, models):
    """使用所有模型进行预测"""
    print("\n预测分子性质...")
    
    predictions = {}
    target_mapping = {
        'wavelength': 'Predicted_Max_wavelength(nm)',
        'PLQY': 'Predicted_PLQY',
        'tau': 'Predicted_tau(s*10^-6)'
    }
    
    for target_key, col_name in target_mapping.items():
        if target_key in models:
            print(f"  预测 {target_key}...")
            pred = batch_predict(X, models[target_key])
            predictions[col_name] = pred
            
            # 统计
            print(f"    范围: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"    均值: {pred.mean():.3f} ± {pred.std():.3f}")
    
    # 一次性添加所有预测列
    for col_name, pred_values in predictions.items():
        df_valid[col_name] = pred_values
    
    return df_valid

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高效生成虚拟数据库')
    
    parser.add_argument('--data', '-d', default='../data/Database_normalized.csv',
                       help='原始数据文件')
    parser.add_argument('--project', '-p', default='paper_table',
                       help='训练项目目录')
    parser.add_argument('--model', '-m', default='xgboost',
                       help='使用的模型')
    parser.add_argument('--output', '-o', default='ir_assemble.csv',
                       help='输出文件名')
    parser.add_argument('--feature-type', default='combined',
                       choices=['morgan', 'descriptors', 'combined'],
                       help='特征类型')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='批处理大小')
    parser.add_argument('--combination-method', default='mean',
                       choices=['mean', 'sum', 'concat'],
                       help='多个配体特征的合并方式')
    parser.add_argument('--descriptor-count', type=int, default=85,
                       help='分子描述符数量')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("=" * 60)
    print("高效生成虚拟数据库")
    print("=" * 60)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"原始数据: {args.data}")
    print(f"项目目录: {args.project}")
    print(f"使用模型: {args.model}")
    
    # 1. 加载并去重配体
    l12_unique, l3_unique, original_df = load_and_deduplicate_ligands(args.data)
    
    # 2. 生成组合（向量化）
    assembled_df = generate_combinations_vectorized(l12_unique, l3_unique)
    
    # 保存组合文件
    combo_file = args.output.replace('.csv', '_combinations.csv')
    assembled_df.to_csv(combo_file, index=False)
    print(f"\n✅ 组合文件已保存: {combo_file}")
    
    # 3. 批量提取特征
    X, df_valid = batch_extract_features(
        assembled_df,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        combination_method=args.combination_method,
        descriptor_count=args.descriptor_count
    )
    
    if X is None:
        print("❌ 特征提取失败")
        return
    
    # 4. 加载所有模型
    models = load_trained_models(args.project, args.model)
    
    if not models:
        print("❌ 没有找到模型")
        return
    
    # 5. 批量预测
    df_predicted = predict_all_properties(X, df_valid, models)
    
    # 6. 保存结果
    print("\n保存虚拟数据库...")
    df_predicted.to_csv(args.output, index=False)
    print(f"✅ 虚拟数据库已保存: {args.output}")
    
    # 7. 分析结果
    print("\n" + "=" * 60)
    print("📊 虚拟数据库分析")
    print("-" * 40)
    print(f"总组合数: {len(df_predicted):,}")
    
    # 找出最优组合
    if 'Predicted_PLQY' in df_predicted.columns:
        # Top 10 PLQY
        top_plqy = df_predicted.nlargest(10, 'Predicted_PLQY')
        print(f"\n🏆 Top 10 PLQY组合:")
        for idx, row in top_plqy.iterrows():
            print(f"  #{idx+1}: PLQY={row['Predicted_PLQY']:.3f}")
            if idx == 0:  # 显示最佳组合的详细信息
                print(f"       L1=L2: {row['L1'][:40]}...")
                print(f"       L3: {row['L3'][:40]}...")
                if 'Predicted_Max_wavelength(nm)' in df_predicted.columns:
                    print(f"       λ={row['Predicted_Max_wavelength(nm)']:.1f}nm")
    
    # 保存Top候选
    if 'Predicted_PLQY' in df_predicted.columns:
        top_file = args.output.replace('.csv', '_top1000.csv')
        top_candidates = df_predicted.nlargest(1000, 'Predicted_PLQY')
        top_candidates.to_csv(top_file, index=False)
        print(f"\n✅ Top 1000候选已保存: {top_file}")
    
    # 时间统计
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n总用时: {duration:.1f} 秒")
    
    print("\n" + "=" * 60)
    print("✅ 虚拟数据库生成完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
