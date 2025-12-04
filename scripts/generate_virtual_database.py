#!/usr/bin/env python3
"""
生成虚拟数据库 - 重组L1、L2、L3的所有组合并预测
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

def load_original_data(data_file):
    """加载原始数据并提取唯一的配体"""
    df = pd.read_csv(data_file)
    
    # 提取所有唯一的L1, L2, L3
    l1_unique = df['L1'].dropna().unique()
    l2_unique = df['L2'].dropna().unique()
    l3_unique = df['L3'].dropna().unique()
    
    print(f"原始数据统计:")
    print(f"  L1: {len(l1_unique)} 个唯一配体")
    print(f"  L2: {len(l2_unique)} 个唯一配体")
    print(f"  L3: {len(l3_unique)} 个唯一配体")
    print(f"  原始组合数: {len(df)}")
    
    return l1_unique, l2_unique, l3_unique, df

def generate_all_combinations(l1_unique, l2_unique, l3_unique, max_combinations=None):
    """生成所有可能的L1、L2、L3组合
    
    注意：L1和L2应该是相同的配体（理论上），所以我们使用L1=L2的组合
    """
    
    # 合并L1和L2的唯一值（因为理论上它们应该是相同的配体集）
    l12_unique = np.unique(np.concatenate([l1_unique, l2_unique]))
    
    print(f"\n组合策略：")
    print(f"  L1/L2共享配体池: {len(l12_unique)} 个配体")
    print(f"  L3配体池: {len(l3_unique)} 个配体")
    
    # 生成所有可能的组合
    all_combinations = []
    
    # L1=L2的情况（对称配体）
    total_possible = len(l12_unique) * len(l3_unique)
    print(f"  理论组合数: {total_possible:,} (L1=L2配对)")
    
    if max_combinations and total_possible > max_combinations:
        print(f"⚠️ 限制组合数为: {max_combinations:,}")
        # 随机采样
        import random
        random.seed(42)
        sampled_indices = random.sample(range(total_possible), min(max_combinations, total_possible))
        sampled_indices.sort()
        
        count = 0
        for idx, (l12, l3) in enumerate(product(l12_unique, l3_unique)):
            if idx in sampled_indices:
                all_combinations.append({
                    'L1': l12,
                    'L2': l12,  # L1和L2相同
                    'L3': l3
                })
                count += 1
                if count >= max_combinations:
                    break
    else:
        # 生成所有L1=L2的组合
        for l12 in l12_unique:
            for l3 in l3_unique:
                all_combinations.append({
                    'L1': l12,
                    'L2': l12,  # L1和L2相同
                    'L3': l3
                })
    
    # 创建DataFrame
    assembled_df = pd.DataFrame(all_combinations)
    print(f"生成组合数: {len(assembled_df):,}")
    
    return assembled_df

def remove_existing_combinations(assembled_df, original_df):
    """移除已存在的组合，只保留新组合"""
    
    # 创建组合键
    assembled_df['combo_key'] = assembled_df['L1'] + '|' + assembled_df['L2'] + '|' + assembled_df['L3']
    original_df['combo_key'] = original_df['L1'] + '|' + original_df['L2'] + '|' + original_df['L3']
    
    # 找出新组合
    existing_keys = set(original_df['combo_key'].dropna())
    new_df = assembled_df[~assembled_df['combo_key'].isin(existing_keys)].copy()
    
    # 删除辅助列
    new_df = new_df.drop('combo_key', axis=1)
    
    print(f"新组合数（排除已有）: {len(new_df):,}")
    
    return new_df

def extract_features_for_prediction(df, feature_type='combined'):
    """为预测提取特征"""
    
    print("\n提取分子特征...")
    
    # 初始化特征提取器
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        combination_method='mean',
        use_cache=True
    )
    
    # 提取特征
    features_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            smiles_list = [row['L1'], row['L2'], row['L3']]
            # 过滤掉NaN
            smiles_list = [s for s in smiles_list if pd.notna(s) and s != '']
            
            if smiles_list:
                features = extractor.extract_features(smiles_list)
                if features is not None:
                    features_list.append(features)
                    valid_indices.append(idx)
                    
                    if len(features_list) % 100 == 0:
                        print(f"  已处理: {len(features_list)} 个组合")
        except Exception as e:
            # 跳过有问题的SMILES
            continue
    
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"✅ 成功提取特征: {len(X)} 个组合")
        return X, df_valid
    else:
        print("❌ 没有成功提取任何特征")
        return None, None

def load_trained_model(project_dir, model_name='xgboost', target='PLQY'):
    """加载训练好的模型"""
    
    project_path = Path(project_dir)
    model_dir = project_path / model_name / 'models'
    
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return None
    
    # 查找对应目标的模型文件
    model_files = list(model_dir.glob(f"*{target}*.joblib"))
    
    if not model_files:
        # 尝试其他可能的命名
        if target == 'wavelength':
            model_files = list(model_dir.glob("*wavelength*.joblib")) + \
                         list(model_dir.glob("*Max_wavelength*.joblib"))
        elif target == 'PLQY':
            model_files = list(model_dir.glob("*PLQY*.joblib")) + \
                         list(model_dir.glob("*plqy*.joblib"))
        elif target == 'tau':
            model_files = list(model_dir.glob("*tau*.joblib")) + \
                         list(model_dir.glob("*lifetime*.joblib"))
    
    if model_files:
        # 使用最新的模型
        model_file = sorted(model_files)[-1]
        print(f"✅ 加载模型: {model_file.name}")
        model = joblib.load(model_file)
        return model
    else:
        print(f"❌ 未找到{target}的模型文件")
        return None

def predict_properties(X, df_valid, project_dir, model_name='xgboost'):
    """使用训练好的模型预测分子性质"""
    
    print("\n加载模型并预测...")
    
    predictions = {}
    
    # 预测三个目标
    targets = {
        'Max_wavelength(nm)': 'wavelength',
        'PLQY': 'PLQY',
        'tau(s*10^-6)': 'tau'
    }
    
    for target_col, target_key in targets.items():
        model = load_trained_model(project_dir, model_name, target_key)
        
        if model:
            print(f"  预测 {target_col}...")
            try:
                pred = model.predict(X)
                predictions[target_col] = pred
                
                # 统计预测结果
                print(f"    范围: [{pred.min():.3f}, {pred.max():.3f}]")
                print(f"    均值: {pred.mean():.3f}")
                print(f"    标准差: {pred.std():.3f}")
            except Exception as e:
                print(f"    ❌ 预测失败: {e}")
                predictions[target_col] = np.zeros(len(X))
        else:
            print(f"  ⚠️ 跳过 {target_col} (无模型)")
            predictions[target_col] = np.zeros(len(X))
    
    # 将预测结果添加到DataFrame
    for col, pred in predictions.items():
        df_valid[f'Predicted_{col}'] = pred
    
    return df_valid

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成虚拟数据库')
    
    parser.add_argument('--data', '-d', default='data/Database_normalized.csv',
                       help='原始数据文件')
    parser.add_argument('--project', '-p', default='paper_table',
                       help='训练项目目录')
    parser.add_argument('--model', '-m', default='xgboost',
                       help='使用的模型')
    parser.add_argument('--output', '-o', default='data/ir_assemble.csv',
                       help='输出文件名')
    parser.add_argument('--max-combinations', type=int,
                       help='最大组合数限制')
    parser.add_argument('--include-existing', action='store_true',
                       help='包含已存在的组合')
    parser.add_argument('--feature-type', default='combined',
                       choices=['morgan', 'descriptors', 'combined'],
                       help='特征类型')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("生成虚拟数据库")
    print("=" * 60)
    print(f"原始数据: {args.data}")
    print(f"项目目录: {args.project}")
    print(f"使用模型: {args.model}")
    print(f"特征类型: {args.feature_type}")
    
    # 1. 加载原始数据
    print("\n" + "-" * 40)
    print("步骤1: 加载原始数据")
    l1_unique, l2_unique, l3_unique, original_df = load_original_data(args.data)
    
    # 2. 生成所有组合
    print("\n" + "-" * 40)
    print("步骤2: 生成所有组合")
    assembled_df = generate_all_combinations(
        l1_unique, l2_unique, l3_unique, 
        max_combinations=args.max_combinations
    )
    
    # 3. 可选：移除已存在的组合
    if not args.include_existing:
        print("\n" + "-" * 40)
        print("步骤3: 移除已存在的组合")
        assembled_df = remove_existing_combinations(assembled_df, original_df)
    
    # 保存组合文件
    assembled_file = args.output.replace('.csv', '_combinations.csv')
    assembled_df.to_csv(assembled_file, index=False)
    print(f"\n✅ 保存组合文件: {assembled_file}")
    
    # 4. 提取特征
    print("\n" + "-" * 40)
    print("步骤4: 提取分子特征")
    X, df_valid = extract_features_for_prediction(assembled_df, args.feature_type)
    
    if X is None:
        print("❌ 特征提取失败")
        return
    
    # 5. 预测性质
    print("\n" + "-" * 40)
    print("步骤5: 预测分子性质")
    df_predicted = predict_properties(X, df_valid, args.project, args.model)
    
    # 6. 保存结果
    print("\n" + "-" * 40)
    print("步骤6: 保存虚拟数据库")
    
    # 保存完整的虚拟数据库
    output_file = args.output
    df_predicted.to_csv(output_file, index=False)
    print(f"✅ 虚拟数据库已保存: {output_file}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("📊 虚拟数据库统计:")
    print("-" * 40)
    print(f"总组合数: {len(df_predicted):,}")
    
    # 找出最优组合
    if 'Predicted_PLQY' in df_predicted.columns:
        # PLQY最高的组合
        best_plqy_idx = df_predicted['Predicted_PLQY'].idxmax()
        best_plqy = df_predicted.loc[best_plqy_idx]
        print(f"\n🏆 最高PLQY组合:")
        print(f"  L1: {best_plqy['L1'][:30]}...")
        print(f"  L2: {best_plqy['L2'][:30]}...")
        print(f"  L3: {best_plqy['L3'][:30]}...")
        print(f"  预测PLQY: {best_plqy['Predicted_PLQY']:.3f}")
    
    if 'Predicted_Max_wavelength(nm)' in df_predicted.columns:
        # 波长最长的组合
        best_wl_idx = df_predicted['Predicted_Max_wavelength(nm)'].idxmax()
        best_wl = df_predicted.loc[best_wl_idx]
        print(f"\n🏆 最长波长组合:")
        print(f"  L1: {best_wl['L1'][:30]}...")
        print(f"  L2: {best_wl['L2'][:30]}...")
        print(f"  L3: {best_wl['L3'][:30]}...")
        print(f"  预测波长: {best_wl['Predicted_Max_wavelength(nm)']:.1f} nm")
    
    # 保存Top候选组合
    top_candidates = df_predicted.nlargest(100, 'Predicted_PLQY')
    top_file = output_file.replace('.csv', '_top100.csv')
    top_candidates.to_csv(top_file, index=False)
    print(f"\n✅ Top 100候选已保存: {top_file}")
    
    print("\n" + "=" * 60)
    print("✅ 虚拟数据库生成完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
