#!/usr/bin/env python3
"""
使用训练好的模型对组合进行预测
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_extractor import FeatureExtractor

def load_models(project_dir, model_name='xgboost'):
    """加载训练好的模型（支持AutoML路径与自动发现最新Paper_*目录）"""
    print("加载模型...")

    models = {}

    project_path = Path(project_dir)
    possible_dirs = [
        project_path / 'all_models' / 'automl_train' / model_name / 'models',
        project_path / model_name / 'models',
        project_path / 'models' / model_name,
    ]

    model_dir = None
    for d in possible_dirs:
        if d.exists():
            model_dir = d
            break

    if model_dir is None:
        root = project_path.parent if project_path.name == 'paper_table' else project_path
        candidates = []
        try:
            for d in root.glob('Paper_*'):
                mdir = d / 'all_models' / 'automl_train' / model_name / 'models'
                if mdir.exists():
                    candidates.append(mdir)
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_dir = candidates[0]
                print(f"🔁 自动切换到最新模型目录: {model_dir}")
        except Exception:
            pass

    if model_dir is None or not model_dir.exists():
        print(f"❌ 模型目录不存在: {project_path}/{model_name}/models")
        return models

    print(f"📁 模型目录: {model_dir}")

    for model_file in model_dir.glob("*.joblib"):
        filename = model_file.stem
        if 'wavelength' in filename.lower():
            models['wavelength'] = joblib.load(model_file)
            print(f"  ✅ 波长模型: {model_file.name}")
        elif 'plqy' in filename.lower():
            models['PLQY'] = joblib.load(model_file)
            print(f"  ✅ PLQY模型: {model_file.name}")

    print(f"成功加载 {len(models)} 个模型")
    return models

def extract_features_batch(df, feature_type='combined', batch_size=1000):
    """批量提取特征"""
    print(f"\n提取特征 (批大小: {batch_size})...")
    
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True
    )
    
    n_samples = len(df)
    features_list = []
    valid_indices = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_df = df.iloc[i:batch_end]
        
        for idx, row in batch_df.iterrows():
            try:
                # 提取组合特征
                smiles_list = [row['L1'], row['L2'], row['L3']]
                features = extractor.extract_combination(smiles_list)
                
                if features is not None:
                    features_list.append(features)
                    valid_indices.append(idx)
            except:
                continue
        
        # 进度显示
        if (i + batch_size) % 10000 == 0 or batch_end == n_samples:
            print(f"  进度: {batch_end:,}/{n_samples:,} ({100*batch_end/n_samples:.1f}%)")
    
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"  ✅ 成功提取: {len(X):,} 个特征")
        return X, df_valid
    else:
        return None, None

def predict_batch(models, X, df_valid, batch_size=10000):
    """批量预测"""
    print("\n预测性质...")
    
    predictions = {}
    
    # 预测每个目标
    for target, model in models.items():
        print(f"  预测 {target}...")
        
        n_samples = len(X)
        preds = []
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            batch_pred = model.predict(batch_X)
            preds.append(batch_pred)
        
        predictions[target] = np.concatenate(preds)
        
        # 统计
        print(f"    范围: [{predictions[target].min():.3f}, {predictions[target].max():.3f}]")
        print(f"    均值: {predictions[target].mean():.3f}")
    
    # 添加预测到DataFrame
    if 'wavelength' in predictions:
        df_valid['Predicted_wavelength'] = predictions['wavelength']
    if 'PLQY' in predictions:
        df_valid['Predicted_PLQY'] = predictions['PLQY']
    
    return df_valid

def analyze_results(df):
    """分析预测结果"""
    print("\n" + "=" * 60)
    print("预测结果分析")
    print("-" * 60)
    
    if 'Predicted_PLQY' in df.columns:
        # Top 10 PLQY
        top10 = df.nlargest(10, 'Predicted_PLQY')
        print("\n🏆 Top 10 PLQY组合:")
        for i, row in enumerate(top10.iterrows(), 1):
            idx, data = row
            print(f"\n  #{i}: PLQY = {data['Predicted_PLQY']:.4f}")
            if i <= 3:  # 显示前3个的详细信息
                print(f"      L1/L2: {data['L1'][:50]}...")
                print(f"      L3: {data['L3'][:50]}...")
                if 'Predicted_wavelength' in df.columns:
                    print(f"      λ = {data['Predicted_wavelength']:.1f} nm")
    
    if 'Predicted_wavelength' in df.columns:
        # 波长分布
        print(f"\n📊 波长分布:")
        print(f"  最短: {df['Predicted_wavelength'].min():.1f} nm")
        print(f"  最长: {df['Predicted_wavelength'].max():.1f} nm")
        print(f"  平均: {df['Predicted_wavelength'].mean():.1f} nm")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='预测组合性质')
    parser.add_argument('--input', '-i', 
                       default='ir_assemble.csv',
                       help='组合文件')
    parser.add_argument('--project', '-p',
                       default='paper_table',
                       help='模型项目目录')
    parser.add_argument('--output', '-o',
                       default='ir_assemble_predicted.csv',
                       help='输出文件')
    parser.add_argument('--top', '-t', type=int, default=1000,
                       help='保存Top N个候选')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='批处理大小')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("预测组合性质")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载组合
    print(f"\n加载组合文件: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  组合数: {len(df):,}")
    
    # 2. 加载模型
    models = load_models(args.project)
    if not models:
        print("❌ 没有找到模型")
        return
    
    # 3. 提取特征
    X, df_valid = extract_features_batch(df, batch_size=args.batch_size)
    if X is None:
        print("❌ 特征提取失败")
        return
    
    # 4. 预测
    df_predicted = predict_batch(models, X, df_valid)
    
    # 5. 保存结果
    print(f"\n保存预测结果...")
    df_predicted.to_csv(args.output, index=False)
    print(f"  ✅ 完整结果: {args.output}")
    
    # 保存Top候选
    if 'Predicted_PLQY' in df_predicted.columns:
        top_file = args.output.replace('.csv', f'_top{args.top}.csv')
        top_df = df_predicted.nlargest(args.top, 'Predicted_PLQY')
        top_df.to_csv(top_file, index=False)
        print(f"  ✅ Top {args.top}: {top_file}")
    
    # 6. 分析结果
    analyze_results(df_predicted)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
