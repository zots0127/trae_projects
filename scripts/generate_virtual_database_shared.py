#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.shared import (
    resolve_model_dir,
    load_models_from_dir,
    default_extractor,
    extract_features_dataframe,
    log_banner,
    log_step,
    log_stats_range,
    generate_combinations,
    remove_existing,
)


def main():
    parser = argparse.ArgumentParser(description='生成虚拟数据库(共享工具版)')
    parser.add_argument('--data', '-d', default='data/Database_normalized.csv')
    parser.add_argument('--project', '-p', default='paper_table')
    parser.add_argument('--model', '-m', default='xgboost')
    parser.add_argument('--output', '-o', default='data/ir_assemble_shared.csv')
    parser.add_argument('--max-combinations', type=int)
    parser.add_argument('--include-existing', action='store_true')
    parser.add_argument('--feature-type', default='combined', choices=['morgan', 'descriptors', 'combined'])
    args = parser.parse_args()

    log_banner('生成虚拟数据库(共享工具版)')
    print(f'原始数据: {args.data}')
    print(f'项目目录: {args.project}')
    print(f'使用模型: {args.model}')
    print(f'特征类型: {args.feature_type}')

    log_step('步骤1: 加载原始数据')
    df = pd.read_csv(args.data)
    l1u = df['L1'].dropna().unique()
    l2u = df['L2'].dropna().unique()
    l3u = df['L3'].dropna().unique()
    print(f'L1唯一: {len(l1u)}')
    print(f'L2唯一: {len(l2u)}')
    print(f'L3唯一: {len(l3u)}')
    print(f'原始组合数: {len(df)}')

    log_step('步骤2: 生成所有组合(L1=L2)')
    assembled = generate_combinations(l1u, l2u, l3u, max_combinations=args.max_combinations)
    print(f'生成组合数: {len(assembled):,}')

    if not args.include_existing:
        log_step('步骤3: 移除已存在的组合')
        assembled = remove_existing(assembled, df)
        print(f'新组合数: {len(assembled):,}')

    combos_file = args.output.replace('.csv', '_combinations.csv')
    assembled.to_csv(combos_file, index=False)
    print(f'保存组合文件: {combos_file}')

    project_path = Path(args.project)
    automl_dir = project_path / 'all_models' / 'automl_train'
    if not automl_dir.exists():
        assembled.to_csv(args.output, index=False)
        log_banner('统计')
        print(f'总组合数: {len(assembled):,}')
        return

    log_step('步骤4: 提取分子特征')
    X, df_valid = extract_features_dataframe(assembled, feature_type=args.feature_type)
    if X is None:
        print('特征提取失败')
        return
    print(f'成功提取特征: {len(X)}')

    log_step('步骤5: 加载模型并预测')
    mdir = resolve_model_dir(args.project, args.model)
    if not mdir or not mdir.exists():
        print('模型目录不存在')
        return
    print(f'模型目录: {mdir}')
    models = load_models_from_dir(mdir)

    predictions = {}
    if 'wavelength' in models:
        pred = models['wavelength'].predict(X)
        predictions['Predicted_Max_wavelength(nm)'] = pred
        log_stats_range(pred)
    if 'PLQY' in models:
        pred = models['PLQY'].predict(X)
        predictions['Predicted_PLQY'] = pred
        log_stats_range(pred)

    for k, v in predictions.items():
        df_valid[k] = v

    out = args.output
    df_valid.to_csv(out, index=False)
    print(f'虚拟数据库已保存: {out}')

    if 'Predicted_PLQY' in df_valid.columns:
        top = df_valid.nlargest(100, 'Predicted_PLQY')
        top_file = out.replace('.csv', '_top100.csv')
        top.to_csv(top_file, index=False)
        print(f'Top100候选已保存: {top_file}')

    log_banner('完成')


if __name__ == '__main__':
    main()
