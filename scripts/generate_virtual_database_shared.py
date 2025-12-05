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
    parser = argparse.ArgumentParser(description='Generate virtual database (shared utils)')
    parser.add_argument('--data', '-d', default='data/Database_normalized.csv')
    parser.add_argument('--project', '-p', default='paper_table')
    parser.add_argument('--model', '-m', default='xgboost')
    parser.add_argument('--output', '-o', default='data/ir_assemble_shared.csv')
    parser.add_argument('--max-combinations', type=int)
    parser.add_argument('--include-existing', action='store_true')
    parser.add_argument('--feature-type', default='combined', choices=['morgan', 'descriptors', 'combined'])
    args = parser.parse_args()

    log_banner('Generate virtual database (shared utils)')
    print(f'INFO: Original data: {args.data}')
    print(f'INFO: Project dir: {args.project}')
    print(f'INFO: Model: {args.model}')
    print(f'INFO: Feature type: {args.feature_type}')

    log_step('Step 1: Load original data')
    df = pd.read_csv(args.data)
    l1u = df['L1'].dropna().unique()
    l2u = df['L2'].dropna().unique()
    l3u = df['L3'].dropna().unique()
    print(f'INFO: L1 unique: {len(l1u)}')
    print(f'INFO: L2 unique: {len(l2u)}')
    print(f'INFO: L3 unique: {len(l3u)}')
    print(f'INFO: Original combinations: {len(df)}')

    log_step('Step 2: Generate all combinations (L1=L2)')
    assembled = generate_combinations(l1u, l2u, l3u, max_combinations=args.max_combinations)
    print(f'INFO: Generated combinations: {len(assembled):,}')

    if not args.include_existing:
        log_step('Step 3: Remove existing combinations')
        assembled = remove_existing(assembled, df)
        print(f'INFO: New combinations: {len(assembled):,}')

    combos_file = args.output.replace('.csv', '_combinations.csv')
    assembled.to_csv(combos_file, index=False)
    print(f'INFO: Saved combinations file: {combos_file}')

    project_path = Path(args.project)
    automl_dir = project_path / 'all_models' / 'automl_train'
    if not automl_dir.exists():
        assembled.to_csv(args.output, index=False)
        log_banner('Stats')
        print(f'Total combinations: {len(assembled):,}')
        return

    log_step('Step 4: Extract molecular features')
    X, df_valid = extract_features_dataframe(assembled, feature_type=args.feature_type)
    if X is None:
        print('ERROR: Feature extraction failed')
        return
    print(f'INFO: Successfully extracted features: {len(X)}')

    log_step('Step 5: Load models and predict')
    mdir = resolve_model_dir(args.project, args.model)
    if not mdir or not mdir.exists():
        print('ERROR: Model directory not found')
        return
    print(f'INFO: Model directory: {mdir}')
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
    print(f'INFO: Virtual database saved: {out}')

    if 'Predicted_PLQY' in df_valid.columns:
        top = df_valid.nlargest(100, 'Predicted_PLQY')
        top_file = out.replace('.csv', '_top100.csv')
        top.to_csv(top_file, index=False)
        print(f'INFO: Top100 candidates saved: {top_file}')

    log_banner('Completed')


if __name__ == '__main__':
    main()
