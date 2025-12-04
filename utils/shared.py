from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def resolve_model_dir(project_dir: str, model_name: str, use_intersection: bool = False) -> Path:
    p = Path(project_dir)
    candidates = [
        p / 'all_models' / 'automl_train' / model_name / 'models',
        p / model_name / 'models',
        p / 'models' / model_name,
    ]
    if use_intersection:
        candidates.extend([
            p / model_name / 'intersection' / f'{model_name}_intersection' / 'models',
            p / model_name / 'intersection' / 'models',
            p / f'{model_name}_intersection' / 'models',
        ])
    for d in candidates:
        if d.exists():
            return d
    root = p.parent if p.name == 'paper_table' else p
    papers = [d for d in root.glob('Paper_*') if d.is_dir()]
    papers.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    for d in papers:
        m = d / 'all_models' / 'automl_train' / model_name / 'models'
        if m.exists():
            return m
    return Path()


def load_models_from_dir(model_dir: Path) -> dict:
    models = {}
    if not model_dir or not model_dir.exists():
        return models
    for f in model_dir.glob('*.joblib'):
        name = f.stem.lower()
        if 'wavelength' in name or 'max_wavelength' in name:
            models['wavelength'] = joblib.load(f)
        elif 'plqy' in name:
            models['PLQY'] = joblib.load(f)
    return models


def default_extractor():
    from core.feature_extractor import FeatureExtractor
    return FeatureExtractor(
        feature_type='combined',
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True,
        descriptor_count=85,
    )


def extract_features_dataframe(df: pd.DataFrame, feature_type: str = 'combined', combination_method: str = 'mean', descriptor_count: int = 85):
    extractor = default_extractor()
    features = []
    idxs = []
    for i, row in df.iterrows():
        smiles = [row.get('L1'), row.get('L2'), row.get('L3')]
        smiles = [s for s in smiles if s is not None and not pd.isna(s) and s != '']
        if not smiles:
            continue
        x = extractor.extract_combination(smiles, feature_type=feature_type, combination_method=combination_method)
        if x is not None:
            features.append(x)
            idxs.append(i)
    if features:
        X = np.vstack(features)
        return X, df.loc[idxs].reset_index(drop=True)
    return None, None


def log_banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("-" * 80)


def log_step(title: str):
    print("\n" + "-" * 40)
    print(title)


def log_stats_range(arr: np.ndarray):
    print(f"    范围: [{arr.min():.3f}, {arr.max():.3f}]")
    print(f"    均值: {arr.mean():.3f}")
    print(f"    标准差: {arr.std():.3f}")


def regression_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.array(actual)
    pred = np.array(pred)
    mae = np.abs(actual - pred).mean()
    rmse = np.sqrt(((actual - pred) ** 2).mean())
    sst = ((actual - actual.mean()) ** 2).sum()
    sse = ((actual - pred) ** 2).sum()
    r2 = 1 - sse / sst if sst != 0 else 0.0
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def generate_combinations(l1_unique: np.ndarray, l2_unique: np.ndarray, l3_unique: np.ndarray, max_combinations: int = None) -> pd.DataFrame:
    l12 = np.unique(np.concatenate([l1_unique, l2_unique]))
    pairs = []
    import random
    if max_combinations and len(l12) * len(l3_unique) > max_combinations:
        random.seed(42)
        total = len(l12) * len(l3_unique)
        sampled = set(random.sample(range(total), max_combinations))
        k = 0
        for a in l12:
            for b in l3_unique:
                if k in sampled:
                    pairs.append({'L1': a, 'L2': a, 'L3': b})
                k += 1
    else:
        for a in l12:
            for b in l3_unique:
                pairs.append({'L1': a, 'L2': a, 'L3': b})
    return pd.DataFrame(pairs)


def remove_existing(assembled_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    a = assembled_df.copy()
    a['combo_key'] = a['L1'] + '|' + a['L2'] + '|' + a['L3']
    o = original_df.copy()
    o['combo_key'] = o['L1'] + '|' + o['L2'] + '|' + o['L3']
    existing = set(o['combo_key'].dropna())
    out = a[~a['combo_key'].isin(existing)].copy()
    return out.drop('combo_key', axis=1)


def find_latest_paper_dir(root: Path = Path.cwd()) -> str:
    papers = [d for d in root.glob('Paper_*') if d.is_dir()]
    if not papers:
        return ''
    papers.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(papers[0])


def load_csv_candidates(paths: list) -> pd.DataFrame:
    for p in paths:
        pth = Path(p)
        if pth.exists():
            return pd.read_csv(pth)
    return pd.DataFrame()


def predict_array(models: dict, X: np.ndarray) -> dict:
    preds = {}
    if 'wavelength' in models:
        preds['wavelength'] = models['wavelength'].predict(X)
    if 'PLQY' in models:
        preds['PLQY'] = models['PLQY'].predict(X)
    return preds


def attach_predictions(df: pd.DataFrame, preds: dict, mapping: dict) -> pd.DataFrame:
    out = df.copy()
    for k, v in preds.items():
        col = mapping.get(k, k)
        out[col] = v
    return out


def plqy_distribution(df: pd.DataFrame, col: str = 'Predicted_PLQY') -> dict:
    if col not in df.columns:
        return {}
    s = df[col]
    q = {
        'min': float(s.min()),
        'q25': float(s.quantile(0.25)),
        'median': float(s.median()),
        'q75': float(s.quantile(0.75)),
        'max': float(s.max()),
        'mean': float(s.mean()),
    }
    print(f"最小值: {q['min']:.4f}")
    print(f"25分位: {q['q25']:.4f}")
    print(f"中位数: {q['median']:.4f}")
    print(f"75分位: {q['q75']:.4f}")
    print(f"最大值: {q['max']:.4f}")
    print(f"平均值: {q['mean']:.4f}")
    return q


def save_predictions(output_df: pd.DataFrame, predicted_df: pd.DataFrame, output_path: str) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    valid_only = output_path.replace('.csv', '_valid_only.csv')
    predicted_df.to_csv(valid_only, index=False)
    return {'full': output_path, 'valid_only': valid_only}


def save_top_plqy(df: pd.DataFrame, n: int, base_path: str) -> str:
    if 'Predicted_PLQY' not in df.columns:
        return ''
    top = df.nlargest(n, 'Predicted_PLQY')
    out = base_path.replace('.csv', f'_top{n}.csv')
    top.to_csv(out, index=False)
    return out


def analyze_removed_combinations(original_df: pd.DataFrame, assembled_df: pd.DataFrame) -> dict:
    l12 = pd.unique(pd.concat([original_df['L1'].dropna(), original_df['L2'].dropna()]))
    l3 = original_df['L3'].dropna().unique()
    th = set()
    for a in l12:
        for b in l3:
            th.add(f"{a}|{a}|{b}")
    ir_keys = set((assembled_df['L1'] + '|' + assembled_df['L2'] + '|' + assembled_df['L3']).tolist())
    removed = th - ir_keys
    orig_keys = set((original_df['L1'] + '|' + original_df['L2'] + '|' + original_df['L3']).dropna().tolist())
    dup = removed & orig_keys
    sample = list(dup)[:10]
    return {
        'theoretical': len(th),
        'current': len(ir_keys),
        'removed': len(removed),
        'removed_existing': len(dup),
        'sample': sample,
    }
