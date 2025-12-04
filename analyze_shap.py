#!/usr/bin/env python3
"""
SHAP可解释性分析工具 - 后处理脚本
用于分析已训练好的模型，不会破坏原有项目

使用方法:
    python analyze_shap.py Paper_0930_222051
    python analyze_shap.py Paper_0930_222051 --models xgboost lightgbm
    python analyze_shap.py Paper_0930_222051 --sample-size 200
"""

import sys
import os
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 添加路径（用于导入RDKit相关）
sys.path.insert(0, str(Path(__file__).parent.parent / 'ir2025'))

# SHAP库直接使用，不依赖项目内部模块
print("✅ SHAP分析工具已加载")


class ModelShapAnalyzer:
    """已训练模型的SHAP分析器"""

    def __init__(self, paper_dir):
        self.paper_dir = Path(paper_dir)
        self.models_dir = self.paper_dir / 'all_models' / 'automl_train'
        self.output_dir = self.paper_dir / 'shap_analysis'
        self.output_dir.mkdir(exist_ok=True)

        # KernelExplainer 加速参数（可在 main 中覆盖）
        self.kernel_k = 20           # 背景聚类数
        self.kernel_nsamples = 200   # 每个解释的采样上限
        self.kernel_max_samples = 40 # kernel类型的最大样本数

        # 读取数据
        self.load_data()

    def load_data(self):
        """加载训练数据用于SHAP背景"""
        print("\n📂 加载数据...")

        # 尝试从多个位置加载数据
        data_paths = [
            # 与论文目录同级/内部的data路径
            self.paper_dir / 'data' / 'Database_normalized.csv',
            Path('/Users/kanshan/IR/ir2025/data/Database_normalized.csv'),
            Path('../ir2025/data/Database_normalized.csv'),
            Path('data/Database_normalized.csv')
        ]

        for data_path in data_paths:
            if data_path.exists():
                self.df = pd.read_csv(data_path)
                print(f"  ✅ 数据加载成功: {data_path}")
                print(f"  📊 数据维度: {self.df.shape}")
                return

        raise FileNotFoundError("❌ 未找到训练数据文件")

    def extract_features(self, smiles_list):
        """提取分子特征"""
        # 与训练管线保持一致（1024-bit Morgan + 85个描述符）
        from core.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor(feature_type="combined", morgan_bits=1024, morgan_radius=2)

        features_list = []
        for smiles in smiles_list:
            if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
                continue
            feat = extractor.extract_from_smiles(smiles, feature_type="combined")
            features_list.append(feat)

        if not features_list:
            return np.array([])

        return np.vstack(features_list)

    def get_feature_names(self):
        from core.feature_extractor import DESCRIPTOR_NAMES
        fp_names = [f'Morgan_{i}' for i in range(1024)]
        desc_names = DESCRIPTOR_NAMES if DESCRIPTOR_NAMES else []
        return fp_names + desc_names

    def find_models(self, model_filter=None):
        """查找所有训练好的模型"""
        print("\n🔍 搜索已训练模型...")

        models_info = []

        if not self.models_dir.exists():
            print(f"  ❌ 模型目录不存在: {self.models_dir}")
            return models_info

        # 遍历所有模型目录
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # 过滤模型
            if model_filter and model_name not in model_filter:
                continue

            # 查找模型文件
            model_files_dir = model_dir / 'models'
            if not model_files_dir.exists():
                continue

            for model_file in model_files_dir.glob('*.joblib'):
                # 解析目标名称
                filename = model_file.stem
                if 'Max_wavelength' in filename:
                    target = 'Max_wavelength(nm)'
                    target_clean = 'Wavelength'
                elif 'PLQY' in filename:
                    target = 'PLQY'
                    target_clean = 'PLQY'
                else:
                    continue

                models_info.append({
                    'model_name': model_name,
                    'model_path': model_file,
                    'target': target,
                    'target_clean': target_clean
                })

        print(f"  ✅ 找到 {len(models_info)} 个模型")
        for info in models_info:
            print(f"     - {info['model_name']:20s} | {info['target_clean']}")

        return models_info

    def _resolve_predictor(self, loaded_model):
        """解析模型对象，返回可用于SHAP的预测函数与底层模型对象"""
        # 直接使用可预测的模型
        if hasattr(loaded_model, 'predict'):
            return loaded_model.predict, loaded_model

        # 字典封装（包含scaler/target_scaler等）
        if isinstance(loaded_model, dict):
            inner = loaded_model.get('model', None)
            scaler = loaded_model.get('scaler', None)
            target_scaler = loaded_model.get('target_scaler', None)

            if inner is None:
                raise ValueError('字典模型缺少 "model" 键')

            # 组装带预处理的预测函数
            def predict_fn(X):
                X_in = X
                if scaler is not None:
                    X_in = scaler.transform(X_in)
                y_pred = inner.predict(X_in)
                if target_scaler is not None:
                    y_pred = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).ravel()
                return y_pred

            return predict_fn, inner

        # 其他包装类型（如pipeline）
        try:
            predict = getattr(loaded_model, 'predict')
            return predict, loaded_model
        except Exception:
            raise AttributeError('无法解析预测函数，模型对象不支持 predict')

    def analyze_model(self, model_info, sample_size=100):
        """分析单个模型"""
        model_name = model_info['model_name']
        target = model_info['target']
        target_clean = model_info['target_clean']

        print(f"\n{'='*70}")
        print(f"🔬 分析模型: {model_name} - {target_clean}")
        print(f"{'='*70}")

        # 加载模型
        try:
            model = joblib.load(model_info['model_path'])
            print(f"  ✅ 模型加载成功")
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            return None

        # 准备数据
        print(f"  📊 准备特征数据...")
        valid_df = self.df.dropna(subset=[target])
        smiles_cols = ['L1', 'L2', 'L3']
        print(f"     正在提取分子特征...")
        from core.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor(feature_type="combined", morgan_bits=1024, morgan_radius=2)
        X = extractor.extract_from_dataframe(valid_df, smiles_columns=smiles_cols, feature_type="combined")

        if len(X) == 0:
            print(f"  ❌ 特征提取失败")
            return None

        print(f"     ✅ 特征维度: {X.shape}")

        # 采样（SHAP计算较慢）
        if len(X) > sample_size:
            print(f"  ⚡ 采样 {sample_size} 个样本进行SHAP分析")
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X

        # 创建SHAP分析器
        print(f"  🧮 计算SHAP值...")
        try:
            # 确定模型类型
            model_type_map = {
                'xgboost': 'tree',
                'lightgbm': 'tree',
                'catboost': 'tree',
                'random_forest': 'tree',
                'gradient_boosting': 'tree',
                'decision_tree': 'tree',
                'adaboost': 'tree',
                'ridge': 'linear',
                'lasso': 'linear',
                'elastic_net': 'linear'
            }

            shap_model_type = model_type_map.get(model_name, 'kernel')

            # 创建explainer
            if shap_model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            elif shap_model_type == 'linear':
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                # Kernel类型：强制快速模式（KMeans背景 + 限制样本数 + 限制nsamples）
                print(f"     ⚡ Kernel快速模式：kmeans={self.kernel_k}, nsamples={self.kernel_nsamples}")
                predict_fn, _ = self._resolve_predictor(model)
                # 限制样本量
                if len(X_sample) > self.kernel_max_samples:
                    sample_idx = np.random.choice(len(X_sample), self.kernel_max_samples, replace=False)
                    X_sample = X_sample[sample_idx]
                    print(f"     ⚠️ 已将样本数限制为 {len(X_sample)} 用于Kernel解释")
                # 使用kmeans摘要作为背景
                try:
                    background = shap.kmeans(X_sample, self.kernel_k)
                except Exception:
                    # 回退到随机采样
                    k = min(self.kernel_k, len(X_sample))
                    background = shap.sample(X_sample, k)
                explainer = shap.KernelExplainer(predict_fn, background)

            # 计算SHAP值
            # Kernel分支已在上方进入，直接限制nsamples提速；其它解释器正常计算
            if shap_model_type == 'kernel':
                shap_values = explainer.shap_values(X_sample, nsamples=self.kernel_nsamples)
            else:
                shap_values = explainer.shap_values(X_sample)

            print(f"     ✅ SHAP值计算完成")

        except Exception as e:
            print(f"  ❌ SHAP分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 分析特征重要性
        feature_names = self.get_feature_names()

        # 确保特征维度匹配
        if len(feature_names) > X_sample.shape[1]:
            feature_names = feature_names[:X_sample.shape[1]]
        elif len(feature_names) < X_sample.shape[1]:
            feature_names = feature_names + [f'Feature_{i}' for i in range(len(feature_names), X_sample.shape[1])]

        # 计算特征重要性
        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # 保存结果
        output_subdir = self.output_dir / model_name / target_clean
        output_subdir.mkdir(parents=True, exist_ok=True)

        # 保存特征重要性CSV
        csv_path = output_subdir / 'shap_feature_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        print(f"  💾 特征重要性已保存: {csv_path}")

        # 保存Top 30特征
        top30 = importance_df.head(30)

        # 创建可视化
        print(f"  📊 生成可视化...")

        # 1. 特征重要性条形图
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top30)), top30['importance'].values)
        plt.yticks(range(len(top30)), top30['feature'].values)
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top 30 Feature Importance - {model_name} ({target_clean})')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        fig_path = output_subdir / 'feature_importance_bar.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"     ✅ 条形图: {fig_path.name}")

        # 2. SHAP summary plot
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                max_display=30,
                show=False
            )
            summary_path = output_subdir / 'shap_summary_plot.png'
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"     ✅ Summary图: {summary_path.name}")
        except Exception as e:
            print(f"     ⚠️ Summary图生成失败: {e}")

        # 保存元数据
        metadata = {
            'model_name': model_name,
            'target': target_clean,
            'sample_size': len(X_sample),
            'n_features': X_sample.shape[1],
            'top_10_features': top30.head(10).to_dict('records'),
            'analysis_time': datetime.now().isoformat()
        }

        json_path = output_subdir / 'shap_metadata.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"  ✅ 分析完成")

        return {
            'model': model_name,
            'target': target_clean,
            'top_features': top30.head(10)
        }

    def generate_summary_report(self, results):
        """生成汇总报告"""
        print(f"\n{'='*70}")
        print(f"📝 生成汇总报告")
        print(f"{'='*70}")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SHAP可解释性分析报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            padding-left: 10px;
            border-left: 4px solid #3498db;
        }}
        .model-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .model-title {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .feature-table th {{
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .feature-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .feature-table tr:hover {{
            background: #f0f0f0;
        }}
        .images {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .images img {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .summary-box {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .file-link {{
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }}
        .file-link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 SHAP可解释性分析报告</h1>

        <div class="summary-box">
            <p><strong>📅 分析时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>📂 项目目录:</strong> {self.paper_dir.name}</p>
            <p><strong>🔬 分析模型数:</strong> {len(results)}</p>
            <p><strong>💡 说明:</strong> SHAP值表示每个特征对模型预测的贡献度。值越大表示该特征对预测结果影响越大。</p>
        </div>

        <h2>📊 分析结果</h2>
"""

        for result in results:
            if result is None:
                continue

            model = result['model']
            target = result['target']
            top_features = result['top_features']

            html += f"""
        <div class="model-section">
            <div class="model-title">🔹 {model} - {target}</div>

            <h3>Top 10 重要特征</h3>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>特征名称</th>
                        <th>重要性 (Mean |SHAP|)</th>
                    </tr>
                </thead>
                <tbody>
"""

            for idx, row in enumerate(top_features.iterrows(), 1):
                feature = row[1]['feature']
                importance = row[1]['importance']
                html += f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{feature}</td>
                        <td>{importance:.6f}</td>
                    </tr>
"""

            html += """
                </tbody>
            </table>

            <h3>可视化结果</h3>
            <div class="images">
"""

            # 添加图片链接
            img_dir = f"{model}/{target}"
            html += f"""
                <div>
                    <p><strong>特征重要性条形图</strong></p>
                    <a href="{img_dir}/feature_importance_bar.png" target="_blank">
                        <img src="{img_dir}/feature_importance_bar.png" alt="Feature Importance">
                    </a>
                </div>
                <div>
                    <p><strong>SHAP Summary Plot</strong></p>
                    <a href="{img_dir}/shap_summary_plot.png" target="_blank">
                        <img src="{img_dir}/shap_summary_plot.png" alt="SHAP Summary">
                    </a>
                </div>
"""

            html += """
            </div>

            <p style="margin-top: 15px;">
                📄 <a class="file-link" href="{}/{}/shap_feature_importance.csv">下载完整特征重要性数据 (CSV)</a> |
                📄 <a class="file-link" href="{}/{}/shap_metadata.json">元数据 (JSON)</a>
            </p>
        </div>
""".format(model, target, model, target)

        html += """
        <div class="summary-box" style="margin-top: 40px;">
            <h3>📖 如何使用这些结果</h3>
            <ol>
                <li><strong>识别关键特征:</strong> Top特征表明哪些分子性质对预测最重要</li>
                <li><strong>指导分子设计:</strong> 关注重要特征来优化分子结构</li>
                <li><strong>模型诊断:</strong> 检查模型是否关注合理的化学特征</li>
                <li><strong>论文写作:</strong> 在讨论部分解释模型的预测依据</li>
            </ol>
        </div>
    </div>
</body>
</html>
"""

        report_path = self.output_dir / 'shap_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"  ✅ 报告已生成: {report_path}")
        print(f"  🌐 用浏览器打开查看: file://{report_path.absolute()}")

        return report_path


def main():
    parser = argparse.ArgumentParser(
        description='SHAP可解释性分析工具 - 分析已训练的模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 分析所有模型
    python analyze_shap.py Paper_0930_222051

    # 只分析XGBoost和LightGBM
    python analyze_shap.py Paper_0930_222051 --models xgboost lightgbm

    # 使用更大的样本量
    python analyze_shap.py Paper_0930_222051 --sample-size 200
        """
    )

    parser.add_argument('paper_dir', help='论文输出目录 (如: Paper_0930_222051)')
    parser.add_argument('--models', nargs='+', help='指定要分析的模型 (如: xgboost lightgbm)')
    parser.add_argument('--sample-size', type=int, default=100, help='SHAP分析的样本数量 (默认: 100)')
    parser.add_argument('--kernel-k', type=int, default=20, help='KernelExplainer的背景kmeans聚类数 (默认: 20)')
    parser.add_argument('--kernel-nsamples', type=int, default=200, help='KernelExplainer每次解释的采样次数上限 (默认: 200)')
    parser.add_argument('--kernel-max-samples', type=int, default=40, help='Kernel模型参与解释的最大样本数 (默认: 40)')

    args = parser.parse_args()

    print("="*70)
    print("🎯 SHAP可解释性分析工具")
    print("="*70)

    # 创建分析器
    analyzer = ModelShapAnalyzer(args.paper_dir)
    # 覆盖Kernel快速参数
    analyzer.kernel_k = max(5, int(args.kernel_k))
    analyzer.kernel_nsamples = max(50, int(args.kernel_nsamples))
    analyzer.kernel_max_samples = max(10, int(args.kernel_max_samples))

    # 查找模型
    models = analyzer.find_models(model_filter=args.models)

    if not models:
        print("\n❌ 未找到任何模型文件")
        return 1

    # 分析每个模型
    results = []
    for model_info in models:
        result = analyzer.analyze_model(model_info, sample_size=args.sample_size)
        results.append(result)

    # 生成汇总报告
    analyzer.generate_summary_report(results)

    print("\n" + "="*70)
    print("✅ 所有分析完成!")
    print(f"📁 结果保存在: {analyzer.output_dir}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
