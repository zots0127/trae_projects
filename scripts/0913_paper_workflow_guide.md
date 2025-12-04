# 0913 论文数据处理完整工作流程指南

> 本指南提供了完整的论文数据处理流程，从模型训练到预测，再到生成所有论文所需的图表和数据。

## 目录
1. [环境准备](#1-环境准备)
2. [数据说明](#2-数据说明)
3. [步骤1：数据准备和验证](#步骤1数据准备和验证)
4. [步骤2：训练所有模型](#步骤2训练所有模型)
5. [步骤3：生成模型对比表格](#步骤3生成模型对比表格)
6. [步骤4：预测虚拟数据库](#步骤4预测虚拟数据库)
7. [步骤5：生成分段性能分析图](#步骤5生成分段性能分析图plqy混淆矩阵)
8. [步骤6：生成论文图表](#步骤6生成论文图表)
9. [步骤7：预测测试数据](#步骤7预测测试数据)
10. [步骤8：生成最终报告](#步骤8生成最终报告)
11. [一键执行脚本](#一键执行脚本)

---

## 1. 环境准备

```bash
# 切换到工作目录
cd /Users/kanshan/IR/ir2025/v3

# 检查Python环境
python --version  # 需要 Python 3.8+

# 安装依赖（如果需要）
pip install -r requirements.txt
```

## 2. 数据说明

### 核心数据文件
- **训练数据**: `data/Database_normalized.csv` - 实验测量的分子数据
- **虚拟数据库**: `data/ir_assemble.csv` - 基于训练数据组装的272,104个分子组合（用于预测）
- **测试数据**: `data/Database_ours_0903update_normalized.csv` - 用于模型预测展示

### 数据结构
```python
# 所有数据包含以下列：
# - L1, L2, L3: SMILES字符串（配体分子结构）
# - Max_wavelength(nm): 最大发射波长
# - PLQY: 光致发光量子产率
# - tau(s*10^-6): 激发态寿命
```

### 训练和预测流程
1. **训练阶段**：使用 `Database_normalized.csv` 进行10折交叉验证，然后训练最终模型
2. **预测阶段**：使用训练好的最终模型预测 `ir_assemble.csv` 中的272,104个组合

---

## 步骤1：数据准备和验证

### 1.1 验证数据文件

```bash
# 检查所有必需的数据文件
echo "=========================================="
echo "检查数据文件"
echo "=========================================="

# 检查训练数据
if [ -f "data/Database_normalized.csv" ]; then
    echo "✅ 训练数据存在"
    wc -l data/Database_normalized.csv
else
    echo "❌ 训练数据不存在"
    exit 1
fi

# 检查测试数据
if [ -f "data/Database_ours_0903update_normalized.csv" ]; then
    echo "✅ 测试数据存在"
    wc -l data/Database_ours_0903update_normalized.csv
else
    echo "❌ 测试数据不存在"
fi

# 检查虚拟数据库
if [ -f "data/ir_assemble.csv" ]; then
    echo "✅ 虚拟数据库存在"
    wc -l data/ir_assemble.csv
else
    echo "⚠️ 虚拟数据库不存在，需要生成"
fi
```

### 1.2 生成虚拟数据库（如果不存在）

```python
# 文件: generate_virtual_database.py
# 如果 ir_assemble.csv 不存在，运行此脚本生成

python -c "
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path

# 读取训练数据
df = pd.read_csv('data/Database_normalized.csv')

# 提取唯一的配体
l1_unique = df['L1'].dropna().unique()
l2_unique = df['L2'].dropna().unique()
l3_unique = df['L3'].dropna().unique()

# 合并L1和L2（理论上应该是相同的配体集）
l12_unique = np.unique(np.concatenate([l1_unique, l2_unique]))

print(f'配体统计:')
print(f'  L1/L2配体池: {len(l12_unique)} 个')
print(f'  L3配体池: {len(l3_unique)} 个')
print(f'  理论组合数: {len(l12_unique) * len(l3_unique):,}')

# 生成所有L1=L2的组合
all_combinations = []
for l12 in l12_unique:
    for l3 in l3_unique:
        all_combinations.append({
            'L1': l12,
            'L2': l12,  # L1和L2相同
            'L3': l3
        })

# 创建DataFrame并保存
assembled_df = pd.DataFrame(all_combinations)
assembled_df.to_csv('data/ir_assemble.csv', index=False)
print(f'✅ 虚拟数据库已生成: {len(assembled_df):,} 个组合')
"
```

### 1.3 数据统计分析

```python
python -c "
import pandas as pd
import numpy as np

print('='*60)
print('数据统计分析')
print('='*60)

# 分析训练数据
df_train = pd.read_csv('data/Database_normalized.csv')
print(f'\n训练数据 (Database_normalized.csv):')
print(f'  总样本数: {len(df_train)}')
print(f'  波长数据: {df_train[\"Max_wavelength(nm)\"].notna().sum()} 个')
print(f'  PLQY数据: {df_train[\"PLQY\"].notna().sum()} 个')
print(f'  寿命数据: {df_train[\"tau(s*10^-6)\"].notna().sum()} 个')

# 同时有波长和PLQY的样本（交集）
intersection = df_train[['Max_wavelength(nm)', 'PLQY']].notna().all(axis=1).sum()
print(f'  波长∩PLQY: {intersection} 个 ({100*intersection/len(df_train):.1f}%)')

# 目标变量统计
if df_train['Max_wavelength(nm)'].notna().any():
    print(f'\n波长统计:')
    print(f'  范围: {df_train[\"Max_wavelength(nm)\"].min():.1f} - {df_train[\"Max_wavelength(nm)\"].max():.1f} nm')
    print(f'  均值: {df_train[\"Max_wavelength(nm)\"].mean():.1f} ± {df_train[\"Max_wavelength(nm)\"].std():.1f} nm')

if df_train['PLQY'].notna().any():
    print(f'\nPLQY统计:')
    print(f'  范围: {df_train[\"PLQY\"].min():.4f} - {df_train[\"PLQY\"].max():.4f}')
    print(f'  均值: {df_train[\"PLQY\"].mean():.4f} ± {df_train[\"PLQY\"].std():.4f}')

# 分析虚拟数据库
try:
    df_virtual = pd.read_csv('data/ir_assemble.csv')
    print(f'\n虚拟数据库 (ir_assemble.csv):')
    print(f'  总组合数: {len(df_virtual):,}')
except:
    print(f'\n虚拟数据库不存在')

# 分析测试数据
try:
    df_test = pd.read_csv('data/Database_ours_0903update_normalized.csv')
    print(f'\n测试数据 (Database_ours_0903update_normalized.csv):')
    print(f'  总样本数: {len(df_test)}')
except:
    print(f'\n测试数据不存在')

print('='*60)
"
```

---

## 步骤2：训练所有模型

### 2.1 设置训练参数

```bash
# 设置输出目录
OUTPUT_DIR="Paper_0913_$(date +%H%M%S)"
echo "输出目录: $OUTPUT_DIR"

# 数据文件
DATA_FILE="data/Database_normalized.csv"
TEST_DATA_FILE="data/Database_ours_0903update_normalized.csv"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
```

### 2.2 训练模型 - 三种模式可选

根据您的时间和精度需求，选择以下三种模式之一：

#### 模式A：快速测试（10-15分钟）
适用于：快速验证流程、调试、初步测试

```bash
echo "=========================================="
echo "快速模式：训练3个关键模型"
echo "=========================================="

python automl.py train \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=all_models_quick \
    models=xgboost,lightgbm,catboost \
    training.n_folds=10 \
    'training.metrics=["r2","rmse","mae"]' \
    training.save_final_model=true \
    training.save_fold_models=false \
    training.save_feature_importance=true \
    training.verbose=1 \
    feature.feature_type=combined \
    feature.morgan_bits=1024 \
    feature.morgan_radius=2 \
    feature.combination_method=mean \
    feature.use_cache=true \
    'data.smiles_columns=["L1","L2","L3"]' \
    'data.target_columns=["Max_wavelength(nm)","PLQY"]' \
    data.multi_target_strategy=independent \
    data.nan_handling=skip \
    data.train_ratio=1.0 \
    data.val_ratio=0.0 \
    data.test_ratio=0.0

echo "✅ 快速训练完成（XGBoost, LightGBM, CatBoost）"
```

#### 模式B：标准论文模式（30-60分钟）【推荐】
适用于：论文数据生成、标准对比分析

**重要说明**：设置 `training.save_final_model=true` 时，系统会：
1. 首先执行10折交叉验证，获得稳健的性能评估指标
2. 然后使用100%的训练数据重新训练一个最终模型
3. 保存这个最终模型用于后续的预测任务（如虚拟数据库预测）

这确保了：评估指标来自严格的交叉验证，而预测使用了所有可用数据训练的模型。

```bash
echo "=========================================="
echo "标准模式：训练所有13个模型（使用默认参数）"
echo "=========================================="

python automl.py train \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=all_models \
    models=adaboost,catboost,decision_tree,elastic_net,extra_trees,gradient_boosting,knn,lasso,lightgbm,random_forest,ridge,svr,xgboost \
    training.n_folds=10 \
    'training.metrics=["r2","rmse","mae"]' \
    training.save_final_model=true \
    training.save_fold_models=true \
    training.save_feature_importance=true \
    training.verbose=1 \
    feature.feature_type=combined \
    feature.morgan_bits=1024 \
    feature.morgan_radius=2 \
    feature.combination_method=mean \
    feature.use_cache=true \
    'data.smiles_columns=["L1","L2","L3"]' \
    'data.target_columns=["Max_wavelength(nm)","PLQY"]' \
    data.multi_target_strategy=independent \
    data.nan_handling=skip \
    data.train_ratio=1.0 \
    data.val_ratio=0.0 \
    data.test_ratio=0.0 \
    comparison.enable=true \
    'comparison.formats=["markdown","html","latex","csv"]' \
    comparison.highlight_best=true \
    comparison.include_std=true \
    comparison.decimal_places.r2=4 \
    comparison.decimal_places.rmse=4 \
    comparison.decimal_places.mae=4 \
    export.enable=true \
    'export.formats=["json","csv","excel"]' \
    export.include_predictions=true \
    export.include_feature_importance=true \
    export.include_cv_details=true \
    export.generate_plots=true \
    export.generate_report=true

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "✅ 所有13个模型训练完成"
else
    echo "❌ 训练失败"
    exit 1
fi
```

#### 模式C：使用预定义配置文件
适用于：使用标准化配置、易于重复实验

```bash
echo "=========================================="
echo "配置文件模式：使用standard配置训练单个模型"
echo "=========================================="

# 训练XGBoost（使用xgboost_standard配置）
python automl.py train \
    config=xgboost_standard \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=xgboost_model

# 训练LightGBM（使用lightgbm_standard配置）
python automl.py train \
    config=lightgbm_standard \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=lightgbm_model

# 训练CatBoost（使用catboost_standard配置）
python automl.py train \
    config=catboost_standard \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=catboost_model

echo "✅ 模型训练完成"
```

#### 模式选择建议

| 模式 | 用途 | 模型数 | 交叉验证 | 配置方式 | 预计时间 |
|------|------|--------|----------|----------|----------|
| A-快速 | 测试流程 | 3个 | 5折 | 命令行参数 | 10-15分钟 |
| B-标准 | **论文对比** | 13个 | 10折 | 命令行参数 | 30-60分钟 |
| C-配置 | 单模型训练 | 1个/次 | 10折 | 配置文件 | 5-10分钟/模型 |

**默认推荐使用模式B**，它能在合理时间内完成所有模型的训练，适合论文数据生成。

### 2.3 训练交集模型（同时有波长和PLQY的样本）

**说明**：此步骤创建一个特殊的数据集，只包含同时有波长和PLQY值的样本，用于训练更精确的联合预测模型。

```bash
echo "=========================================="
echo "创建交集数据并训练"
echo "=========================================="

# 创建交集数据
python -c "
import pandas as pd

# 读取数据
df = pd.read_csv('$DATA_FILE')

# 筛选同时有波长和PLQY的样本
mask = df[['Max_wavelength(nm)', 'PLQY']].notna().all(axis=1)
df_intersection = df[mask].copy()

# 移除tau列（交集模型不预测tau）
if 'tau(s*10^-6)' in df_intersection.columns:
    df_intersection = df_intersection.drop('tau(s*10^-6)', axis=1)

# 保存
df_intersection.to_csv('$OUTPUT_DIR/intersection_data.csv', index=False)
print(f'✅ 交集数据创建完成')
print(f'   原始样本: {len(df)}')
print(f'   交集样本: {len(df_intersection)} ({100*len(df_intersection)/len(df):.1f}%)')
"

# 训练最佳模型（XGBoost）的交集版本
# 注意：同样会进行10折CV评估，然后训练最终模型
python automl.py train \
    data=$OUTPUT_DIR/intersection_data.csv \
    project=$OUTPUT_DIR \
    name=xgboost_intersection \
    model.model_type=xgboost \
    model.hyperparameters.n_estimators=1000 \
    model.hyperparameters.max_depth=10 \
    model.hyperparameters.learning_rate=0.03 \
    model.hyperparameters.subsample=0.8 \
    model.hyperparameters.colsample_bytree=0.8 \
    training.n_folds=10 \
    training.metrics=[r2,rmse,mae] \
    training.save_final_model=true \
    training.save_fold_models=false \
    training.save_feature_importance=true \
    feature.feature_type=combined \
    feature.morgan_bits=1024 \
    feature.morgan_radius=2 \
    feature.combination_method=mean \
    data.smiles_columns=[L1,L2,L3] \
    'data.target_columns=["Max_wavelength(nm)","PLQY"]' \
    data.nan_handling=skip
```

---

## 步骤3：生成模型对比表格

### 3.1 收集所有模型结果

```python
python -c "
import pandas as pd
import json
import numpy as np
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
results = []

# 查找所有训练结果
for train_dir in output_dir.glob('all_models*/'):
    results_file = train_dir / 'results' / 'training_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
            
            # 提取每个模型和目标的结果
            for model_name, model_data in data.items():
                if isinstance(model_data, dict) and 'targets' in model_data:
                    for target, target_data in model_data['targets'].items():
                        if 'fold_results' in target_data:
                            # 计算10-fold的统计
                            r2_scores = [fold['metrics']['r2'] for fold in target_data['fold_results']]
                            rmse_scores = [fold['metrics']['rmse'] for fold in target_data['fold_results']]
                            mae_scores = [fold['metrics']['mae'] for fold in target_data['fold_results']]
                            
                            results.append({
                                'Model': model_name.upper(),
                                'Target': target.replace('(nm)', '').replace('(s*10^-6)', ''),
                                'R2_mean': np.mean(r2_scores),
                                'R2_std': np.std(r2_scores),
                                'RMSE_mean': np.mean(rmse_scores),
                                'RMSE_std': np.std(rmse_scores),
                                'MAE_mean': np.mean(mae_scores),
                                'MAE_std': np.std(mae_scores),
                                'Samples': target_data.get('training_samples', 0)
                            })

# 创建DataFrame并保存
df_results = pd.DataFrame(results)
df_results.to_csv(output_dir / 'model_comparison_full.csv', index=False)

# 生成格式化的对比表格
print('='*120)
print('模型性能对比表 (10-fold Cross-Validation)')
print('='*120)

for target in df_results['Target'].unique():
    print(f'\n目标: {target}')
    print('-'*100)
    
    target_df = df_results[df_results['Target'] == target].copy()
    
    # 格式化显示
    target_df['R²'] = target_df.apply(lambda x: f\"{x['R2_mean']:.4f} ± {x['R2_std']:.4f}\", axis=1)
    target_df['RMSE'] = target_df.apply(lambda x: f\"{x['RMSE_mean']:.2f} ± {x['RMSE_std']:.2f}\", axis=1)
    target_df['MAE'] = target_df.apply(lambda x: f\"{x['MAE_mean']:.2f} ± {x['MAE_std']:.2f}\", axis=1)
    
    display_df = target_df[['Model', 'R²', 'RMSE', 'MAE', 'Samples']]
    display_df = display_df.sort_values('Model')
    
    print(display_df.to_string(index=False))

# 找出最佳模型
print('\n' + '='*120)
print('最佳模型总结')
print('='*120)

best_models = []
for target in df_results['Target'].unique():
    target_df = df_results[df_results['Target'] == target]
    best_idx = target_df['R2_mean'].idxmax()
    best = target_df.loc[best_idx]
    
    best_models.append({
        'Target': target,
        'Best Model': best['Model'],
        'R²': f\"{best['R2_mean']:.4f} ± {best['R2_std']:.4f}\",
        'RMSE': f\"{best['RMSE_mean']:.2f} ± {best['RMSE_std']:.2f}\",
        'MAE': f\"{best['MAE_mean']:.2f} ± {best['MAE_std']:.2f}\"
    })

best_df = pd.DataFrame(best_models)
print(best_df.to_string(index=False))
best_df.to_csv(output_dir / 'best_models_summary.csv', index=False)
"
```

### 3.2 生成LaTeX和Markdown表格

```python
python -c "
import pandas as pd
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
df = pd.read_csv(output_dir / 'model_comparison_full.csv')

# 生成LaTeX表格
latex_content = r'''\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{multirow}
\\begin{document}

\\begin{table}[htbp]
\\centering
\\caption{Model Performance Comparison (10-fold Cross-Validation)}
\\label{tab:model_comparison}
\\begin{tabular}{llcccc}
\\toprule
Model & Target & R\\textsuperscript{2} & RMSE & MAE & Samples \\\\
\\midrule
'''

for _, row in df.iterrows():
    model = row['Model']
    target = row['Target'].replace('_', '\\\\_')
    r2 = f\"{row['R2_mean']:.4f} \\\\pm {row['R2_std']:.4f}\"
    rmse = f\"{row['RMSE_mean']:.2f} \\\\pm {row['RMSE_std']:.2f}\"
    mae = f\"{row['MAE_mean']:.2f} \\\\pm {row['MAE_std']:.2f}\"
    samples = int(row['Samples'])
    
    latex_content += f\"{model} & {target} & \\${r2}\\$ & \\${rmse}\\$ & \\${mae}\\$ & {samples} \\\\\\\\\\n\"

latex_content += r'''\\bottomrule
\\end{tabular}
\\end{table}

\\end{document}'''

# 保存LaTeX文件
with open(output_dir / 'model_comparison_table.tex', 'w') as f:
    f.write(latex_content)

print(f'✅ LaTeX表格已生成: {output_dir}/model_comparison_table.tex')

# 生成Markdown表格
markdown_content = '# Model Performance Comparison (10-fold Cross-Validation)\\n\\n'
markdown_content += '| Model | Target | R² | RMSE | MAE | Samples |\\n'
markdown_content += '|-------|--------|-----|------|-----|---------|\\n'

for _, row in df.iterrows():
    r2 = f\"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}\"
    rmse = f\"{row['RMSE_mean']:.2f} ± {row['RMSE_std']:.2f}\"
    mae = f\"{row['MAE_mean']:.2f} ± {row['MAE_std']:.2f}\"
    
    markdown_content += f\"| {row['Model']} | {row['Target']} | {r2} | {rmse} | {mae} | {int(row['Samples'])} |\\n\"

with open(output_dir / 'model_comparison_table.md', 'w') as f:
    f.write(markdown_content)

print(f'✅ Markdown表格已生成: {output_dir}/model_comparison_table.md')
"
```

---

## 步骤4：预测虚拟数据库

### 4.1 使用最终训练模型预测

**说明**：这一步使用步骤2中训练的最终模型（基于100%训练数据）来预测虚拟数据库中的272,104个分子组合。

```bash
echo "=========================================="
echo "预测虚拟数据库 (272,104个组合)"
echo "使用最终训练模型（100%数据训练）"
echo "=========================================="

# 查找最终模型文件（这些是用完整训练集训练的模型）
BEST_MODEL_DIR=$(ls -td $OUTPUT_DIR/all_models*/models 2>/dev/null | head -1)

if [ -z "$BEST_MODEL_DIR" ]; then
    # 如果没有all_models，尝试使用xgboost_intersection
    BEST_MODEL_DIR=$(ls -td $OUTPUT_DIR/xgboost_intersection*/models 2>/dev/null | head -1)
fi

echo "模型目录: $BEST_MODEL_DIR"

# 预测波长（使用最终模型）
WAVELENGTH_MODEL=$(ls -t $BEST_MODEL_DIR/*wavelength*.joblib 2>/dev/null | head -1)
if [ -n "$WAVELENGTH_MODEL" ]; then
    echo "预测波长（使用100%数据训练的最终模型）..."
    echo "处理272,104个分子组合，批次大小=10000"
    python automl.py predict \
        model=$WAVELENGTH_MODEL \
        data=data/ir_assemble.csv \
        output=$OUTPUT_DIR/virtual_predict_wavelength.csv \
        feature.feature_type=combined \
        feature.morgan_bits=1024 \
        feature.morgan_radius=2 \
        feature.combination_method=mean \
        data.smiles_columns=[L1,L2,L3] \
        batch_size=10000
fi

# 预测PLQY（使用最终模型）
PLQY_MODEL=$(ls -t $BEST_MODEL_DIR/*PLQY*.joblib 2>/dev/null | head -1)
if [ -n "$PLQY_MODEL" ]; then
    echo "预测PLQY（使用100%数据训练的最终模型）..."
    echo "处理272,104个分子组合，批次大小=10000"
    python automl.py predict \
        model=$PLQY_MODEL \
        data=data/ir_assemble.csv \
        output=$OUTPUT_DIR/virtual_predict_plqy.csv \
        feature.feature_type=combined \
        feature.morgan_bits=1024 \
        feature.morgan_radius=2 \
        feature.combination_method=mean \
        data.smiles_columns=[L1,L2,L3] \
        batch_size=10000
fi
```

### 4.2 合并预测结果并筛选高性能候选

```python
python -c "
import pandas as pd
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')

# 读取原始数据
df_virtual = pd.read_csv('data/ir_assemble.csv')

# 合并预测结果
try:
    wavelength = pd.read_csv(output_dir / 'virtual_predict_wavelength.csv')
    df_virtual['Predicted_wavelength'] = wavelength['predictions']
except:
    print('⚠️ 波长预测文件不存在')

try:
    plqy = pd.read_csv(output_dir / 'virtual_predict_plqy.csv')
    df_virtual['Predicted_PLQY'] = plqy['predictions']
except:
    print('⚠️ PLQY预测文件不存在')

# 保存完整预测结果
df_virtual.to_csv(output_dir / 'virtual_predict_all.csv', index=False)
print(f'✅ 虚拟数据库预测完成: {len(df_virtual):,} 个组合')

# 筛选高PLQY候选
if 'Predicted_PLQY' in df_virtual.columns:
    # PLQY >= 0.9
    high_plqy_09 = df_virtual[df_virtual['Predicted_PLQY'] >= 0.9].sort_values('Predicted_PLQY', ascending=False)
    if len(high_plqy_09) > 0:
        high_plqy_09.to_csv(output_dir / 'virtual_predict_plqy_0.9+.csv', index=False)
        print(f'✅ PLQY >= 0.9: {len(high_plqy_09):,} 个候选')
    
    # PLQY >= 0.8
    high_plqy_08 = df_virtual[df_virtual['Predicted_PLQY'] >= 0.8].sort_values('Predicted_PLQY', ascending=False)
    if len(high_plqy_08) > 0:
        high_plqy_08.to_csv(output_dir / 'virtual_predict_plqy_0.8+.csv', index=False)
        print(f'✅ PLQY >= 0.8: {len(high_plqy_08):,} 个候选')
    
    # PLQY >= 0.7
    high_plqy_07 = df_virtual[df_virtual['Predicted_PLQY'] >= 0.7].sort_values('Predicted_PLQY', ascending=False)
    if len(high_plqy_07) > 0:
        high_plqy_07.to_csv(output_dir / 'virtual_predict_plqy_0.7+.csv', index=False)
        print(f'✅ PLQY >= 0.7: {len(high_plqy_07):,} 个候选')
    
    # 统计分析
    print(f'\nPLQY预测统计:')
    print(f'  范围: {df_virtual[\"Predicted_PLQY\"].min():.4f} - {df_virtual[\"Predicted_PLQY\"].max():.4f}')
    print(f'  均值: {df_virtual[\"Predicted_PLQY\"].mean():.4f} ± {df_virtual[\"Predicted_PLQY\"].std():.4f}')
    
    # Top 20 候选
    top20 = df_virtual.nlargest(20, 'Predicted_PLQY')
    top20.to_csv(output_dir / 'virtual_predict_top20.csv', index=False)
    
    print(f'\nTop 10 高PLQY候选:')
    for i, row in top20.head(10).iterrows():
        print(f'  {i+1}. PLQY={row[\"Predicted_PLQY\"]:.4f}')
        if 'Predicted_wavelength' in row:
            print(f'      λ={row[\"Predicted_wavelength\"]:.1f}nm')

if 'Predicted_wavelength' in df_virtual.columns:
    print(f'\n波长预测统计:')
    print(f'  范围: {df_virtual[\"Predicted_wavelength\"].min():.1f} - {df_virtual[\"Predicted_wavelength\"].max():.1f} nm')
    print(f'  均值: {df_virtual[\"Predicted_wavelength\"].mean():.1f} ± {df_virtual[\"Predicted_wavelength\"].std():.1f} nm')
"
```

---

## 步骤5：生成分段性能分析图（PLQY混淆矩阵）

### 5.1 生成PLQY范围准确性分析

这一步生成模型在不同PLQY范围内的预测准确性分析，包括混淆矩阵热图。

```bash
echo "=========================================="
echo "生成分段性能分析图"
echo "=========================================="

# 使用训练结果生成分段分析
python -c "
import sys
sys.path.append('/Users/kanshan/IR/ir2025/v3')
from visualization.stratified_analysis import generate_stratified_analysis
from pathlib import Path
import pandas as pd
import numpy as np
import json

# 设置项目目录
project_dir = Path('$OUTPUT_DIR')

# 查找所有模型的预测结果
predictions = {}

# 遍历所有训练结果目录
for model_dir in project_dir.glob('all_models*/'):
    # 查找CV预测结果
    for result_file in model_dir.glob('results/cv_predictions_*.csv'):
        target = result_file.stem.replace('cv_predictions_', '')
        
        # 读取预测结果
        df = pd.read_csv(result_file)
        if 'actual' in df.columns and 'predicted' in df.columns:
            predictions[target] = {
                'actual': df['actual'].values,
                'predicted': df['predicted'].values
            }
            print(f'  加载 {target}: {len(df)} 个预测')

# 如果没有找到cv_predictions文件，尝试从training_results.json提取
if not predictions:
    for model_dir in project_dir.glob('all_models*/'):
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
                                    all_actual.extend(fold['predictions'].get('actual', []))
                                    all_predicted.extend(fold['predictions'].get('predicted', []))
                            
                            if all_actual:
                                key = f'{target}_{model_name}'
                                predictions[key] = {
                                    'actual': np.array(all_actual),
                                    'predicted': np.array(all_predicted)
                                }
                                print(f'  加载 {key}: {len(all_actual)} 个预测')

# 生成分段分析
if predictions:
    output_dir = project_dir / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成分析图表
    results = generate_stratified_analysis(predictions, output_dir)
    
    print(f'✅ 分段性能分析已生成: {output_dir}/stratified_analysis/')
else:
    print('⚠️ 未找到预测数据，无法生成分段分析')
"
```

### 5.2 使用配置开关启用分段分析（可选）

如果你想通过配置文件控制是否生成分段分析，可以在训练时添加配置：

```bash
# 在训练命令中添加分段分析配置
python automl.py train \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=all_models \
    models=xgboost,lightgbm,catboost \
    # ... 其他参数 ...
    export.stratified_analysis=true  # 启用分段分析
```

---

## 步骤6：生成论文图表

### 6.1 生成模型性能对比图

```python
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

output_dir = Path('$OUTPUT_DIR')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# 读取模型对比数据
df = pd.read_csv(output_dir / 'model_comparison_full.csv')

# 创建图表目录
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# 图1: R²对比（按目标分组）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
targets = df['Target'].unique()

for idx, target in enumerate(targets[:3]):
    target_df = df[df['Target'] == target].sort_values('R2_mean', ascending=False)
    
    ax = axes[idx]
    x = range(len(target_df))
    ax.bar(x, target_df['R2_mean'], yerr=target_df['R2_std'], 
           capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(target_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title(f'{target}')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

plt.suptitle('Model Performance Comparison (R² with Std Dev)', fontsize=14)
plt.tight_layout()
plt.savefig(figures_dir / 'model_comparison_r2.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'✅ 生成图表: {figures_dir}/model_comparison_r2.png')

# 图2: RMSE对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, target in enumerate(targets[:3]):
    target_df = df[df['Target'] == target].sort_values('RMSE_mean')
    
    ax = axes[idx]
    x = range(len(target_df))
    ax.bar(x, target_df['RMSE_mean'], yerr=target_df['RMSE_std'], 
           capsize=5, alpha=0.7, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(target_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('RMSE')
    ax.set_title(f'{target}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Model Performance Comparison (RMSE with Std Dev)', fontsize=14)
plt.tight_layout()
plt.savefig(figures_dir / 'model_comparison_rmse.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'✅ 生成图表: {figures_dir}/model_comparison_rmse.png')
"
```

### 6.2 生成预测分布图

```python
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

output_dir = Path('$OUTPUT_DIR')
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# 读取虚拟数据库预测结果
df_virtual = pd.read_csv(output_dir / 'virtual_predict_all.csv')

# 图3: PLQY预测分布
if 'Predicted_PLQY' in df_virtual.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直方图
    ax = axes[0]
    ax.hist(df_virtual['Predicted_PLQY'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted PLQY')
    ax.set_ylabel('Count')
    ax.set_title(f'PLQY Distribution (n={len(df_virtual):,})')
    ax.axvline(0.9, color='red', linestyle='--', label='PLQY=0.9')
    ax.axvline(0.8, color='orange', linestyle='--', label='PLQY=0.8')
    ax.axvline(0.7, color='yellow', linestyle='--', label='PLQY=0.7')
    ax.legend()
    
    # 累积分布
    ax = axes[1]
    sorted_plqy = np.sort(df_virtual['Predicted_PLQY'])
    cumulative = np.arange(1, len(sorted_plqy) + 1) / len(sorted_plqy)
    ax.plot(sorted_plqy, cumulative, linewidth=2)
    ax.set_xlabel('Predicted PLQY')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('PLQY Cumulative Distribution')
    ax.grid(True, alpha=0.3)
    ax.axvline(0.9, color='red', linestyle='--', alpha=0.5)
    ax.axvline(0.8, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(0.7, color='yellow', linestyle='--', alpha=0.5)
    
    plt.suptitle('Virtual Database PLQY Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'plqy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'✅ 生成图表: {figures_dir}/plqy_distribution.png')

# 图4: 波长预测分布
if 'Predicted_wavelength' in df_virtual.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直方图
    ax = axes[0]
    ax.hist(df_virtual['Predicted_wavelength'], bins=50, alpha=0.7, 
            edgecolor='black', color='skyblue')
    ax.set_xlabel('Predicted Wavelength (nm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Wavelength Distribution (n={len(df_virtual):,})')
    
    # 箱线图
    ax = axes[1]
    ax.boxplot(df_virtual['Predicted_wavelength'], vert=True)
    ax.set_ylabel('Predicted Wavelength (nm)')
    ax.set_title('Wavelength Box Plot')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Virtual Database Wavelength Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'wavelength_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'✅ 生成图表: {figures_dir}/wavelength_distribution.png')

# 图5: 波长vs PLQY散点图
if 'Predicted_wavelength' in df_virtual.columns and 'Predicted_PLQY' in df_virtual.columns:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建2D直方图
    hexbin = ax.hexbin(df_virtual['Predicted_wavelength'], 
                       df_virtual['Predicted_PLQY'],
                       gridsize=50, cmap='YlOrRd', mincnt=1)
    
    ax.set_xlabel('Predicted Wavelength (nm)')
    ax.set_ylabel('Predicted PLQY')
    ax.set_title(f'Wavelength vs PLQY Predictions (n={len(df_virtual):,})')
    
    # 添加颜色条
    cb = plt.colorbar(hexbin)
    cb.set_label('Count')
    
    # 添加参考线
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='PLQY=0.9')
    ax.axhline(0.8, color='orange', linestyle='--', alpha=0.5, label='PLQY=0.8')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'wavelength_vs_plqy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'✅ 生成图表: {figures_dir}/wavelength_vs_plqy.png')
"
```

---

## 步骤7：预测测试数据

### 7.1 预测Database_ours_0903update_normalized.csv

```bash
echo "=========================================="
echo "预测测试数据"
echo "=========================================="

# 使用最佳模型预测
# 预测波长
if [ -n "$WAVELENGTH_MODEL" ]; then
    echo "预测测试数据波长..."
    python automl.py predict \
        model=$WAVELENGTH_MODEL \
        data=data/Database_ours_0903update_normalized.csv \
        output=$OUTPUT_DIR/test_predict_wavelength.csv \
        feature.feature_type=combined \
        feature.morgan_bits=1024 \
        feature.morgan_radius=2 \
        feature.combination_method=mean \
        data.smiles_columns=[L1,L2,L3]
fi

# 预测PLQY
if [ -n "$PLQY_MODEL" ]; then
    echo "预测测试数据PLQY..."
    python automl.py predict \
        model=$PLQY_MODEL \
        data=data/Database_ours_0903update_normalized.csv \
        output=$OUTPUT_DIR/test_predict_plqy.csv \
        feature.feature_type=combined \
        feature.morgan_bits=1024 \
        feature.morgan_radius=2 \
        feature.combination_method=mean \
        data.smiles_columns=[L1,L2,L3]
fi

# 预测tau（如果有模型）
TAU_MODEL=$(ls -t $BEST_MODEL_DIR/*tau*.joblib 2>/dev/null | head -1)
if [ -n "$TAU_MODEL" ]; then
    echo "预测测试数据寿命..."
    python automl.py predict \
        model=$TAU_MODEL \
        data=data/Database_ours_0903update_normalized.csv \
        output=$OUTPUT_DIR/test_predict_tau.csv \
        feature.feature_type=combined \
        feature.morgan_bits=1024 \
        feature.morgan_radius=2 \
        feature.combination_method=mean \
        data.smiles_columns=[L1,L2,L3]
fi
```

### 7.2 合并测试预测结果

```python
python -c "
import pandas as pd
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')

# 读取测试数据
df_test = pd.read_csv('data/Database_ours_0903update_normalized.csv')

# 合并预测结果
try:
    wavelength = pd.read_csv(output_dir / 'test_predict_wavelength.csv')
    df_test['Predicted_wavelength'] = wavelength['predictions']
except:
    print('⚠️ 波长预测文件不存在')

try:
    plqy = pd.read_csv(output_dir / 'test_predict_plqy.csv')
    df_test['Predicted_PLQY'] = plqy['predictions']
except:
    print('⚠️ PLQY预测文件不存在')

try:
    tau = pd.read_csv(output_dir / 'test_predict_tau.csv')
    df_test['Predicted_tau'] = tau['predictions']
except:
    print('⚠️ 寿命预测文件不存在')

# 保存完整结果
df_test.to_csv(output_dir / 'test_predict_all.csv', index=False)
print(f'✅ 测试数据预测完成: {len(df_test)} 个样本')

# 统计分析
print('\n测试数据预测统计:')
if 'Predicted_wavelength' in df_test.columns:
    print(f'  波长范围: {df_test[\"Predicted_wavelength\"].min():.1f} - {df_test[\"Predicted_wavelength\"].max():.1f} nm')
    print(f'  波长均值: {df_test[\"Predicted_wavelength\"].mean():.1f} ± {df_test[\"Predicted_wavelength\"].std():.1f} nm')

if 'Predicted_PLQY' in df_test.columns:
    print(f'  PLQY范围: {df_test[\"Predicted_PLQY\"].min():.4f} - {df_test[\"Predicted_PLQY\"].max():.4f}')
    print(f'  PLQY均值: {df_test[\"Predicted_PLQY\"].mean():.4f} ± {df_test[\"Predicted_PLQY\"].std():.4f}')

if 'Predicted_tau' in df_test.columns:
    print(f'  寿命范围: {df_test[\"Predicted_tau\"].min():.2f} - {df_test[\"Predicted_tau\"].max():.2f} μs')
    print(f'  寿命均值: {df_test[\"Predicted_tau\"].mean():.2f} ± {df_test[\"Predicted_tau\"].std():.2f} μs')

# 如果有真实值，计算评估指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

if 'Max_wavelength(nm)' in df_test.columns and 'Predicted_wavelength' in df_test.columns:
    mask = df_test['Max_wavelength(nm)'].notna()
    if mask.sum() > 0:
        y_true = df_test.loc[mask, 'Max_wavelength(nm)']
        y_pred = df_test.loc[mask, 'Predicted_wavelength']
        print(f'\n波长预测评估 (n={len(y_true)}):')
        print(f'  R²: {r2_score(y_true, y_pred):.4f}')
        print(f'  RMSE: {mean_squared_error(y_true, y_pred, squared=False):.2f}')
        print(f'  MAE: {mean_absolute_error(y_true, y_pred):.2f}')
"
```

---

## 步骤8：生成最终报告

```python
python -c "
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')

print('='*120)
print('论文数据处理完成报告')
print('='*120)
print(f'完成时间: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print(f'输出目录: {output_dir}')
print('')

# 模型训练总结
try:
    comparison_df = pd.read_csv(output_dir / 'model_comparison_full.csv')
    best_df = pd.read_csv(output_dir / 'best_models_summary.csv')
    
    print('【1】模型训练总结')
    print('-'*80)
    print(f'  训练模型数: {len(comparison_df.groupby(\"Model\"))}')
    print(f'  评估目标数: {len(comparison_df[\"Target\"].unique())}')
    print('')
    print('  最佳模型:')
    for _, row in best_df.iterrows():
        print(f'    {row[\"Target\"]}: {row[\"Best Model\"]} (R²={row[\"R²\"]})')
except:
    pass

# 虚拟数据库预测总结
try:
    virtual_df = pd.read_csv(output_dir / 'virtual_predict_all.csv')
    print('')
    print('【2】虚拟数据库预测')
    print('-'*80)
    print(f'  总组合数: {len(virtual_df):,}')
    
    if 'Predicted_PLQY' in virtual_df.columns:
        print(f'  PLQY分布:')
        print(f'    >= 0.9: {(virtual_df[\"Predicted_PLQY\"] >= 0.9).sum():,} ({100*(virtual_df[\"Predicted_PLQY\"] >= 0.9).sum()/len(virtual_df):.2f}%)')
        print(f'    >= 0.8: {(virtual_df[\"Predicted_PLQY\"] >= 0.8).sum():,} ({100*(virtual_df[\"Predicted_PLQY\"] >= 0.8).sum()/len(virtual_df):.2f}%)')
        print(f'    >= 0.7: {(virtual_df[\"Predicted_PLQY\"] >= 0.7).sum():,} ({100*(virtual_df[\"Predicted_PLQY\"] >= 0.7).sum()/len(virtual_df):.2f}%)')
except:
    pass

# 测试数据预测总结
try:
    test_df = pd.read_csv(output_dir / 'test_predict_all.csv')
    print('')
    print('【3】测试数据预测')
    print('-'*80)
    print(f'  样本数: {len(test_df)}')
    
    if 'Predicted_wavelength' in test_df.columns:
        print(f'  波长: {test_df[\"Predicted_wavelength\"].mean():.1f} ± {test_df[\"Predicted_wavelength\"].std():.1f} nm')
    if 'Predicted_PLQY' in test_df.columns:
        print(f'  PLQY: {test_df[\"Predicted_PLQY\"].mean():.4f} ± {test_df[\"Predicted_PLQY\"].std():.4f}')
except:
    pass

# 生成文件列表
print('')
print('【4】生成的文件')
print('-'*80)

files_generated = {
    '模型对比表格': [
        'model_comparison_full.csv',
        'best_models_summary.csv',
        'model_comparison_table.tex',
        'model_comparison_table.md'
    ],
    '虚拟数据库预测': [
        'virtual_predict_all.csv',
        'virtual_predict_plqy_0.9+.csv',
        'virtual_predict_plqy_0.8+.csv',
        'virtual_predict_plqy_0.7+.csv',
        'virtual_predict_top20.csv'
    ],
    '测试数据预测': [
        'test_predict_all.csv',
        'test_predict_wavelength.csv',
        'test_predict_plqy.csv'
    ],
    '图表': list((output_dir / 'figures').glob('*.png')) if (output_dir / 'figures').exists() else []
}

for category, files in files_generated.items():
    print(f'  {category}:')
    for file in files:
        if isinstance(file, Path):
            file = file.name
        file_path = output_dir / file
        if file_path.exists():
            print(f'    ✅ {file}')

print('')
print('='*120)
print('完成！所有数据和图表已生成。')
print('='*120)

# 生成JSON报告
report = {
    'timestamp': datetime.now().isoformat(),
    'output_dir': str(output_dir),
    'summary': {
        'models_trained': len(comparison_df.groupby('Model')) if 'comparison_df' in locals() else 0,
        'virtual_database_size': len(virtual_df) if 'virtual_df' in locals() else 0,
        'test_samples': len(test_df) if 'test_df' in locals() else 0
    },
    'files_generated': {k: [str(f) for f in v] for k, v in files_generated.items()}
}

with open(output_dir / 'final_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f'\n综合报告已保存: {output_dir}/final_report.json')
"
```

---

## 一键执行脚本

将上述所有步骤整合到一个脚本中：

```bash
#!/bin/bash
# 一键执行完整工作流程

echo "=========================================="
echo "0913 论文数据处理 - 一键执行"
echo "=========================================="
echo ""

# 设置工作目录
cd /Users/kanshan/IR/ir2025/v3

# 设置输出目录
OUTPUT_DIR="Paper_0913_$(date +%H%M%S)"
echo "输出目录: $OUTPUT_DIR"

# 数据文件
DATA_FILE="data/Database_normalized.csv"
TEST_DATA_FILE="data/Database_ours_0903update_normalized.csv"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ========== 步骤1：数据准备 ==========
echo ""
echo "[步骤1] 数据准备和验证"
# 检查数据文件是否存在
if [ -f "$DATA_FILE" ]; then
    echo "✅ 训练数据存在: $(wc -l $DATA_FILE)"
else
    echo "❌ 训练数据不存在"
    exit 1
fi

if [ -f "$TEST_DATA_FILE" ]; then
    echo "✅ 测试数据存在: $(wc -l $TEST_DATA_FILE)"
else
    echo "⚠️ 测试数据不存在"
fi

if [ -f "data/ir_assemble.csv" ]; then
    echo "✅ 虚拟数据库存在: $(wc -l data/ir_assemble.csv)"
else
    echo "⚠️ 虚拟数据库不存在，需要生成"
    python scripts/generate_virtual_database.py
fi

# ========== 步骤2：训练模型 ==========
echo ""
echo "[步骤2] 训练模型"

# 选择训练模式（修改MODE变量来选择：quick/standard）
MODE="standard"  # 可选: quick, standard

if [ "$MODE" = "quick" ]; then
    echo "使用快速模式（3个模型，5折验证）"
    python automl.py train \
        data=$DATA_FILE \
        test_data=$TEST_DATA_FILE \
        project=$OUTPUT_DIR \
        name=all_models \
        'models=["xgboost","lightgbm","catboost"]' \
        training.n_folds=5 \
        'training.metrics=["r2","rmse","mae"]' \
        training.save_final_model=true \
        training.save_fold_models=false \
        training.save_feature_importance=true \
        feature.feature_type=combined \
        feature.morgan_bits=1024 \
        feature.morgan_radius=2 \
        feature.combination_method=mean \
        'data.smiles_columns=["L1","L2","L3"]' \
        'data.target_columns=["Max_wavelength(nm)","PLQY"]' \
        data.nan_handling=skip
else
    # 默认standard模式
    echo "使用标准模式（13个模型，10折验证）"
    python automl.py train \
        data=$DATA_FILE \
        test_data=$TEST_DATA_FILE \
        project=$OUTPUT_DIR \
        name=all_models \
        models=adaboost,catboost,decision_tree,elastic_net,extra_trees,gradient_boosting,knn,lasso,lightgbm,random_forest,ridge,svr,xgboost \
        training.n_folds=10 \
        'training.metrics=["r2","rmse","mae"]' \
        training.save_final_model=true \
        training.save_fold_models=false \
        training.save_feature_importance=true \
        feature.feature_type=combined \
        feature.morgan_bits=1024 \
        feature.morgan_radius=2 \
        feature.combination_method=mean \
        'data.smiles_columns=["L1","L2","L3"]' \
        'data.target_columns=["Max_wavelength(nm)","PLQY"]' \
        data.nan_handling=skip \
        comparison.enable=true \
        'comparison.formats=["markdown","html","latex","csv"]' \
        export.generate_plots=true \
        export.generate_report=true
fi

# ========== 步骤3：生成对比表格 ==========
echo ""
echo "[步骤3] 生成模型对比表格"
python scripts/generate_model_comparison.py --project $OUTPUT_DIR

# ========== 步骤4：预测虚拟数据库 ==========
echo ""
echo "[步骤4] 预测虚拟数据库"
python scripts/predict_all_combinations.py \
    --project $OUTPUT_DIR \
    --data data/ir_assemble.csv \
    --output $OUTPUT_DIR/virtual_predictions

# ========== 步骤5：生成分段性能分析 ==========
echo ""
echo "[步骤5] 生成分段性能分析图"
# 运行内嵌的Python代码生成分段分析
python scripts/generate_stratified_analysis.py \
    --project $OUTPUT_DIR \
    --output $OUTPUT_DIR/figures

# ========== 步骤6：生成图表 ==========
echo ""
echo "[步骤6] 生成论文图表"
python scripts/generate_paper_figures.py \
    --project $OUTPUT_DIR \
    --output $OUTPUT_DIR/figures

# ========== 步骤7：预测测试数据 ==========
echo ""
echo "[步骤7] 预测测试数据"
python scripts/predict_test_data.py \
    --project $OUTPUT_DIR \
    --data $TEST_DATA_FILE \
    --output $OUTPUT_DIR/test_predictions

# ========== 步骤8：生成最终报告 ==========
echo ""
echo "[步骤8] 生成最终报告"
# 使用内嵌Python代码生成最终报告（参考步骤8的代码）
python -c "
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')

print('='*80)
print('生成最终报告...')
print('='*80)

# 收集信息
report = {
    'timestamp': datetime.now().isoformat(),
    'output_dir': str(output_dir),
    'files_generated': []
}

# 检查生成的文件
if (output_dir / 'model_comparison_full.csv').exists():
    report['files_generated'].append('model_comparison_full.csv')
if (output_dir / 'virtual_predict_all.csv').exists():
    report['files_generated'].append('virtual_predict_all.csv')
if (output_dir / 'test_predict_all.csv').exists():
    report['files_generated'].append('test_predict_all.csv')

# 保存报告
with open(output_dir / 'final_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f'✅ 最终报告已生成: {output_dir}/final_report.json')
"

# ========== 完成 ==========
echo ""
echo "=========================================="
echo "✅ 所有任务完成！"
echo "=========================================="
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "重要文件:"
echo "  1. 模型对比: $OUTPUT_DIR/model_comparison_full.csv"
echo "  2. 最佳模型: $OUTPUT_DIR/best_models_summary.csv"
echo "  3. 虚拟预测: $OUTPUT_DIR/virtual_predict_all.csv"
echo "  4. 高PLQY候选: $OUTPUT_DIR/virtual_predict_plqy_0.9+.csv"
echo "  5. 测试预测: $OUTPUT_DIR/test_predict_all.csv"
echo "  6. 分段分析: $OUTPUT_DIR/figures/stratified_analysis/"
echo "  7. 论文图表: $OUTPUT_DIR/figures/"
echo "  8. 最终报告: $OUTPUT_DIR/final_report.json"
echo "  9. 时间报告: $OUTPUT_DIR/timing/"
```

---

## 使用说明

1. **完整执行**：运行一键脚本或按步骤执行每个部分的代码
2. **部分执行**：可以单独运行任何步骤
3. **自定义参数**：修改脚本中的参数以适应不同需求
4. **结果检查**：每步都有验证和统计输出

## 主要更新

### 2024-09-13 更新
- 澄清了训练流程：10折CV用于评估，最终模型使用100%数据训练
- 强调了预测使用的是最终训练模型（基于完整训练集）
- 添加了批处理说明以处理大规模预测任务（272,104个分子）

### 之前的更新
- 移除所有优化相关参数（optimization.enable等）
- 模式C改为使用预定义配置文件
- 直接使用 `models` 参数指定要训练的模型列表
- 所有模型都使用默认参数训练

## 输出文件说明

### 核心输出
- `model_comparison_full.csv` - 所有模型的完整性能对比
- `best_models_summary.csv` - 每个目标的最佳模型
- `virtual_predict_all.csv` - 272,104个虚拟分子的完整预测
- `test_predict_all.csv` - 测试数据的预测结果

### 论文表格
- `model_comparison_table.tex` - LaTeX格式的模型对比表
- `model_comparison_table.md` - Markdown格式的模型对比表

### 高性能候选
- `virtual_predict_plqy_0.9+.csv` - PLQY ≥ 0.9的分子
- `virtual_predict_plqy_0.8+.csv` - PLQY ≥ 0.8的分子
- `virtual_predict_top20.csv` - Top 20高PLQY分子

### 可视化图表
- `figures/model_comparison_r2.png` - R²对比图
- `figures/model_comparison_rmse.png` - RMSE对比图
- `figures/plqy_distribution.png` - PLQY分布图
- `figures/stratified_analysis/` - 分段性能分析图表

### 时间统计报告
- `timing/timing_*.json` - 详细时间统计（JSON格式）
- `timing/timing_*.txt` - 时间统计摘要（文本格式）
- `batch_predictions_*/timing_report.json` - 批量预测时间报告

### 其他可视化
- `figures/wavelength_distribution.png` - 波长分布图
- `figures/wavelength_vs_plqy.png` - 波长vs PLQY散点图

## 时间性能统计

系统会自动记录和分析各个阶段的执行时间：

### 训练阶段时间统计
- **数据加载时间**: CSV文件读取和解析
- **特征提取时间**: 分子指纹和描述符计算
  - 每个样本的平均提取时间
  - 特征提取吞吐量（样本/秒）
- **交叉验证时间**: 
  - 每折的训练时间
  - 每折的预测时间
  - 平均折时间
- **最终模型训练时间**: 使用全部数据训练
- **测试集评估时间**: 测试数据预测和评估

### 预测阶段时间统计
- **批量预测时间**: 每个模型的预测耗时
- **预测吞吐量**: 样本/秒
- **文件I/O时间**: 结果保存时间

### 时间报告内容
```json
{
  "summary": {
    "total_time": 156.78,
    "records": {
      "data_loading": {
        "duration": 0.45,
        "percentage": 0.29,
        "metrics": {
          "samples": 1516,
          "throughput": 3368.89
        }
      },
      "feature_extraction": {
        "duration": 23.67,
        "percentage": 15.09,
        "metrics": {
          "samples": 1516,
          "throughput": 64.04,
          "per_sample_time": 15.61
        }
      }
    }
  },
  "performance_metrics": {
    "training": {
      "total_time": 120.34,
      "avg_fold_time": 12.03
    },
    "prediction": {
      "throughput": "256.78 samples/sec"
    }
  }
}
```

---

**完成时间**: 2024-09-13
**作者**: AutoML System
**版本**: 2.0 (移除优化功能版)