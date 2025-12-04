# 论文级模型对比功能使用指南

## 概述

本功能允许您一键训练所有支持的机器学习模型，并自动生成论文级别的性能对比表格，类似于学术论文中的 Table 1。

## 快速开始

### 最简单的用法

```bash
python automl.py train config=paper_comparison data=your_data.csv
```

这个命令会：
1. 训练所有13个支持的模型
2. 自动生成多格式的对比表格
3. 标记每个目标的最佳模型

## 功能特点

### 支持的模型（13个）

- **树模型**: XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting, Extra Trees, Decision Tree
- **线性模型**: Ridge, Lasso, Elastic Net
- **其他**: SVR, KNN, AdaBoost

### 生成的表格格式

- **Markdown** (.md) - 用于文档和GitHub
- **HTML** (.html) - 用于网页展示
- **LaTeX** (.tex) - 用于学术论文
- **CSV** (.csv) - 用于进一步分析

### 表格内容

每个表格包含：
- 模型名称 (Algorithms)
- R² ± 标准差
- RMSE ± 标准差
- MAE ± 标准差
- 最佳模型高亮标记 (⭐)

## 使用示例

### 1. 基础用法（使用默认列名）

```bash
python automl.py train config=paper_comparison data=Database_normalized.csv
```

### 2. 自定义输入和目标列

```bash
python automl.py train config=paper_comparison \
    data=my_data.csv \
    smiles_columns=ligand1,ligand2,ligand3 \
    targets=wavelength,quantum_yield
```

### 3. 指定项目名称

```bash
python automl.py train config=paper_comparison \
    data=data.csv \
    project=NaturePaper \
    name=final_results
```

### 4. 快速测试（只训练3个主要模型）

```bash
python automl.py train config=paper_comparison \
    data=data.csv \
    optimization.automl_models=[xgboost,catboost,lightgbm] \
    training.n_folds=5
```

### 5. 包含测试集评估

```bash
python automl.py train config=paper_comparison \
    data=train_data.csv \
    test_data=test_data.csv \
    project=Paper
```

## 配置选项

### 通过命令行覆盖配置

您可以通过命令行参数覆盖配置文件中的任何设置：

```bash
# 修改交叉验证折数
training.n_folds=5

# 修改特征类型
feature.feature_type=morgan

# 修改数值精度
comparison.decimal_places.r2=3
comparison.decimal_places.rmse=2

# 选择特定模型
optimization.automl_models=[xgboost,lightgbm,catboost]

# 修改表格输出格式
comparison.formats=[latex,csv]
```

### 完整命令示例

```bash
python automl.py train config=paper_comparison \
    data=../data/Database_normalized.csv \
    test_data=Database_ours_0903update_normalized.csv \
    project=NaturePaper \
    name=submission_v1 \
    smiles_columns=L1,L2,L3 \
    targets=Max_wavelength(nm),PLQY \
    training.n_folds=10 \
    feature.feature_type=combined \
    feature.morgan_bits=1024 \
    comparison.formats=[markdown,latex] \
    comparison.highlight_best=true
```

## 输出结构

训练完成后，会生成以下文件结构：

```
NaturePaper/
├── submission_v1/
│   ├── xgboost/
│   │   ├── models/
│   │   └── results/
│   ├── catboost/
│   │   ├── models/
│   │   └── results/
│   ├── lightgbm/
│   │   ├── models/
│   │   └── results/
│   ├── ... (其他模型)
│   ├── comparison_table_20250912_150000.md
│   ├── comparison_table_20250912_150000.html
│   ├── comparison_table_20250912_150000.tex
│   └── comparison_table_20250912_150000.csv
```

## 生成的表格示例

### Markdown格式

```markdown
## Target: λem

| Algorithm | R² | RMSE | MAE |
|-----------|-----|------|-----|
| **CatBoost Regressor** ⭐ | **0.8643±0.0258** | **26.9270±2.9826** | **19.1066±1.6162** |
| XGBoost Regressor | 0.8554±0.0276 | 27.7885±2.9424 | 18.9790±1.7357 |
| LightGBM Regressor | 0.8385±0.0401 | 29.3269±4.0560 | 19.9741±2.1457 |
...
```

### LaTeX格式（用于论文）

```latex
\begin{table}[htbp]
\centering
\caption{Prediction performance with different models}
\begin{tabular}{llrrr}
\toprule
Objects & Algorithms & R\textsuperscript{2} & RMSE & MAE \\
\midrule
λem & \textbf{CatBoost Regressor} & \textbf{0.8643$\pm$0.0258} & \textbf{26.927$\pm$2.983} & \textbf{19.107$\pm$1.616} \\
 & XGBoost Regressor & 0.8554$\pm$0.0276 & 27.789$\pm$2.942 & 18.979$\pm$1.736 \\
 & LightGBM Regressor & 0.8385$\pm$0.0401 & 29.327$\pm$4.056 & 19.974$\pm$2.146 \\
\bottomrule
\end{tabular}
\end{table}
```

## Python API 使用

您也可以在Python脚本中使用：

```python
from utils.comparison_table import ComparisonTableGenerator

# 创建生成器
generator = ComparisonTableGenerator('path/to/results')

# 收集所有结果
df = generator.collect_all_results()

# 生成特定格式
markdown = generator.generate_markdown_table(df)
latex = generator.generate_latex_table(df)

# 导出所有格式
exported_files = generator.export_all_formats(
    formats=['markdown', 'html', 'latex', 'csv']
)

# 获取最佳模型
best_models = generator.get_best_models(metric='r2')
```

## 测试脚本

使用提供的测试脚本快速验证功能：

```bash
# 完整测试
./scripts/test_paper_comparison.sh

# Python演示
python demo_paper_comparison.py --quick
```

## 常见问题

### Q: 如何只训练特定的模型？
A: 使用 `optimization.automl_models` 参数：
```bash
optimization.automl_models=[xgboost,catboost,lightgbm]
```

### Q: 如何修改表格的数值精度？
A: 使用 `comparison.decimal_places` 参数：
```bash
comparison.decimal_places.r2=3 comparison.decimal_places.rmse=2
```

### Q: 如何关闭最佳模型高亮？
A: 设置 `comparison.highlight_best=false`

### Q: 训练时间太长怎么办？
A: 
- 减少模型数量
- 减少交叉验证折数 (`training.n_folds=5`)
- 使用快速模式

## 性能建议

1. **快速测试**: 先用少量模型和5折验证测试
2. **正式训练**: 使用全部模型和10折验证
3. **大数据集**: 考虑使用 `data.sample_size` 参数先测试
4. **特征缓存**: 保持 `feature.use_cache=true` 加速重复训练

## 总结

这个功能让您能够：
- ✅ 一键训练所有模型
- ✅ 自动生成论文级表格
- ✅ 灵活配置输入输出
- ✅ 支持多种导出格式
- ✅ 自动选择最佳模型

非常适合用于：
- 学术论文撰写
- 模型性能比较
- 项目报告生成
- 快速原型验证