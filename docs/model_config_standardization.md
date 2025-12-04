# 模型配置标准化说明

## 概述
所有模型配置已按照XGBoost的规格进行标准化，每个模型都有完整的配置级别。
**重要更新：所有配置级别统一使用10折交叉验证，以确保结果的稳定性和可比性。**

## 标准配置级别

每个模型都包含以下4个标准配置级别：

| 级别 | 文件名后缀 | 训练时间 | 用途 | K折 | Morgan位数 |
|------|------------|----------|------|-----|------------|
| **debug** | `_debug.yaml` | <1分钟 | 调试测试 | 10 | 512 |
| **quick** | `_quick.yaml` | ~5分钟 | 快速验证 | 10 | 1024 |
| **standard** | `_standard.yaml` | ~15分钟 | 标准训练 | 10 | 1024 |
| **full** | `_full.yaml` | ~30分钟 | 完整训练 | 10 | 2048 |

## 模型配置统计

| 模型 | 配置数量 | 特殊配置 |
|------|----------|----------|
| XGBoost | 8 | example, optuna, fast, large |
| LightGBM | 5 | optuna |
| CatBoost | 5 | optuna |
| Random Forest | 4 | 标准4个 |
| Gradient Boosting | 4 | 标准4个 |
| AdaBoost | 4 | 标准4个 |
| Extra Trees | 4 | 标准4个 |
| Decision Tree | 4 | 标准4个 |
| SVR | 4 | 标准4个 |
| KNN | 4 | 标准4个 |
| Ridge | 4 | 标准4个 |
| Lasso | 4 | 标准4个 |
| ElasticNet | 4 | 标准4个 |
| **总计** | **61** | - |

## 特殊配置

除了标准配置外，还有以下特殊配置：

### AutoML配置 (2个)
- `automl_quick.yaml` - 快速测试主要模型
- `automl_full.yaml` - 测试所有模型

### Special配置 (2个)
- `debug.yaml` - 最小调试配置
- `quick_optimize.yaml` - 快速优化配置

### 优化配置 (3个)
- `xgboost_optuna.yaml`
- `lightgbm_optuna.yaml`
- `catboost_optuna.yaml`

## 使用示例

### 1. 快速测试
```bash
# 使用debug配置快速验证
python automl.py train config=random_forest_debug data=mydata.csv
```

### 2. 标准训练
```bash
# 使用standard配置进行正式训练
python automl.py train config=xgboost_standard data=mydata.csv
```

### 3. 比较不同模型
```bash
# 测试多个模型的standard配置
python automl.py train config=xgboost_standard data=data.csv project=compare name=xgb
python automl.py train config=lightgbm_standard data=data.csv project=compare name=lgb
python automl.py train config=catboost_standard data=data.csv project=compare name=cat
```

### 4. 从快到慢逐步验证
```bash
# 步骤1：debug测试（1分钟）
python automl.py train config=xgboost_debug data=data.csv

# 步骤2：quick验证（5分钟）
python automl.py train config=xgboost_quick data=data.csv

# 步骤3：standard训练（15分钟）
python automl.py train config=xgboost_standard data=data.csv

# 步骤4：full完整训练（30分钟）
python automl.py train config=xgboost_full data=data.csv
```

## 配置参数标准化

### 树模型通用参数
- `n_estimators`: 树的数量（debug:10, quick:100, standard:300, full:500）
- `max_depth`: 最大深度（根据模型调整）
- `random_state`: 42（所有模型统一）
- `n_jobs`: -1（使用所有CPU核心）

### 线性模型通用参数
- `alpha`: 正则化强度
- `max_iter`: 最大迭代次数（debug:100, quick:500, standard:1000, full:2000）
- `random_state`: 42（所有模型统一）

### 特征配置统一
- `feature_type`: combined（Morgan指纹 + 分子描述符）
- `morgan_radius`: 2（所有配置统一）
- `use_cache`: true（启用缓存）
- `combination_method`: mean（多分子平均）

## 优势

1. **标准化**：所有模型遵循相同的配置级别
2. **可预测**：用户知道每个级别的预期训练时间
3. **渐进式**：从debug到full逐步增加复杂度
4. **一致性**：命名规范统一，易于理解
5. **完整性**：每个模型都有完整的配置集

## 配置文件总数

- 标准模型配置：52个（13个模型 × 4个级别）
- XGBoost额外配置：4个
- LightGBM/CatBoost优化配置：2个
- AutoML配置：2个
- Special配置：2个
- **总计：62个核心配置文件**

## 快速选择指南

| 需求 | 推荐配置 |
|------|----------|
| 初次尝试 | `*_debug` |
| 快速验证 | `*_quick` |
| 正式实验 | `*_standard` |
| 论文发表 | `*_full` |
| 超参数优化 | `*_optuna` |
| 多模型比较 | `automl_quick` |