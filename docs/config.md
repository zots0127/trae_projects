# 配置系统完整指南

## 目录
1. [快速开始](#快速开始)
2. [配置优先级](#配置优先级)
3. [预设配置模板](#预设配置模板)
4. [参数详解](#参数详解)
5. [使用示例](#使用示例)
6. [最佳实践](#最佳实践)

---

## 快速开始

### 基本命令格式
```bash
python automl.py train [config=模板] [参数=值] ...
```

### 最简单的训练命令
```bash
# 使用默认配置
python automl.py train data=mydata.csv

# 使用预设模板
python automl.py train config=xgboost_quick data=mydata.csv

# 自定义参数
python automl.py train config=xgboost_standard \
    data=mydata.csv \
    n_folds=5 \
    project=MyProject \
    name=experiment_001
```

---

## 配置优先级

配置参数的优先级从高到低：

1. **命令行参数** - 最高优先级，覆盖所有其他设置
2. **配置文件** - 通过`config=path/to/config.yaml`指定
3. **预设模板** - 通过`config=template_name`指定
4. **默认值** - 系统内置默认值

### 示例
```bash
# 模板设置n_folds=10，但命令行参数覆盖为5
python automl.py train config=xgboost_standard n_folds=5 data=mydata.csv
# 实际使用: n_folds=5
```

---

## 预设配置模板

### 可用模板列表

| 模板名称 | 训练时间 | 适用场景 | 主要参数 |
|---------|---------|---------|---------|
| **debug** | <1分钟 | 调试和测试 | n_folds=2, n_estimators=10, max_depth=3 |
| **xgboost_quick** | ~5分钟 | 快速验证 | n_folds=5, n_estimators=100, max_depth=5 |
| **xgboost_fast** | ~10分钟 | 快速训练 | n_folds=5, n_estimators=200, max_depth=6 |
| **xgboost_standard** | ~15分钟 | 标准训练 | n_folds=10, n_estimators=300, max_depth=7 |
| **xgboost_full** | ~30分钟 | 完整训练 | n_folds=10, n_estimators=500, max_depth=8 |
| **xgboost_optuna** | ~60分钟 | 超参数优化 | n_trials=50, optimization=true |
| **lightgbm_standard** | ~15分钟 | LightGBM标准 | num_leaves=31, n_estimators=300 |
| **catboost_standard** | ~15分钟 | CatBoost标准 | iterations=300, depth=6 |
| **automl** | ~2小时 | 自动选择最佳模型 | 测试所有模型 |

### 查看模板内容
```bash
# 列出所有可用模板
python automl.py config list

# 查看特定模板详情
python automl.py config show xgboost_standard
```

---

## 参数详解

### 1. 数据参数 (Data)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **data** | str | 必需 | 训练数据CSV文件路径 |
| **test_data** | str | None | 测试数据CSV文件路径（可选） |
| **target_columns** | list | ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)'] | 目标列名 |
| **smiles_columns** | list | ['L1', 'L2', 'L3'] | SMILES列名 |

#### 示例
```bash
python automl.py train \
    data=../data/train.csv \
    test_data=../data/test.csv \
    target_columns="['wavelength','plqy']"
```

### 2. 多目标策略 (Multi-target Strategy)

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| **multi_target** | intersection, independent, union | independent | 多目标数据选择策略 |

- **intersection**: 只使用所有目标都有值的数据（~1354行）
- **independent**: 每个目标独立使用其有效数据（默认）
- **union**: 使用所有数据，配合缺失值填充

#### 示例
```bash
# 严格模式：所有目标使用相同数据
python automl.py train config=xgboost_standard \
    data=data.csv \
    multi_target=intersection

# 独立模式：最大化每个目标的数据利用
python automl.py train config=xgboost_standard \
    data=data.csv \
    multi_target=independent

# 并集模式：使用所有数据
python automl.py train config=xgboost_standard \
    data=data.csv \
    multi_target=union \
    nan_handling=mean
```

### 3. 缺失值处理 (NaN Handling)

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| **nan_handling** | skip, mean, median, zero, forward, interpolate | skip | 缺失值处理策略 |

- **skip**: 跳过含缺失值的行
- **mean**: 均值填充
- **median**: 中位数填充
- **zero**: 零值填充
- **forward**: 前向填充
- **interpolate**: 插值填充

#### 示例
```bash
# 跳过缺失值
python automl.py train data=data.csv nan_handling=skip

# 均值填充
python automl.py train data=data.csv nan_handling=mean

# 中位数填充（适合有异常值的数据）
python automl.py train data=data.csv nan_handling=median
```

### 4. 模型参数 (Model)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **model** | str | xgboost | 模型类型 |
| **n_folds** | int | 10 | 交叉验证折数 |
| **random_seed** | int | 42 | 随机种子 |

#### 可用模型
- **树模型**: xgboost, lightgbm, catboost, random_forest, gradient_boosting, ada_boost, extra_trees, decision_tree
- **线性模型**: ridge, lasso, elastic_net
- **其他**: svr, knn

#### 示例
```bash
# 使用LightGBM
python automl.py train model=lightgbm data=data.csv

# 使用5折交叉验证
python automl.py train n_folds=5 data=data.csv

# 设置随机种子
python automl.py train random_seed=123 data=data.csv
```

### 5. 特征参数 (Features)

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| **feature** | morgan, descriptors, combined | combined | 特征类型 |
| **morgan_bits** | int | 1024 | Morgan指纹位数 |
| **morgan_radius** | int | 2 | Morgan指纹半径 |
| **combination** | mean, sum, concat | mean | 多分子组合方法 |
| **use_cache** | bool | true | 是否使用特征缓存 |

#### 示例
```bash
# 只使用Morgan指纹
python automl.py train feature=morgan data=data.csv

# 只使用分子描述符
python automl.py train feature=descriptors data=data.csv

# 使用组合特征（默认）
python automl.py train feature=combined data=data.csv

# 自定义Morgan指纹参数
python automl.py train \
    feature=morgan \
    morgan_bits=2048 \
    morgan_radius=3 \
    data=data.csv
```

### 6. 超参数 (Hyperparameters)

通过`model.hyperparameters.参数名=值`格式设置：

#### XGBoost参数
```bash
python automl.py train \
    model=xgboost \
    model.hyperparameters.n_estimators=500 \
    model.hyperparameters.max_depth=8 \
    model.hyperparameters.learning_rate=0.05 \
    model.hyperparameters.subsample=0.8 \
    data=data.csv
```

#### LightGBM参数
```bash
python automl.py train \
    model=lightgbm \
    model.hyperparameters.num_leaves=31 \
    model.hyperparameters.n_estimators=300 \
    model.hyperparameters.learning_rate=0.05 \
    data=data.csv
```

### 7. 输出参数 (Output)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **project** | str | runs | 项目目录名 |
| **name** | str | 自动生成 | 实验名称 |
| **output_dir** | str | runs | 输出基础目录 |

#### 示例
```bash
# 指定项目和实验名称
python automl.py train \
    project=MyResearch \
    name=exp_001 \
    data=data.csv

# 结果保存在: MyResearch/exp_001/
```

### 8. 优化参数 (Optimization)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **optimization** | bool | false | 是否启用超参数优化 |
| **n_trials** | int | 50 | Optuna试验次数 |
| **timeout** | int | 3600 | 优化超时（秒） |

#### 示例
```bash
# 启用超参数优化
python automl.py train \
    optimization=true \
    n_trials=100 \
    data=data.csv

# 使用预设优化模板
python automl.py train config=xgboost_optuna data=data.csv
```

---

## 使用示例

### 示例1：快速测试
```bash
python automl.py train \
    config=debug \
    data=../data/Database_normalized.csv \
    project=test \
    name=quick_test
```

### 示例2：标准训练
```bash
python automl.py train \
    config=xgboost_standard \
    data=../data/Database_normalized.csv \
    test_data=../data/test.csv \
    project=Nature \
    name=standard_exp
```

### 示例3：严格数据模式
```bash
python automl.py train \
    config=xgboost_standard \
    data=../data/Database_normalized.csv \
    multi_target=intersection \
    nan_handling=skip \
    project=Nature \
    name=strict_data
```

### 示例4：最大数据利用
```bash
python automl.py train \
    config=xgboost_standard \
    data=../data/Database_normalized.csv \
    multi_target=union \
    nan_handling=mean \
    project=Nature \
    name=max_data
```

### 示例5：超参数优化
```bash
python automl.py train \
    config=xgboost_optuna \
    data=../data/Database_normalized.csv \
    n_trials=100 \
    project=Nature \
    name=optimized
```

### 示例6：多模型比较
```bash
# 测试XGBoost
python automl.py train \
    model=xgboost \
    data=data.csv \
    project=comparison \
    name=xgb_test

# 测试LightGBM
python automl.py train \
    model=lightgbm \
    data=data.csv \
    project=comparison \
    name=lgbm_test

# 测试CatBoost
python automl.py train \
    model=catboost \
    data=data.csv \
    project=comparison \
    name=cat_test
```

### 示例7：完整实验流程
```bash
# 步骤1：快速验证数据
python automl.py train \
    config=debug \
    data=data.csv \
    project=MyProject \
    name=validation

# 步骤2：标准训练
python automl.py train \
    config=xgboost_standard \
    data=data.csv \
    test_data=test.csv \
    multi_target=intersection \
    project=MyProject \
    name=standard

# 步骤3：超参数优化
python automl.py train \
    config=xgboost_optuna \
    data=data.csv \
    test_data=test.csv \
    multi_target=intersection \
    n_trials=100 \
    project=MyProject \
    name=optimized

# 步骤4：分析结果
python automl.py analyze \
    dir=MyProject/optimized \
    format=html
```

---

## 最佳实践

### 1. 数据准备
- 确保CSV文件格式正确
- SMILES列命名为L1, L2, L3
- 目标列使用标准命名
- 检查缺失值分布

### 2. 实验策略
```bash
# 推荐的实验流程
1. debug模式验证数据 (1分钟)
2. quick模式快速测试 (5分钟)
3. standard模式正式训练 (15分钟)
4. optuna模式优化超参数 (60分钟)
```

### 3. 数据策略选择
- **高质量数据**：使用`multi_target=intersection`
- **数据量充足**：使用`multi_target=independent`（默认）
- **数据稀缺**：使用`multi_target=union` + `nan_handling=mean`

### 4. 模型选择
- **快速基线**：XGBoost（默认）
- **大数据集**：LightGBM（更快）
- **类别特征多**：CatBoost
- **需要解释性**：RandomForest

### 5. 交叉验证
- **探索阶段**：n_folds=5（快速）
- **正式实验**：n_folds=10（标准）
- **论文发表**：n_folds=10 + 重复实验

### 6. 结果管理
```bash
# 组织结构
ProjectName/
├── validation/     # 数据验证
├── baseline/       # 基线模型
├── optimized/      # 优化模型
└── final/          # 最终模型
```

---

## 配置文件格式

### YAML格式
```yaml
# config.yaml
data:
  data_path: ../data/Database_normalized.csv
  test_data_path: ../data/test.csv
  target_columns:
    - Max_wavelength(nm)
    - PLQY
    - tau(s*10^-6)
  multi_target_strategy: intersection
  nan_handling: skip

model:
  model_type: xgboost
  hyperparameters:
    n_estimators: 300
    max_depth: 7
    learning_rate: 0.07

training:
  n_folds: 10
  save_final_model: true

feature:
  feature_type: combined
  morgan_bits: 1024
  morgan_radius: 2
```

### 使用配置文件
```bash
python automl.py train config=path/to/config.yaml
```

---

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少并行度
   python automl.py train n_jobs=1 data=data.csv
   ```

2. **训练太慢**
   ```bash
   # 使用快速模板
   python automl.py train config=xgboost_quick data=data.csv
   ```

3. **缺失值错误**
   ```bash
   # 检查数据策略
   python automl.py train multi_target=intersection data=data.csv
   ```

4. **特征提取失败**
   ```bash
   # 清除缓存
   rm -rf feature_cache/
   python automl.py train use_cache=false data=data.csv
   ```

---

## 附录：完整参数列表

```bash
# 数据相关
data=                    # 训练数据路径
test_data=              # 测试数据路径
target_columns=         # 目标列
smiles_columns=         # SMILES列
multi_target=           # 多目标策略
nan_handling=           # 缺失值处理

# 模型相关
model=                  # 模型类型
n_folds=               # 交叉验证折数
random_seed=           # 随机种子
model.hyperparameters.*= # 模型超参数

# 特征相关
feature=               # 特征类型
morgan_bits=           # Morgan位数
morgan_radius=         # Morgan半径
combination=           # 组合方法
use_cache=            # 使用缓存

# 输出相关
project=              # 项目名
name=                 # 实验名
output_dir=           # 输出目录

# 优化相关
optimization=         # 启用优化
n_trials=            # 试验次数
timeout=             # 超时时间

# 其他
verbose=             # 详细输出
save_plots=          # 保存图表
save_importance=     # 保存特征重要性
```

---

## 更新日志

- **v3.0**: 添加多目标策略和缺失值处理
- **v2.5**: 添加Optuna超参数优化
- **v2.0**: 添加预设配置模板
- **v1.0**: 基础训练功能

---

## 获取帮助

```bash
# 查看帮助
python automl.py --help

# 查看版本
python automl.py --version

# 列出可用模型
python automl.py models list

# 验证配置
python automl.py validate config=myconfig.yaml
```