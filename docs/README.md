AutoML 完整命令参考指南
================================

本文档完整记录AutoML系统的所有命令、参数和功能，按照从基础到高级的顺序组织。

目录
--------------------------------
1. [快速入门（基础功能）](#快速入门基础功能)
2. [核心命令详解（中级功能）](#核心命令详解中级功能)
3. [配置模板系统](#配置模板系统)
4. [高级功能](#高级功能)
5. [完整参数参考](#完整参数参考)
6. [实战示例](#实战示例)

================================
# 第一部分：快速入门（基础功能）
================================

## 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 如需分子特征，安装RDKit（推荐conda）
conda install -c conda-forge rdkit
```

## 最简单的命令

### 1. 训练模型（最基础）
```bash
# 使用默认配置训练XGBoost
python automl.py train data=data/Database_normalized.csv

# 指定模型类型
python automl.py train model=lightgbm data=data/Database_normalized.csv

# 使用预定义模板（快速训练）
python automl.py train config=xgboost_quick
```

### 2. 预测（最基础）
```bash
# 使用训练好的模型预测
python automl.py predict model=models/best.joblib data=test.csv

# 预测并保存结果
python automl.py predict model=models/best.joblib data=test.csv output=predictions.csv
```

### 3. 分析结果（最基础）
```bash
# 分析最近一次训练
python automl.py analyze dir=last format=html

# 分析指定目录
python automl.py analyze dir=runs/train/myproject
```

## 基础参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| data | 训练数据路径 | data/train.csv |
| model | 模型类型或路径 | xgboost, models/best.joblib |
| config | 配置模板名称 | xgboost_quick |
| output | 输出文件路径 | predictions.csv |
| dir | 目录路径 | runs/train/project1 |
| format | 输出格式 | html, text |

================================
# 第二部分：核心命令详解（中级功能）
================================

## 1. train - 训练命令

### 基本用法
```bash
python automl.py train [参数]
```

### 常用参数组合
```bash
# 指定项目名和实验名
python automl.py train config=xgboost_quick project=myproj name=exp1

# 指定目标列
python automl.py train data=data.csv target=PLQY

# 设置交叉验证折数
python automl.py train config=xgboost_quick n_folds=10

# 带测试集评估
python automl.py train config=xgboost_quick test_data=test.csv
```

### train命令参数表

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| data | 训练数据路径 | 必需 | data/train.csv |
| model | 模型类型 | xgboost | lightgbm, catboost |
| config | 配置模板 | - | xgboost_quick |
| project | 项目名称 | default | my_experiment |
| name | 实验名称 | 自动生成 | exp_001 |
| target | 目标列 | 自动检测 | PLQY, Max_wavelength(nm) |
| n_folds | 交叉验证折数 | 10 | 5, 10 |
| test_data | 测试集路径 | - | test.csv |
| feature | 特征类型 | combined | morgan, descriptors |

## 2. predict - 预测命令

### 基本用法
```bash
python automl.py predict model=<模型路径> data=<数据> [参数]
```

### 预测模式

#### A. CSV文件预测
```bash
# 分子数据预测（自动识别SMILES列）
python automl.py predict model=models/best.joblib data=test.csv feature=combined

# 表格数据预测
python automl.py predict model=models/tabular.joblib data=test.csv feature=tabular
```

#### B. 直接输入预测
```bash
# 单个SMILES预测
python automl.py predict model=models/best.joblib input='["CCO","c1ccccc1"]' feature=morgan

# 多配体预测（L1,L2,L3）
python automl.py predict model=models/best.joblib \
    input='[["CCO","c1ccccc1",null],["O","N",null]]' feature=combined

# 数值数组预测
python automl.py predict model=models/tabular.joblib \
    input='[[0.1,0.2,0.3],[0.5,0.6,0.7]]' feature=tabular
```

### predict命令参数表

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| model | 模型文件路径 | 必需 | models/best.joblib |
| data | 数据文件路径 | - | test.csv |
| input | 直接输入数据 | - | '["CCO"]' |
| feature | 特征类型 | auto | morgan, descriptors, combined, tabular |
| smiles_columns | SMILES列名 | L1,L2,L3 | mol1,mol2,mol3 |
| morgan_bits | 指纹位数 | 1024 | 512, 2048 |
| morgan_radius | 指纹半径 | 2 | 2, 3 |
| output | 输出文件 | predictions.csv | results.csv |

## 3. analyze - 分析命令

### 基本用法
```bash
python automl.py analyze dir=<目录> [参数]
```

### 分析选项
```bash
# 分析最近训练
python automl.py analyze dir=last format=html

# 文本格式输出（终端查看）
python automl.py analyze dir=last format=text

# 比较多个实验
python automl.py analyze dir=runs/train1,runs/train2

# 生成论文图表
python automl.py analyze dir=last export_for_paper=true
```

### analyze命令参数表

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| dir | 实验目录 | last | runs/train/exp1 |
| format | 输出格式 | html | text, json |
| export_for_paper | 导出论文图表 | false | true |
| compare | 比较多个实验 | false | true |

## 4. validate - 验证命令

### 基本用法
```bash
# 验证数据文件（推荐）
python automl.py validate data=data/train.csv

# 验证配置文件
python automl.py validate config=configs/myconfig.yaml
```

### 数据验证功能
验证数据文件时会检查：
- ✅ 文件是否存在和可读
- ✅ 数据行数和列数
- ✅ SMILES列（L1, L2, L3）
- ✅ 目标列（Max_wavelength, PLQY, tau）
- ✅ 数据质量（缺失值、重复行）
- ✅ SMILES格式有效性（如果RDKit可用）

### 示例输出
```
📊 验证数据文件: data/Database_normalized.csv
✅ 数据加载成功
数据信息:
行数: 1667
列数: 13
✅ SMILES列: L1, L2, L3
✅ 目标列: Max_wavelength(nm), PLQY, tau(s*10^-6)
缺失值总数: 2960
重复行数: 0
✅ SMILES格式检查通过
✅ 数据验证完成!
```

## 5. export - 导出命令

### 基本用法
```bash
# 导出为ONNX格式
python automl.py export model=models/best.joblib format=onnx output=exports/model

# 导出为Pickle格式
python automl.py export model=models/best.joblib format=pickle output=exports/model
```

## 6. warmup - 缓存预热命令

### 基本用法
```bash
# 预计算特征缓存
python automl.py warmup data=data/train.csv feature=combined

# 清理缓存
python automl.py warmup clean=true
```

## 7. info - 信息命令

### 基本用法
```bash
# 显示系统信息
python automl.py info

# 显示可用模型
python automl.py info models

# 显示可用模板
python automl.py info templates
```

================================
# 第三部分：配置模板系统
================================

## 预定义模板列表

系统提供20+预定义配置模板，覆盖从调试到生产的各种场景：

### 快速训练模板

| 模板名 | 说明 | 训练时间 | 适用场景 |
|--------|------|----------|----------|
| debug | 最小化调试模板 | <1分钟 | 代码测试 |
| xgboost_quick | XGBoost快速训练 | ~5分钟 | 快速验证 |
| lightgbm_quick | LightGBM快速训练 | ~5分钟 | 快速验证 |
| catboost_quick | CatBoost快速训练 | ~5分钟 | 快速验证 |

### 标准训练模板

| 模板名 | 说明 | 训练时间 | 适用场景 |
|--------|------|----------|----------|
| xgboost_full | XGBoost完整训练 | ~30分钟 | 生产环境 |
| lightgbm_full | LightGBM完整训练 | ~30分钟 | 生产环境 |
| lightgbm | LightGBM基础配置 | ~15分钟 | 标准训练 |

### 优化模板

| 模板名 | 说明 | 训练时间 | 适用场景 |
|--------|------|----------|----------|
| xgboost_optuna | XGBoost+Optuna优化 | 1-2小时 | 超参数搜索 |
| quick_optimize | 快速优化（20次试验） | ~30分钟 | 快速调优 |
| automl | 多模型自动选择 | 2-4小时 | 自动化ML |

### 经典算法模板

| 模板名 | 说明 | 模型类型 |
|--------|------|----------|
| random_forest | 随机森林回归 | 集成学习 |
| gradient_boosting | 梯度提升回归 | 集成学习 |
| adaboost | AdaBoost回归 | 集成学习 |
| extra_trees | Extra Trees回归 | 集成学习 |
| svr_rbf | 支持向量回归（RBF核） | SVM |
| knn | K近邻回归 | 基于实例 |
| decision_tree | 决策树回归 | 树模型 |
| ridge | Ridge回归（L2正则） | 线性模型 |
| lasso | Lasso回归（L1正则） | 线性模型 |
| elastic_net | ElasticNet回归 | 线性模型 |

## 使用模板示例

```bash
# 调试模式（最快）
python automl.py train config=debug

# 快速训练
python automl.py train config=xgboost_quick

# 标准训练
python automl.py train config=xgboost_full

# 带优化的训练
python automl.py train config=xgboost_optuna

# AutoML（测试所有模型）
python automl.py train config=automl

# 使用模板并覆盖参数
python automl.py train config=random_forest \
    model.hyperparameters.n_estimators=500
```

================================
# 第四部分：高级功能
================================

## 1. 超参数优化（Optuna）

### 基础优化
```bash
# 启用优化，100次试验
python automl.py train model=xgboost optimization=true n_trials=100

# 使用预定义优化模板
python automl.py train config=xgboost_optuna
```

### 高级优化配置
```bash
python automl.py train model=xgboost \
    optimization.enable=true \
    optimization.n_trials=200 \
    optimization.n_folds=5 \
    optimization.metric=r2 \
    optimization.direction=maximize \
    optimization.timeout=3600
```

### 优化参数说明

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| optimization.enable | 启用优化 | false | true/false |
| optimization.n_trials | 试验次数 | 100 | 任意正整数 |
| optimization.n_folds | 优化时折数 | 5 | 2-10 |
| optimization.metric | 优化指标 | rmse | rmse, mae, r2, mape |
| optimization.direction | 优化方向 | minimize | minimize/maximize |
| optimization.timeout | 超时（秒） | None | 任意正整数 |

## 2. AutoML - 自动模型选择

### 基础AutoML
```bash
# 使用AutoML模板
python automl.py train config=automl

# 自定义AutoML配置
python automl.py train \
    optimization.automl=true \
    optimization.automl_models='["xgboost","lightgbm","catboost"]' \
    optimization.automl_trials_per_model=50
```

### 完整AutoML（所有13个模型）
```bash
python automl.py train config=automl \
    data=data/Database_normalized.csv \
    test_data=test.csv \
    optimization.automl_models='["xgboost","lightgbm","catboost","random_forest","gradient_boosting","adaboost","extra_trees","svr","knn","decision_tree","ridge","lasso","elastic_net"]' \
    optimization.automl_trials_per_model=50 \
    optimization.n_folds=10
```

## 3. NUMA优化和并行训练

### 基础并行
```bash
# 启用NUMA优化，2个并行任务，每个2核心
python automl.py train config=automl \
    numa=true parallel=2 cores=2
```

### 大规模并行
```bash
# 256核服务器配置
python automl.py train config=automl \
    numa=true \
    parallel=32 \
    cores=8 \
    bind_cpu=true \
    project=large_scale
```

### NUMA参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| numa | 启用NUMA优化 | false | true |
| parallel | 并行任务数 | 1 | 8, 16, 32 |
| cores | 每任务核心数 | 1 | 4, 8 |
| bind_cpu | CPU亲和性绑定 | false | true |

## 4. 特征工程配置

### 分子特征配置
```bash
# Morgan指纹（1024位，半径2）
python automl.py train data=data.csv \
    feature.feature_type=morgan \
    feature.morgan_bits=1024 \
    feature.morgan_radius=2

# 分子描述符
python automl.py train data=data.csv \
    feature.feature_type=descriptors

# 组合特征（指纹+描述符）
python automl.py train data=data.csv \
    feature.feature_type=combined \
    feature.combination_method=mean
```

### 多配体组合方法
```bash
# 平均值组合
feature.combination_method=mean

# 求和组合
feature.combination_method=sum  

# 拼接组合
feature.combination_method=concat
```

## 5. 数据处理配置

### 数据分割
```bash
python automl.py train data=data.csv \
    data.train_ratio=0.7 \
    data.val_ratio=0.2 \
    data.test_ratio=0.1
```

### 目标列配置
```bash
# 单目标
python automl.py train data=data.csv target=PLQY

# 多目标（自动分别训练）
python automl.py train data=data.csv \
    target='Max_wavelength(nm),PLQY,tau(s*10^-6)'
```

### SMILES列配置
```bash
python automl.py train data=data.csv \
    data.smiles_columns='mol1,mol2,mol3'
```

================================
# 第五部分：完整参数参考
================================

## 数据参数（data.*）

| 参数 | 说明 | 类型 | 默认值 | 示例 |
|------|------|------|--------|------|
| data | 数据文件路径 | str | 必需 | data/train.csv |
| data.data_path | 同data | str | - | data/train.csv |
| data.test_data_path | 测试集路径 | str | None | data/test.csv |
| data.smiles_columns | SMILES列名 | list | [L1,L2,L3] | mol1,mol2,mol3 |
| data.target_columns | 目标列名 | list | 自动检测 | PLQY,wavelength |
| data.train_ratio | 训练集比例 | float | 0.8 | 0.7 |
| data.val_ratio | 验证集比例 | float | 0.2 | 0.2 |
| data.test_ratio | 测试集比例 | float | 0.0 | 0.1 |
| data.random_seed | 随机种子 | int | 42 | 123 |

## 特征参数（feature.*）

| 参数 | 说明 | 类型 | 默认值 | 可选值 |
|------|------|------|--------|---------|
| feature | 特征类型 | str | combined | morgan/descriptors/combined/tabular/auto |
| feature.feature_type | 同feature | str | combined | 同上 |
| feature.morgan_bits | Morgan指纹位数 | int | 1024 | 512/1024/2048 |
| feature.morgan_radius | Morgan指纹半径 | int | 2 | 2/3 |
| feature.combination_method | 多配体组合方法 | str | mean | mean/sum/concat |
| feature.use_cache | 使用特征缓存 | bool | true | true/false |
| feature.cache_dir | 缓存目录 | str | feature_cache | 任意路径 |

## 模型参数（model.*）

| 参数 | 说明 | 类型 | 默认值 | 可选值 |
|------|------|------|--------|---------|
| model | 模型类型 | str | xgboost | 13种模型 |
| model.model_type | 同model | str | xgboost | 同上 |
| model.hyperparameters.* | 模型超参数 | dict | 模型相关 | 见各模型文档 |

### 支持的13种模型
- **梯度提升**: xgboost, lightgbm, catboost
- **集成学习**: random_forest, gradient_boosting, adaboost, extra_trees
- **经典算法**: svr, knn, decision_tree
- **线性模型**: ridge, lasso, elastic_net

## 训练参数（training.*）

| 参数 | 说明 | 类型 | 默认值 | 示例 |
|------|------|------|--------|------|
| n_folds | 交叉验证折数 | int | 10 | 5, 10 |
| training.n_folds | 同n_folds | int | 10 | 5, 10 |
| training.metrics | 评估指标 | list | [rmse,mae,r2,mape] | rmse,r2 |
| training.early_stopping | 早停 | bool | false | true |
| training.early_stopping_rounds | 早停轮数 | int | 10 | 50 |
| training.verbose | 详细输出 | int | 1 | 0/1/2 |
| training.save_fold_models | 保存折模型 | bool | true | true/false |
| training.save_final_model | 保存最终模型 | bool | true | true/false |

## 优化参数（optimization.*）

| 参数 | 说明 | 类型 | 默认值 | 示例 |
|------|------|------|--------|------|
| optimization | 启用优化 | bool | false | true |
| optimization.enable | 同optimization | bool | false | true |
| optimization.optimizer | 优化器类型 | str | optuna | optuna/grid/random |
| optimization.n_trials | 试验次数 | int | 100 | 50, 200 |
| optimization.n_folds | 优化折数 | int | 5 | 3, 5 |
| optimization.timeout | 超时(秒) | int | None | 3600 |
| optimization.metric | 优化指标 | str | rmse | rmse/mae/r2/mape |
| optimization.direction | 优化方向 | str | minimize | minimize/maximize |
| optimization.automl | 启用AutoML | bool | false | true |
| optimization.automl_models | AutoML模型列表 | list | [xgboost,lightgbm,catboost] | 所有13种模型 |
| optimization.automl_trials_per_model | 每模型试验数 | int | 50 | 20, 100 |

## 项目管理参数

| 参数 | 说明 | 类型 | 默认值 | 示例 |
|------|------|------|--------|------|
| project | 项目名称 | str | default | my_project |
| name | 实验名称 | str | 自动生成 | exp_001 |
| output_dir | 输出目录 | str | runs/ | experiments/ |
| config | 配置模板 | str | None | xgboost_quick |

## 日志参数（logging.*）

| 参数 | 说明 | 类型 | 默认值 | 示例 |
|------|------|------|--------|------|
| logging.log_level | 日志级别 | str | INFO | DEBUG/INFO/WARNING |
| logging.save_plots | 保存图表 | bool | true | true/false |
| logging.generate_report | 生成报告 | bool | true | true/false |
| logging.export_for_paper | 导出论文图表 | bool | false | true |

================================
# 第六部分：实战示例
================================

## 示例1：最简单的训练
```bash
# 使用默认配置
python automl.py train data=data/Database_normalized.csv
```

## 示例2：快速实验
```bash
# 使用快速模板，5折交叉验证
python automl.py train config=xgboost_quick \
    data=data/Database_normalized.csv \
    n_folds=5 \
    project=quick_test
```

## 示例3：标准生产训练
```bash
# 完整训练配置
python automl.py train config=xgboost_full \
    data=data/Database_normalized.csv \
    test_data=data/test.csv \
    project=production \
    name=final_model \
    n_folds=10 \
    training.early_stopping=true \
    training.early_stopping_rounds=50
```

## 示例4：超参数优化
```bash
# Optuna优化
python automl.py train config=xgboost_optuna \
    data=data/Database_normalized.csv \
    target=PLQY \
    optimization.n_trials=100 \
    optimization.metric=r2 \
    optimization.direction=maximize
```

## 示例5：AutoML完整流程
```bash
# 测试所有模型，自动选择最优
python automl.py train config=automl \
    data=data/Database_normalized.csv \
    test_data=test.csv \
    optimization.automl_models='["xgboost","lightgbm","catboost","random_forest"]' \
    optimization.automl_trials_per_model=50 \
    optimization.n_folds=10
```

## 示例6：大规模并行训练
```bash
# NUMA优化 + 并行训练
python automl.py train config=automl \
    data=data/Database_normalized.csv \
    numa=true \
    parallel=8 \
    cores=4 \
    bind_cpu=true \
    project=parallel_exp
```

## 示例7：分子特征配置
```bash
# 高维Morgan指纹
python automl.py train model=xgboost \
    data=data/molecules.csv \
    feature.feature_type=morgan \
    feature.morgan_bits=2048 \
    feature.morgan_radius=3 \
    target=activity
```

## 示例8：完整的生产流水线
```bash
# 包含所有高级特性的完整命令
python automl.py train \
    config=automl \
    data=data/Database_normalized.csv \
    data.test_data_path=test_set.csv \
    data.smiles_columns='L1,L2,L3' \
    data.target_columns='Max_wavelength(nm),PLQY,tau(s*10^-6)' \
    feature.feature_type=combined \
    feature.morgan_bits=2048 \
    feature.morgan_radius=2 \
    feature.combination_method=mean \
    optimization.enable=true \
    optimization.automl=true \
    optimization.automl_models='["xgboost","lightgbm","catboost","random_forest","gradient_boosting"]' \
    optimization.automl_trials_per_model=100 \
    optimization.n_folds=10 \
    optimization.metric=r2 \
    optimization.direction=maximize \
    training.n_folds=10 \
    training.early_stopping=true \
    training.early_stopping_rounds=50 \
    numa=true \
    parallel=16 \
    cores=8 \
    bind_cpu=true \
    project=production_v1 \
    name=final_ensemble \
    logging.generate_report=true \
    logging.export_for_paper=true
```

## 示例9：批量预测流水线
```bash
# 步骤1：训练最优模型
python automl.py train config=xgboost_optuna \
    data=data/train.csv \
    project=batch_pred

# 步骤2：批量预测
python automl.py predict \
    model=runs/train/batch_pred/models/best_model.joblib \
    data=data/new_molecules.csv \
    feature=combined \
    morgan_bits=2048 \
    output=predictions_batch.csv

# 步骤3：分析预测结果
python automl.py analyze dir=runs/train/batch_pred format=html
```

## 示例10：过拟合缓解配置
```bash
# 针对PLQY过拟合的优化配置
python automl.py train config=xgboost_quick \
    data=data/Database_normalized.csv \
    target=PLQY \
    feature.feature_type=morgan \
    feature.morgan_bits=512 \
    feature.morgan_radius=2 \
    n_folds=10 \
    training.early_stopping=true \
    training.early_stopping_rounds=50 \
    model.hyperparameters.max_depth=5 \
    model.hyperparameters.min_child_weight=8 \
    model.hyperparameters.gamma=0.3 \
    model.hyperparameters.subsample=0.7 \
    model.hyperparameters.colsample_bytree=0.7 \
    model.hyperparameters.reg_alpha=0.6 \
    model.hyperparameters.reg_lambda=1.0 \
    model.hyperparameters.learning_rate=0.05 \
    model.hyperparameters.n_estimators=600
```

================================
# 附录：常见问题与解决方案
================================

## Q1: RDKit未安装导致分子特征报错
**解决**: 使用conda安装RDKit
```bash
conda install -c conda-forge rdkit
```
或切换到表格特征：
```bash
python automl.py train data=data.csv feature=tabular
```

## Q2: 预测维度不匹配
**解决**: 确保预测时特征参数与训练时一致
```bash
# 训练时
python automl.py train feature.morgan_bits=2048 feature.morgan_radius=2

# 预测时必须使用相同参数
python automl.py predict morgan_bits=2048 morgan_radius=2
```

## Q3: R²从0.8+降至0.1+
**可能原因与解决**:
1. 特征参数不一致 - 检查morgan_bits/morgan_radius
2. 数据预处理不同 - 确认PLQY单位处理
3. 随机种子变化 - 固定random_seed=42
4. 使用降维和正则化缓解过拟合

## Q4: 内存不足
**解决**: 
- 减少n_folds数量
- 使用较小的morgan_bits
- 减少parallel任务数
- 使用feature=morgan代替combined

## Q5: 训练时间过长
**解决**:
- 使用快速模板：config=xgboost_quick
- 减少n_trials数量
- 启用early_stopping
- 使用NUMA并行加速

================================
# 命令速查表
================================

```bash
# ===== 训练 =====
python automl.py train model=xgboost data=data.csv
python automl.py train config=xgboost_quick project=test
python automl.py train config=automl test_data=test.csv

# ===== 预测 =====
python automl.py predict model=model.joblib data=test.csv
python automl.py predict model=model.joblib input='["CCO"]' feature=morgan

# ===== 分析 =====
python automl.py analyze dir=last format=html
python automl.py analyze dir=runs/exp1,runs/exp2 compare=true

# ===== 验证 =====
python automl.py validate config=config.yaml
python automl.py validate data=data.csv

# ===== 导出 =====
python automl.py export model=model.joblib format=onnx

# ===== 系统信息 =====
python automl.py info
python automl.py info models
python automl.py info templates

# ===== 特征缓存 =====
python automl.py warmup data=data.csv feature=combined
python automl.py warmup clean=true
```

================================
# 第七部分：自适应生产环境训练脚本（8核到256核）
================================

## 智能自适应训练脚本

自动检测硬件配置，从8核开发机到256核服务器都能优化运行，先测试所有功能再执行训练。

### 脚本特性
- ✅ **自动硬件检测**：根据CPU核心数自动选择最优配置
- ✅ **完整功能测试**：训练前测试所有功能，确保环境正常
- ✅ **四种运行模式**：debug/development/standard/production
- ✅ **智能并行配置**：根据核心数自动调整并行参数
- ✅ **全面错误处理**：测试失败时提供详细诊断信息
- ✅ **美观HTML报告**：自动生成综合分析报告

### 运行模式

| 模式 | CPU核心 | 适用场景 | 训练规模 |
|------|---------|----------|----------|
| **Debug** | 8核以下 | 开发测试 | 最小化训练，快速验证 |
| **Development** | 16-32核 | 日常开发 | 标准训练，5折CV |
| **Standard** | 64-128核 | 标准服务器 | 完整训练，10折CV |
| **Production** | 256核+ | 生产环境 | 全量训练+AutoML |

### 使用方法

```bash
# 1. 赋予执行权限
chmod +x production_train_adaptive.sh

# 2. 自动检测硬件并选择最优模式
./production_train_adaptive.sh

# 3. 强制使用调试模式（8核开发机）
./production_train_adaptive.sh --debug

# 4. 强制使用生产模式（256核服务器）
./production_train_adaptive.sh --production

# 5. 查看结果
firefox production_runs/*/reports/index.html
```

### 脚本执行流程

#### 第1阶段：快速验证（5分钟）
- 数据验证
- 配置检查
- 2折快速测试

#### 第2阶段：主要模型训练（30-60分钟）
```bash
# XGBoost完整训练（32并行×8核）
parallel=32 cores=8 → 256核心

# LightGBM完整训练（32并行×8核）
parallel=32 cores=8 → 256核心  

# CatBoost完整训练（16并行×8核）
parallel=16 cores=8 → 128核心
```

#### 第3阶段：超参数优化（1-2小时）
- XGBoost Optuna（200次试验）
- LightGBM Optuna（200次试验）
- 自动选择最佳参数

#### 第4阶段：AutoML全模型测试（2-3小时）
- 测试全部13个模型
- 每模型50次试验
- 自动模型选择

#### 第5阶段：分析报告生成
- HTML可视化报告
- 文本分析报告
- JSON数据导出

#### 第6阶段：测试集预测和对比
- 使用最佳模型预测测试数据
- **生成预测值与真实值对比报告**
- 计算R²、MAE、RMSE、MAPE指标
- 生成交互式散点图（Plotly）
- 详细对比表显示每个样本的预测误差

#### 第7阶段：综合报告
- 生成index.html主页
- 汇总所有结果
- 性能统计
- **包含预测对比分析链接**

#### 第8阶段：打包归档
- 压缩所有结果
- 清理临时文件
- 生成最终包

### 输出目录结构

```
production_runs/production_YYYYMMDD_HHMMSS/
├── logs/                      # 日志文件
│   ├── main_*.log            # 主日志
│   ├── errors_*.log          # 错误日志
│   ├── performance_*.log     # 性能日志
│   ├── xgboost_full.log     # XGBoost训练日志
│   ├── lightgbm_full.log    # LightGBM训练日志
│   └── catboost_full.log    # CatBoost训练日志
├── reports/                   # 分析报告
│   ├── index.html            # 综合报告主页
│   ├── prediction_comparison.html  # 🎯 预测对比分析报告
│   ├── prediction_comparison.csv   # 预测对比数据
│   ├── *_report.html         # HTML报告
│   ├── *_report.txt          # 文本报告
│   └── *_results.json        # JSON数据
├── exports/                   # 导出文件
│   ├── best_model.pkl        # 最佳模型
│   └── test_predictions.csv  # 预测结果
├── visualizations/            # 可视化图表
│   ├── feature_importance/   # 特征重要性
│   ├── learning_curves/      # 学习曲线
│   └── predictions/          # 预测散点图
└── production_*/              # 各实验子目录
    ├── xgboost_full/         # XGBoost完整训练
    ├── lightgbm_full/        # LightGBM完整训练
    ├── catboost_full/        # CatBoost完整训练
    ├── xgboost_optuna/       # XGBoost优化
    ├── lightgbm_optuna/      # LightGBM优化
    └── automl_complete/      # AutoML结果
```

### 硬件配置建议

#### 256核服务器配置
```bash
# CPU配置
CPU: AMD EPYC 或 Intel Xeon (256核心)
内存: 512GB+ DDR4 ECC
存储: NVMe SSD 2TB+

# NUMA配置
NUMA节点: 8个
每节点核心: 32个
每节点内存: 64GB

# 并行配置
推荐并行数: 32
每任务核心: 8
总使用核心: 256
```

### 性能优化参数

```bash
# NUMA优化
numa=true              # 启用NUMA感知
bind_cpu=true         # CPU亲和性绑定

# 并行训练
parallel=32           # 32个并行任务
cores=8              # 每任务8核心

# 内存优化
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

### 监控和调试

```bash
# 实时监控训练进度
tail -f production_runs/*/logs/main_*.log

# 查看错误
tail -f production_runs/*/logs/errors_*.log

# 监控系统资源
htop -d 1
nvidia-smi -l 1  # 如果有GPU

# 查看NUMA状态
numactl --hardware
numastat
```

### 定制化配置

#### 修改模型参数
```bash
# 编辑脚本中的模型配置
model.hyperparameters.n_estimators=2000
model.hyperparameters.max_depth=10
model.hyperparameters.learning_rate=0.05
```

#### 修改并行配置
```bash
# 128核配置
parallel=16 cores=8

# 64核配置  
parallel=8 cores=8

# 32核配置
parallel=4 cores=8
```

#### 修改优化配置
```bash
# 更多优化试验
optimization.n_trials=500

# 更多交叉验证折数
optimization.n_folds=10

# 不同优化指标
optimization.metric=mape
optimization.direction=minimize
```

### 预计运行时间

| 阶段 | 时间估计 | 说明 |
|------|---------|------|
| 快速验证 | 5分钟 | 数据和配置检查 |
| 主要模型训练 | 30-60分钟 | 3个主要模型并行 |
| 超参数优化 | 1-2小时 | 200次试验×2模型 |
| AutoML | 2-3小时 | 13模型×50试验 |
| 报告生成 | 10-20分钟 | 分析和可视化 |
| **总计** | **4-6小时** | 完整流程 |

### 故障排除

#### 内存不足
```bash
# 减少并行数
parallel=16 cores=8

# 减少交叉验证折数
n_folds=5
```

#### 训练太慢
```bash
# 减少优化试验
optimization.n_trials=50

# 使用快速模板
config=xgboost_quick
```

#### NUMA问题
```bash
# 禁用NUMA优化
numa=false
bind_cpu=false
```

### 最佳实践

1. **数据准备**
   - 确保数据已清洗和标准化
   - 使用feature_cache加速特征提取
   - 预先分割训练/验证/测试集

2. **资源分配**
   - 为操作系统保留10-20%资源
   - 监控内存使用防止OOM
   - 使用nice命令调整优先级

3. **结果管理**
   - 定期备份重要模型
   - 使用版本控制管理配置
   - 记录实验参数和结果

4. **生产部署**
   - 选择最佳模型部署
   - 使用ONNX格式提高兼容性
   - 实施A/B测试验证效果

================================
# 更新日志
================================

- v3.0: 完整的AutoML系统，支持13种模型
- v3.1: 添加NUMA优化和并行训练支持
- v3.2: 增强特征工程和过拟合缓解
- v3.3: 完善文档和示例
- v3.4: 添加256核生产环境训练脚本

---
文档版本：2024.01
作者：AutoML Development Team