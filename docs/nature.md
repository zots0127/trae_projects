# Nature 项目完整训练与预测指南

本文档提供 Nature 项目的完整训练和预测命令，可直接复制粘贴运行。

## 数据准备

### 1. 确认数据文件
```bash
# 检查训练数据
ls ../data/Database_normalized.csv

# 检查测试数据  
ls ours.csv

# 如果文件不存在，请先准备数据
```

## 第一部分：模型训练

### 1. XGBoost 模型训练

#### 1.1 快速训练（5分钟，用于测试）
```bash
python automl.py train \
    model=xgboost \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    name=xgboost_quick \
    n_folds=5 \
    model.hyperparameters.n_estimators=100 \
    model.hyperparameters.max_depth=6 \
    model.hyperparameters.learning_rate=0.1 \
    training.save_final_model=true
```

#### 1.2 标准训练（15分钟，推荐）
```bash
python automl.py train \
    model=xgboost \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    name=xgboost_standard \
    n_folds=10 \
    model.hyperparameters.n_estimators=500 \
    model.hyperparameters.max_depth=8 \
    model.hyperparameters.learning_rate=0.05 \
    model.hyperparameters.subsample=0.8 \
    model.hyperparameters.colsample_bytree=0.8 \
    training.save_final_model=true
```

#### 1.3 完整训练（30分钟，最佳性能）
```bash
python automl.py train \
    model=xgboost \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    name=xgboost_full \
    n_folds=10 \
    model.hyperparameters.n_estimators=1000 \
    model.hyperparameters.max_depth=10 \
    model.hyperparameters.learning_rate=0.03 \
    model.hyperparameters.subsample=0.8 \
    model.hyperparameters.colsample_bytree=0.8 \
    model.hyperparameters.reg_alpha=0.1 \
    model.hyperparameters.reg_lambda=1.0 \
    training.save_final_model=true
```

### 2. LightGBM 模型训练

#### 2.1 快速训练
```bash
python automl.py train \
    model=lightgbm \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    name=lightgbm_quick \
    n_folds=5 \
    model.hyperparameters.n_estimators=100 \
    model.hyperparameters.num_leaves=31 \
    model.hyperparameters.learning_rate=0.1 \
    training.save_final_model=true
```

#### 2.2 标准训练
```bash
python automl.py train \
    model=lightgbm \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    name=lightgbm_standard \
    n_folds=10 \
    model.hyperparameters.n_estimators=500 \
    model.hyperparameters.num_leaves=50 \
    model.hyperparameters.learning_rate=0.05 \
    model.hyperparameters.feature_fraction=0.8 \
    model.hyperparameters.bagging_fraction=0.8 \
    model.hyperparameters.bagging_freq=5 \
    training.save_final_model=true
```

### 3. CatBoost 模型训练

#### 3.1 快速训练
```bash
python automl.py train \
    model=catboost \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    name=catboost_quick \
    n_folds=5 \
    model.hyperparameters.iterations=100 \
    model.hyperparameters.depth=6 \
    model.hyperparameters.learning_rate=0.1 \
    training.save_final_model=true
```

#### 3.2 标准训练
```bash
python automl.py train \
    model=catboost \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    name=catboost_standard \
    n_folds=10 \
    model.hyperparameters.iterations=500 \
    model.hyperparameters.depth=8 \
    model.hyperparameters.learning_rate=0.05 \
    model.hyperparameters.l2_leaf_reg=3.0 \
    training.save_final_model=true
```

### 4. Random Forest 模型训练

```bash
python automl.py train \
    model=random_forest \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    name=random_forest \
    n_folds=10 \
    model.hyperparameters.n_estimators=200 \
    model.hyperparameters.max_depth=20 \
    model.hyperparameters.min_samples_split=5 \
    model.hyperparameters.min_samples_leaf=2 \
    training.save_final_model=true
```

### 5. 一键训练所有模型（使用配置模板）

```bash
# 使用 AutoML 配置训练所有模型
python automl.py train \
    config=automl \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature \
    n_folds=10
```

## 第二部分：模型预测

训练完成后，模型会保存在 `Nature/train{n}/models/` 目录下。

### 1. 查看可用模型

```bash
# 列出所有训练好的模型
ls -la Nature/*/models/*.joblib
```

### 2. 预测命令

#### 2.1 预测 Max_wavelength (最大波长)

```bash
# 使用 XGBoost 模型预测
python automl.py predict \
    model=Nature/train1/models/xgboost_Max_wavelength_nm_final.joblib \
    data=ours.csv \
    output=predictions_wavelength.csv \
    output_column=Predicted_Max_wavelength \
    smiles_columns=L1,L2,L3 \
    batch_size=5000 \
    use_file_cache=true
```

#### 2.2 预测 PLQY (量子产率)

```bash
# 使用 XGBoost 模型预测
python automl.py predict \
    model=Nature/train1/models/xgboost_PLQY_final.joblib \
    data=ours.csv \
    output=predictions_plqy.csv \
    output_column=Predicted_PLQY \
    smiles_columns=L1,L2,L3 \
    batch_size=5000 \
    use_file_cache=true
```


### 3. 批量预测（波长和PLQY）

```bash
# 创建批量预测脚本
cat > batch_predict.sh << 'EOF'
#!/bin/bash
# 批量预测波长和PLQY

# 设置模型目录（根据实际训练结果调整）
MODEL_DIR="Nature/train1/models"

echo "开始批量预测..."

# 预测 Max_wavelength
echo "1. 预测 Max_wavelength..."
python automl.py predict \
    model=$MODEL_DIR/xgboost_Max_wavelength_nm_final.joblib \
    data=ours.csv \
    output=predictions_wavelength.csv \
    output_column=Predicted_Max_wavelength

# 预测 PLQY
echo "2. 预测 PLQY..."
python automl.py predict \
    model=$MODEL_DIR/xgboost_PLQY_final.joblib \
    data=ours.csv \
    output=predictions_plqy.csv \
    output_column=Predicted_PLQY

echo "批量预测完成！"
echo "结果文件："
echo "  - predictions_wavelength.csv"
echo "  - predictions_plqy.csv"
EOF

# 运行批量预测
chmod +x batch_predict.sh
./batch_predict.sh
```

### 4. 预测新数据

#### 4.1 预测单个SMILES

```bash
# 预测单个分子
python automl.py predict \
    model=Nature/train1/models/xgboost_Max_wavelength_nm_final.joblib \
    input='[["CCO","c1ccccc1","C1=CC=CC=C1"]]' \
    feature=combined \
    output=single_prediction.csv
```

#### 4.2 预测CSV文件

```bash
# 准备新数据文件 new_molecules.csv，格式如下：
# L1,L2,L3
# SMILES1,SMILES2,SMILES3
# ...

# 执行预测
python automl.py predict \
    model=Nature/train1/models/xgboost_Max_wavelength_nm_final.joblib \
    data=new_molecules.csv \
    output=new_predictions.csv \
    smiles_columns=L1,L2,L3
```

## 第三部分：结果分析

### 1. 查看训练结果

```bash
# 分析最新训练结果
python automl.py analyze dir=Nature/train1 format=html

# 以文本格式查看
python automl.py analyze dir=Nature/train1 format=text
```

### 2. 比较多个模型

```bash
# 比较不同模型的性能
python automl.py analyze dir=Nature/train1,Nature/train2,Nature/train3 format=html
```

### 3. 查看缓存统计

```bash
# 查看特征缓存使用情况
python automl.py cache stats

# 清理旧缓存（30天前的）
python automl.py cache clear 30
```

## 快速开始示例

### 最简单的完整流程（训练+预测）

```bash
# 步骤1: 快速训练一个模型（5分钟）
python automl.py train \
    config=xgboost_quick \
    data=../data/Database_normalized.csv \
    project=Nature

# 步骤2: 查看生成的模型
ls Nature/train1/models/

# 步骤3: 预测（自动命名输出文件）
python automl.py predict \
    model=Nature/train1/models/xgboost_Max_wavelength_nm_final_*.joblib \
    data=ours.csv

# 预测结果会自动保存为: predictions_20250912_HHMMSS.csv
```

## 常见问题

### Q1: 如何知道模型训练完成？
查看输出中的 "✅ 训练完成" 信息，或检查 `Nature/train{n}/models/` 目录下是否有 `.joblib` 文件。

### Q2: 预测很慢怎么办？
使用文件缓存（默认开启），第二次预测相同文件会快100倍。

### Q3: 如何使用不同的特征类型？
添加参数 `feature=morgan` 或 `feature=descriptors` 或 `feature=combined`（默认）。

### Q4: 预测结果保存在哪里？
- 不指定output时：`predictions_时间戳.csv`
- 指定output时：保存到指定文件，存在时自动加编号

## 性能参考

- 训练时间：5-30分钟（取决于配置）
- 预测时间：
  - 小文件（<1000行）：1秒
  - 大文件（272K行）：首次45秒，缓存后6秒

## 联系支持

如有问题，请检查：
1. 数据文件路径是否正确
2. 模型文件是否存在
3. Python环境是否正确（需要rdkit等依赖）