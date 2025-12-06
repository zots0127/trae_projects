# Nature 项目快速指南（复制即用）

## 🚀 最快速开始（5分钟）

```bash
# 1. 训练模型（使用快速配置）
python automl.py train config=xgboost_quick data=../data/Database_normalized.csv project=Nature

# 2. 查看模型
ls Nature/train1/models/

# 3. 预测（自动保存到带时间戳的文件）
python automl.py predict model=Nature/train1/models/*_Max_wavelength_*.joblib data=ours.csv
```

## 📋 完整训练命令（直接复制）

### XGBoost 标准训练（推荐）
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
    training.save_final_model=true
```

### LightGBM 标准训练
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
    training.save_final_model=true
```

### CatBoost 标准训练
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
    training.save_final_model=true
```

## 🎯 预测命令（直接复制）

### 预测最大波长
```bash
python automl.py predict \
    model=Nature/train1/models/xgboost_Max_wavelength_nm_final*.joblib \
    data=ours.csv \
    output=predictions_wavelength.csv \
    output_column=Predicted_Max_wavelength
```

### 预测PLQY
```bash
python automl.py predict \
    model=Nature/train1/models/xgboost_PLQY_final*.joblib \
    data=ours.csv \
    output=predictions_plqy.csv \
    output_column=Predicted_PLQY
```


## 📊 一键批量脚本

创建并运行以下脚本：

```bash
cat > run_all.sh << 'EOF'
#!/bin/bash
echo "=== Nature 项目自动化脚本 ==="

# 训练
echo "开始训练..."
python automl.py train \
    config=xgboost_standard \
    data=../data/Database_normalized.csv \
    test_data=ours.csv \
    project=Nature

# 等待训练完成
echo "训练完成！"

# 找到最新的模型目录
MODEL_DIR=$(ls -td Nature/train* | head -1)/models
echo "使用模型目录: $MODEL_DIR"

# 预测波长和PLQY
echo "预测 Max_wavelength..."
python automl.py predict \
    model=$MODEL_DIR/*Max_wavelength*.joblib \
    data=ours.csv \
    output=pred_wavelength.csv

echo "预测 PLQY..."
python automl.py predict \
    model=$MODEL_DIR/*PLQY*.joblib \
    data=ours.csv \
    output=pred_plqy.csv

echo "=== 完成！==="
echo "结果文件："
ls pred_*.csv
EOF

chmod +x run_all.sh
./run_all.sh
```

## 🔍 查看结果

```bash
# 查看训练结果
python automl.py analyze dir=Nature/train1 format=text

# 查看缓存
python automl.py cache stats

# 列出所有预测结果
ls predictions_*.csv pred_*.csv
```

## ⚡ 性能提示

1. **使用缓存**：第二次预测相同文件快100倍
2. **批量处理**：`batch_size=5000` 处理大文件
3. **自动命名**：不指定output避免覆盖

## 📝 文件格式要求

输入CSV必须包含：
- `L1` - 第一个配体SMILES
- `L2` - 第二个配体SMILES  
- `L3` - 第三个配体SMILES

可选目标列：
- `Max_wavelength(nm)` - 最大波长
- `PLQY` - 量子产率

## 🆘 问题排查

```bash
# 检查文件
ls ../data/Database_normalized.csv
ls ours.csv

# 检查模型
ls Nature/*/models/*.joblib

# 检查Python环境
python -c "import rdkit, xgboost, lightgbm; print('环境OK')"
```