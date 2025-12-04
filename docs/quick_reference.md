# 快速参考卡片

## 🚀 常用命令

### 调试测试（1分钟）
```bash
python automl.py train config=debug data=data.csv
```

### 快速训练（5分钟）
```bash
python automl.py train config=xgboost_quick data=data.csv
```

### 标准训练（15分钟）
```bash
python automl.py train config=xgboost_standard \
    data=data.csv \
    test_data=test.csv \
    project=MyProject \
    name=exp001
```

### 超参数优化（60分钟）
```bash
python automl.py train config=xgboost_optuna \
    data=data.csv \
    n_trials=100
```

---

## 📊 多目标策略

### 严格模式（1354行）
```bash
multi_target=intersection  # 所有目标都有值
```

### 独立模式（默认）
```bash
multi_target=independent   # 每个目标独立
```

### 并集模式（1667行）
```bash
multi_target=union nan_handling=mean  # 填充缺失值
```

---

## 🔧 缺失值处理

```bash
nan_handling=skip        # 跳过（默认）
nan_handling=mean        # 均值填充
nan_handling=median      # 中位数填充
nan_handling=zero        # 零值填充
```

---

## 🎯 模型选择

```bash
model=xgboost           # 默认，性能好
model=lightgbm          # 速度快
model=catboost          # 类别特征
model=random_forest     # 可解释性
```

---

## 📁 输出控制

```bash
project=ProjectName     # 项目目录
name=experiment_001     # 实验名称
# 结果保存在: ProjectName/experiment_001/
```

---

## ⚡ 组合示例

### 最严格训练
```bash
python automl.py train \
    config=xgboost_standard \
    data=data.csv \
    multi_target=intersection \
    nan_handling=skip \
    project=Strict \
    name=exp001
```

### 最大数据利用
```bash
python automl.py train \
    config=xgboost_standard \
    data=data.csv \
    multi_target=union \
    nan_handling=mean \
    project=MaxData \
    name=exp001
```

### 快速对比实验
```bash
# XGBoost
python automl.py train model=xgboost data=data.csv project=Compare name=xgb

# LightGBM
python automl.py train model=lightgbm data=data.csv project=Compare name=lgb

# CatBoost
python automl.py train model=catboost data=data.csv project=Compare name=cat
```

---

## 📈 结果分析

```bash
# 分析最后一次训练
python automl.py analyze dir=last format=html

# 分析指定实验
python automl.py analyze dir=ProjectName/exp001 format=html

# 对比多个实验
python automl.py analyze dir=exp1,exp2,exp3 format=html
```

---

## 🎮 预测使用

```bash
# 单个预测
python automl.py predict \
    model=path/to/model.joblib \
    input='[["SMILES1","SMILES2","SMILES3"]]'

# 批量预测
python automl.py predict \
    model=path/to/model.joblib \
    data=new_data.csv \
    output=predictions.csv
```

---

## 💡 提示

1. **先用debug测试**：验证数据格式
2. **逐步增加复杂度**：debug → quick → standard → optuna
3. **保存配置**：重要实验保存配置文件
4. **使用项目管理**：相关实验放在同一project下
5. **记录实验**：使用有意义的name参数