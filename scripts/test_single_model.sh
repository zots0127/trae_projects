#!/bin/bash
# 测试单个模型训练

echo "测试XGBoost模型训练..."

python automl.py train \
    config=xgboost_debug \
    data=../data/Database_normalized.csv \
    project=test_single \
    name=xgboost_test \
    n_folds=2 \
    multi_target=independent \
    nan_handling=skip

echo "完成！检查 test_single/ 目录"