#!/bin/bash
# Test single model training

echo "Testing XGBoost model training..."

python automl.py train \
    config=xgboost_debug \
    data=../data/Database_normalized.csv \
    project=test_single \
    name=xgboost_test \
    n_folds=2 \
    multi_target=independent \
    nan_handling=skip

echo "Done! Check test_single/ directory"
