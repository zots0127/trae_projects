#!/usr/bin/env python3
"""
测试Optuna优化功能
"""

import os
import sys

# 测试1: 使用快速优化模板
print("="*60)
print("测试1: 快速优化XGBoost")
print("="*60)
os.system("python automl.py train config=quick_optimize project=test_optuna_quick")

# # 测试2: 使用完整优化模板（可选，耗时较长）
# print("\n" + "="*60)
# print("测试2: 完整XGBoost+Optuna优化")
# print("="*60)
# os.system("python automl.py train config=xgboost_optuna project=test_optuna_full")

# # 测试3: 使用AutoML模板（可选，耗时很长）
# print("\n" + "="*60)
# print("测试3: AutoML自动选择最佳模型")
# print("="*60)
# os.system("python automl.py train config=automl project=test_automl")

print("\n✅ 测试完成！")