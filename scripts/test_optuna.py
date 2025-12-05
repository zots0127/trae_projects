#!/usr/bin/env python3
"""
Test Optuna optimization
"""

import os
import sys
from pathlib import Path

# Test 1: quick optimization template
print("="*60)
print("INFO: Test 1 - Quick XGBoost optimization")
print("="*60)
automl_path = Path(__file__).parent.parent / 'automl.py'
cmd = f'"{sys.executable}" "{automl_path}" train config=quick_optimize project=test_optuna_quick'
os.system(cmd)

# # Test 2: full optimization template (optional, longer)
# # print("\n" + "="*60)
# # print("INFO: Test 2 - XGBoost + Optuna full optimization")
# # print("="*60)
# # os.system(f'"{sys.executable}" "{automl_path}" train config=xgboost_optuna project=test_optuna_full')

# # Test 3: AutoML template (optional, very long)
# # print("\n" + "="*60)
# # print("INFO: Test 3 - AutoML selects best model")
# # print("="*60)
# # os.system(f'"{sys.executable}" "{automl_path}" train config=automl project=test_automl')

print("\nINFO: Tests completed!")
