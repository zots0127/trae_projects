#!/usr/bin/env python3
"""
AutoML系统主入口
统一的命令行接口
"""

import sys
from pathlib import Path

# 确保automl.py可以被导入
sys.path.insert(0, str(Path(__file__).parent))

from automl import main

if __name__ == "__main__":
    sys.exit(main())