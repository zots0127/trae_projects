#!/usr/bin/env python3
"""
AutoML main entry point
Unified command-line interface
"""

import sys
from pathlib import Path

# Ensure automl.py can be imported
sys.path.insert(0, str(Path(__file__).parent))

from automl import main

if __name__ == "__main__":
    sys.exit(main())
