#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ -f "$SCRIPT_DIR/0913.sh" ]; then
  bash "$SCRIPT_DIR/0913.sh" "$@"
else
  echo "0913.sh not found. Please run python scripts/train.py paper -d data/Database_normalized.csv"
  exit 1
fi
