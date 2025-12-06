#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ -f "$SCRIPT_DIR/project_workflow.sh" ]; then
  bash "$SCRIPT_DIR/project_workflow.sh" "$@"
elif [ -f "$SCRIPT_DIR/ml_pipeline.sh" ]; then
  bash "$SCRIPT_DIR/ml_pipeline.sh" "$@"
else
  echo "[ERROR] No workflow script found. Expected one of: project_workflow.sh, ml_pipeline.sh"
  echo "[HINT] Ensure the repo is up to date or run: python automl.py train data=data/Database_normalized.csv project=experiments name=exp"
  exit 1
fi
