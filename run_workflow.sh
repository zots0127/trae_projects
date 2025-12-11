#!/bin/bash
# Single entry to run unified workflow
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKFLOW="$SCRIPT_DIR/workflow.sh"

if [ ! -f "$WORKFLOW" ]; then
  echo "[ERROR] workflow.sh not found at $WORKFLOW"
  exit 1
fi

bash "$WORKFLOW" "$@"
