#!/bin/bash
# Clean uv-based environment bootstrap for Nature release
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$ROOT_DIR/.venv"
PY_VER="${UV_PYTHON:-3.9}"

if [ "${UV_LOCAL_TEST:-0}" = "1" ]; then
  mkdir -p "$ROOT_DIR/Project_Output"
  exit 0
fi

echo "==========================================="
echo "Setting up Python environment with uv"
echo "==========================================="

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  # Try common install locations
  if [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  elif [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
  fi

  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to install or locate uv. Please install it manually."
    exit 1
  fi
}

ensure_uv
echo "uv version: $(uv --version)"

# Recreate venv only when explicitly requested
if [ "${UV_RECREATE:-0}" = "1" ] && [ -d "$VENV_PATH" ]; then
  echo "UV_RECREATE=1 detected, removing existing venv..."
  rm -rf "$VENV_PATH"
fi

if [ ! -d "$VENV_PATH" ]; then
  echo "Creating virtual environment at $VENV_PATH (Python $PY_VER)..."
  uv venv "$VENV_PATH" --python "$PY_VER" || uv venv "$VENV_PATH"
else
  echo "Using existing virtual environment at $VENV_PATH"
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo ""
echo "Pinning NumPy for RDKit compatibility (numpy<2)..."
uv pip install 'numpy<2'
echo "Installing core requirements..."
uv pip install -r <(grep -vi '^\s*rdkit' "$ROOT_DIR/requirements.txt")

echo ""
echo "Ensuring RDKit availability..."
python -c "import rdkit" >/dev/null 2>&1 || uv pip install rdkit-pypi
python -c "import rdkit" >/dev/null 2>&1 || echo "WARNING: RDKit not available; molecular features may fail"

echo ""
echo "Installing PyTorch CPU wheel..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing extra ML utilities..."
EXTRA_PKGS=(
  imbalanced-learn
  statsmodels
)
uv pip install "${EXTRA_PKGS[@]}"

echo ""
echo "==========================================="
echo "Environment setup complete!"
echo "==========================================="
echo "Python version: $(python --version)"
echo ""
echo "Activate with: source \"$VENV_PATH/bin/activate\""
echo "To recreate: UV_RECREATE=1 UV_PYTHON=$PY_VER bash uv.sh"

# Auto-run full workflow unless explicitly skipped
if [ "${UV_SKIP_WORKFLOW:-0}" != "1" ]; then
  WORKFLOW="$ROOT_DIR/run_workflow.sh"
  if [ -x "$WORKFLOW" ]; then
    echo ""
    echo "==========================================="
    echo "Running workflow: $WORKFLOW"
    echo "==========================================="
    bash "$WORKFLOW"
  elif [ -f "$WORKFLOW" ]; then
    echo "Workflow script found but not executable: $WORKFLOW"
    echo "Run: chmod +x run_workflow.sh && bash uv.sh"
    exit 1
  else
    echo "No workflow script found (run_workflow.sh)."
    echo "Set UV_SKIP_WORKFLOW=1 to suppress this check."
  fi
else
  echo "UV_SKIP_WORKFLOW=1 set; skipping workflow run."
fi
