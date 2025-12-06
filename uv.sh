#!/bin/bash

# UV-based Python environment setup script

set -e  # Exit on any error

echo "==========================================="
echo "Setting up Python environment with uv"
echo "==========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    # Try sourcing uv environment first
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    # Check again after sourcing
    if ! command -v uv &> /dev/null; then
        echo "uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || true

        # Add to PATH manually
        export PATH="$HOME/.local/bin:$PATH"

        # Verify installation
        if ! command -v uv &> /dev/null; then
            echo "Failed to install or locate uv. Please install it manually."
            exit 1
        fi
    fi
fi

echo "uv version: $(uv --version)"

# Create virtual environment (clear if exists)
echo ""
echo "Creating virtual environment (.venv)..."
uv venv .venv --clear --python 3.9

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."

# Core ML libraries
uv pip install numpy pandas scikit-learn xgboost lightgbm catboost

# Deep learning (CPU)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Data visualization
uv pip install matplotlib seaborn plotly

# Utilities
uv pip install jupyter ipython tqdm pyyaml openpyxl tabulate

# Model evaluation and metrics
uv pip install optuna shap

# Additional ML tools
uv pip install imbalanced-learn scipy statsmodels

# Chemistry and molecular tools
uv pip install rdkit

echo ""
echo "==========================================="
echo "Environment setup complete!"
echo "==========================================="
echo ""
echo "Python version: $(python --version)"
echo ""

# Run workflow if available
if [ -f "./run_workflow.sh" ]; then
    echo "Running workflow: run_workflow.sh"
    echo "==========================================="
    bash ./run_workflow.sh
else
    echo "No workflow script found (run_workflow.sh)."
    echo "To activate the environment manually, run:"
    echo "  source .venv/bin/activate"
fi
