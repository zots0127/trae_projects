#!/bin/bash
# Test script for paper-level comparison
# Train all models and generate comparison tables

echo "============================================================"
echo "          Paper-level Model Comparison Test Script"
echo "============================================================"
echo ""

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
DATA_FILE="${1:-../data/Database_normalized.csv}"
TEST_DATA="${2:-Database_ours_0903update_normalized.csv}"
PROJECT_NAME="${3:-PaperComparison}"

# Check data files
echo -e "${BLUE}[1/4] Checking data files...${NC}"
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${YELLOW}WARNING: Training data file not found: $DATA_FILE${NC}"
    echo "Trying sample data..."
    DATA_FILE="data/synthetic_molecules.csv"
    if [ ! -f "$DATA_FILE" ]; then
        echo -e "${RED}ERROR: No usable data file found${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}INFO: Data file: $DATA_FILE${NC}"

# Run training
echo ""
echo -e "${BLUE}[2/4] Starting training for all models...${NC}"
echo "Config: paper_comparison"
echo "Project: $PROJECT_NAME"
echo ""

# Train with paper_comparison config
python automl.py train \
    config=paper_comparison \
    data="$DATA_FILE" \
    project="$PROJECT_NAME" \
    name=paper_test \
    training.n_folds=5 \
    optimization.automl_models=[xgboost,lightgbm,catboost,random_forest,gradient_boosting]

# Check training result
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Training failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}INFO: Training completed${NC}"

# Generate comparison table
echo ""
echo -e "${BLUE}[3/4] Generating comparison table...${NC}"

# Find latest training directory
LATEST_DIR=$(ls -td "$PROJECT_NAME"/paper_test* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo -e "${YELLOW}WARNING: No training result directory found${NC}"
    exit 1
fi

echo "Result directory: $LATEST_DIR"

# Generate tables with Python
python -c "
import sys
sys.path.append('.')
from utils.comparison_table import ComparisonTableGenerator

generator = ComparisonTableGenerator('$LATEST_DIR')

exported = generator.export_all_formats(
    formats=['markdown', 'html', 'latex', 'csv']
)

print('')
print('='*60)
print('Best model summary')
print('='*60)
best_models = generator.get_best_models()
for target, info in best_models.items():
    print(f'{target}:')
    print(f'  Best model: {info[\"algorithm\"]}')
    print(f'  R2: {info[\"r2\"]}')
    print(f'  RMSE: {info[\"rmse\"]}')
    print('')
"

# Show generated files
echo ""
echo -e "${BLUE}[4/4] Generated files:${NC}"
ls -la "$LATEST_DIR"/comparison_table_* 2>/dev/null

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}                    Test completed${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "View results:"
echo "  - Markdown: cat $LATEST_DIR/comparison_table_*.md"
echo "  - HTML: open $LATEST_DIR/comparison_table_*.html"
echo "  - LaTeX: cat $LATEST_DIR/comparison_table_*.tex"
echo "  - CSV: cat $LATEST_DIR/comparison_table_*.csv"
echo ""
echo "Tips:"
echo "  - Use full config for production training:"
echo "    python automl.py train config=paper_comparison data=your_data.csv"
echo "  - Customize column names:"
echo "    python automl.py train config=paper_comparison \\
      data=data.csv \\
      smiles_columns=L1,L2,L3 \\
      targets=wavelength,plqy"
