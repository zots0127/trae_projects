#!/bin/bash
# Nature project batch prediction script (wavelength & PLQY)
# Usage: ./predict_nature.sh [MODEL_DIR] [DATA_FILE]

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default parameters
MODEL_DIR=${1:-"Nature/train1/models"}
DATA_FILE=${2:-"Database_ours_0903update_normalized.csv"}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    Nature Batch Prediction (Wavelength & PLQY)    ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check data file
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${YELLOW}WARNING: Data file not found: $DATA_FILE${NC}"
    exit 1
fi

# Check model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}WARNING: Model directory not found: $MODEL_DIR${NC}"
    echo "Available model directories:"
    ls -d Nature/*/models/ 2>/dev/null
    exit 1
fi

echo "Model directory: $MODEL_DIR"
echo "Data file: $DATA_FILE"
echo ""

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Predict wavelength
echo -e "${GREEN}[1/2] Predicting Max_wavelength...${NC}"
WAVELENGTH_MODEL=$(ls $MODEL_DIR/*Max_wavelength*.joblib 2>/dev/null | head -1)
if [ -n "$WAVELENGTH_MODEL" ]; then
    OUTPUT_FILE="predictions_wavelength_${TIMESTAMP}.csv"
    python automl.py predict \
        model=$WAVELENGTH_MODEL \
        data=$DATA_FILE \
        output=$OUTPUT_FILE \
        output_column=Predicted_Max_wavelength \
        batch_size=5000 \
        use_file_cache=true
    echo "INFO: Wavelength prediction completed: $OUTPUT_FILE"
else
    echo -e "${YELLOW}WARNING: Wavelength model not found${NC}"
fi

echo ""

# Predict PLQY
echo -e "${GREEN}[2/2] Predicting PLQY...${NC}"
PLQY_MODEL=$(ls $MODEL_DIR/*PLQY*.joblib 2>/dev/null | head -1)
if [ -n "$PLQY_MODEL" ]; then
    OUTPUT_FILE="predictions_plqy_${TIMESTAMP}.csv"
    python automl.py predict \
        model=$PLQY_MODEL \
        data=$DATA_FILE \
        output=$OUTPUT_FILE \
        output_column=Predicted_PLQY \
        batch_size=5000 \
        use_file_cache=true
    echo "INFO: PLQY prediction completed: $OUTPUT_FILE"
else
    echo -e "${YELLOW}WARNING: PLQY model not found${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}           Batch prediction completed              ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Generated files:"
ls predictions_*_${TIMESTAMP}.csv 2>/dev/null
echo ""
echo "Tips:"
echo "  - Running the same file a second time uses cache (up to 100x faster)"
echo "  - Use python automl.py cache stats to view cache"
