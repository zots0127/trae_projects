#!/bin/bash

# ============================================================
# MASTER ML PIPELINE SCRIPT
# All-in-one script for training, prediction, and analysis
# ============================================================

# ============================================================
# CONFIGURATION SECTION - EDIT ALL PARAMETERS HERE
# ============================================================

# === DATA CONFIGURATION ===
TRAIN_DATA="data/Database_normalized.csv"              # Training dataset
TEST_DATA="Database_ours_0903update_normalized.csv"    # Test dataset for predictions
VIRTUAL_DATA="ir_assemble.csv"                        # Virtual database (272k combinations)

# === MODEL CONFIGURATION ===
MODELS="xgboost lightgbm catboost"                    # Space-separated list of models
CONFIG_LEVEL="standard"                               # quick(5min)/standard(15min)/full(30min)
N_FOLDS=10                                           # Number of cross-validation folds
TRAIN_REGULAR=true                                   # Train on full dataset
TRAIN_INTERSECTION=true                              # Train on wavelength+PLQY intersection

# === OUTPUT CONFIGURATION ===
OUTPUT_BASE="experiments"                            # Base directory for all outputs
OUTPUT_PREFIX="exp"                                  # Prefix for experiment folders
USE_TIMESTAMP=true                                   # Append timestamp to folder names
CUSTOM_NAME=""                                       # Optional custom name (overrides auto-naming)

# === PREDICTION CONFIGURATION ===
PREDICT_TEST=true                                    # Predict on test dataset
PREDICT_VIRTUAL=false                                # Predict on virtual database (272k)
BATCH_SIZE=10000                                    # Batch size for large predictions
USE_BEST_MODEL=true                                 # Use best model from training for predictions

# === FEATURE CONFIGURATION ===
FEATURE_TYPE="combined"                             # morgan/descriptors/combined
MORGAN_BITS=1024                                   # Morgan fingerprint bits
MORGAN_RADIUS=2                                     # Morgan fingerprint radius
COMBINATION_METHOD="mean"                           # mean/sum/concat for multi-ligand

# === REPORT CONFIGURATION ===
GENERATE_COMPARISON=true                            # Generate model comparison table
GENERATE_LATEX=false                                # Generate LaTeX tables
GENERATE_PLOTS=false                                # Generate visualization plots
GENERATE_HTML=false                                 # Generate HTML reports
SHOW_TOP_N=10                                      # Show top N candidates in reports

# === PERFORMANCE CONFIGURATION ===
VERBOSE=1                                           # 0=quiet, 1=normal, 2=debug
PARALLEL_JOBS=-1                                    # Number of parallel jobs (-1=all cores)
USE_CACHE=true                                      # Use feature extraction cache
SAVE_FINAL_MODEL=true                               # Save final trained models

# === ADVANCED OPTIONS ===
NAN_HANDLING="skip"                                 # skip/mean/median/constant
MULTI_TARGET="independent"                          # independent/chain/simultaneous
RANDOM_STATE=42                                     # Random seed for reproducibility
MAX_ITER=100                                       # Maximum iterations for iterative models

# ============================================================
# FUNCTION DEFINITIONS
# ============================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_step() {
    echo ""
    echo -e "${GREEN}>>> $1${NC}"
    echo "------------------------------------------------------------"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Setup output directory
setup_output_dir() {
    if [ -n "$CUSTOM_NAME" ]; then
        OUTPUT_DIR="$OUTPUT_BASE/$CUSTOM_NAME"
    elif [ "$USE_TIMESTAMP" = true ]; then
        OUTPUT_DIR="$OUTPUT_BASE/${OUTPUT_PREFIX}_$(date +%Y%m%d_%H%M%S)"
    else
        OUTPUT_DIR="$OUTPUT_BASE/${OUTPUT_PREFIX}"
    fi
    
    mkdir -p "$OUTPUT_DIR"
    echo "$OUTPUT_DIR"
}

# Validate inputs
validate_inputs() {
    local valid=true
    
    if [ ! -f "$TRAIN_DATA" ]; then
        print_error "Training data not found: $TRAIN_DATA"
        valid=false
    fi
    
    if [ "$PREDICT_TEST" = true ] && [ ! -f "$TEST_DATA" ]; then
        print_warning "Test data not found: $TEST_DATA (will skip test prediction)"
        PREDICT_TEST=false
    fi
    
    if [ "$PREDICT_VIRTUAL" = true ] && [ ! -f "$VIRTUAL_DATA" ]; then
        print_warning "Virtual data not found: $VIRTUAL_DATA (will skip virtual prediction)"
        PREDICT_VIRTUAL=false
    fi
    
    if [ "$valid" = false ]; then
        exit 1
    fi
}

# Create intersection data
create_intersection_data() {
    local output_dir=$1
    
    python3 ml_helpers.py create_intersection \
        --input "$TRAIN_DATA" \
        --output "$output_dir/intersection_data.csv"
}

# Train models
train_models() {
    local output_dir=$1
    local data_file=$2
    local name_suffix=$3
    
    for model in $MODELS; do
        print_step "Training $model$name_suffix"
        
        local model_dir="$output_dir/${model}${name_suffix}"
        
        python main.py train \
            config=${model}_${CONFIG_LEVEL} \
            data="$data_file" \
            project="$model_dir" \
            name="${model}${name_suffix}" \
            n_folds=$N_FOLDS \
            multi_target=$MULTI_TARGET \
            nan_handling=$NAN_HANDLING \
            save_final_model=$SAVE_FINAL_MODEL \
            verbose=$VERBOSE \
            feature=$FEATURE_TYPE \
            morgan_bits=$MORGAN_BITS \
            morgan_radius=$MORGAN_RADIUS \
            random_state=$RANDOM_STATE
        
        if [ $? -eq 0 ]; then
            print_success "$model$name_suffix training completed"
        else
            print_error "$model$name_suffix training failed"
        fi
    done
}

# Find best model
find_best_model() {
    local output_dir=$1
    
    python3 ml_helpers.py find_best_model \
        --output-dir "$output_dir" \
        --models "$MODELS"
}

# Run predictions
run_predictions() {
    local output_dir=$1
    local model_dir=$2
    
    # Find model files
    local wavelength_model=$(ls -t $model_dir/*/models/*Max_wavelength*final*.joblib 2>/dev/null | head -1)
    local plqy_model=$(ls -t $model_dir/*/models/*PLQY*final*.joblib 2>/dev/null | head -1)
    
    if [ -z "$wavelength_model" ] || [ -z "$plqy_model" ]; then
        print_error "Model files not found in $model_dir"
        return 1
    fi
    
    # Predict test data
    if [ "$PREDICT_TEST" = true ]; then
        print_step "Predicting test dataset"
        
        python main.py predict \
            model="$wavelength_model" \
            data="$TEST_DATA" \
            output="$output_dir/test_wavelength.csv" \
            feature=$FEATURE_TYPE
        
        python main.py predict \
            model="$plqy_model" \
            data="$TEST_DATA" \
            output="$output_dir/test_plqy.csv" \
            feature=$FEATURE_TYPE
        
        # Merge predictions
        python3 ml_helpers.py merge_predictions \
            --data "$TEST_DATA" \
            --wavelength "$output_dir/test_wavelength.csv" \
            --plqy "$output_dir/test_plqy.csv" \
            --output "$output_dir/test_predictions.csv"
        
        print_success "Test predictions saved to $output_dir/test_predictions.csv"
    fi
    
    # Predict virtual database
    if [ "$PREDICT_VIRTUAL" = true ]; then
        print_step "Predicting virtual database (272,104 combinations)"
        
        python main.py predict \
            model="$wavelength_model" \
            data="$VIRTUAL_DATA" \
            output="$output_dir/virtual_wavelength.csv" \
            feature=$FEATURE_TYPE
        
        python main.py predict \
            model="$plqy_model" \
            data="$VIRTUAL_DATA" \
            output="$output_dir/virtual_plqy.csv" \
            feature=$FEATURE_TYPE
        
        # Merge and filter high PLQY
        python3 ml_helpers.py merge_predictions \
            --data "$VIRTUAL_DATA" \
            --wavelength "$output_dir/virtual_wavelength.csv" \
            --plqy "$output_dir/virtual_plqy.csv" \
            --output "$output_dir/virtual_predictions.csv" \
            --filter-plqy 0.9 \
            --top-n $SHOW_TOP_N
        
        print_success "Virtual predictions saved to $output_dir/virtual_predictions.csv"
    fi
}

# Generate comparison table
generate_comparison() {
    local output_dir=$1
    
    python3 ml_helpers.py generate_comparison \
        --output-dir "$output_dir" \
        --models "$MODELS" \
        --generate-latex $GENERATE_LATEX \
        --generate-plots $GENERATE_PLOTS \
        --generate-html $GENERATE_HTML
}

# Generate final report
generate_report() {
    local output_dir=$1
    
    python3 ml_helpers.py generate_report \
        --output-dir "$output_dir" \
        --config "$0"
}

# ============================================================
# MAIN EXECUTION
# ============================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --train-only)
                PREDICT_TEST=false
                PREDICT_VIRTUAL=false
                shift
                ;;
            --predict-only)
                TRAIN_REGULAR=false
                TRAIN_INTERSECTION=false
                shift
                ;;
            --models)
                MODELS="$2"
                shift 2
                ;;
            --folds)
                N_FOLDS="$2"
                shift 2
                ;;
            --quick)
                CONFIG_LEVEL="quick"
                N_FOLDS=5
                shift
                ;;
            --output)
                CUSTOM_NAME="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --train-only      Only train models, skip predictions"
                echo "  --predict-only    Only run predictions, skip training"
                echo "  --models LIST     Models to train (e.g., 'xgboost lightgbm')"
                echo "  --folds N         Number of CV folds"
                echo "  --quick           Quick mode (5 folds, quick config)"
                echo "  --output NAME     Custom output directory name"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Header
    print_header "ML PIPELINE EXECUTION"
    echo "Configuration:"
    echo "  Models: $MODELS"
    echo "  Config: $CONFIG_LEVEL"
    echo "  Folds: $N_FOLDS"
    echo "  Features: $FEATURE_TYPE"
    
    # Setup
    cd /Users/kanshan/IR/ir2025/v3
    validate_inputs
    OUTPUT_DIR=$(setup_output_dir)
    echo "  Output: $OUTPUT_DIR"
    
    # Save configuration
    cat > "$OUTPUT_DIR/config.txt" << EOF
# Pipeline Configuration
MODELS=$MODELS
CONFIG_LEVEL=$CONFIG_LEVEL
N_FOLDS=$N_FOLDS
FEATURE_TYPE=$FEATURE_TYPE
TRAIN_DATA=$TRAIN_DATA
TEST_DATA=$TEST_DATA
VIRTUAL_DATA=$VIRTUAL_DATA
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
EOF
    
    # Step 1: Create intersection data
    if [ "$TRAIN_INTERSECTION" = true ]; then
        print_step "Step 1: Creating intersection data"
        create_intersection_data "$OUTPUT_DIR"
    fi
    
    # Step 2: Train models
    if [ "$TRAIN_REGULAR" = true ] || [ "$TRAIN_INTERSECTION" = true ]; then
        print_step "Step 2: Training models"
        
        if [ "$TRAIN_REGULAR" = true ]; then
            train_models "$OUTPUT_DIR" "$TRAIN_DATA" ""
        fi
        
        if [ "$TRAIN_INTERSECTION" = true ]; then
            train_models "$OUTPUT_DIR" "$OUTPUT_DIR/intersection_data.csv" "_intersection"
        fi
    fi
    
    # Step 3: Find best model
    if [ "$USE_BEST_MODEL" = true ]; then
        print_step "Step 3: Finding best model"
        BEST_MODEL_DIR=$(find_best_model "$OUTPUT_DIR")
        echo "Best model: $BEST_MODEL_DIR"
    fi
    
    # Step 4: Run predictions
    if [ "$PREDICT_TEST" = true ] || [ "$PREDICT_VIRTUAL" = true ]; then
        print_step "Step 4: Running predictions"
        
        if [ -n "$BEST_MODEL_DIR" ]; then
            run_predictions "$OUTPUT_DIR" "$BEST_MODEL_DIR"
        else
            # Use first available model
            MODEL_DIR="$OUTPUT_DIR/xgboost_intersection"
            if [ -d "$MODEL_DIR" ]; then
                run_predictions "$OUTPUT_DIR" "$MODEL_DIR"
            fi
        fi
    fi
    
    # Step 5: Generate comparison
    if [ "$GENERATE_COMPARISON" = true ]; then
        print_step "Step 5: Generating model comparison"
        generate_comparison "$OUTPUT_DIR"
    fi
    
    # Step 6: Generate final report
    print_step "Step 6: Generating final report"
    generate_report "$OUTPUT_DIR"
    
    # Summary
    print_header "PIPELINE COMPLETED"
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Key files:"
    [ -f "$OUTPUT_DIR/model_comparison.csv" ] && echo "  • Model comparison: model_comparison.csv"
    [ -f "$OUTPUT_DIR/best_models.csv" ] && echo "  • Best models: best_models.csv"
    [ -f "$OUTPUT_DIR/test_predictions.csv" ] && echo "  • Test predictions: test_predictions.csv"
    [ -f "$OUTPUT_DIR/virtual_predictions.csv" ] && echo "  • Virtual predictions: virtual_predictions.csv"
    [ -f "$OUTPUT_DIR/final_report.json" ] && echo "  • Final report: final_report.json"
    echo ""
}

# Run main function
main "$@"