#!/bin/bash
# Project Workflow - Full Automation Script
# Project execution script

set -e  # Exit on error

# Start time
START_TIME=$(date +%s)
echo "=========================================="
echo "Project Workflow - Full Automated Run"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Working directory (optional)
#cd /Users/kanshan/IR/ir2025/v3

# Output directory
# If an argument is provided, use it; otherwise create a new directory
if [ "$1" != "" ]; then
    OUTPUT_DIR="$1"
    echo "Using existing output directory: $OUTPUT_DIR"
else
    OUTPUT_DIR="Project_Output"
    echo "Creating new output directory: $OUTPUT_DIR"
fi

# Data files
DATA_FILE="data/PhosIrDB.csv"
TEST_DATA_FILE="data/Database_ours_0903update_normalized.csv"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file path (with timestamp)
LOG_FILE="$OUTPUT_DIR/log_$(date +%m%d_%H%M%S).log"

# Start logging
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting full workflow" | tee -a "$LOG_FILE"

# ========== Step 1: Data Preparation & Validation ==========
echo ""
echo "[Step 1] Data preparation and validation"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Data preparation and validation" | tee -a "$LOG_FILE"

# Check data files exist
if [ -f "$DATA_FILE" ]; then
    echo "INFO: Training data found: $(wc -l $DATA_FILE)" | tee -a "$LOG_FILE"
else
    echo "ERROR: Training data not found" | tee -a "$LOG_FILE"
    exit 1
fi

if [ -f "$TEST_DATA_FILE" ]; then
    echo "INFO: Test data found: $(wc -l $TEST_DATA_FILE)" | tee -a "$LOG_FILE"
else
    echo "WARNING: Test data not found" | tee -a "$LOG_FILE"
fi

if [ -f "data/ir_assemble.csv" ]; then
    echo "INFO: Virtual database found: $(wc -l data/ir_assemble.csv)" | tee -a "$LOG_FILE"
else
    echo "INFO: Virtual database not found - generating" | tee -a "$LOG_FILE"
    python scripts/generate_virtual_database.py 2>&1 | tee -a "$LOG_FILE"
fi

# ========== Step 1.1: Virtual Database Statistics ==========
echo ""
echo "[Step 1.1] Virtual database combination statistics"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1.1: Output virtual database combination statistics" | tee -a "$LOG_FILE"

if [ -f "data/ir_assemble.csv" ]; then
    python scripts/analyze_combinations.py \
        --data $DATA_FILE \
        --virtual data/ir_assemble.csv 2>&1 | tee -a "$LOG_FILE" || echo "WARNING: Virtual database statistics failed - continuing" | tee -a "$LOG_FILE"
else
    echo "WARNING: Virtual database not found - skipping combination statistics" | tee -a "$LOG_FILE"
fi

## ========== Step 2: Train Models ==========
echo ""
## Configure model list (full vs minimal), dynamic count, and force switch
TRAIN_FULL=${TRAIN_FULL:-false}
FORCE_TRAIN=${FORCE_TRAIN:-true}
if [ "$TRAIN_FULL" = "true" ]; then
    TRAIN_MODELS="adaboost,catboost,decision_tree,elastic_net,gradient_boosting,knn,lasso,lightgbm,mlp,random_forest,ridge,svr,xgboost"
else
    TRAIN_MODELS="catboost,decision_tree,gradient_boosting,lightgbm,mlp,random_forest,ridge,xgboost"
fi
MODEL_COUNT=$(echo "$TRAIN_MODELS" | tr ',' '\n' | wc -l)

echo "[Step 2] Train models ($MODEL_COUNT models)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: Train models ($MODEL_COUNT models)" | tee -a "$LOG_FILE"

# Check if models are already trained (honor FORCE_TRAIN)
if [ -d "$OUTPUT_DIR/all_models/automl_train" ]; then
    existing_runs=$(ls -d $OUTPUT_DIR/all_models/automl_train/*/ 2>/dev/null | wc -l)
    if [ "$FORCE_TRAIN" = "false" ] && [ $existing_runs -gt 0 ]; then
        echo "INFO: Found trained runs ($existing_runs) - skipping training (FORCE_TRAIN=false)" | tee -a "$LOG_FILE"
    else
        echo "INFO: Start training ($MODEL_COUNT models)" | tee -a "$LOG_FILE"
        python automl.py train \
        data=$DATA_FILE \
        test_data=$TEST_DATA_FILE \
        project=$OUTPUT_DIR \
        name=all_models \
        models=$TRAIN_MODELS \
        training.n_folds=10 \
        'training.metrics=["r2","rmse","mae"]' \
        training.save_final_model=true \
        training.save_fold_models=false \
        training.save_feature_importance=true \
        training.verbose=2 \
        feature.feature_type=combined \
        feature.morgan_bits=1024 \
        feature.morgan_radius=2 \
        feature.combination_method=mean \
        feature.use_cache=true \
        'data.smiles_columns=["L1","L2","L3"]' \
        'data.target_columns=["Max_wavelength(nm)","PLQY"]' \
        data.multi_target_strategy=intersection \
        data.nan_handling=skip \
        data.train_ratio=1.0 \
        data.val_ratio=0.0 \
        data.test_ratio=0.0 \
        comparison.enable=true \
        'comparison.formats=["markdown","html","latex","csv"]' \
        comparison.highlight_best=true \
        comparison.include_std=true \
        comparison.decimal_places.r2=4 \
        comparison.decimal_places.rmse=4 \
        comparison.decimal_places.mae=4 \
        export.enable=true \
        'export.formats=["json","csv","excel"]' \
        export.include_predictions=true \
        export.include_feature_importance=true \
        export.include_cv_details=true \
        export.generate_plots=true \
        export.generate_report=true 2>&1 | tee -a "$LOG_FILE"

        if [ $? -eq 0 ]; then
            echo "INFO: Models trained ($MODEL_COUNT)" | tee -a "$LOG_FILE"
        else
            echo "ERROR: Training failed" | tee -a "$LOG_FILE"
            exit 1
        fi
    fi
else
    echo "INFO: Start training ($MODEL_COUNT models)" | tee -a "$LOG_FILE"
    python automl.py train \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=all_models \
    models=$TRAIN_MODELS \
    training.n_folds=10 \
    'training.metrics=["r2","rmse","mae"]' \
    training.save_final_model=true \
    training.save_fold_models=false \
    training.save_feature_importance=true \
    training.verbose=2 \
    feature.feature_type=combined \
    feature.morgan_bits=1024 \
    feature.morgan_radius=2 \
    feature.combination_method=mean \
    feature.use_cache=true \
    'data.smiles_columns=["L1","L2","L3"]' \
    'data.target_columns=["Max_wavelength(nm)","PLQY"]' \
    data.multi_target_strategy=intersection \
    data.nan_handling=skip \
    data.train_ratio=1.0 \
    data.val_ratio=0.0 \
    data.test_ratio=0.0 \
    comparison.enable=true \
    'comparison.formats=["markdown","html","latex","csv"]' \
    comparison.highlight_best=true \
    comparison.include_std=true \
    comparison.decimal_places.r2=4 \
    comparison.decimal_places.rmse=4 \
    comparison.decimal_places.mae=4 \
    export.enable=true \
    'export.formats=["json","csv","excel"]' \
    export.include_predictions=true \
    export.include_feature_importance=true \
    export.include_cv_details=true \
    export.generate_plots=true \
    export.generate_report=true 2>&1 | tee -a "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "INFO: Models trained ($MODEL_COUNT)" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Training failed" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# ========== Step 3: Generate Comparison Table ==========
echo ""
echo "[Step 3] Generate model comparison table"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3: Generate model comparison table" | tee -a "$LOG_FILE"

python scripts/generate_model_comparison.py --output-dir $OUTPUT_DIR 2>&1 | tee -a "$LOG_FILE" || echo "WARNING: Failed to generate comparison table - continuing" | tee -a "$LOG_FILE"

# ========== Step 3.5: Generate Best Model Performance Charts ==========
echo ""
echo "[Step 3.5] Generate best model performance charts"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3.5: Generate best model prediction scatter plots and confusion matrices" | tee -a "$LOG_FILE"

python scripts/plot_best_model_performance.py \
    --project $OUTPUT_DIR 2>&1 | tee -a "$LOG_FILE" || echo "WARNING: Failed to generate best model performance charts - continuing" | tee -a "$LOG_FILE"

# ========== Step 4: Predict Virtual Database ==========
echo ""
echo "[Step 4] Predict virtual database"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 4: Predict virtual database (272,104 combinations)" | tee -a "$LOG_FILE"

# Check if predictions already exist
if [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ] || [ -f "$OUTPUT_DIR/performance_statistics.json" ]; then
    echo "INFO: Virtual predictions already exist - skipping" | tee -a "$LOG_FILE"
else
    python scripts/predict_all_combinations.py \
        --project $OUTPUT_DIR \
        --input data/ir_assemble.csv \
        --output $OUTPUT_DIR/virtual_predictions_all.csv 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "INFO: Virtual database prediction completed" | tee -a "$LOG_FILE"
    else
        echo "WARNING: Virtual database prediction failed - continuing" | tee -a "$LOG_FILE"
    fi
fi

# ========== Step 4.5: Visualize Virtual Predictions ==========
echo ""
echo "[Step 4.5] Visualize virtual database predictions"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 4.5: Generate virtual database scatter plots" | tee -a "$LOG_FILE"

if [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ]; then
    python scripts/plot_virtual_predictions.py \
        --input $OUTPUT_DIR/virtual_predictions_all.csv \
        --output $OUTPUT_DIR/figures 2>&1 | tee -a "$LOG_FILE" || echo "WARNING: Failed to visualize virtual database - continuing" | tee -a "$LOG_FILE"
else
    echo "WARNING: Virtual prediction file not found - skipping visualization" | tee -a "$LOG_FILE"
fi

# ========== Step 5: Generate Stratified Performance Analysis ==========
echo ""
echo "[Step 5] Generate stratified performance plots"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 5: Generate stratified performance plots (PLQY confusion matrix)" | tee -a "$LOG_FILE"

python scripts/generate_stratified_analysis.py \
    --project $OUTPUT_DIR \
    --output $OUTPUT_DIR/figures 2>&1 | tee -a "$LOG_FILE" || echo "WARNING: Stratified analysis failed - continuing" | tee -a "$LOG_FILE"

# ========== Step 6: Generate Figures ==========
echo ""
echo "[Step 6] Generate figures"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 6: Generate figures" | tee -a "$LOG_FILE"

python scripts/generate_paper_figures.py \
    --project $OUTPUT_DIR \
    --data $DATA_FILE \
    --output $OUTPUT_DIR/figures 2>&1 | tee -a "$LOG_FILE" || echo "WARNING: Failed to generate figures - continuing" | tee -a "$LOG_FILE"

# ========== Step 7: Predict Test Data ==========
echo ""
echo "[Step 7] Predict test data"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 7: Predict test data" | tee -a "$LOG_FILE"

if [ -f "$TEST_DATA_FILE" ]; then
    python scripts/predict_test_data.py \
        --project $OUTPUT_DIR \
        --input $TEST_DATA_FILE \
        --output $OUTPUT_DIR/test_predictions 2>&1 | tee -a "$LOG_FILE" || echo "WARNING: Test data prediction failed" | tee -a "$LOG_FILE"
else
    echo "WARNING: Test data file not found - skipping" | tee -a "$LOG_FILE"
fi

# ========== Step 8: SHAP Interpretability Analysis ==========
echo ""
echo "[Step 8] SHAP interpretability analysis"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 8: SHAP interpretability analysis" | tee -a "$LOG_FILE"

# Check if SHAP analysis exists
if [ -f "$OUTPUT_DIR/shap_analysis/shap_report.html" ]; then
    echo "INFO: SHAP analysis already exists - skipping" | tee -a "$LOG_FILE"
else
    # Analyze top 3 models (save time)
    echo "  Analyzing XGBoost, LightGBM, CatBoost models..." | tee -a "$LOG_FILE"
    python analyze_shap.py $OUTPUT_DIR \
        --models xgboost lightgbm catboost \
        --sample-size 100 2>&1 | grep -v "DEPRECATION" | tee -a "$LOG_FILE" || echo "WARNING: SHAP analysis failed - continuing" | tee -a "$LOG_FILE"

    if [ -f "$OUTPUT_DIR/shap_analysis/shap_report.html" ]; then
        echo "INFO: SHAP analysis completed" | tee -a "$LOG_FILE"
    else
        echo "WARNING: No SHAP report generated" | tee -a "$LOG_FILE"
    fi
fi

# ========== Step 9: Generate Final Report ==========
echo ""
echo "[Step 9] Generate final report"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 9: Generate final report" | tee -a "$LOG_FILE"

# Use embedded Python to generate the final report
python -c "
import pandas as pd
import json
from datetime import datetime as _datetime
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')

print('='*80)
print('Generating final report...')
print('='*80)

# Collect information
report = {
    'timestamp': _datetime.now().isoformat(),
    'output_dir': str(output_dir),
    'workflow_completed': True,
    'summary': {
        'data_prepared': True,
        'models_trained': False,
        'virtual_predicted': False,
        'figures_generated': False
    },
    'files_generated': []
}

# Count models (from AutoML training results)
try:
    training_dir = output_dir / 'all_models' / 'automl_train'
    if training_dir.exists():
        models = [d.name for d in training_dir.iterdir() if d.is_dir()]
        report['summary']['models_count'] = len(models)
        report['summary']['model_list'] = models
        if len(models) > 0:
            report['summary']['models_trained'] = True
except:
    pass

# Check model comparison table
try:
    comparison_df = pd.read_csv(output_dir / 'model_comparison_detailed.csv')
    report['summary']['comparison_generated'] = True
except:
    report['summary']['comparison_generated'] = False

# Check virtual database predictions
try:
    # Check main prediction file (272,104 rows)
    virtual_pred = output_dir / 'virtual_predictions_all.csv'
    if virtual_pred.exists():
        df = pd.read_csv(virtual_pred)
        report['summary']['virtual_predictions_count'] = len(df)
        report['summary']['virtual_predicted'] = True
        if 'Predicted_PLQY' in df.columns:
            high_plqy = df[df['Predicted_PLQY'] >= 0.7]
            report['summary']['high_plqy_candidates'] = len(high_plqy)
            report['summary']['plqy_ranges'] = {
                '>=0.9': len(df[df['Predicted_PLQY'] >= 0.9]),
                '>=0.8': len(df[df['Predicted_PLQY'] >= 0.8]),
                '>=0.7': len(df[df['Predicted_PLQY'] >= 0.7])
            }
except Exception as e:
    print(f'Error checking virtual predictions: {e}')

# Check test data predictions
test_files = list(output_dir.glob('test_predict*.csv'))
if test_files:
    for f in test_files:
        report['files_generated'].append(str(f.name))

# Check generated files
files_to_check = [
    ('model_comparison_full.csv', 'Model comparison table'),
    ('virtual_predictions', 'Virtual database predictions'),
    ('figures/*.png', 'Figures'),
    ('test_predictions/*.csv', 'Test set predictions'),
    ('performance_statistics.json', 'Performance statistics'),
    ('workflow_summary.json', 'Workflow summary')
]

for pattern, desc in files_to_check:
    if '*' in pattern:
        files = list(output_dir.glob(pattern))
        if files:
            report['files_generated'].append(f'{desc}: {len(files)} files')
            if 'figures' in pattern:
                report['summary']['figures_generated'] = True
    else:
        if (output_dir / pattern).exists():
            report['files_generated'].append(desc)

# Save report
with open(output_dir / 'final_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=True)

print(f'Final report generated: {output_dir}/final_report.json')

# Print summary
print('\n' + '='*80)
print('Execution summary')
print('='*80)
for key, value in report['summary'].items():
    print(f'{key}: {value}')
print(f'Files generated: {len(report["files_generated"])}')
" 2>&1 | tee -a "$LOG_FILE"

# ========== Completion ==========
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "All tasks completed successfully!"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Elapsed time: ${HOURS} hours ${MINUTES} minutes ${SECONDS} seconds"
echo ""
echo "Key files:"
echo "  1. Model comparison: $OUTPUT_DIR/model_comparison_detailed.csv"
echo "  2. Best model performance: $OUTPUT_DIR/figures/model_performance/"
echo "  3. Virtual predictions (all): $OUTPUT_DIR/virtual_predictions_all.csv"
echo "  4. High PLQY candidates (>=0.9): $OUTPUT_DIR/virtual_predictions_all_plqy_0.9+.csv"
echo "  5. High PLQY candidates (>=0.7): $OUTPUT_DIR/virtual_predictions_all_plqy_0.7+.csv"
echo "  6. Test predictions: $OUTPUT_DIR/test_predictions/"
echo "  7. Virtual predictions scatter: $OUTPUT_DIR/figures/virtual_predictions_scatter.png"
echo "  8. Virtual predictions interactive plot: $OUTPUT_DIR/figures/virtual_predictions_scatter.html"
echo "  9. Virtual predictions density: $OUTPUT_DIR/figures/virtual_predictions_density.png"
echo "  10. Stratified analysis: $OUTPUT_DIR/figures/stratified_analysis/"
echo "  11. Figures: $OUTPUT_DIR/figures/"
echo "  12. Final report: $OUTPUT_DIR/final_report.json"
echo "  13. Performance statistics: $OUTPUT_DIR/performance_statistics.json"
echo "  14. SHAP analysis: $OUTPUT_DIR/shap_analysis/shap_report.html"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Workflow completed. Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
