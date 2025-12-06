#!/bin/bash
# Unified Nature workflow - single entry for training, evaluation, prediction
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
START_TIME=$(date +%s)

# -----------------------------
# User-tunable env vars (defaults)
# -----------------------------
OUTPUT_DIR="${OUTPUT_DIR:-Project_Output}"
DATA_FILE="${DATA_FILE:-data/PhosIrDB.csv}"
TEST_DATA_FILE="${TEST_DATA_FILE:-data/ours.csv}"
VIRTUAL_FILE="${VIRTUAL_FILE:-data/ir_assemble.csv}"

TRAIN_MODELS_DEFAULT="catboost,lightgbm,xgboost,random_forest"
TRAIN_MODELS_FULL="adaboost,catboost,decision_tree,elastic_net,gradient_boosting,knn,lasso,lightgbm,mlp,random_forest,ridge,svr,xgboost"
TRAIN_MODELS="${TRAIN_MODELS:-$TRAIN_MODELS_DEFAULT}"
TRAIN_FULL="${TRAIN_FULL:-0}"
TRAIN_FOLDS="${TRAIN_FOLDS:-10}"
FORCE_TRAIN="${FORCE_TRAIN:-1}"     # 1: always train; 0: reuse existing

SKIP_VIRTUAL="${SKIP_VIRTUAL:-0}"
SKIP_SHAP="${SKIP_SHAP:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

LOG_FILE="$OUTPUT_DIR/log_$(date +%m%d_%H%M%S).log"

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Nature unified workflow"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# -----------------------------
# Step 0: Data checks
# -----------------------------
if [ ! -f "$DATA_FILE" ]; then
  error "Training data not found: $DATA_FILE"
fi

if [ -f "$TEST_DATA_FILE" ]; then
  info "Test data found: $TEST_DATA_FILE"
else
  warn "Test data not found: $TEST_DATA_FILE (step 7 will be skipped)"
fi

if [ ! -f "$VIRTUAL_FILE" ]; then
  warn "Virtual DB not found ($VIRTUAL_FILE), generating..."
  python "$ROOT_DIR/scripts/generate_virtual_database.py" 2>&1 | tee -a "$LOG_FILE"
fi

# -----------------------------
# Step 1: Virtual DB stats (optional if file exists)
# -----------------------------
if [ -f "$VIRTUAL_FILE" ]; then
  python "$ROOT_DIR/scripts/analyze_combinations.py" \
    --data "$DATA_FILE" \
    --virtual "$VIRTUAL_FILE" 2>&1 | tee -a "$LOG_FILE" || warn "Virtual DB stats failed"
fi

# -----------------------------
# Step 2: Train models
# -----------------------------
if [ "$TRAIN_FULL" = "1" ]; then
  TRAIN_MODELS="$TRAIN_MODELS_FULL"
fi

TRAIN_DIR="$OUTPUT_DIR/all_models/automl_train"
if [ -d "$TRAIN_DIR" ] && [ "$FORCE_TRAIN" = "0" ]; then
  info "Found existing training runs at $TRAIN_DIR (FORCE_TRAIN=0), skipping training."
else
  info "Training models: $TRAIN_MODELS"
  python "$ROOT_DIR/automl.py" train \
    data="$DATA_FILE" \
    test_data="$TEST_DATA_FILE" \
    project="$OUTPUT_DIR" \
    name=all_models \
    models="$TRAIN_MODELS" \
    training.n_folds="$TRAIN_FOLDS" \
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
fi

# -----------------------------
# Step 3: Comparison table
# -----------------------------
python "$ROOT_DIR/scripts/generate_model_comparison.py" --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE" || warn "Comparison table generation failed"

# -----------------------------
# Step 3.5: Best model plots
# -----------------------------
python "$ROOT_DIR/scripts/plot_best_model_performance.py" --project "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE" || warn "Best model plots failed"

# -----------------------------
# Step 4: Virtual predictions
# -----------------------------
if [ "$SKIP_VIRTUAL" = "1" ]; then
  info "SKIP_VIRTUAL=1 set; skipping virtual DB prediction."
else
  if [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ]; then
    info "Virtual predictions already exist, skipping."
  else
    python "$ROOT_DIR/scripts/predict_all_combinations.py" \
      --project "$OUTPUT_DIR" \
      --input "$VIRTUAL_FILE" \
      --output "$OUTPUT_DIR/virtual_predictions_all.csv" 2>&1 | tee -a "$LOG_FILE" || warn "Virtual prediction failed"
  fi
fi

# -----------------------------
# Step 4.5: Virtual prediction plots
# -----------------------------
if [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ]; then
  python "$ROOT_DIR/scripts/plot_virtual_predictions.py" \
    --input "$OUTPUT_DIR/virtual_predictions_all.csv" \
    --output "$OUTPUT_DIR/figures" 2>&1 | tee -a "$LOG_FILE" || warn "Virtual prediction plotting failed"
fi

# -----------------------------
# Step 5: Stratified performance
# -----------------------------
python "$ROOT_DIR/scripts/generate_stratified_analysis.py" \
  --project "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/figures" 2>&1 | tee -a "$LOG_FILE" || warn "Stratified analysis failed"

# -----------------------------
# Step 6: Paper figures
# -----------------------------
if [ "$SKIP_FIGURES" = "1" ]; then
  info "SKIP_FIGURES=1 set; skipping figure generation."
else
  python "$ROOT_DIR/scripts/generate_paper_figures.py" \
    --project "$OUTPUT_DIR" \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR/figures" 2>&1 | tee -a "$LOG_FILE" || warn "Paper figures generation failed"
fi

# -----------------------------
# Step 7: Predict test data
# -----------------------------
if [ -f "$TEST_DATA_FILE" ]; then
  python "$ROOT_DIR/scripts/predict_test_data.py" \
    --project "$OUTPUT_DIR" \
    --input "$TEST_DATA_FILE" \
    --output "$OUTPUT_DIR/test_predictions" 2>&1 | tee -a "$LOG_FILE" || warn "Test data prediction failed"
fi

# -----------------------------
# Step 8: SHAP analysis
# -----------------------------
if [ "$SKIP_SHAP" = "1" ]; then
  info "SKIP_SHAP=1 set; skipping SHAP."
else
  if [ -f "$OUTPUT_DIR/shap_analysis/shap_report.html" ]; then
    info "SHAP report exists; skipping."
  else
    python "$ROOT_DIR/analyze_shap.py" "$OUTPUT_DIR" \
      --models xgboost lightgbm catboost \
      --sample-size 100 2>&1 | grep -v "DEPRECATION" | tee -a "$LOG_FILE" || warn "SHAP analysis failed"
  fi
fi

# -----------------------------
# Step 9: Final report summary
# -----------------------------
python - "$OUTPUT_DIR" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
out_dir = Path(__import__("sys").argv[1])
report = {
    "timestamp": dt.now().isoformat(),
    "output_dir": str(out_dir),
    "files": [],
    "summary": {}
}

def exists(name):
    p = out_dir / name
    if p.exists():
        report["files"].append(str(p))
        return True
    return False

report["summary"]["training_dir"] = str(out_dir / "all_models" / "automl_train")
report["summary"]["comparison"] = exists("model_comparison_detailed.csv")
report["summary"]["virtual_predictions"] = exists("virtual_predictions_all.csv")
report["summary"]["final_report"] = exists("final_report.json")

if (out_dir / "virtual_predictions_all.csv").exists():
    try:
        df = pd.read_csv(out_dir / "virtual_predictions_all.csv")
        report["summary"]["virtual_rows"] = len(df)
        if "Predicted_PLQY" in df.columns:
            report["summary"]["plqy_ge_0.9"] = int((df["Predicted_PLQY"] >= 0.9).sum())
            report["summary"]["plqy_ge_0.8"] = int((df["Predicted_PLQY"] >= 0.8).sum())
    except Exception as e:
        report["summary"]["virtual_error"] = str(e)

with open(out_dir / "workflow_summary.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=True)

print("Workflow summary saved:", out_dir / "workflow_summary.json")
PY

# -----------------------------
# Done
# -----------------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
printf "\n==========================================\n"
printf "Workflow completed in %02dh:%02dm:%02ds\n" $((DURATION/3600)) $(((DURATION%3600)/60)) $((DURATION%60))
printf "Output directory: %s\n" "$OUTPUT_DIR"
printf "Log file: %s\n" "$LOG_FILE"
printf "==========================================\n\n"

