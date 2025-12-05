#!/bin/bash

# Batch prediction script
# Use all models in a project to predict on test data

# Default values
PROJECT_NAME=${1:-"TestPaperComparison"}
TEST_DATA=${2:-"data/Database_ours_0903update_normalized.csv"}
MODE=${3:-"best"}  # all, best, ensemble

echo "=========================================="
echo "Batch prediction script"
echo "=========================================="
echo "Project: $PROJECT_NAME"
echo "Test data: $TEST_DATA"
echo "Prediction mode: $MODE"
echo ""

# Check project exists
if [ ! -d "$PROJECT_NAME" ]; then
    echo "ERROR: Project directory not found: $PROJECT_NAME"
    exit 1
fi

# Check test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data not found: $TEST_DATA"
    exit 1
fi

echo "Project info:"
python automl.py project info project=$PROJECT_NAME

echo ""
echo "Starting batch prediction..."
echo ""

# Predict according to mode
case $MODE in
    all)
        echo "Predicting with all models..."
        python automl.py project predict \
            project=$PROJECT_NAME \
            data=$TEST_DATA \
            mode=all \
            output=$PROJECT_NAME/batch_predictions
        ;;
    
    best)
        echo "Predicting with best model..."
        python automl.py project predict \
            project=$PROJECT_NAME \
            data=$TEST_DATA \
            mode=best \
            output=$PROJECT_NAME/best_predictions.csv
        ;;
    
    ensemble)
        echo "Predicting with ensemble method..."
        python automl.py project predict \
            project=$PROJECT_NAME \
            data=$TEST_DATA \
            mode=ensemble \
            method=weighted \
            output=$PROJECT_NAME/ensemble_predictions.csv
        ;;
    
    *)
        echo "ERROR: Unknown mode: $MODE"
        echo "       Available modes: all, best, ensemble"
        exit 1
        ;;
esac

echo ""
echo "INFO: Batch prediction completed"
echo ""

# Show result location
echo "Result location:"
if [ "$MODE" = "all" ]; then
    echo "   $PROJECT_NAME/batch_predictions/"
    ls -la $PROJECT_NAME/batch_predictions/*.csv 2>/dev/null | head -5
elif [ "$MODE" = "best" ]; then
    echo "   $PROJECT_NAME/best_predictions.csv"
    if [ -f "$PROJECT_NAME/best_predictions.csv" ]; then
    echo "   Row count: $(wc -l $PROJECT_NAME/best_predictions.csv | awk '{print $1}')"
    fi
elif [ "$MODE" = "ensemble" ]; then
    echo "   $PROJECT_NAME/ensemble_predictions.csv"
    if [ -f "$PROJECT_NAME/ensemble_predictions.csv" ]; then
    echo "   Row count: $(wc -l $PROJECT_NAME/ensemble_predictions.csv | awk '{print $1}')"
    fi
fi

echo ""
echo "=========================================="
echo "Batch prediction finished"
echo "=========================================="
