#!/bin/bash

# 批量预测脚本
# 使用项目中的所有模型对测试数据进行预测

# 设置默认值
PROJECT_NAME=${1:-"TestPaperComparison"}
TEST_DATA=${2:-"data/Database_ours_0903update_normalized.csv"}
MODE=${3:-"best"}  # all, best, ensemble

echo "=========================================="
echo "批量预测脚本"
echo "=========================================="
echo "项目: $PROJECT_NAME"
echo "测试数据: $TEST_DATA"
echo "预测模式: $MODE"
echo ""

# 检查项目是否存在
if [ ! -d "$PROJECT_NAME" ]; then
    echo "❌ 项目目录不存在: $PROJECT_NAME"
    exit 1
fi

# 检查测试数据是否存在
if [ ! -f "$TEST_DATA" ]; then
    echo "❌ 测试数据不存在: $TEST_DATA"
    exit 1
fi

# 显示项目信息
echo "📊 项目信息:"
python automl.py project info project=$PROJECT_NAME

echo ""
echo "🚀 开始批量预测..."
echo ""

# 根据模式执行预测
case $MODE in
    all)
        echo "使用所有模型进行预测..."
        python automl.py project predict \
            project=$PROJECT_NAME \
            data=$TEST_DATA \
            mode=all \
            output=$PROJECT_NAME/batch_predictions
        ;;
    
    best)
        echo "使用最佳模型进行预测..."
        python automl.py project predict \
            project=$PROJECT_NAME \
            data=$TEST_DATA \
            mode=best \
            output=$PROJECT_NAME/best_predictions.csv
        ;;
    
    ensemble)
        echo "使用集成方法进行预测..."
        python automl.py project predict \
            project=$PROJECT_NAME \
            data=$TEST_DATA \
            mode=ensemble \
            method=weighted \
            output=$PROJECT_NAME/ensemble_predictions.csv
        ;;
    
    *)
        echo "❌ 未知模式: $MODE"
        echo "   可用模式: all, best, ensemble"
        exit 1
        ;;
esac

echo ""
echo "✅ 批量预测完成!"
echo ""

# 显示结果位置
echo "📁 结果位置:"
if [ "$MODE" = "all" ]; then
    echo "   $PROJECT_NAME/batch_predictions/"
    ls -la $PROJECT_NAME/batch_predictions/*.csv 2>/dev/null | head -5
elif [ "$MODE" = "best" ]; then
    echo "   $PROJECT_NAME/best_predictions.csv"
    if [ -f "$PROJECT_NAME/best_predictions.csv" ]; then
        echo "   文件大小: $(wc -l $PROJECT_NAME/best_predictions.csv | awk '{print $1}') 行"
    fi
elif [ "$MODE" = "ensemble" ]; then
    echo "   $PROJECT_NAME/ensemble_predictions.csv"
    if [ -f "$PROJECT_NAME/ensemble_predictions.csv" ]; then
        echo "   文件大小: $(wc -l $PROJECT_NAME/ensemble_predictions.csv | awk '{print $1}') 行"
    fi
fi

echo ""
echo "=========================================="
echo "批量预测完成"
echo "=========================================="