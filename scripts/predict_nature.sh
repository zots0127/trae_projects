#!/bin/bash
# Nature 项目批量预测脚本（仅波长和PLQY）
# 用法: ./predict_nature.sh [MODEL_DIR] [DATA_FILE]

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
MODEL_DIR=${1:-"Nature/train1/models"}
DATA_FILE=${2:-"Database_ours_0903update_normalized.csv"}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    Nature 项目批量预测（波长&PLQY）    ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查数据文件
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${YELLOW}⚠️  数据文件不存在: $DATA_FILE${NC}"
    exit 1
fi

# 检查模型目录
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}⚠️  模型目录不存在: $MODEL_DIR${NC}"
    echo "可用的模型目录："
    ls -d Nature/*/models/ 2>/dev/null
    exit 1
fi

echo "📁 使用模型目录: $MODEL_DIR"
echo "📄 使用数据文件: $DATA_FILE"
echo ""

# 获取时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 预测波长
echo -e "${GREEN}[1/2] 预测 Max_wavelength...${NC}"
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
    echo "✅ 波长预测完成: $OUTPUT_FILE"
else
    echo -e "${YELLOW}⚠️  未找到波长模型${NC}"
fi

echo ""

# 预测PLQY
echo -e "${GREEN}[2/2] 预测 PLQY...${NC}"
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
    echo "✅ PLQY预测完成: $OUTPUT_FILE"
else
    echo -e "${YELLOW}⚠️  未找到PLQY模型${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}           批量预测完成！              ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "📊 结果文件："
ls predictions_*_${TIMESTAMP}.csv 2>/dev/null
echo ""
echo "💡 提示："
echo "  • 第二次运行相同文件会使用缓存（速度提升100倍）"
echo "  • 可以使用 python automl.py cache stats 查看缓存"