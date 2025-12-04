#!/bin/bash
# 测试论文对比功能的脚本
# 训练所有模型并生成对比表格

echo "============================================================"
echo "          论文级模型对比测试脚本"
echo "============================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 默认参数
DATA_FILE="${1:-../data/Database_normalized.csv}"
TEST_DATA="${2:-Database_ours_0903update_normalized.csv}"
PROJECT_NAME="${3:-PaperComparison}"

# 检查数据文件
echo -e "${BLUE}[1/4] 检查数据文件...${NC}"
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${YELLOW}⚠️  训练数据文件不存在: $DATA_FILE${NC}"
    echo "尝试使用示例数据..."
    DATA_FILE="data/synthetic_molecules.csv"
    if [ ! -f "$DATA_FILE" ]; then
        echo -e "${RED}❌ 没有找到可用的数据文件${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✅ 数据文件: $DATA_FILE${NC}"

# 运行训练
echo ""
echo -e "${BLUE}[2/4] 开始训练所有模型...${NC}"
echo "配置: paper_comparison"
echo "项目: $PROJECT_NAME"
echo ""

# 使用paper_comparison配置训练
python automl.py train \
    config=paper_comparison \
    data="$DATA_FILE" \
    project="$PROJECT_NAME" \
    name=paper_test \
    training.n_folds=5 \
    optimization.automl_models=[xgboost,lightgbm,catboost,random_forest,gradient_boosting]

# 检查训练结果
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 训练失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ 训练完成！${NC}"

# 生成对比表格
echo ""
echo -e "${BLUE}[3/4] 生成对比表格...${NC}"

# 找到最新的训练目录
LATEST_DIR=$(ls -td "$PROJECT_NAME"/paper_test* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo -e "${YELLOW}⚠️  未找到训练结果目录${NC}"
    exit 1
fi

echo "结果目录: $LATEST_DIR"

# 使用Python生成表格
python -c "
import sys
sys.path.append('.')
from utils.comparison_table import ComparisonTableGenerator

# 创建生成器
generator = ComparisonTableGenerator('$LATEST_DIR')

# 导出所有格式
exported = generator.export_all_formats(
    formats=['markdown', 'html', 'latex', 'csv']
)

# 显示最佳模型
print('')
print('='*60)
print('最佳模型总结')
print('='*60)
best_models = generator.get_best_models()
for target, info in best_models.items():
    print(f'{target}:')
    print(f'  最佳模型: {info[\"algorithm\"]}')
    print(f'  R²: {info[\"r2\"]}')
    print(f'  RMSE: {info[\"rmse\"]}')
    print('')
"

# 显示生成的文件
echo ""
echo -e "${BLUE}[4/4] 生成的文件：${NC}"
ls -la "$LATEST_DIR"/comparison_table_* 2>/dev/null

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}                    测试完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "📊 查看结果："
echo "  - Markdown: cat $LATEST_DIR/comparison_table_*.md"
echo "  - HTML: open $LATEST_DIR/comparison_table_*.html"
echo "  - LaTeX: cat $LATEST_DIR/comparison_table_*.tex"
echo "  - CSV: cat $LATEST_DIR/comparison_table_*.csv"
echo ""
echo "💡 提示："
echo "  - 使用完整配置进行生产训练："
echo "    python automl.py train config=paper_comparison data=your_data.csv"
echo "  - 自定义列名："
echo "    python automl.py train config=paper_comparison \\"
echo "      data=data.csv \\"
echo "      smiles_columns=L1,L2,L3 \\"
echo "      targets=wavelength,plqy"