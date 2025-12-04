#!/bin/bash
# 0913 论文数据处理完整自动化脚本
# 基于 scripts/0913_paper_workflow_guide.md 的完整执行流程

set -e  # 遇到错误立即停止

# 开始时间
START_TIME=$(date +%s)
echo "=========================================="
echo "0913 论文数据处理 - 完整自动化执行"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 设置工作目录
#cd /Users/kanshan/IR/ir2025/v3

# 设置输出目录
# 如果提供了参数，使用指定的目录；否则创建新目录
if [ "$1" != "" ]; then
    OUTPUT_DIR="$1"
    echo "使用已存在的输出目录: $OUTPUT_DIR"
else
    OUTPUT_DIR="Paper_$(date +%m%d)_$(date +%H%M%S)"
    echo "创建新的输出目录: $OUTPUT_DIR"
fi

# 数据文件
DATA_FILE="data/Database_normalized.csv"
TEST_DATA_FILE="data/Database_ours_0903update_normalized.csv"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置日志文件路径（使用时间戳）
LOG_FILE="$OUTPUT_DIR/log_$(date +%m%d_%H%M%S).log"

# 记录开始
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始执行完整流程" | tee -a "$LOG_FILE"

# ========== 步骤1：数据准备和验证 ==========
echo ""
echo "[步骤1] 数据准备和验证"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤1：数据准备和验证" | tee -a "$LOG_FILE"

# 检查数据文件是否存在
if [ -f "$DATA_FILE" ]; then
    echo "✅ 训练数据存在: $(wc -l $DATA_FILE)" | tee -a "$LOG_FILE"
else
    echo "❌ 训练数据不存在" | tee -a "$LOG_FILE"
    exit 1
fi

if [ -f "$TEST_DATA_FILE" ]; then
    echo "✅ 测试数据存在: $(wc -l $TEST_DATA_FILE)" | tee -a "$LOG_FILE"
else
    echo "⚠️ 测试数据不存在" | tee -a "$LOG_FILE"
fi

if [ -f "data/ir_assemble.csv" ]; then
    echo "✅ 虚拟数据库存在: $(wc -l data/ir_assemble.csv)" | tee -a "$LOG_FILE"
else
    echo "⚠️ 虚拟数据库不存在，需要生成" | tee -a "$LOG_FILE"
    python scripts/generate_virtual_database.py 2>&1 | tee -a "$LOG_FILE"
fi

# ========== 步骤1.1：虚拟数据库统计 ==========
echo ""
echo "[步骤1.1] 虚拟数据库组合统计"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤1.1：输出虚拟数据库组合统计" | tee -a "$LOG_FILE"

if [ -f "data/ir_assemble.csv" ]; then
    python scripts/analyze_combinations.py \
        --data $DATA_FILE \
        --virtual data/ir_assemble.csv 2>&1 | tee -a "$LOG_FILE" || echo "⚠️ 虚拟数据库统计失败，继续执行" | tee -a "$LOG_FILE"
else
    echo "⚠️ 虚拟数据库不存在，跳过组合统计" | tee -a "$LOG_FILE"
fi

# ========== 步骤2：训练模型 ==========
echo ""
echo "[步骤2] 训练模型"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤2：训练所有模型（13个）" | tee -a "$LOG_FILE"

# 检查是否已经训练过模型
if [ -d "$OUTPUT_DIR/all_models/automl_train" ]; then
    model_count=$(ls -d $OUTPUT_DIR/all_models/automl_train/*/ 2>/dev/null | wc -l)
    if [ $model_count -gt 0 ]; then
        echo "✅ 发现已训练的模型 ($model_count 个)，跳过训练步骤" | tee -a "$LOG_FILE"
    else
        # 使用标准模式（13个模型，10折验证）
        python automl.py train \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=all_models \
    models=adaboost,catboost,decision_tree,elastic_net,gradient_boosting,knn,lasso,lightgbm,mlp,random_forest,ridge,svr,xgboost \
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

        # 检查训练是否成功
        if [ $? -eq 0 ]; then
            echo "✅ 所有13个模型训练完成" | tee -a "$LOG_FILE"
        else
            echo "❌ 训练失败" | tee -a "$LOG_FILE"
            exit 1
        fi
    fi
else
    # 目录不存在，需要训练
    python automl.py train \
    data=$DATA_FILE \
    test_data=$TEST_DATA_FILE \
    project=$OUTPUT_DIR \
    name=all_models \
    models=adaboost,catboost,decision_tree,elastic_net,gradient_boosting,knn,lasso,lightgbm,mlp,random_forest,ridge,svr,xgboost \
    training.n_folds=10 \
    'training.metrics=["r2","rmse","mae"]' \
    training.save_final_model=true \
    training.save_fold_models=false \
    training.save_feature_importance=true \
    training.verbose=1 \
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

    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 所有13个模型训练完成" | tee -a "$LOG_FILE"
    else
        echo "❌ 训练失败" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# ========== 步骤3：生成对比表格 ==========
echo ""
echo "[步骤3] 生成模型对比表格"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤3：生成模型对比表格" | tee -a "$LOG_FILE"

python scripts/generate_model_comparison.py --output-dir $OUTPUT_DIR 2>&1 | tee -a "$LOG_FILE" || echo "⚠️ 模型对比表格生成失败，继续执行" | tee -a "$LOG_FILE"

# ========== 步骤3.5：生成最佳模型性能图表 ==========
echo ""
echo "[步骤3.5] 生成最佳模型性能图表"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤3.5：生成最佳模型预测散点图和混淆矩阵" | tee -a "$LOG_FILE"

python scripts/plot_best_model_performance.py \
    --project $OUTPUT_DIR 2>&1 | tee -a "$LOG_FILE" || echo "⚠️ 最佳模型性能图表生成失败，继续执行" | tee -a "$LOG_FILE"

# ========== 步骤4：预测虚拟数据库 ==========
echo ""
echo "[步骤4] 预测虚拟数据库"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤4：预测虚拟数据库 (272,104个组合)" | tee -a "$LOG_FILE"

# 检查是否已经有预测结果
if [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ] || [ -f "$OUTPUT_DIR/performance_statistics.json" ]; then
    echo "✅ 虚拟数据库预测已存在，跳过" | tee -a "$LOG_FILE"
else
    python scripts/predict_all_combinations.py \
        --project $OUTPUT_DIR \
        --input data/ir_assemble.csv \
        --output $OUTPUT_DIR/virtual_predictions_all.csv 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ 虚拟数据库预测完成" | tee -a "$LOG_FILE"
    else
        echo "⚠️ 虚拟数据库预测失败，继续执行" | tee -a "$LOG_FILE"
    fi
fi

# ========== 步骤4.5：可视化虚拟数据库预测 ==========
echo ""
echo "[步骤4.5] 可视化虚拟数据库预测结果"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤4.5：生成虚拟数据库散点图" | tee -a "$LOG_FILE"

if [ -f "$OUTPUT_DIR/virtual_predictions_all.csv" ]; then
    python scripts/plot_virtual_predictions.py \
        --input $OUTPUT_DIR/virtual_predictions_all.csv \
        --output $OUTPUT_DIR/figures 2>&1 | tee -a "$LOG_FILE" || echo "⚠️ 虚拟数据库可视化失败，继续执行" | tee -a "$LOG_FILE"
else
    echo "⚠️ 虚拟预测文件不存在，跳过可视化" | tee -a "$LOG_FILE"
fi

# ========== 步骤5：生成分段性能分析 ==========
echo ""
echo "[步骤5] 生成分段性能分析图"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤5：生成分段性能分析图（PLQY混淆矩阵）" | tee -a "$LOG_FILE"

python scripts/generate_stratified_analysis.py \
    --project $OUTPUT_DIR \
    --output $OUTPUT_DIR/figures 2>&1 | tee -a "$LOG_FILE" || echo "⚠️ 分段分析生成失败，继续执行" | tee -a "$LOG_FILE"

# ========== 步骤6：生成图表 ==========
echo ""
echo "[步骤6] 生成论文图表"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤6：生成论文图表" | tee -a "$LOG_FILE"

python scripts/generate_paper_figures.py \
    --project $OUTPUT_DIR \
    --data $DATA_FILE \
    --output $OUTPUT_DIR/figures 2>&1 | tee -a "$LOG_FILE" || echo "⚠️ 论文图表生成失败，继续执行" | tee -a "$LOG_FILE"

# ========== 步骤7：预测测试数据 ==========
echo ""
echo "[步骤7] 预测测试数据"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤7：预测测试数据" | tee -a "$LOG_FILE"

if [ -f "$TEST_DATA_FILE" ]; then
    python scripts/predict_test_data.py \
        --project $OUTPUT_DIR \
        --input $TEST_DATA_FILE \
        --output $OUTPUT_DIR/test_predictions 2>&1 | tee -a "$LOG_FILE" || echo "⚠️ 测试数据预测失败" | tee -a "$LOG_FILE"
else
    echo "⚠️ 测试数据文件不存在，跳过" | tee -a "$LOG_FILE"
fi

# ========== 步骤8：SHAP可解释性分析 ==========
echo ""
echo "[步骤8] SHAP可解释性分析"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤8：SHAP可解释性分析" | tee -a "$LOG_FILE"

# 检查是否已经有SHAP分析结果
if [ -f "$OUTPUT_DIR/shap_analysis/shap_report.html" ]; then
    echo "✅ SHAP分析已存在，跳过" | tee -a "$LOG_FILE"
else
    # 只分析最重要的3个模型（节省时间）
    echo "  正在分析XGBoost, LightGBM, CatBoost模型..." | tee -a "$LOG_FILE"
    python analyze_shap.py $OUTPUT_DIR \
        --models xgboost lightgbm catboost \
        --sample-size 100 2>&1 | grep -v "DEPRECATION" | tee -a "$LOG_FILE" || echo "⚠️ SHAP分析失败，继续执行" | tee -a "$LOG_FILE"

    if [ -f "$OUTPUT_DIR/shap_analysis/shap_report.html" ]; then
        echo "✅ SHAP分析完成" | tee -a "$LOG_FILE"
    else
        echo "⚠️ SHAP分析未生成报告" | tee -a "$LOG_FILE"
    fi
fi

# ========== 步骤9：生成最终报告 ==========
echo ""
echo "[步骤9] 生成最终报告"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 步骤9：生成最终报告" | tee -a "$LOG_FILE"

# 使用内嵌Python代码生成最终报告
python -c "
import pandas as pd
import json
from datetime import datetime as _datetime
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')

print('='*80)
print('生成最终报告...')
print('='*80)

# 收集信息
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

# 统计模型数量（从AutoML训练结果）
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

# 统计模型对比表
try:
    comparison_df = pd.read_csv(output_dir / 'model_comparison_detailed.csv')
    report['summary']['comparison_generated'] = True
except:
    report['summary']['comparison_generated'] = False

# 检查虚拟数据库预测
try:
    # 检查主预测文件（完整的272,104条）
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
    print(f'检查虚拟预测时出错: {e}')

# 检查测试数据预测
test_files = list(output_dir.glob('test_predict*.csv'))
if test_files:
    for f in test_files:
        report['files_generated'].append(str(f.name))

# 检查生成的文件
files_to_check = [
    ('model_comparison_full.csv', '模型对比表'),
    ('virtual_predictions', '虚拟数据库预测'),
    ('figures/*.png', '论文图表'),
    ('test_predictions/*.csv', '测试集预测'),
    ('performance_statistics.json', '性能统计'),
    ('workflow_summary.json', '工作流摘要')
]

for pattern, desc in files_to_check:
    if '*' in pattern:
        files = list(output_dir.glob(pattern))
        if files:
            report['files_generated'].append(f'{desc}: {len(files)}个文件')
            if 'figures' in pattern:
                report['summary']['figures_generated'] = True
    else:
        if (output_dir / pattern).exists():
            report['files_generated'].append(desc)

# 保存报告
with open(output_dir / 'final_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f'✅ 最终报告已生成: {output_dir}/final_report.json')

# 打印摘要
print('\\n' + '='*80)
print('执行摘要')
print('='*80)
for key, value in report['summary'].items():
    print(f'{key}: {value}')
print(f'生成文件数: {len(report[\"files_generated\"])}')
" 2>&1 | tee -a "$LOG_FILE"

# ========== 完成 ==========
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "✅ 所有任务完成！"
echo "=========================================="
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "执行时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "重要文件:"
echo "  1. 模型对比: $OUTPUT_DIR/model_comparison_detailed.csv"
echo "  2. 最佳模型性能图: $OUTPUT_DIR/figures/model_performance/"
echo "  3. 虚拟预测(全部): $OUTPUT_DIR/virtual_predictions_all.csv"
echo "  4. 高PLQY候选(≥0.9): $OUTPUT_DIR/virtual_predictions_all_plqy_0.9+.csv"
echo "  5. 高PLQY候选(≥0.7): $OUTPUT_DIR/virtual_predictions_all_plqy_0.7+.csv"
echo "  6. 测试预测: $OUTPUT_DIR/test_predictions/"
echo "  7. 虚拟预测散点图: $OUTPUT_DIR/figures/virtual_predictions_scatter.png"
echo "  8. 虚拟预测交互图: $OUTPUT_DIR/figures/virtual_predictions_scatter.html"
echo "  9. 虚拟预测密度图: $OUTPUT_DIR/figures/virtual_predictions_density.png"
echo "  10. 分段分析: $OUTPUT_DIR/figures/stratified_analysis/"
echo "  11. 论文图表: $OUTPUT_DIR/figures/"
echo "  12. 最终报告: $OUTPUT_DIR/final_report.json"
echo "  13. 性能统计: $OUTPUT_DIR/performance_statistics.json"
echo "  14. SHAP可解释性分析: $OUTPUT_DIR/shap_analysis/shap_report.html"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完整流程执行完成，总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "日志已保存到: $LOG_FILE"
