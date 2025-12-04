# SHAP分析已集成到0913.sh 🎉

## ✅ 集成状态

### 1. **依赖安装** - 已包含
```bash
# uv.sh 第65行
uv pip install optuna shap  ✅
```

### 2. **自动化流程** - 已集成
```bash
# 0913.sh 新增步骤8：SHAP可解释性分析
./0913.sh  # 会自动运行SHAP分析
```

### 3. **分析文件** - 已部署
- `analyze_shap.py` - SHAP分析主脚本
- `SHAP_README.md` - 详细使用文档

---

## 🚀 使用方式

### **方式1：全自动流程（推荐）**
```bash
# 一键运行完整流程（包括SHAP分析）
./uv.sh

# 或者如果环境已配置
./0913.sh
```

**流程说明：**
```
1. 数据准备 ✓
2. 训练13个模型 ✓
3. 生成对比表格 ✓
4. 虚拟数据库预测 ✓
5. 生成论文图表 ✓
6. 测试数据预测 ✓
7. 虚拟数据库可视化 ✓
8. 🆕 SHAP可解释性分析 ← 新增！
   └─ 分析 XGBoost, LightGBM, CatBoost
9. 生成最终报告 ✓
```

### **方式2：手动运行SHAP（补充分析）**
```bash
# 对已有结果进行SHAP分析
python analyze_shap.py Paper_0930_222051

# 只分析特定模型
python analyze_shap.py Paper_0930_222051 --models xgboost

# 使用更大样本（更准确）
python analyze_shap.py Paper_0930_222051 --sample-size 200
```

### **方式3：对现有结果补充SHAP**
```bash
# 如果已经运行过0913.sh，单独运行SHAP
python analyze_shap.py Paper_0930_222051 --models xgboost lightgbm catboost
```

---

## 📊 输出结果

### 完整流程后的目录结构：
```
Paper_0930_222051/
├── all_models/                    # 训练的13个模型
├── figures/                       # 论文图表
│   ├── virtual_predictions_scatter.png
│   ├── model_performance/
│   └── stratified_analysis/
├── shap_analysis/                 # 🆕 SHAP分析结果
│   ├── shap_report.html          # 📊 主报告（浏览器打开）
│   ├── xgboost/
│   │   ├── PLQY/
│   │   │   ├── shap_feature_importance.csv
│   │   │   ├── feature_importance_bar.png
│   │   │   └── shap_summary_plot.png
│   │   └── Wavelength/
│   │       └── (同上)
│   ├── lightgbm/
│   │   └── ...
│   └── catboost/
│       └── ...
├── virtual_predictions_all.csv
├── model_comparison_detailed.csv
├── final_report.json
└── log_XXXX.log
```

---

## ⚙️ 自动化配置说明

### 0913.sh 中的SHAP配置
```bash
# 步骤8：SHAP可解释性分析（约5-10分钟）
python analyze_shap.py $OUTPUT_DIR \
    --models xgboost lightgbm catboost \  # 只分析Top 3模型
    --sample-size 100                      # 100个样本（平衡速度与精度）
```

### 为什么只分析3个模型？
- **XGBoost**: R² = 0.84 (波长), 0.56 (PLQY) - 最佳性能
- **LightGBM**: R² = 0.82 (波长), 0.54 (PLQY) - 第二名
- **CatBoost**: R² = 0.77 (波长), 0.48 (PLQY) - 第三名

其他10个模型性能较低，分析意义不大。

### 如果想分析所有模型？
编辑0913.sh第303行：
```bash
# 改为
python analyze_shap.py $OUTPUT_DIR --sample-size 100
# （不指定--models参数会分析所有13个模型，耗时约30-45分钟）
```

---

## 🎯 智能跳过机制

### 重复运行保护
```bash
./0913.sh Paper_0930_222051  # 对现有结果目录运行
```

**行为：**
1. ✅ 检测到SHAP报告已存在 → 跳过SHAP分析
2. ✅ 检测到模型已训练 → 跳过训练步骤
3. ✅ 检测到预测已完成 → 跳过预测步骤
4. 🔄 只生成缺失的部分

### 强制重新分析
```bash
# 删除SHAP结果后重新运行
rm -rf Paper_0930_222051/shap_analysis
./0913.sh Paper_0930_222051
```

---

## 📝 论文撰写应用

### Results 部分
```markdown
### Model Interpretability Analysis

To understand the decision-making process of the models, we performed
SHAP (SHapley Additive exPlanations) analysis on the top three models
(XGBoost, LightGBM, and CatBoost).

**Figure X** shows the feature importance ranking for PLQY prediction.
Morgan fingerprint bits [list top bits] and molecular descriptors
[list top descriptors] contributed most significantly to the predictions.
```

### 插入图表
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{shap_analysis/xgboost/PLQY/feature_importance_bar.png}
\caption{SHAP feature importance for PLQY prediction using XGBoost.}
\label{fig:shap_plqy}
\end{figure}
```

### Discussion 部分
```markdown
The SHAP analysis revealed that [specific features] were the most
influential factors in PLQY prediction. This finding aligns with
chemical intuition, as [chemical explanation]. The importance of
[molecular descriptor] suggests that [mechanistic insight].
```

---

## ⏱️ 性能参数调优

### 快速测试（1-2分钟）
```bash
python analyze_shap.py Paper_0930_222051 \
    --models xgboost \
    --sample-size 50
```

### 标准分析（5-10分钟）- 默认配置
```bash
python analyze_shap.py Paper_0930_222051 \
    --models xgboost lightgbm catboost \
    --sample-size 100
```

### 高精度分析（15-20分钟）
```bash
python analyze_shap.py Paper_0930_222051 \
    --models xgboost lightgbm catboost \
    --sample-size 200
```

### 完整分析（30-45分钟）
```bash
python analyze_shap.py Paper_0930_222051 \
    --sample-size 100  # 分析所有13个模型
```

---

## 🔧 故障排除

### 问题1: SHAP分析失败
```bash
# 检查SHAP是否安装
python -c "import shap; print(shap.__version__)"

# 如果未安装
pip install shap
```

### 问题2: 内存不足
```bash
# 减少样本数
python analyze_shap.py Paper_0930_222051 \
    --models xgboost \
    --sample-size 50
```

### 问题3: 特征维度错误
```bash
# 删除特征缓存重新提取
rm -rf file_feature_cache/
python analyze_shap.py Paper_0930_222051 --models xgboost
```

### 问题4: RDKit警告太多
```bash
# 已在0913.sh中过滤（第305行）
2>&1 | grep -v "DEPRECATION"
```

---

## 📚 相关文档

- **SHAP_README.md** - 详细使用说明
- **analyze_shap.py** - 源代码（带详细注释）
- **0913.sh** - 完整自动化流程

---

## ✨ 下一步工作

### 1. 运行完整流程
```bash
./uv.sh  # 包含SHAP分析
```

### 2. 查看SHAP报告
```bash
open Paper_XXXX/shap_analysis/shap_report.html
```

### 3. 论文撰写
- 复制 `feature_importance_bar.png` 到论文
- 引用Top特征进行讨论
- 添加可解释性章节

### 4. （可选）深度分析
```bash
# 对特定模型进行更详细分析
python analyze_shap.py Paper_XXXX --models xgboost --sample-size 200
```

---

**✅ SHAP分析已完全集成，无需手动干预！**

运行 `./uv.sh` 即可获得包含可解释性分析的完整结果。