# SHAP可解释性分析使用指南

## 📌 功能说明

这个工具为你的机器学习模型提供**可解释性分析**，帮助理解：
- 哪些分子特征对预测最重要？
- 每个特征如何影响预测结果？
- 模型的决策依据是什么？

## ✅ 优势

### 1. **完全无侵入**
- 不修改任何原有代码
- 作为独立脚本运行
- 直接读取已训练好的模型

### 2. **易于使用**
```bash
# 基本用法 - 分析所有模型
python analyze_shap.py Paper_0930_222051

# 只分析最重要的模型
python analyze_shap.py Paper_0930_222051 --models xgboost lightgbm

# 使用更大样本（更准确但更慢）
python analyze_shap.py Paper_0930_222051 --sample-size 200
```

### 3. **丰富的输出**
- ✅ 特征重要性排名（CSV格式）
- ✅ 可视化图表（PNG格式）
- ✅ 交互式HTML报告
- ✅ 详细元数据（JSON格式）

## 📂 输出结构

```
Paper_0930_222051/shap_analysis/
├── shap_report.html                 # 📊 主报告（浏览器打开）
├── xgboost/
│   ├── PLQY/
│   │   ├── shap_feature_importance.csv      # 完整特征排名
│   │   ├── feature_importance_bar.png        # 条形图
│   │   ├── shap_summary_plot.png            # SHAP摘要图
│   │   └── shap_metadata.json               # 元数据
│   └── Wavelength/
│       └── ...
├── lightgbm/
│   └── ...
└── ...
```

## 🎯 使用场景

### 1. **论文写作**
在Discussion部分添加：
> "通过SHAP分析，我们发现Morgan指纹的第X位和分子描述符Y对PLQY预测贡献最大，表明..."

### 2. **模型诊断**
检查模型是否关注了合理的化学特征：
- ✅ 如果重要特征包含共轭度、HOMO-LUMO gap等 → 模型合理
- ❌ 如果只依赖几个指纹位 → 可能过拟合

### 3. **分子设计指导**
根据Top特征优化候选分子：
- 增强重要特征的表达
- 避免负面影响的结构

## 🔧 参数说明

| 参数 | 说明 | 默认值 | 推荐设置 |
|------|------|--------|----------|
| `paper_dir` | 论文输出目录 | 必填 | `Paper_0930_222051` |
| `--models` | 指定分析的模型 | 全部 | `xgboost lightgbm catboost` |
| `--sample-size` | SHAP计算的样本数 | 100 | 50-200 |

### ⚡ 性能建议

**快速分析**（1-2分钟）：
```bash
python analyze_shap.py Paper_0930_222051 --models xgboost --sample-size 50
```

**标准分析**（5-10分钟）：
```bash
python analyze_shap.py Paper_0930_222051 --models xgboost lightgbm --sample-size 100
```

**深度分析**（15-30分钟）：
```bash
python analyze_shap.py Paper_0930_222051 --sample-size 200
```

## 📊 结果解读

### 特征重要性CSV
```csv
feature,importance
Morgan_512,0.025634
MolLogP,0.018923
NumHDonors,0.012456
...
```

- **importance**: 平均绝对SHAP值，越大表示特征越重要
- **feature**: 特征名称
  - `Morgan_X`: Morgan指纹第X位（分子结构片段）
  - 其他: RDKit分子描述符（如分子量、LogP等）

### SHAP Summary Plot

<details>
<summary>图表说明</summary>

- **横轴**: SHAP值（正值=增加预测，负值=减少预测）
- **纵轴**: 特征排序（上方=最重要）
- **颜色**: 特征值大小（红=高，蓝=低）
- **密度**: 数据点分布

**解读示例**：
- 如果`MolLogP`的红点集中在右侧 → 高LogP增加PLQY
- 如果`NumHDonors`的蓝点集中在右侧 → 低氢键供体数增加PLQY
</details>

## ⚠️ 注意事项

### 1. 计算时间
- 树模型（XGBoost/LightGBM）：快（~1分钟/模型）
- 线性模型（Ridge/Lasso）：中等（~2-3分钟/模型）
- 其他模型（SVR/KNN）：慢（~5-10分钟/模型）

### 2. 内存使用
- 样本数×特征数 ≈ 100 × 2200 = 220K floats ≈ 1.7MB
- 建议内存 >= 4GB

### 3. 特征维度
- Morgan指纹: 2048位
- 分子描述符: ~200个
- 总计: ~2250维

### 4. RDKit警告
忽略以下警告（不影响结果）：
```
DEPRECATION WARNING: please use MorganGenerator
```

## 🚀 快速开始

```bash
# 1. 安装依赖（如果还没安装）
pip install shap

# 2. 运行分析（推荐从最佳模型开始）
python analyze_shap.py Paper_0930_222051 --models xgboost

# 3. 打开报告查看
open Paper_0930_222051/shap_analysis/shap_report.html
```

## 💡 常见问题

**Q: 为什么不分析所有13个模型？**
A: SHAP计算较耗时。建议先分析性能最好的3-5个模型（XGBoost, LightGBM, CatBoost, Random Forest）。

**Q: sample-size应该设多少？**
A:
- 论文用: 100（平衡速度和准确性）
- 快速预览: 50
- 高精度: 200

**Q: 如何引用SHAP方法？**
A:
```
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
Advances in neural information processing systems, 30.
```

**Q: 结果会改变原有模型吗？**
A: **不会！**这是纯分析工具，只读取模型，不修改任何文件。

## 📚 扩展阅读

- [SHAP官方文档](https://shap.readthedocs.io/)
- [SHAP论文](https://arxiv.org/abs/1705.07874)
- [可解释AI综述](https://christophm.github.io/interpretable-ml-book/)

## 🎓 论文应用建议

### Results部分
> "To understand the feature importance, we performed SHAP (SHapley Additive exPlanations) analysis on the best-performing models..."

### 添加图表
- Figure X: Top 20 features by SHAP importance for PLQY prediction
- Figure Y: SHAP summary plot showing feature contributions

### Discussion要点
1. 识别出的关键特征是否符合化学直觉？
2. 不同模型的重要特征是否一致？
3. 能否从重要特征推导分子设计规则？

---

**✨ 祝分析顺利！有问题随时问。**