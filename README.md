## Project Overview

PhosIrDB is a curated dataset of phosphorescent Ir(III) complexes designed to enable end‑to‑end data‑driven discovery. This project provides:
- Automated training across multiple ML models (XGBoost, LightGBM, CatBoost, Random Forest, etc.) with cross‑validation and unified comparison.
- Molecular feature extraction using RDKit (Morgan fingerprints + configurable descriptor sets), supporting multi‑ligand combination (`L1`, `L2`, `L3`) via mean/sum/concat.
- Virtual database assembly and large‑scale prediction to identify high‑PLQY candidates, with figures and summary tables.
- Analysis utilities, including stratified performance analysis and SHAP‑based interpretability for key models.
- Reproducible workflow scripts that produce reports, plots, and exportable results in one run.

## Getting Started

- Recommended OS: Linux or macOS.
- Environment setup: run `uv.sh` to create a Python 3.9 virtual environment and install all dependencies.
- Full workflow: run `run_workflow.sh` to execute end‑to‑end training, evaluation, virtual database prediction, figures, and reports.

## Quick Start

The workflow is designed for one-click execution:

```bash
# The run.sh script handles environment setup (via uv) and workflow execution automatically.
bash run.sh
```

## Recommended Configuration

- Python: `3.9` (managed automatically)
- CPU: `4+ cores` recommended; `8+` for faster virtual predictions
- RAM: `8+ GB` (virtual DB prediction can be memory‑intensive)
- Disk: `>40 GB` free for models, figures and reports
- OS: Linux/macOS

## Dataset

- Training data: `data/PhosIrDB.csv`
- Optional test data: `data/ours.csv`
- Virtual database: `data/ir_assemble.csv`
- Columns: SMILES in `L1`, `L2`, `L3`; targets include `Max_wavelength(nm)` and `PLQY`
- Values are normalized for robust training and fair model comparison

## Workflow Outputs

- Model comparison tables (`model_comparison_*.csv`)
- Virtual predictions (`virtual_predictions_all.csv` and filtered candidates)
- Figures (`figures/`), including performance and virtual DB plots
- Final JSON summary (`final_report.json`)

## Key Contributions & Highlights

This project represents a rigorous engineering effort to bridge machine learning and materials science:

1.  **Industrial-Grade Reproducibility**:
    *   By strictly managing environments (`uv`) and random seeds, we ensure that every result—from model training to figure generation—is fully reproducible on any machine.
    
2.  **Interpretable Discovery**:
    *   We go beyond "black box" metrics by integrating SHAP (SHapley Additive exPlanations), providing chemical insights into *why* a molecule performs well.

3.  **High-Throughput Architecture**:
    *   The system includes an optimized batch predictor capable of screening large-scale virtual libraries, directly facilitating the discovery of novel high-performance candidates.

4.  **Automated Reporting**:
    *   The workflow automatically generates publication-ready figures and LaTeX tables, significantly reducing the overhead of converting raw data into research insights.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{PhosIrDB2025,
  title={PhosIrDB: A Data-Driven Workflow for Discovery of Phosphorescent Iridium(III) Emitters},
  author={Publication Pending},
  journal={Under Review},
  year={2025}
}
```
