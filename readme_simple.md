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

## Recommended Configuration

- Python: `3.9` (managed by `uv.sh`)
- CPU: `4+ cores` recommended; `8+` for faster virtual predictions
- RAM: `8+ GB` (virtual DB prediction can be memory‑intensive)
- Disk: `>40 GB` free for models, figures and reports
- OS: Linux/macOS; Windows users are recommended to run under WSL2 for RDKit compatibility

## Dataset

- PhosIrDB packaged as `data/PhosIrDB.csv`
- Columns: SMILES in `L1`, `L2`, `L3`; targets include `Max_wavelength(nm)` and `PLQY`
- Values are normalized for robust training and fair model comparison

## Steps

```bash
# 1) Setup environment
bash uv.sh

# 2) Run the full workflow (outputs to an auto‑named directory)
bash run_workflow.sh
```


## Workflow Outputs

- Model comparison tables (`model_comparison_*.csv`)
- Virtual predictions (`virtual_predictions_all.csv` and filtered candidates)
- Figures (`figures/`), including performance and virtual DB plots
- Final JSON summary (`final_report.json`)
