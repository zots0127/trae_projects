# Bioresorbable, Light-Emitting Triboelectric Implants Enabled by Nano-Engineered Ir(III) Emitters

## Machine-Learning-Assisted Molecular Design Workflow

This repository contains the **computational and machine learning workflow** supporting the research paper *"Bioresorbable, light-emitting triboelectric implants enabled by nano-engineered Ir(III) emitters"*.

While the broader research introduces a new paradigm for transient bioelectronics, this code specifically implements the **data-driven molecular design** component. It enables the discovery of efficient Ir(III) emitters by:
- Automated training of ensemble models (XGBoost, CatBoost, LightGBM) to predict photoluminescence quantum yield (PLQY) and emission wavelength.
- Virtual screening of combinatorial libraries to identify candidates with human-vision-aligned phosphorescence.
- Providing interpretable SHAP analysis to guide molecular engineering.

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

## Computational Highlights

As the auxiliary computational engine for the material discovery process, this codebase emphasizes:

1.  **Reproducible Discovery**:
    *   Strict environment management (`uv`) ensures that the molecular screening results are fully reproducible.
2.  **Interpretable AI**:
    *   Integration of SHAP (SHapley Additive exPlanations) to provide chemical insights into the structure-property relationships of Ir(III) complexes.
3.  **High-Throughput Screening**:
    *   An optimized batch predictor capable of screening large-scale virtual libraries to pinpoint optimal emitter candidates.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{BioresorbableImplants2025,
  title={Bioresorbable, light-emitting triboelectric implants enabled by nano-engineered Ir(III) emitters},
  author={Publication Pending},
  journal={Under Review},
  year={2025}
}
```
