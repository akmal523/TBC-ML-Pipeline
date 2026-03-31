# TBC-ML-Pipeline

Machine learning pipeline for predicting the effective thermal conductivity of Thermal Barrier Coating (TBC) systems composed of Yttria-Stabilized Zirconia (YSZ) over a CMSX-4 single-crystal nickel superalloy substrate.

---

## Overview

The pipeline consists of three sequential stages:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `generate_cards.py` | Generates randomized YSZ material cards as ABAQUS `.inp` files |
| 2 | `abaqus_ml_pipeline.py` | Runs all simulations in ABAQUS, extracts thermal results from `.odb` files |
| 3 | `train_model.py` | Trains a Random Forest model to predict effective thermal conductivity |

---
```text
TBC-ML-Pipeline/
├── run_pipeline.py                # Main orchestrator; handles environmental auto-detection
├── src/                           # Logic layer (The creative intellect of the project)
│   ├── generate_cards.py          # Stage 1: Randomized ABAQUS .inp generation
│   ├── abaqus_ml_pipeline.py      # Stage 2: Simulation execution and ODB data parsing
│   ├── train_model.py             # Stage 3: Random Forest training and evaluation
│   ├── comp.py                    # Mathematical comparison and validation logic
│   └── tbc_ml_plots.py            # Logic for generating analytical visualizations
├── data/                          # Fixed input parameters
│   └── materials.json             # Reference physical properties for YSZ and CMSX-4
├── results/                       # Artifact layer (Transient outputs of the logic)
│   ├── dataset.json               # Primary simulation database (512 samples)
│   ├── ml_results.json            # Model metrics and feature importance weights
│   └── ml_results_description.txt # Textual interpretation of simulation outcomes
├── requirements.txt               # Dependency manifest
├── .gitignore                     # Rules to exclude machine-specific/binary bloat
└── README.md                      # Project documentation and usage instructions
```
---

## Material System

### YSZ (Yttria-Stabilized Zirconia) — Thermal Barrier Coating
- Thickness: 0.5, 1.0, 1.5, 2.0 mm
- Properties randomized ±20% per variant:
  - Thermal conductivity (dense or porous curves)
  - Elastic modulus and Poisson's ratio (orientations: 100, 110, 111, 113)
  - Specific heat capacity
  - Density

### CMSX-4 — Single-Crystal Nickel Superalloy Substrate
- Fixed thickness: 10.0 mm
- Temperature-dependent properties: conductivity, specific heat, density, elastic constants (orientations: 001, 101, 111)
- No randomization applied

---

## Simulation Setup

- Solver: ABAQUS (steady-state heat transfer)
- Element type: DC2D4 (2D heat transfer quadrilateral)
- Boundary conditions:
  - Hot side (YSZ bottom): **T = 1400 K**
  - Cold side (CMSX-4 top): **T = 600 K**
- Effective thermal conductivity computed as:

```
k_eff = (q * L_total) / ΔT
```

where `q` is average heat flux [W/mm²], `L_total` is total coating thickness [mm], and `ΔT = 800 K`.

---

## Dataset

- Total variants generated: **512** (4 thicknesses × 128 random samples)
- Successful simulations: **512 / 512**
- Training set: 412 samples
- Test set: 100 samples

### ML Features

| Feature | Description |
|---------|-------------|
| `ysz_thickness_mm` | YSZ layer thickness |
| `ysz_density` | YSZ density [tonne/mm³] |
| `ysz_k_avg` | Average YSZ thermal conductivity [W/mm·K] |
| `ysz_cp_avg` | Average YSZ specific heat [mJ/tonne·K] |

**Target:** `k_effective` — effective thermal conductivity of the full TBC system [W/mm·K]

---

## Results Summary

### Feature Importance (Random Forest)

| Feature | Importance |
|---------|------------|
| `YSZ k_avg` | **91.3%** |
| `YSZ Thickness` | 8.3% |
| `YSZ Cp_avg` | 0.2% |
| `YSZ Density` | 0.2% |

### Model Performance

| Metric | Training | Test |
|--------|----------|------|
| R² | 0.9984 | 0.9904 |
| MAE | 6.33×10⁻⁵ | 1.57×10⁻⁴ |
| RMSE | 8.60×10⁻⁵ | 2.05×10⁻⁴ |

Full result details in [`results/ml_results_description.txt`](results/ml_results_description.txt).

---

## Usage

### Recommended: single entry point

```bash
python run_pipeline.py
```

`run_pipeline.py` automatically checks whether ABAQUS is installed and routes accordingly:

| Situation | What happens |
|-----------|-------------|
| ABAQUS found on PATH | Full pipeline: generate cards → simulate → train model |
| ABAQUS not found | Stages 1–2 skipped; pre-built `results/dataset.json` is used for training |

No manual configuration required.

---

### Manual stage-by-stage execution

**Stage 1 — Generate ABAQUS input files** *(requires ABAQUS)*

```bash
python src/generate_cards.py data/materials.json out_dir
```

Output: `out_dir/` containing 512 `.inp` files and `variants_manifest.json`.

# Manual stage-by-stage execution

**Stage 2 — Run simulations and extract results**
Output: `results/dataset.json`

**Stage 3 — Train ML model**
Output: `results/ml_results.json`, `results/ml_results.png`

> If running Stage 3 manually without ABAQUS, ensure `results/dataset.json` exists.


---

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list. ABAQUS is required for Stage 2 and is not installable via pip.

---

## Units

All quantities follow the ABAQUS consistent unit system used in this project:

| Quantity | Unit |
|----------|------|
| Length | mm |
| Force | N |
| Mass | tonne |
| Stress | MPa |
| Thermal conductivity | W/(mm·K) |
| Specific heat | mJ/(tonne·K) |
| Temperature | K |

---

## Data Sources

- CMSX-4 thermal conductivity: ResearchGate / Epishin et al.
- CMSX-4 elastic properties: ScienceDirect / Epishin et al.
- YSZ thermal conductivity: ResearchGate / porous and dense ceramics data
- YSZ elastic properties: ScienceDirect
- YSZ density and specific heat: Morgan Technical Ceramics / ENEA
