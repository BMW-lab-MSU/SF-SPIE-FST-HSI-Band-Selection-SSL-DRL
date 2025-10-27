# SF-SPIE-FST-HSI-Band-Selection-SSL-DRL

### Self-Supervised and Reinforcement-Driven Band Selection for UAV-Based Post-Fire Vegetation Analysis

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Paper](https://img.shields.io/badge/Paper-SPIE%202025%20(MLSP)-green.svg)]() -->

---

## Overview

**BurnSSL-DRL** is a label-efficient framework combining **Self-Supervised Learning (SSL)** and **Deep Reinforcement Learning (DRL)** for **spectral band selection** and **3D spectral–spatial classification** on UAV-based **VNIR hyperspectral imagery (HSI)** after prescribed burns.  

Developed under the **SMART FIRES NSF EPSCoR Project (OIA-2242802)** at **Montana State University**, this framework reduces HSI dimensionality while preserving key spectral cues for vegetation recovery analysis.

---

## Key Contributions

- **Self-Supervised Pretraining:** SimCLR-based 3D-CNN learns spectral–spatial features from unlabeled data.  
- **Reinforcement-Driven Band Selection:** A DQN agent prioritizes bands with high discriminative power and low redundancy.  
- **Balanced Learning:** Incorporates SMOTE + class-weighted loss to handle class imbalance.  
- **Comprehensive Evaluation:** Supports KNN, SVM, RF, and 3D-CNN with visualized per-class F1 trends and confusion matrices.  
- **Comparative Study Extension:** Enables SSEP, SRPA, and DRL band-selection comparison for journal-level reproducibility.

---

## Repository Structure

```text
SF-SPIE-FST-HSI-Band-Selection-SSL-DRL/
│
├── SSL/                                   # Self-Supervised Learning (SimCLR3DCNN)
│   └── ssl/
│       ├── SimCLR3DCNN.py
│       ├── train_ssl.py
│       ├── contrastive_loss.py
│       └── dataset.py
│
├── drl_band_selection/                    # Deep Reinforcement Learning (DRL) pipeline
│   ├── agent_dqn.py
│   ├── env_band_selection.py
│   ├── agniNet.py
│   ├── train_drl.py
│   ├── trainBalancedClassifiers.py
│   ├── trainUnbalancedClassifiers.py
│   ├── extract_features.py
│   ├── baseline_vs_smote_comparison.xlsx
│   └── agni_best.pth
│
├── Visuals/                               # Figures, confusion matrices, and CSVs
│   ├── 3dcnn_drl_classification_results.xlsx
│   ├── baseline_vs_smote_comparison.xlsx
│   ├── conf_matrix_*.png
│   ├── macro_f1_vs_topk_all_classifiers.png
│   ├── Per_Class_F1_Trends_All_Classifiers.png
│   └── before_balance_comparison.xlsx
│
├── data/                                  # User-provided hyperspectral cube & ground truth
├── outputs/                               # Band scores, models, and metrics
├── plots/                                 # Python plotting scripts (e.g., codeToFig3.py)
├── scripts/                               # Patch extraction & utilities
└── README.md

```

---

## Environment Setup

### 1. Clone and Create Environment

```bash
git clone https://github.com/BMW-lab-MSU/SF-SPIE-FST-HSI-Band-Selection-SSL-DRL.git
cd SF-SPIE-FST-HSI-Band-Selection-SSL-DRL

conda create -n burnssl-drl python=3.10
conda activate burnssl-drl

pip install -r requirements.txt
```

### 2. Dependencies

```
torch >= 2.0
torchvision
numpy
pandas
scikit-learn
matplotlib
rasterio
spectral
opencv-python
```

---

## Usage

### Step 1 — Self-Supervised Pretraining

```bash
cd SSL/ssl
python train_ssl.py --epochs 100 --batch_size 32
```

---

### Step 2 — DRL Band Selection

```bash
cd ../../drl_band_selection
python train_drl.py --episodes 200 --reward spectral_redundancy
```

**Outputs**
```
outputs/band_scores/drl_scores.npy
Visuals/3dcnn_drl_classification_results.xlsx
```

---

### Step 3 — Train Classifiers

```bash
# Unbalanced
python trainUnbalancedClassifiers.py --topk 30

# Balanced (SMOTE + weighted loss)
python trainBalancedClassifiers.py --topk 30
```

---

### Step 4 — Reproduce SPIE Figures (Fig 1–3)

```bash
cd ../plots
python codeToFig3.py
```

| Figure | Description | Output |
|:-------|:-------------|:--------|
| **Fig 1** | Macro-F1 vs Top-K (all classifiers) | `Visuals/macro_f1_vs_topk_all_classifiers.png` |
| **Fig 2** | Per-class F1 trends | `Visuals/Per_Class_F1_Trends_All_Classifiers.png` |
| **Fig 3** | Confusion matrices | `Visuals/conf_matrix_*.png` |

---

##  Example Results

| Classifier | Balancing | Top-K | Macro-F1 | Weighted-F1 |
|:------------|:-----------|:------|:----------|:-------------|
| KNN | No | 30 | 0.68 | 0.70 |
| SVM | No | 30 | 0.71 | 0.72 |
| RF | No | 40 | 0.74 | 0.75 |
| 3D-CNN | No | 30 | 0.74 | 0.75 |
| **3D-CNN** | **Yes (SMOTE)** | **30** | **0.80** | **0.82** |

> **Observation:** SMOTE balancing enhanced *Grass* and *Soil* F1-scores, while *Tree* class remained stable—indicating robust canopy detection.

---

## Technical Summary

| Component | Description |
|:-----------|:-------------|
| **SSL** | SimCLR 3D-CNN encoder learns spectral–spatial embeddings from unlabeled patches. |
| **DRL** | DQN agent optimizes band selection using reward = classification gain – redundancy. |
| **Classifier** | Random Forest + 3D-CNN evaluate selected bands. |
| **Metrics** | Macro-F1, Weighted-F1, Per-class F1, Confusion Matrix. |
| **Balancing** | SMOTE resampling + class-weighted loss for imbalanced data. |

---

## Citation

```bibtex
@inproceedings{karankot2025burnssldrl,
  title={Hyperspectral band selection via self-supervised and reinforcement learning for prescribed burn impact analysis},
  author={Mahmad Isaq Karankot, Ethan M. Glenn and Bradley Whitaker},
  booktitle={SPIE Future Sensing Technologies},
  year={2025}
}
```

---

## Acknowledgments

This research was conducted at **Montana State University** within the  
**NSF EPSCoR SMART FIRES Project (OIA-2242802)** under the  
**BMW Lab (Burns, Machine Learning & Wildfire)**.  

We acknowledge UAV and field teams at **Lubrecht Experimental Forest** and the collaborating fire-science groups for data support.

---

## Contact

**Author:** Mahmad Isaq Karankot  
**Email:** mahmad.isaq@outlook.com  
**Institution:** Montana State University  
**Lab:** BMW Lab – Burns, Machine Learning & Wildfire  
**Year:** 2025
