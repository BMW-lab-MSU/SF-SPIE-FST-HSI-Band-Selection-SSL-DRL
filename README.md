# SF-SPIE-FST-HSI-Band-Selection-SSL-DRL

### Self-Supervised and Reinforcement-Driven Band Selection for UAV-Based Post-Fire Vegetation Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Paper](https://img.shields.io/badge/Paper-SPIE%202025%20(MLSP)-green.svg)]()

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

## Environment Setup

### 1️ Clone & create environment
```bash
git clone https://github.com/BMW-lab-MSU/SF-SPIE-FST-HSI-Band-Selection-SSL-DRL.git
cd SF-SPIE-FST-HSI-Band-Selection-SSL-DRL

conda create -n burnssl-drl python=3.10
conda activate burnssl-drl
pip install -r requirements.txt
2️⃣ Dependencies
nginx
Copy code
torch >= 2.0
torchvision
numpy
pandas
scikit-learn
matplotlib
rasterio
spectral
opencv-python
🚀 Usage
Step 1 – Self-Supervised Pretraining
bash
Copy code
cd SSL/ssl
python train_ssl.py --epochs 100 --batch_size 32
Step 2 – DRL Band Selection
bash
Copy code
cd ../../drl_band_selection
python train_drl.py --episodes 200 --reward spectral_redundancy
Outputs:

bash
Copy code
outputs/band_scores/drl_scores.npy
Visuals/3dcnn_drl_classification_results.xlsx
Step 3 – Train Classifiers
bash
Copy code
# Unbalanced
python trainUnbalancedClassifiers.py --topk 30
# Balanced (SMOTE + weighted loss)
python trainBalancedClassifiers.py --topk 30
Step 4 – Reproduce Figures (Fig 1–3)
bash
Copy code
cd ../plots
python codeToFig3.py
Figure	Description	Output
Fig 1	Macro-F1 vs Top-K (All classifiers)	Visuals/macro_f1_vs_topk_all_classifiers.png
Fig 2	Per-class F1 trends	Visuals/Per_Class_F1_Trends_All_Classifiers.png
Fig 3	Confusion matrices	Visuals/conf_matrix_*

Example Results
Classifier	Balancing	Top-K	Macro-F1	Weighted-F1
KNN	No	30	0.68	0.70
SVM	No	30	0.71	0.72
RF	No	40	0.74	0.75
3D-CNN	No	30	0.74	0.75
3D-CNN	Yes (SMOTE)	30	0.80	0.82

Insight: Balancing improved Grass & Soil F1 scores, while Tree class performance remained stable—indicating consistent canopy detection.

Extended Study – SSEP vs SRPA vs DRL Comparison
This section reproduces the comparative analysis between:

SSEP (Spectral-Spatial Edge Preservation)

SRPA (Spectral-Redundancy Penalized Attention)

DRL (BurnSSL-DRL agent)

Input Files
bash
Copy code
outputs/band_scores/ssep_scores.npy
outputs/band_scores/srpa_scores.npy
outputs/band_scores/drl_scores.npy
🧠 Run Comparative Training
bash
Copy code
python scripts/compare_band_selection.py \
    --methods SSEP SRPA DRL \
    --topk_list 10 20 30 40 50 \
    --models RF 3DCNN
Generate Result Tables & Plots
Outputs saved to:

bash
Copy code
outputs/comparison_results/comparison_results.csv
Visuals/SSEP_SRPA_DRL_comparison.png
Method	Classifier	Top-K	Accuracy	Macro-F1
SSEP	RF	30	0.72	0.71
SRPA	RF	30	0.74	0.73
DRL (BurnSSL-DRL)	3D-CNN	30	0.81	0.80

Observation: DRL outperformed both SSEP and SRPA by effectively emphasizing low-wavelength VNIR regions related to chlorophyll degradation and soil exposure.

🧠 Technical Summary
Component	Description
SSL	SimCLR 3D-CNN encoder learns spectral–spatial embeddings.
DRL	DQN agent selects spectral bands optimizing reward = accuracy – redundancy.
Classifier	Random Forest + 3D-CNN used for downstream evaluation.
Metrics	Macro-F1, Weighted-F1, Per-class F1, Confusion Matrix.
Balancing	SMOTE resampling + class-weighted loss.

🧾 Citation
bibtex
Copy code
@inproceedings{karankot2025burnssldrl,
  title={BurnSSL-DRL: Self-Supervised and Reinforcement-Driven Band Selection for UAV-Based Post-Fire Vegetation Analysis},
  author={Mahmad Isaq Karankot and Bradley Whitaker and others},
  booktitle={IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2025}
}
🙏 Acknowledgments
This work was supported by the NSF EPSCoR SMART FIRES Project (OIA-2242802)
at Montana State University, within the BMW Lab (Burns, Machine Learning, and Wildfire).
We acknowledge the UAV and field teams at Lubrecht Experimental Forest for data collection and the fire-science collaborators for their assistance.

📬 Contact
Author: Mahmad Isaq Karankot
Year: 2025

yaml
Copy code

---




