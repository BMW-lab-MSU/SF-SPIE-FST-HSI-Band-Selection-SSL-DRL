# SF-SPIE-FST-HSI-Band-Selection-SSL-DRL

### Self-Supervised and Reinforcement-Driven Band Selection for UAV-Based Post-Fire Vegetation Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Paper](https://img.shields.io/badge/Paper-SPIE%202025%20(MLSP)-green.svg)]()

---

##  Overview

**BurnSSL-DRL** is a label-efficient framework that integrates **Self-Supervised Learning (SSL)** and **Deep Reinforcement Learning (DRL)** for **spectral band selection** and **3D spectral–spatial classification** of **VNIR hyperspectral imagery (HSI)** collected via UAVs after prescribed burns.  

Developed under the **SMART FIRES NSF Project (OIA-2242802)** at **Montana State University**, this framework supports efficient vegetation recovery assessment and canopy stress mapping in post-fire environments.

---

## Key Contributions

- **Self-Supervised Feature Learning:** Uses SimCLR-based 3D CNN pretraining to learn spectral–spatial features without labels.  
- **Deep Reinforcement Band Selection:** A DQN agent learns optimal spectral subsets using redundancy-penalized rewards.  
- **Dimensionality Reduction:** Retains key VNIR bands while reducing redundancy.  
- **Balanced Classification:** Incorporates SMOTE + class-weighted loss for label imbalance mitigation.  
- **Explainable Evaluation:** Provides per-class F1 trends, confusion matrices, and macro-F1 performance visualizations.

---

## Repository Structure

SF-SPIE-FST-HSI-Band-Selection-SSL-DRL/
│
├── SSL/ # Self-Supervised Learning (SimCLR3DCNN)
│ └── ssl/
│ ├── SimCLR3DCNN.py # 3D CNN backbone for SSL
│ ├── train_ssl.py # SSL training script
│ ├── contrastive_loss.py # NT-Xent contrastive loss
│ └── dataset.py # Patch-based dataset and augmentations
│
├── drl_band_selection/ # Deep Reinforcement Learning (DRL) module
│ ├── agent_dqn.py # DQN agent for band selection
│ ├── env_band_selection.py # DRL environment setup
│ ├── agniNet.py # 3D CNN classifier (AgniNet)
│ ├── train_drl.py # DRL training loop
│ ├── trainBalancedClassifiers.py
│ ├── trainUnbalancedClassifiers.py
│ ├── extract_features.py
│ ├── baseline_vs_smote_comparison.xlsx
│ └── agni_best.pth # Best trained model weights
│
├── Visuals/ # Results and figures
│ ├── 3dcnn_drl_classification_results.xlsx
│ ├── baseline_vs_smote_comparison.xlsx
│ ├── conf_matrix_* # Confusion matrices per classifier × Top-K
│ ├── macro_f1_vs_topk_all_classifiers.png
│ ├── Per_Class_F1_Trends_All_Classifiers.png
│ └── before_balance_comparison.xlsx
│
├── data/ # VNIR hyperspectral cube and ground truth (user-supplied)
├── outputs/ # Band scores, model checkpoints, metrics
├── plots/ # Python scripts for reproducing figures
├── scripts/ # Utility scripts for patch extraction, preprocessing
└── README.md


---

## ⚙️ Environment Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/BMW-lab-MSU/SF-SPIE-FST-HSI-Band-Selection-SSL-DRL.git
cd SF-SPIE-FST-HSI-Band-Selection-SSL-DRL ```

2️⃣ Create environment
conda create -n burnssl-drl python=3.10
conda activate burnssl-drl
pip install -r requirements.txt

3️⃣ Core dependencies
torch >= 2.0
torchvision
numpy
pandas
scikit-learn
matplotlib
rasterio
spectral
opencv-python

 How to Run
Step 1 – Self-Supervised Pretraining
cd SSL/ssl
python train_ssl.py --epochs 100 --batch_size 32

This trains a SimCLR3D-CNN encoder using unlabeled hyperspectral patches.

Step 2 – DRL Band Selection
cd ../../drl_band_selection
python train_drl.py --episodes 200 --reward spectral_redundancy


Outputs:

-- outputs/band_scores/drl_scores.npy

-- Band-ranking CSV for Top-K evaluation

Step 3 – Train Classifiers

Unbalanced:

python trainUnbalancedClassifiers.py --topk 30


