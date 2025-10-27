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

