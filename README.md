# Lightweight-Vision-Models
Knowledge Distillation, Pruning, and Quantization on Lightweight Vision Models
---

```markdown
# Model Compression for Vision on Satellite Imagery  
**An Empirical Study of Knowledge Distillation, Pruning, and Quantization**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 1. Project Overview

This repository contains the full codebase and experimental results for a comprehensive study on model compression for satellite image classification. The research systematically evaluates the effectiveness of:

- **Knowledge Distillation (KD)**
- **Network Pruning**
- **Quantization**

on various architectures, including **CNNs (ResNet, MobileNetV2)** and **Vision Transformers (ViT, CLIP)**.

The primary goal is to develop a practical framework for creating efficient, yet accurate, models suitable for deployment on resource-constrained edge devices like satellites and UAVs.

Two benchmark datasets are used:
- **EuroSAT**
- **UC Merced Land-Use**

A major contribution is the **Budget-Aware Iterative Compression (BAIC)** pipeline, an automated framework for discovering the optimal combination of pruning and quantization under a given model size constraint.

---

## 2. Key Findings & Visualizations

### 🔁 Accuracy vs. Model Size Trade-off  
Combining **Knowledge Distillation** with **Quantization-Aware Training (QAT)** consistently produces small models with minimal accuracy loss.

---

### 🧠 Dataset-Dependent Compression Robustness  
Compression effectiveness varies with dataset complexity:
- ResNet-18 is resilient on UC Merced but suffers degradation on EuroSAT.

---


## 5. Setup and Installation

### ✅ Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA
- [Conda](https://docs.conda.io/) (recommended)

### 🧩 Installation Steps

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a conda environment
conda create -n vision-compression python=3.10
conda activate vision-compression

# Install dependencies
pip install -r requirements.txt
````

### 📂 Download Datasets

* **EuroSAT**: [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) → extract into `data/EuroSAT/`
* **UC Merced Land-Use**: [Official website](http://weegee.vision.ucmerced.edu/datasets/landuse.html) → place into `data/UCMerced_LandUse/`

---

## 6. How to Run

### 🧪 Run the Adaptive Compression Pipeline

```bash
python adaptive_compression_pipeline.py
```

The best pruned & quantized model will be saved in `saved_models/`.

---

### 🧬 Run Individual Experiments

```bash
# Example: Run KD on MobileNetV2
python experiments/experiment_6_kd_mobilenet.py
```

---

### 📊 Generate Visualizations

```bash
python generate_visualizations.py
```

Figures will be saved in the `visuals/` directory.

---

## 7. Summary of Key Results

* ✅ **Knowledge Distillation**: MobileNetV2 student (95.74%) nearly matches ResNet-18 baseline (95.81%) on EuroSAT.
* ⚠️ **QAT > PTQ**: QAT MobileNetV2 (97.13%) outperforms full-precision and PTQ versions.
* 🚫 **Transformer Pruning**: ViT and CLIP models are more sensitive to pruning than CNNs.
* 🧩 **Task Complexity Matters**: 2-bit QAT achieves >99% accuracy on UC Merced but underperforms on EuroSAT.

---

## 8. Citation

If this work helps your research, please cite it:

```bibtex
@inproceedings{PatrickIIT,
  title={Model Compression for Lightweight Vision Models},
  author={Patrick Vincent},
  year={2025}
}
```

---

## 9. License

This project is licensed under the [MIT License](LICENSE).

---

## 10. Acknowledgments

We thank **Prof. Tushar Shinde** for invaluable guidance.

This research uses:

* [EuroSAT Dataset](https://github.com/phelber/eurosat)
* [UC Merced Land-Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

---

