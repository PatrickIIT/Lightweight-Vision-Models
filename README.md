# Lightweight-Vision-Models
Knowledge Distillation, Pruning, and Quantization on Lightweight Vision Models

---

# EuroSAT and UC Merced Land Use Dataset Experiments

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains code and results for experiments on the **EuroSAT** and **UC Merced Land Use** datasets, focusing on model compression techniques using PyTorch and Brevitas. The experiments evaluate **ResNet-18**, **ResNet-9**, and **Vision Transformer (ViT-B/16)** models with techniques like Knowledge Distillation (KD), Pruning, Quantization-Aware Training (QAT), and combined Pruning + QAT.

## 📖 Overview

The goal is to optimize deep learning models for satellite (EuroSAT, 10 classes) and aerial (UC Merced, 21 classes) imagery classification, achieving high accuracy with reduced model size. The experiments include:

1. **Knowledge Distillation (KD)**: Transfer knowledge from a larger teacher model (e.g., EfficientNet, ResNet-18) to a smaller student model (e.g., ResNet-18, ResNet-9).
2. **Pruning**: Remove less important weights or channels to reduce model size.
3. **Quantization-Aware Training (QAT)**: Train models with low-bit precision (8-bit, 4-bit, 3-bit, 2-bit, 1-bit) using Brevitas.
4. **Combined Pruning + QAT**: Combine pruning and QAT for optimal compression.

## 📂 Repository Structure

- `experiments.py`: Main script containing code for all experiments (EuroSAT: Experiments 1 & 2, UC Merced: Experiment 8).
  - **Experiment 1**: ResNet-18 and ResNet-9 on EuroSAT with KD, fine-tuning, and training from scratch.
  - **Experiment 2**: ResNet-18 and ViT-B/16 on EuroSAT with pruning, static quantization, and ViT layer importance analysis.
  - **Experiment 8**: ResNet-18 with EfficientNet-V2-S as teacher on UC Merced, using KD, pruning, QAT, and combined approaches.
- `data/`: Placeholder for dataset paths (EuroSAT: `/kaggle/input/eurosat-dataset/EuroSAT`, UC Merced: `/kaggle/input/uc-merced-land-use-dataset/UCMerced_LandUse`).
- `checkpoints/`: Directory for model checkpoints (e.g., `resnet18_eurosat_best_64x64.pth`, `qat_resnet18_4bit.pth`).
- `plots/`: Directory for generated plots (e.g., `comparison_metrics.png`).

## 🚀 Setup

### Prerequisites
- Python 3.11+
- CUDA-enabled GPU (optional, CPU fallback supported)
- Datasets: EuroSAT and UC Merced Land Use (download and place in `data/` or update paths)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/eurosat-ucmerced-experiments.git
   cd eurosat-ucmerced-experiments
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio pandas numpy matplotlib scipy brevitas tabulate
   ```
3. Ensure datasets are accessible:
   - EuroSAT: Requires `train.csv`, `val.csv`, `test.csv`, and `label_map.json`.
   - UC Merced: ImageFolder structure at specified path.

## 🛠️ Usage

1. **Run All Experiments**:
   ```bash
   python experiments.py
   ```
   This executes the `main()` function, running KD, pruning, QAT, and combined experiments.

2. **Run Specific Tasks**:
   Modify the `main()` function to call individual tasks:
   - `knowledge_distillation()` for KD
   - `pruning_training()` for pruning
   - `quantization_aware_training()` for QAT
   - `combined_pruning_qat()` for combined pruning + QAT

3. **Key Configurations**:
   - **EuroSAT**:
     - Image size: 64x64 (Experiment 2) or 224x224 (Experiment 1)
     - Batch size: 32 (Experiment 2) or 128 (QAT in Experiment 1)
     - Epochs: 10–15 (training), 3–10 (QAT/pruning)
   - **UC Merced**:
     - Image size: 224x224
     - Batch size: 16
     - Epochs: 25 (KD), 3 (QAT), 10 (pruning)
   - Adjust hyperparameters in the script (e.g., `LEARNING_RATE`, `PRUNING_AMOUNTS`, `bit_widths`).

4. **Outputs**:
   - **Checkpoints**: Saved in `checkpoints/` (e.g., `qat_resnet18_4bit.pth`).
   - **Plots**: Training history and comparison plots saved in `plots/` (e.g., `comparison_metrics.png`).
   - **Results Tables**: Printed to console with accuracy and model size.

## 📊 Key Results

### Experiment 1 (EuroSAT)
| Model                     | Test Accuracy (%) | Notes                     |
|---------------------------|-------------------|---------------------------|
| Fine-tuned ResNet-18      | 95.81             | Pretrained, fine-tuned    |
| KD ResNet-18              | 93.96             | Distilled from ResNet-18  |
| KD ResNet-9               | 55.56             | Distilled from ResNet-18  |
| ResNet-18 (Scratch)       | 89.89             | Trained from scratch      |
| Pretrained ResNet-18      | 15.52             | No fine-tuning            |

### Experiment 2 (EuroSAT)
| Model                     | Test Accuracy (%) | Size (MB) | Sparsity (%) | Notes                     |
|---------------------------|-------------------|-----------|--------------|---------------------------|
| ResNet-18 Baseline        | 95.81             | 44.82     | 0            | FP32                      |
| ResNet-18 Pruned (75%)    | 91.41             | 46.84     | 75           | Structured L1             |
| ResNet-18 Quantized (8-bit) | 87.56           | 44.82     | N/A          | Static INT8               |
| ViT-B/16 Baseline         | ~95               | ~330      | 0            | FP32                      |
| ViT-B/16 Pruned (40%)     | ~94               | ~330      | 40           | Structured L1, No FT      |

### Experiment 8 (UC Merced)
| Model                     | Accuracy (%) | Size (MB) | Sparsity (%) | Bit Width | Notes                     |
|---------------------------|--------------|-----------|--------------|-----------|---------------------------|
| KD ResNet-18              | -            | 46.84     | -            | -         | Baseline KD               |
| QAT (8-bit)               | 100.00       | 44.84     | N/A          | 8         | Quantized                 |
| QAT (4-bit)               | 98.81        | 44.84     | N/A          | 4         | Quantized                 |
| Pruned (50%)              | 100.00       | 46.84     | 50           | -         | Magnitude Pruning         |
| Pruned (75%) + QAT (2-bit)| 96.67        | 44.84     | 75           | 2         | Combined                  |

## 📝 Analysis

- **Knowledge Distillation**: Highly effective for ResNet-18, achieving near-baseline performance on both datasets.
- **Pruning**: Maintains 100% accuracy on UC Merced up to 87.5% sparsity; EuroSAT sees a drop at 97% sparsity.
- **QAT**: 8-bit and 4-bit QAT preserve high accuracy; 1-bit QAT is unstable on EuroSAT (NaN losses).
- **Combined Pruning + QAT**: 50% sparsity with 4-bit QAT offers a good balance of accuracy and efficiency.
- **Model Size**: Reported sizes (~44–46 MB) may include serialization overhead; actual deployment size could be lower with optimized formats.

## 🔮 Future Work

- Address 1-bit QAT instability with advanced quantization techniques.
- Explore dynamic quantization or mixed-precision training.
- Test on additional datasets for generalization.
- Optimize model serialization for accurate size reporting (e.g., ONNX export).

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙌 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

---
