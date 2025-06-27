# Lightweight-Vision-Models
Knowledge Distillation, Pruning, and Quantization on Lightweight Vision Models
EuroSAT and UC Merced Land Use Dataset Experiments
This repository contains code and results for experiments conducted on the EuroSAT and UC Merced Land Use datasets, focusing on model compression techniques such as Knowledge Distillation, Pruning, and Quantization-Aware Training (QAT) using PyTorch and Brevitas. The experiments aim to evaluate the performance of ResNet-18, ResNet-9, and Vision Transformer (ViT) models under various compression strategies.
Project Overview
The experiments explore model compression techniques to optimize deep learning models for the EuroSAT (10 classes, satellite imagery) and UC Merced Land Use (21 classes, aerial imagery) datasets. The goal is to achieve efficient models with minimal accuracy loss through:

Knowledge Distillation (KD): Transferring knowledge from a larger teacher model (e.g., ResNet-18 or EfficientNet) to a smaller student model (e.g., ResNet-18 or ResNet-9).
Pruning: Reducing model size by removing less important weights or channels.
Quantization-Aware Training (QAT): Training models with simulated low-bit precision (e.g., 8-bit, 4-bit, 3-bit, 2-bit, 1-bit) using Brevitas for efficient deployment.
Combined Pruning and QAT: Combining pruning and QAT to further optimize model size and performance.

Repository Structure

EXPERIMENTS I DID.txt: Contains the main code and results for experiments on the EuroSAT dataset (Experiment 1 and Experiment 2) and UC Merced Land Use dataset (Experiment 8).

Experiment 1: Evaluates ResNet-18 and ResNet-9 on EuroSAT with KD, fine-tuning, and training from scratch.
Experiment 2: Explores ResNet-18 and ViT-B/16 on EuroSAT with pruning, static quantization, and layer importance analysis for ViT.
Experiment 8: Focuses on UC Merced Land Use dataset with KD, pruning, QAT, and combined pruning + QAT using ResNet-18 and EfficientNet as the teacher.


Data Requirements:

EuroSAT: Available at /kaggle/input/eurosat-dataset/EuroSAT with train.csv, val.csv, test.csv, and label_map.json.
UC Merced Land Use: Available at /kaggle/input/uc-merced-land-use-dataset/UCMerced_LandUse.


Output Files:

Model checkpoints (e.g., resnet18_eurosat_best_64x64.pth, qat_resnet18_4bit.pth).
Comparison plots (e.g., comparison_metrics.png for EuroSAT).
Results tables printed in the console for accuracy and model size.



Prerequisites
To run the experiments, ensure the following dependencies are installed:
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib scipy
pip install brevitas tabulate

Additional requirements:

Python 3.11 or compatible version.
CUDA-enabled GPU for accelerated training (CPU fallback supported).
Access to the EuroSAT and UC Merced Land Use datasets.

How to Run

Setup Environment:

Install dependencies as listed above.
Ensure the datasets are available at the specified paths or update the paths in the code.


Run Experiments:

The code is modularized into tasks for KD, pruning, QAT, and combined pruning + QAT.
To execute all experiments, run the main() function in EXPERIMENTS I DID.txt:python experiments.py


Individual tasks can be run by calling their respective functions (e.g., knowledge_distillation(), pruning_training()).


Key Configurations:

EuroSAT:
Image size: 64x64 (Experiment 2) or 224x224 (Experiment 1).
Batch size: 32 (Experiment 2) or 128 (QAT in Experiment 1).
Epochs: 10–15 for training, 3–10 for QAT/pruning.


UC Merced:
Image size: 224x224.
Batch size: 16.
Epochs: 25 for KD, 3 for QAT, 10 for pruning.


Adjust hyperparameters (e.g., learning rate, sparsity levels, bit widths) in the configuration sections of the code.


Output:

Model checkpoints are saved (e.g., qat_resnet18_4bit.pth, pruned_resnet18_0.5.pth).
Results tables are printed to the console, summarizing accuracy and model size.
Plots are generated for training history and model comparisons (e.g., comparison_metrics.png).



Key Results
Experiment 1 (EuroSAT)

Models: ResNet-18, ResNet-9.
Techniques: KD, fine-tuning, training from scratch.
Results:
Fine-tuned ResNet-18: 95.81% test accuracy.
KD ResNet-18: 93.96% test accuracy.
KD ResNet-9: 55.56% test accuracy.
Pretrained ResNet-18 (no fine-tuning): 15.52% test accuracy.
ResNet-18 from scratch: 89.89% test accuracy.



Experiment 2 (EuroSAT)

Models: ResNet-18, ViT-B/16.
Techniques: Pruning, static quantization, combined pruning + QAT.
Results:
ResNet-18 Baseline: 95.81% accuracy, ~44.82 MB.
ResNet-18 Pruned (75% sparsity): 91.41% accuracy.
ResNet-18 Quantized (8-bit): 87.56% accuracy.
ViT-B/16 Baseline: Higher accuracy but larger size (~330 MB).
ViT-B/16 Pruned (40% sparsity): Reduced size with minimal accuracy loss.



Experiment 8 (UC Merced)

Models: ResNet-18 (student), EfficientNet-V2-S (teacher).
Techniques: KD, pruning, QAT, combined pruning + QAT.
Results:
KD Baseline: 46.84 MB.
QAT (8-bit, 3-bit, 2-bit): 98.81–100% accuracy, ~44.84 MB.
Pruning (50–87.5% sparsity): 100% accuracy, ~46.84 MB.
Combined Pruning (50%) + QAT (2-bit): 100% accuracy, ~44.84 MB.



Analysis

Knowledge Distillation: Effective for transferring knowledge to smaller models, with ResNet-18 achieving high accuracy on both datasets.
Pruning: Maintains high accuracy up to 75–87.5% sparsity on UC Merced, but performance drops significantly at 97% sparsity on EuroSAT.
Quantization-Aware Training: 8-bit and 4-bit QAT retain high accuracy; 2-bit and 1-bit QAT show instability (e.g., NaN losses on EuroSAT).
Combined Pruning + QAT: Mild pruning (50%) with 4-bit QAT offers a good balance of accuracy and model size.
Dataset Differences: UC Merced results show higher accuracy (near 100%) due to simpler classification tasks or smaller dataset size compared to EuroSAT.

Notes

Model Size Discrepancy: Reported model sizes (~44–46 MB) may not reflect expected compression due to PyTorch/Brevitas serialization overhead. Actual deployment size may be smaller with optimized formats (e.g., ONNX).
1-bit Quantization Issues: NaN losses observed in 1-bit QAT on EuroSAT, indicating numerical instability at ultra-low precision.
ViT Pruning: Adaptive pruning based on layer importance preserves accuracy better than uniform pruning.

Future Work

Experiment with lower-bit quantization (e.g., 1-bit) using advanced techniques to mitigate instability.
Explore dynamic quantization or mixed-precision training for further optimization.
Test on additional datasets to validate generalization.
Optimize model serialization to reflect true compression benefits.

License
This project is licensed under the MIT License. See the LICENSE file for details.
