# Lightweight-Vision-Models
Knowledge Distillation, Pruning, and Quantization on Lightweight Vision Models
Of course. A high-quality README.md is essential for any research project on GitHub. It serves as the front door, guiding visitors, explaining the project's significance, and enabling others to reproduce your work.

Based on all your experiments and our discussions, I have created a comprehensive, professional README.md file. It is structured to be clear, concise, and immediately useful to anyone from your professor to a potential collaborator.

You can copy and paste the following content directly into a file named README.md in the root of your GitHub repository.

Model Compression for Vision on Satellite Imagery
An Empirical Study of Knowledge Distillation, Pruning, and Quantization

![alt text](https://img.shields.io/badge/python-3.10+-blue.svg)


![alt text](https://img.shields.io/badge/framework-PyTorch-orange.svg)


![alt text](https://img.shields.io/badge/license-MIT-green.svg)

1. Project Overview

This repository contains the full codebase and experimental results for a comprehensive study on model compression for satellite image classification. The research systematically evaluates the effectiveness of Knowledge Distillation (KD), Network Pruning, and Quantization on a diverse set of architectures, including CNNs (ResNet, MobileNetV2) and Vision Transformers (ViT, CLIP).

The primary goal is to develop a practical framework for creating efficient, yet highly accurate, models suitable for deployment on resource-constrained edge devices like satellites and UAVs. Experiments are conducted on two distinct benchmarks—EuroSAT and UC Merced Land-Use—to analyze how task complexity influences the effectiveness of different compression techniques.

A key contribution of this work is the Budget-Aware Iterative Compression (BAIC) pipeline, an automated framework designed to find the optimal combination of pruning and quantization for a given model size budget.

2. Key Findings & Visualizations

Our research yields several critical insights into the practical application of model compression for Earth observation tasks.

Accuracy vs. Model Size Trade-off

The combination of Knowledge Distillation and Quantization-Aware Training (QAT) consistently provides the best trade-off, producing small models that maintain or even exceed the accuracy of larger baselines.

![alt text](https://github.com/user/repo/blob/main/visuals/accuracy_vs_size_compression_results.png?raw=true)

(Note: Please generate and place your accuracy_vs_size_compression_results.png in a visuals folder)

Dataset-Dependent Compression Robustness

A model's resilience to compression is highly dependent on the complexity of the target dataset. A distilled ResNet-18 is nearly impervious to aggressive compression on the simpler UC Merced dataset but shows significant degradation on the more challenging EuroSAT benchmark.

![alt text](https://github.com/user/repo/blob/main/visuals/dataset_compression_comparison.png?raw=true)

(Note: Please generate and place your dataset_compression_comparison.png in a visuals folder)

3. Methodology at a Glance

Our experimental process follows a systematic pipeline to evaluate and combine different compression methods.

![alt text](https://github.com/user/repo/blob/main/visuals/framework_flowchart.png?raw=true)

(Note: Please generate and place a PNG of your TikZ framework diagram in a visuals folder)

4. Repository Structure
Generated code
.
├── adaptive_compression_pipeline.py  # Main script for the BAIC pipeline
├── generate_visualizations.py        # Code to generate the plots for the paper
├── experiments/                      # Individual scripts for each experiment
│   ├── experiment_1_kd.py
│   ├── experiment_2_pruning_vit.py
│   └── ... (and so on for all 8 experiments)
├── data/                             # Placeholder for datasets
│   ├── EuroSAT/
│   └── UCMerced_LandUse/
├── visuals/                          # Output folder for generated plots
├── saved_models/                     # Directory for storing trained model weights
├── requirements.txt                  # Python package dependencies
└── README.md                         # This file

5. Setup and Installation
Prerequisites

Python 3.10+

NVIDIA GPU with CUDA (for faster training)

Conda (recommended for environment management)

Installation Steps

Clone the repository:

Generated bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Create and activate a Conda environment:

Generated bash
conda create -n vision-compression python=3.10
conda activate vision-compression
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install the required packages:
The requirements.txt file contains all necessary libraries from your experiments.

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Download the Datasets:

EuroSAT: Download from Kaggle and unzip into the data/EuroSAT directory. The structure should contain the 2750 folder with all class subdirectories.

UC Merced Land-Use: Download from the official source and place the UCMerced_LandUse folder into data/.

6. How to Run
Running the Main Adaptive Pipeline

The adaptive_compression_pipeline.py script is the primary entry point for reproducing the core automated compression results. It will search for the best pruned and quantized ResNet-18 model under a specified size budget.

Generated bash
python adaptive_compression_pipeline.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The best-performing model that meets the budget will be saved in the saved_models/ directory.

Running Individual Experiments

The original scripts for each of the eight experiments are located in the experiments/ directory. These can be run individually to reproduce specific results.

Generated bash
# Example for running the knowledge distillation experiment
python experiments/experiment_6_kd_mobilenet.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Generating Visualizations

The plots used in the paper can be regenerated by running:

Generated bash
python generate_visualizations.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

This script will use the saved model files and experimental logs to create the final figures in the visuals/ directory.

7. Summary of Key Results

Knowledge Distillation is highly effective: A distilled MobileNetV2 student (95.74%) nearly matched a fine-tuned ResNet-18 baseline (95.81%) on EuroSAT, demonstrating massive efficiency gains.

Quantization-Aware Training (QAT) is essential: QAT consistently outperformed Post-Training Quantization (PTQ), which often led to catastrophic accuracy loss. An 8-bit QAT MobileNetV2 (97.13%) even surpassed its full-precision counterpart.

Transformers are more sensitive to pruning: Unstructured pruning was far more detrimental to ViT and CLIP models compared to ResNets, highlighting fundamental architectural differences.

Task complexity dictates compression limits: Models trained on the simpler UC Merced dataset were extremely robust to compression (maintaining >99% accuracy with 2-bit QAT), while the same techniques caused significant degradation on the more complex EuroSAT dataset.

8. Citation

If you use this work in your research, please cite our paper:

Generated bibtex
@inproceedings{yourname2025compression,
  title={Model Compression for Vision Transformers and CNNs on Diverse Satellite Imagery Benchmarks},
  author={Your Name and Co-author Name},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bibtex
IGNORE_WHEN_COPYING_END
9. License

This project is licensed under the MIT License. See the LICENSE file for details.

10. Acknowledgments

We thank [Professor's Name] for their invaluable guidance and feedback.

This work utilizes the EuroSAT and UC Merced Land-Use datasets.
