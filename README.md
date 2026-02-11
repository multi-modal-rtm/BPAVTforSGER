# Benchmarking Pretrained Transformers for Student Group Engagement Recognition (BPAVTforSGER)

This repository contains the implementation and benchmarking results for student engagement recognition across three major datasets: EngageNet, OUC-CGE, and DAiSEE. The project evaluates various state-of-the-art pretrained Transformers and analyzes the impact of advanced training strategies such as Mixup, Focal Loss, and Fine-tuning.

## Project Overview

This benchmark provides a standardized environment to compare models like ViT, Swin, TimeSformer, and others across diverse datasets with varying challenges.

Datasets Included:

EngageNet: Large-scale dataset focused on individual student engagement in-the-wild.

OUC-CGE: Focused on small-group engagement with complex social interactions in classroom setting.

DAiSEE: Multi-label dataset for academic engagement.
## Repository Structure

The project is organized by dataset, with sub-projects corresponding to specific experimental strategies:

.
├── EngageNet/
│   ├── engagenet_benchmark/            # Baseline and Mixup experiments
│   ├── engagenet_benchmark_finetune/   # Standard fine-tuning strategies
│   └── engagenet_benchmark_fl/         # Focal Loss implementation
├── OUC-CGE/
│   ├── ouc-cge_benchmark/              # Primary OUC-CGE implementation
│   └── ouc-cge_benchmark_fine_tuning/  # Targeted fine-tuning strategies
└── DAiSEE/
    ├── daisee_benchmark/               # Baseline DAiSEE implementation
    ├── daisee_benchmark_finetuning/    # Fine-tuning experiments
    └── daisee_benchmark_focalloss/     # Focal Loss implementation


## Getting Started

### Prerequisites

 - Python 3.11+

 - CUDA-enabled GPU

### Installation

Clone the repository and install dependencies for the relevant dataset folder:

 - `git clone [https://github.com/your-username/BPAVTforSGER.git](https://github.com/your-username/BPAVTforSGER.git)
cd BPAVTforSGER`
- go to target project folder
 - `pip install -r requirements.txt`


### Usage

Each sub-project utilizes Hydra for configuration management. Navigate to a specific sub-project and run the training script:

#### Example: Running ViT on EngageNet with Mixup
`cd EngageNet/engagenet_benchmark`
`python -m src.train --config-name=vit`


## Experimental Strategies

Standard Fine-Tuning: Evaluating pretrained weights (ImageNet-21k, Kinetics-400) adapted for engagement tasks.

Focal Loss: Addressing class imbalance by down-weighting well-classified examples.

Mixup Augmentation: Improving generalization and reducing overfitting by training on convex combinations of input pairs and their labels.

## Results

Full benchmarking results and model performance tables can be found in the results/ folder or within the specific dataset sub-directories.

## Citation

If you use this code or benchmark in your research, please cite our paper:

