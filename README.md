# Multimodal Robust Prompt Distillation for 3D Point Cloud Models

This repository contains the official PyTorch implementation for the experiments in our paper, "Multimodal Robust Prompt Distillation for 3D Point Cloud Models".

**Please Note:** This repository currently provides the raw code used for our experiments to ensure full reproducibility for the review process. We are committed to releasing a refactored, fully-documented, and user-friendly version of the code upon the acceptance of our paper.

## Environment Setup

Our code is built upon the experimental environment of the **Uni3D** baseline.
## Reproduction Steps
if you want to prompt distill under uni3d,pls follow https://github.com/baaivision/Uni3D
otherwise pls choose the model you desire 

### 1. Data Preparation

-   **Datasets:** Please download the ModelNet40 and ScanObjectNN datasets. You may need to specify the path to your datasets in the training/testing scripts.
-   **Attack Data Generation:** For a fair and direct comparison with prior work, our adversarial attack generation protocol strictly follows the public benchmark established by methods like **AOF** (https://github.com/code-roamer/AOF). We utilize their methodologies for generating all adversarial samples.
    -   Our script `test.py` includes functionalities for generating attack data.
    -   We have only made minor adjustments to the data format to align with our processing pipeline; the core attack logic remains identical to ensure fairness.

### 2. Training

To train our MRPD model from scratch, run the `train_multi_prompt.py` script. This script handles the multi-teacher distillation and prompt learning process.

python train_multi_prompt.py #params can be seen in the file.
