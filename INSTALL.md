# Inf-Net Installation & Setup Guide

## Quick Start

This guide covers running Inf-Net for COVID-19 lung infection segmentation on A100 GPUs.

## Prerequisites

- Linux system with A100 GPU access
- Conda package manager
- SLURM job scheduler

## 1. Dataset Setup

### Download COVID-SemiSeg Dataset
```bash
# Download from Google Drive (manual)
# Link: https://drive.google.com/open?id=1bbKAqUuk7Y1q3xsDSwP07oOXN_GL3SQM

# Extract to Dataset directory
cd code/inf-net
unzip COVID-SemiSeg.zip
mv COVID-SemiSeg/Dataset/TrainingSet/* Dataset/TrainingSet/
mv COVID-SemiSeg/Dataset/TestingSet/* Dataset/TestingSet/
rm -rf COVID-SemiSeg COVID-SemiSeg.zip
```

### Download Pretrained Backbones
```bash
# Create directory
mkdir -p Snapshots/pre_trained

# Download backbone models
cd Snapshots/pre_trained
curl -L -o vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth
curl -L -o resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
curl -L -o res2net50_v1b_26w_4s-3cf99910.pth https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth
```

## 3. Training

1. Create Training Script - [train_infnet.sh](./train_infnet.sh)

2. `chmod +x train_infnet.sh`

3. Submit Training Job - `sbatch train_infnet.sh`

4. Monitor Training

    ```bash
    # Check job status
    squeue -u $USER

    # View training progress
    tail -f logs/train_*.out

    # Check for errors
    tail -f logs/train_*.err
    ```

## 4. Testing/Inference

1. Create Test Script - [test_infnet.sh](./test_infnet.sh)

2. `chmod +x test_infnet.sh`

3. `sbatch test_infnet.sh`


## 5. Expected Results

### Training Output
- **Duration**: ~6 minutes for 100 epochs
- **Models**: Saved every 10 epochs in `Snapshots/save_weights/Inf-Net/`
- **Loss**: Should decrease from ~1.4 to ~0.3-0.6

### Testing Output
- **Predictions**: 48 segmentation masks in `Results/Lung_infection_segmentation/Inf-Net/`
- **Format**: PNG files with same names as input images
