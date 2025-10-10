# Inf-Net Installation & Setup Guide

## Quick Start

This guide covers installing and running Inf-Net for COVID-19 lung infection segmentation on A100 GPUs.

## Prerequisites

- Linux system with A100 GPU access
- Conda package manager
- SLURM job scheduler

## 1. Environment Setup

### Create Conda Environment
```bash
# Create environment with Python 3.6
conda create -p ./env python=3.6 -y

# Activate environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/your/project/env
```

### Install Dependencies
```bash
# Install PyTorch 1.9.0 with CUDA 11.1 (A100 compatible)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install other requirements
pip install scipy==1.7.3 thop

# For analysis (optional)
pip install pandas matplotlib tabulate
```

## 2. Dataset Setup

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

### Create Training Script
```bash
# Create train_infnet.sh
cat > train_infnet.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infnet_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours

mkdir -p logs
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/your/project/env
cd /path/to/your/project/code/inf-net

python MyTrain_LungInf.py \
    --epoch 100 \
    --batchsize 8 \
    --num_workers 4 \
    --backbone Res2Net50 \
    --trainsize 352 \
    --lr 1e-4
EOF

chmod +x train_infnet.sh
```

### Submit Training Job
```bash
sbatch train_infnet.sh
```

### Monitor Training
```bash
# Check job status
squeue -u $USER

# View training progress
tail -f logs/train_*.out

# Check for errors
tail -f logs/train_*.err
```

## 4. Testing/Inference

### Create Test Script
```bash
# Create test_infnet.sh
cat > test_infnet.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=infnet_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100-30min

mkdir -p logs
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/your/project/env
cd /path/to/your/project/code/inf-net

python MyTest_LungInf.py \
    --testsize 352 \
    --data_path "./Dataset/TestingSet/LungInfection-Test/" \
    --pth_path "./Snapshots/save_weights/Inf-Net/Inf-Net-100.pth" \
    --save_path "./Results/Lung_infection_segmentation/Inf-Net/"
EOF

chmod +x test_infnet.sh
```

### Run Testing
```bash
sbatch test_infnet.sh
```

## 5. Expected Results

### Training Output
- **Duration**: ~6 minutes for 100 epochs
- **Models**: Saved every 10 epochs in `Snapshots/save_weights/Inf-Net/`
- **Loss**: Should decrease from ~1.4 to ~0.3-0.6

### Testing Output
- **Predictions**: 48 segmentation masks in `Results/Lung_infection_segmentation/Inf-Net/`
- **Format**: PNG files with same names as input images

## 6. Troubleshooting

### Common Issues

**1. CUDA Error: "no kernel image available"**
- **Cause**: PyTorch version incompatible with A100
- **Solution**: Use PyTorch 1.9.0+ with CUDA 11.1

**2. cuDNN Error: "CUDNN_STATUS_EXECUTION_FAILED"**
- **Cause**: Old PyTorch version
- **Solution**: Upgrade to PyTorch 1.9.0+

**3. SSL Certificate Error**
- **Cause**: Network issues downloading pretrained models
- **Solution**: Download manually or use local weights

**4. Job Fails with QOS Error**
- **Cause**: Wrong partition or time limit
- **Solution**: Check available partitions: `sinfo -p a100`

### Performance Tips

- **Batch Size**: Start with 8, increase if memory allows
- **Image Size**: 352x352 works well, can try 320x320 for faster training
- **Workers**: Use 4-8 for data loading
- **Epochs**: 100 is usually sufficient, can stop early if converged

## 7. File Structure

```
code/inf-net/
├── Dataset/
│   ├── TrainingSet/LungInfection-Train/  # 50 training images
│   └── TestingSet/LungInfection-Test/    # 48 test images
├── Snapshots/
│   ├── pre_trained/                      # Backbone models
│   └── save_weights/Inf-Net/            # Trained models
├── Results/
│   └── Lung_infection_segmentation/Inf-Net/  # Predictions
├── Code/
│   ├── model_lung_infection/            # Model definitions
│   └── utils/                           # Utilities
├── MyTrain_LungInf.py                   # Training script
├── MyTest_LungInf.py                    # Testing script
└── logs/                                # Training logs
```

## 8. Quick Commands

```bash
# Activate environment
conda activate /path/to/your/project/env

# Check GPU availability
nvidia-smi

# Submit training
sbatch train_infnet.sh

# Submit testing
sbatch test_infnet.sh

# Monitor jobs
squeue -u $USER

# View results
ls -la Snapshots/save_weights/Inf-Net/
ls -la Results/Lung_infection_segmentation/Inf-Net/
```

## 9. Next Steps

- **Evaluation**: Use MATLAB evaluation toolbox for metrics
- **Semi-supervised**: Try Semi-Inf-Net with pseudo-labels
- **Multi-class**: Extend to GGO and consolidation segmentation
- **Differential Privacy**: Add DP-SGD for privacy-preserving training

---

**Total Setup Time**: ~30 minutes  
**Training Time**: ~6 minutes  
**Testing Time**: ~2 minutes  

**Requirements**: A100 GPU, 32GB RAM, 50GB storage
