#!/bin/bash
#SBATCH --job-name=nestedunet_groupnorm_batch24_run1
#SBATCH --output=logs/train/nestedunet_groupnorm_batch12_run1_%j.out
#SBATCH --error=logs/train/nestedunet_groupnorm_batch12_run1_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090
#SBATCH --qos=gpu30min

# Create logs directory if it doesn't exist
mkdir -p logs/train

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Set PyTorch memory allocation to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Navigate to inf-net code directory
cd code/inf-net

# Run training with reduced batch size for RTX4090 memory constraints
python MyTrain_LungInf_NestedUNet_GroupNorm.py --batchsize 12 --run 1 --epoch 70

echo "Training completed at $(date)"

