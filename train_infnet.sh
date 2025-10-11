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

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /scicore/home/wagner0024/shandi0000/miniconda3/etc/profile.d/conda.sh
conda activate /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/env

# Navigate to the code directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net

# Train Inf-Net from scratch with PyTorch 1.9.0
# Training on 50 doctor-labeled images
# Note: is_semi and is_pseudo default to False, so we don't pass them
python MyTrain_LungInf.py \
    --epoch 100 \
    --batchsize 8 \
    --num_workers 4 \
    --backbone Res2Net50 \
    --trainsize 352 \
    --lr 1e-4

echo "Training completed!"

