#!/bin/bash
#SBATCH --job-name=infnet_dp_nm0.7_batch64_run2
#SBATCH --output=logs/train/infnet_dp_nm0.7_batch64_run2_%j.out
#SBATCH --error=logs/train/infnet_dp_nm0.7_batch64_run2_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
#SBATCH --qos=gpu30min

# Create logs directory if it doesn't exist
mkdir -p logs/train

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Run training
python MyTrain_LungInfDP_Morph.py --batchsize 64 --run 2 --epoch 70 --enable_privacy --noise_multiplier 0.7 --max_grad_norm 1.2

echo "Training completed at $(date)"
