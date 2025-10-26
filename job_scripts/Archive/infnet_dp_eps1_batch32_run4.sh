#!/bin/bash
#SBATCH --job-name=infnet_dp_eps1_batch32_run1
#SBATCH --output=train_logs/infnet_dp_eps1_batch32_run4_%j.out
#SBATCH --error=train_logs/infnet_dp_eps1_batch32_run4_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090
#SBATCH --qos=gpu30min

# Create train_logs directory if it doesn't exist
mkdir -p train_logs

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Run training
python MyTrain_LungInfDP_Morph.py --batchsize 32 --run 4 --enable_privacy --noise_multiplier 1.5 --max_grad_norm 1.2

echo "Training completed at $(date)"
