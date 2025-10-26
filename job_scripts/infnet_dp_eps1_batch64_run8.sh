#!/bin/bash
#SBATCH --job-name=infnet_dp_eps1_batch64_run_8
#SBATCH --output=train_logs/infnet_dp_eps1_batch64_run_8_%j.out
#SBATCH --error=train_logs/infnet_dp_eps1_batch64_run_8_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
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
python MyTrain_LungInfDP_Morph.py --batchsize 64 --run 2 --enable_privacy --noise_multiplier 0.5 --epoch 70 --max_grad_norm 1.2

echo "Training completed at $(date)"
