#!/bin/bash
#SBATCH --job-name=infnet_dp_eps1_batch24_run2
#SBATCH --output=logs/infnet_dp_eps1_batch24_run2_%j.out
#SBATCH --error=logs/infnet_dp_eps1_batch24_run2_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090
#SBATCH --qos=gpu30min

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Run training
python MyTrain_LungInfDP_Morph.py --batchsize 24 --run 1 --enable_privacy --noise_multiplier 10

echo "Training completed at $(date)"
