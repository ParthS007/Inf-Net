#!/bin/bash
#SBATCH --job-name=infnet_train_with_dp
#SBATCH --output=logs/train_dp_r%j.out
#SBATCH --error=logs/train_dp_r%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=gpu30min

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to thesis project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to the code directory
cd code/inf-net

python MyTrain_LungInfDP.py --run 1

echo "Training completed!"
