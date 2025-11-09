#!/bin/bash
#SBATCH --job-name=infnet_groupnorm_morph_close_batch72_run2
#SBATCH --output=logs/train/infnet_groupnorm_morph_close_batch72_run2_%j.out
#SBATCH --error=logs/train/infnet_groupnorm_morph_close_batch72_run2_%j.err
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
python MyTrain_LungInf_GroupNorm.py --batchsize 72 --run 2 --epoch 70 --enable_morphology --morph_operation close

echo "Training completed at $(date)"
