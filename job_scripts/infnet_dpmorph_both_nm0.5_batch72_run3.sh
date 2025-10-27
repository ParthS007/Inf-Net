#!/bin/bash
#SBATCH --job-name=infnet_dpmorph_both_nm0.5_batch72_run3
#SBATCH --output=logs/train/infnet_dpmorph_both_nm0.5_batch72_run3_%j.out
#SBATCH --error=logs/train/infnet_dpmorph_both_nm0.5_batch72_run3_%j.err
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
python MyTrain_LungInfDP_Morph.py --batchsize 72 --run 3 --epoch 70 --enable_privacy --noise_multiplier 0.5 --max_grad_norm 1.2 --enable_morphology --morph_operation both

echo "Training completed at $(date)"
