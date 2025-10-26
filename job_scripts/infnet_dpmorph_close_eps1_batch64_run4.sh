#!/bin/bash
#SBATCH --job-name=infnet_dpmorph_close_eps1_batch64_run4
#SBATCH --output=train_logs/infnet_dpmorph_close_eps1_batch64_run4_%j.out
#SBATCH --error=train_logs/infnet_dpmorph_close_eps1_batch64_run4_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
#SBATCH --qos=gpu30min

# Create logs directory if it doesn't exist
mkdir -p train_logs/

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Run training
python MyTrain_LungInfDP_Morph.py --batchsize 64 --run 1 --enable_privacy --noise_multiplier 0.5 --epoch 70 --max_grad_norm 1.2 --enable_morphology --morph_operation close

echo "Training completed at $(date)"
