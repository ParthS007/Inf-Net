#!/bin/bash
#SBATCH --job-name=infnet_dpmorph_erosion_eps1_batch64_run2
#SBATCH --output=logs/infnet_dpmorph_erosion_eps1_batch64_run2_%j.out
#SBATCH --error=logs/infnet_dpmorph_erosion_eps1_batch64_run2_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Run training
python MyTrain_LungInfDP_Morph.py --batchsize 64 --run 2 --enable_privacy --noise_multiplier 1.5 --enable_morphology --morph_operation erosion

echo "Training completed at $(date)"
