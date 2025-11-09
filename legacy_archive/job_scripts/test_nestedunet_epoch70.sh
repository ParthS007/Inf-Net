#!/bin/bash
#SBATCH --job-name=test_nestedunet_epoch70
#SBATCH --output=logs/test/test_nestedunet_epoch70_%j.out
#SBATCH --error=logs/test/test_nestedunet_epoch70_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090
#SBATCH --qos=gpu30min

# Create logs directory if it doesn't exist
mkdir -p logs/test

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Run testing for NestedUNet epoch 70
# Note: batchsize is 12 because that's what was used for training
python MyTest_LungInf_UNet_NestedUNet.py \
    --model_type NestedUNet_GroupNorm \
    --batchsize 12 \
    --run 1 \
    --epoch 70

echo "Testing completed at $(date)"

