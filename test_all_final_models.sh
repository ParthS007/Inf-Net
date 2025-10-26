#!/bin/bash
#SBATCH --job-name=infnet_test_all
#SBATCH --output=test_logs/test_all_%j.out
#SBATCH --error=test_logs/test_all_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090
#SBATCH --qos=gpu30min

# Create logs directory if it doesn't exist
mkdir -p test_logs

# Navigate to thesis project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to the code directory
cd code/inf-net

# Run batch testing on all final epoch models (100.pth)
python MyTest_LungInf_All.py \
    --data_path "./Dataset/TestingSet/LungInfection-Test/" \
    --testsize 352 \
    --model_type Inf-Net_DP_Morph \
    --batchsize 64 \
    --run 1 \
    --noise_multiplier 0.5 \
    --epoch 70 \
    --gpu_device 0

echo "Batch testing of all models completed!"
