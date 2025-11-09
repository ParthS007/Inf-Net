#!/bin/bash
#SBATCH --job-name=infnet_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to thesis project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to the code directory
cd code/inf-net

# Run testing with standard Inf-Net model
python MyTest_LungInf_all.py \
    --model_type "Inf-Net" \
    --batchsize 64 \
    --run 1 \
    --epoch 100 \
    --data_path "./Dataset/TestingSet/LungInfection-Test/" \
    --testsize 352 \
    --gpu_device 0

echo "Testing completed!"
