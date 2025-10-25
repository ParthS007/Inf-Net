#!/bin/bash
#SBATCH --job-name=infnet_test_all
#SBATCH --output=test_logs/test_all_%j.out
#SBATCH --error=test_logs/test_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours

# Create logs directory if it doesn't exist
mkdir -p test_logs

# Navigate to thesis project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to the code directory
cd code/inf-net

# Run batch testing on all final epoch models (100.pth)
python MyTest_LungInf_all.py \
    --test_all_final \
    --data_path "./Dataset/TestingSet/LungInfection-Test/" \
    --testsize 352 \
    --gpu_device 0

echo "Batch testing of all models completed!"
