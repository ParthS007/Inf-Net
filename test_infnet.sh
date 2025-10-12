#!/bin/bash
#SBATCH --job-name=infnet_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100-30min

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to thesis project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to the code directory
cd code/inf-net

# Run testing with Inf-Net pre-trained weights
python MyTest_LungInf.py \
    --data_path "./Dataset/TestingSet/LungInfection-Test/" \
    --pth_path "./Snapshots/save_weights/Inf-Net/6/Inf-Net-100.pth" \
    --save_path "./Results/Lung_infection_segmentation/Inf-Net/" \
    --run 6

echo "Testing completed!"
