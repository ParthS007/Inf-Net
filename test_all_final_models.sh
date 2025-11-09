#!/bin/bash
#SBATCH --job-name=infnet_test_all
#SBATCH --output=logs/test/test_all_%j.out
#SBATCH --error=logs/test/test_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
#SBATCH --qos=gpu6hours

# Create logs directory if it doesn't exist
mkdir -p logs/test

# Navigate to thesis project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to the code directory
cd code/inf-net

# Run batch testing on all final epoch models
echo "Starting batch testing of all final epoch models..."
echo "Timestamp: $(date)"
python MyTest_LungInf_All.py --test_all_final

echo ""
echo "Batch testing of all models completed!"
echo "Timestamp: $(date)"
