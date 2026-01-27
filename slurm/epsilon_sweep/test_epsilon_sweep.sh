#!/bin/bash
#SBATCH --job-name=test_eps_sweep
#SBATCH --output=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/logs/test_eps_sweep_%j.out
#SBATCH --error=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/logs/test_eps_sweep_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
#SBATCH --qos=gpu6hours

# Create logs directory if it doesn't exist
mkdir -p /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/logs

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya

# Activate virtual environment
source .venv/bin/activate

# Navigate to inf-net directory
cd code/inf-net

echo "=========================================="
echo "Testing Epsilon Sweep Models"
echo "Start time: $(date)"
echo "=========================================="

# Run the test script
python slurm/epsilon_sweep/test_epsilon_sweep.py

echo "=========================================="
echo "Testing completed at $(date)"
echo "=========================================="
