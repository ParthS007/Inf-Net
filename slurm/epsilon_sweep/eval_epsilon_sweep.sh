#!/bin/bash
#SBATCH --job-name=eval_eps_sweep
#SBATCH --output=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/logs/eval_eps_sweep_%j.out
#SBATCH --error=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/logs/eval_eps_sweep_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
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
echo "Evaluating Epsilon Sweep Predictions (ONLY)"
echo "Start time: $(date)"
echo "=========================================="

# Run evaluation ONLY for epsilon sweep models (epsilon 20-180)
python slurm/epsilon_sweep/eval_epsilon_sweep_only.py

echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Reports saved to: ./EvaluateResults/Lung_infection_segmentation"
echo "=========================================="
