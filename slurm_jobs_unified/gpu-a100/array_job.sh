#!/bin/bash
#SBATCH --job-name=infnet_gpu-a100
#SBATCH --output=logs/train/infnet_gpu-a100_%A_%a.out
#SBATCH --error=logs/train/infnet_gpu-a100_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
#SBATCH --qos=gpu6hours
#SBATCH --array=1-162%20

# Create logs directory if it doesn't exist
mkdir -p logs/train

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Get the command for this array task
SEEDFILE=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm_jobs_unified/gpu-a100/commands.cmd
SEED=$(sed -n ${SLURM_ARRAY_TASK_ID}p $SEEDFILE)

# Execute the command
eval $SEED

echo "Training completed at $(date)"
