#!/bin/bash
#SBATCH --job-name=unet_eps_sweep
#SBATCH --output=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/logs/unet-eps-sweep/unet_eps_sweep_%A_%a.out
#SBATCH --error=/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/logs/unet-eps-sweep/unet_eps_sweep_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-80g
#SBATCH --qos=gpu6hours
#SBATCH --array=1-9%16

# Create logs directory if it doesn't exist
mkdir -p /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/logs/unet-eps-sweep

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya

# Activate virtual environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Get the command for this array task
COMMANDS_FILE="/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/slurm/epsilon_sweep/unet-eps-sweep.txt"
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMANDS_FILE")

echo "=========================================="
echo "Job Name: unet_eps_sweep"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Command: $COMMAND"
echo "=========================================="

# Execute the command
eval $COMMAND

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
