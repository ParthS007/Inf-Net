#!/bin/bash
#SBATCH --job-name=infnet_eval_all
#SBATCH --output=eval_logs/eval_all_%j.out
#SBATCH --error=eval_logs/eval_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours

# Create logs directory if it doesn't exist
mkdir -p eval_logs

# Navigate to thesis project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate virtual environment
source .venv/bin/activate

# Navigate to evaluation tool directory
cd code/inf-net/EvaluationToolPython

# Run evaluation over all predictions found in ../Results/Lung_infection_segmentation
python main_all.py \
  --gt_path "../Dataset/TestingSet/LungInfection-Test/GT/" \
  --verbose

echo "Evaluation completed! Reports saved under ../EvaluateResults/Lung_infection_segmentation"