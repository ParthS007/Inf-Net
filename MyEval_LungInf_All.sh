#!/bin/bash
#SBATCH --job-name=infnet_eval_all
#SBATCH --output=eval_logs/eval_all_%j.out
#SBATCH --error=eval_logs/eval_all_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090
#SBATCH --qos=gpu30min

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
  --model_filter "Inf-Net_DP_Morph" \
  --batch_filter "64" \
  --run_filter "1" \
  --verbose

echo "Evaluation completed! Reports saved under ../EvaluateResults/Lung_infection_segmentation"