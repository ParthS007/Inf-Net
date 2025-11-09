#!/bin/bash
#SBATCH --job-name=eval_unet_nestedunet
#SBATCH --output=logs/eval/eval_unet_nestedunet_%j.out
#SBATCH --error=logs/eval/eval_unet_nestedunet_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4090
#SBATCH --qos=gpu30min

# Create logs directory if it doesn't exist
mkdir -p logs/eval

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to evaluation tool directory
cd code/inf-net/EvaluationToolPython

# Run evaluation for UNet results
echo "Evaluating UNet_GroupNorm..."
python main_all.py \
  --gt_path "../Dataset/TestingSet/LungInfection-Test/GT/" \
  --model_filter "UNet_GroupNorm" \
  --verbose

# Run evaluation for NestedUNet results
echo "Evaluating NestedUNet_GroupNorm..."
python main_all.py \
  --gt_path "../Dataset/TestingSet/LungInfection-Test/GT/" \
  --model_filter "NestedUNet_GroupNorm" \
  --verbose

echo "Evaluation completed! Reports saved under ../EvaluateResults/Lung_infection_segmentation"

