#!/usr/bin/env python3
"""
Evaluate ONLY the epsilon sweep predictions for Lung CT.

This script only evaluates the epsilon sweep models (epsilon 20-180),
not the existing epsilon 8 and 200 results.

Configuration matches the epsilon sweep training:
- Models: UNet_DP, NestedUNet_DP, Inf-Net_DP
- Epsilons: 20, 40, 60, 80, 100, 120, 140, 160, 180
- Batch Size: 24
- Run: 1
"""

import subprocess
import sys
import os

# Configuration matching the epsilon sweep
MODELS = ["UNet_DP", "NestedUNet_DP", "Inf-Net_DP"]
EPSILONS = [20, 40, 60, 80, 100, 120, 140, 160, 180]
BATCH_SIZE = 24
MAX_GRAD_NORM = 1.5
CLIPPING_STRATEGY = "automatic"
RUN = 1


def main():
    # Change to inf-net directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inf_net_dir = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(inf_net_dir)
    print(f"Working directory: {os.getcwd()}")

    # Use absolute path for GT
    gt_path = os.path.join(os.getcwd(), "Dataset/TestingSet/LungInfection-Test/GT/")
    print(f"Ground truth path: {gt_path}")

    # Change to EvaluationToolPython directory
    eval_dir = os.path.join(os.getcwd(), "EvaluationToolPython")
    os.chdir(eval_dir)
    print(f"Evaluation directory: {os.getcwd()}")

    total = len(MODELS) * len(EPSILONS)
    current = 0
    failed = []
    succeeded = []

    for model in MODELS:
        for epsilon in EPSILONS:
            current += 1
            # Build the result directory path
            result_dir = f"{model}/batch_{BATCH_SIZE}/run_{RUN}/epsilon_{epsilon}/maxgrad_{MAX_GRAD_NORM}/{CLIPPING_STRATEGY}"

            print(f"\n{'='*60}")
            print(f"[{current}/{total}] Evaluating: {model}, epsilon={epsilon}")
            print(f"Result dir: {result_dir}")
            print(f"{'='*60}")

            cmd = [
                sys.executable,
                "main_all.py",
                "--gt_path",
                gt_path,
                "--result_dir",
                result_dir,
                "--verbose",
            ]

            print(f"Command: {' '.join(cmd)}")

            try:
                result = subprocess.run(cmd, check=True)
                succeeded.append((model, epsilon))
                print(f"[SUCCESS] {model} epsilon={epsilon}")
            except subprocess.CalledProcessError as e:
                failed.append((model, epsilon, str(e)))
                print(f"[FAILED] {model} epsilon={epsilon}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed evaluations:")
        for model, epsilon, error in failed:
            print(f"  - {model} epsilon={epsilon}: {error}")

    print(
        "\nEvaluation results saved to: ./EvaluateResults/Lung_infection_segmentation/"
    )


if __name__ == "__main__":
    main()
