#!/usr/bin/env python3
"""
Run testing/inference on all epsilon sweep models for Lung CT.

This script loops through all epsilon sweep configurations and runs
MyTest_LungInf_All.py with the correct parameters for each model.

Configuration matches the epsilon sweep training:
- Models: UNet, NestedUNet, Inf_Net
- Epsilons: 20, 40, 60, 80, 100, 120, 140, 160, 180
- Batch Size: 24
- Max Grad Norm: 1.5
- Clipping Strategy: automatic
- Run: 1
- Epoch: 70
"""

import subprocess
import sys
import os

# Configuration matching the epsilon sweep training
MODELS = [
    ("UNet", "UNet_DP"),
    ("NestedUNet", "NestedUNet_DP"),
    ("Inf_Net", "Inf-Net_DP"),
]
EPSILONS = [20, 40, 60, 80, 100, 120, 140, 160, 180]
BATCH_SIZE = 24
MAX_GRAD_NORM = 1.5
CLIPPING_STRATEGY = "automatic"
RUN = 1
EPOCH = 70

DATA_PATH = "./Dataset/TestingSet/LungInfection-Test/"


def main():
    # Change to inf-net directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inf_net_dir = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(inf_net_dir)
    print(f"Working directory: {os.getcwd()}")

    total = len(MODELS) * len(EPSILONS)
    current = 0
    failed = []
    succeeded = []

    for network, model_type in MODELS:
        for epsilon in EPSILONS:
            current += 1
            print(f"\n{'='*60}")
            print(f"[{current}/{total}] Testing: {model_type}, epsilon={epsilon}")
            print(f"{'='*60}")

            cmd = [
                sys.executable,
                "MyTest_LungInf_All.py",
                "--model_type",
                model_type,
                "--batchsize",
                str(BATCH_SIZE),
                "--run",
                str(RUN),
                "--epsilon",
                str(epsilon),
                "--max_grad_norm",
                str(MAX_GRAD_NORM),
                "--clipping_strategy",
                CLIPPING_STRATEGY,
                "--epoch",
                str(EPOCH),
                "--data_path",
                DATA_PATH,
            ]

            print(f"Command: {' '.join(cmd)}")

            try:
                result = subprocess.run(cmd, check=True)
                succeeded.append((model_type, epsilon))
                print(f"[SUCCESS] {model_type} epsilon={epsilon}")
            except subprocess.CalledProcessError as e:
                failed.append((model_type, epsilon, str(e)))
                print(f"[FAILED] {model_type} epsilon={epsilon}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for model_type, epsilon, error in failed:
            print(f"  - {model_type} epsilon={epsilon}: {error}")

    print("\nPredictions saved to: ./Results/Lung_infection_segmentation/")
    print("Run evaluation with: sbatch slurm/epsilon_sweep/eval_epsilon_sweep.sh")


if __name__ == "__main__":
    main()
