#!/usr/bin/env python3
"""
Generate SLURM experiment scripts for epsilon sweep experiments (Lung CT).

This script generates experiments to study how Dice score varies with epsilon
(privacy budget) from 20 to 180 in increments of 20.

Two variants:
1. Standard (no morph): DP with clipping_strategy automatic only.
2. Morph close k5: DP with automatic + morphology close, kernel size 5.

Configuration:
- Clipping Strategy: automatic
- Batch Size: 24
- Max Grad Norm: 1.5
- Run: 1
- Epsilons: 20, 40, 60, 80, 100, 120, 140, 160, 180 (skip 8 and 200 - already exist)
- Results Base: /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/results_epsilon_sweep
- Script: MyTrainTestEval_Unified.py (train + test + eval in one job)

Usage:
    python generate_epsilon_sweep.py           # Generate all experiments
    python generate_epsilon_sweep.py --dry-run # Show what would be generated
"""

import os
import argparse
from pathlib import Path

# Base paths
SLURM_DIR = Path(__file__).parent
PROJECT_ROOT = "/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya"

# Experiment configurations
MODELS = ["UNet", "NestedUNet", "Inf_Net"]
MODEL_NAMES = {"UNet": "unet", "NestedUNet": "nestedunet", "Inf_Net": "inf_net"}

# Fixed parameters for epsilon sweep
BATCH_SIZE = 24
MAX_GRAD_NORM = 1.5
RUN_NUMBER = 1
CLIPPING_STRATEGY = "automatic"
RESULTS_BASE = "/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/results_epsilon_sweep"

# Epsilon values to test (skip 8 and 200 as they already exist in results/)
EPSILONS = [20, 40, 60, 80, 100, 120, 140, 160, 180]


def generate_sh_script(
    job_name, log_dir, array_size, txt_file, time="01:00:00", partition="a100-80g"
):
    """Generate SLURM shell script content. Morph close k5 uses partition=rtx4090."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={PROJECT_ROOT}/code/inf-net/slurm/epsilon_sweep/logs/{log_dir}/{job_name}_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/code/inf-net/slurm/epsilon_sweep/logs/{log_dir}/{job_name}_%A_%a.err
#SBATCH --time={time}
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition={partition}
#SBATCH --qos=gpu6hours
#SBATCH --array=1-{array_size}%16

# Create logs directory if it doesn't exist
mkdir -p {PROJECT_ROOT}/code/inf-net/slurm/epsilon_sweep/logs/{log_dir}

# Navigate to project directory
cd {PROJECT_ROOT}

# Activate virtual environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Get the command for this array task
COMMANDS_FILE="{PROJECT_ROOT}/code/inf-net/slurm/epsilon_sweep/{txt_file}"
COMMAND=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$COMMANDS_FILE")

echo "=========================================="
echo "Job Name: {job_name}"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Command: $COMMAND"
echo "=========================================="

# Execute the command
eval $COMMAND

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
"""


def generate_epsilon_sweep_commands(model, morph_close_k5=False):
    """Generate epsilon sweep commands for a given model.

    morph_close_k5: if True, add --enable_morphology --morph_operation close --morph_kernel_size 5.
    """
    commands = []
    for epsilon in EPSILONS:
        cmd = (
            f"python MyTrainTestEval_Unified.py "
            f"--run {RUN_NUMBER} "
            f"--network {model} "
            f"--batchsize {BATCH_SIZE} "
            f"--enable_privacy "
            f"--epsilon {epsilon} "
            f"--max_grad_norm {MAX_GRAD_NORM} "
            f"--clipping_strategy {CLIPPING_STRATEGY} "
            f"--results_base {RESULTS_BASE} "
            f"--skip_aggregation"
        )
        if morph_close_k5:
            cmd += (
                " --enable_morphology --morph_operation close --morph_kernel_size 5"
            )
        commands.append(cmd)
    return commands


def write_files(
    job_name, log_dir, txt_file, sh_file, commands, dry_run=False, partition=None
):
    """Write command file and SLURM script. partition defaults to a100-80g."""
    if dry_run:
        print(f"  Would create: {txt_file} ({len(commands)} commands)")
        print(f"  Would create: {sh_file}")
        for i, cmd in enumerate(commands[:3], 1):
            print(f"    Command {i}: {cmd}")
        if len(commands) > 3:
            print(f"    ... and {len(commands) - 3} more commands")
        return 2

    # Write txt file
    txt_path = SLURM_DIR / txt_file
    with open(txt_path, "w") as f:
        f.write("\n".join(commands) + "\n")
    print(f"  Created: {txt_file} ({len(commands)} commands)")

    # Write sh file
    sh_path = SLURM_DIR / sh_file
    part = partition if partition is not None else "a100-80g"
    with open(sh_path, "w") as f:
        f.write(
            generate_sh_script(job_name, log_dir, len(commands), txt_file, partition=part)
        )
    print(f"  Created: {sh_file}")

    return 2


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for epsilon sweep experiments (Lung CT)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating files",
    )
    args = parser.parse_args()

    total_experiments = 0
    files_created = 0

    print("=" * 60)
    print("Epsilon Sweep Experiment Generator (Lung CT)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Clipping Strategy: {CLIPPING_STRATEGY}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Grad Norm: {MAX_GRAD_NORM}")
    print(f"  Run Number: {RUN_NUMBER}")
    print(f"  Results Base: {RESULTS_BASE}")
    print(f"  Epsilons: {EPSILONS}")
    print(f"  Models: {list(MODEL_NAMES.values())}")
    print(f"  Variants: standard (no morph), morph_close_k5 (automatic)")

    for model in MODELS:
        model_short = MODEL_NAMES[model]

        # Standard epsilon sweep (no morph)
        job_name = f"{model_short}_eps_sweep"
        log_dir = f"{model_short}-eps-sweep"
        txt_file = f"{model_short}-eps-sweep.txt"
        sh_file = f"{model_short}_eps_sweep.sh"

        print(f"\n{model_short} (standard):")

        commands = generate_epsilon_sweep_commands(model, morph_close_k5=False)
        files_created += write_files(
            job_name, log_dir, txt_file, sh_file, commands, args.dry_run
        )
        total_experiments += len(commands)

        # Morph close kernel 5 epsilon sweep (automatic)
        job_name_morph = f"{model_short}_eps_sweep_morph_close_k5"
        log_dir_morph = f"{model_short}-eps-sweep-morph-close-k5"
        txt_file_morph = f"{model_short}-eps-sweep-morph-close-k5.txt"
        sh_file_morph = f"{model_short}_eps_sweep_morph_close_k5.sh"

        print(f"{model_short} (morph close k5):")

        commands_morph = generate_epsilon_sweep_commands(
            model, morph_close_k5=True
        )
        files_created += write_files(
            job_name_morph,
            log_dir_morph,
            txt_file_morph,
            sh_file_morph,
            commands_morph,
            args.dry_run,
            partition="rtx4090",
        )
        total_experiments += len(commands_morph)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Files {'would be ' if args.dry_run else ''}created: {files_created}")
    print(f"\nExperiments per model: {len(EPSILONS)} (standard) + {len(EPSILONS)} (morph close k5)")

    if not args.dry_run:
        print(f"\nFiles written to: {SLURM_DIR}")
        print("\nTo submit all experiments:")
        print(f"  cd {SLURM_DIR}")
        print("  for f in *_eps_sweep*.sh; do sbatch $f; done")
        print("\nOr submit individually:")
        for model in MODELS:
            model_short = MODEL_NAMES[model]
            print(f"  sbatch {model_short}_eps_sweep.sh")
            print(f"  sbatch {model_short}_eps_sweep_morph_close_k5.sh")


if __name__ == "__main__":
    main()
