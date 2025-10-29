#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster Job Submission Script for Inf-Net Training Variants
Submits jobs for:
- Inf-Net (standard with BatchNorm)
- Inf-Net-Morph (with morphological operations and BatchNorm)
- Inf-Net-GroupNorm (GroupNorm without DP)
- Inf-Net-Morph-GroupNorm (GroupNorm + Morphology without DP)
- Inf-Net-DP (with Differential Privacy)
- Inf-Net-DP-Morph (DP + Morphology)
- Different batch sizes (24, 48, 64, 72)
- Different morphology operations (open, close, dilation, erosion, both)
"""

import os
import argparse
import subprocess
from datetime import datetime


def create_sbatch_script(
    job_name,
    output_log,
    error_log,
    python_script,
    python_args,
    time_limit="00:15:00",
    mem="64G",
    cpus=4,
    gpus=1,
    partition="a100-80g",
    qos="gpu30min",
):
    """Create an SBATCH script content"""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_log}
#SBATCH --error={error_log}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --partition={partition}
#SBATCH --qos={qos}

# Create logs directory if it doesn't exist
mkdir -p logs/train

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Run training
python {python_script} {python_args}

echo "Training completed at $(date)"
"""
    return script


def submit_job(script_content, script_path):
    """Submit a job to the cluster"""
    # Write script to file
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make script executable
    os.chmod(script_path, 0o755)

    # Submit job
    try:
        result = subprocess.run(
            ["sbatch", script_path], capture_output=True, text=True, check=True
        )
        job_id = result.stdout.strip().split()[-1]
        print(f"✓ Submitted: {script_path}")
        print(f"  Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit {script_path}")
        print(f"  Error: {e.stderr}")
        return None


def generate_training_configs():
    """Generate all training configurations"""
    configs = []

    batch_sizes = [24, 48, 64, 72]
    morph_operations = ["open", "close", "both", "dilation", "erosion"]
    noise_multiplier = [0.3, 0.5, 0.7]
    epoch = 70
    max_grad_norm = 1.2

    # Standard Inf-Net (no morphology, no DP)
    for batch_size in batch_sizes:
        for run in range(1, 4):  # 3 runs per configuration
            config = {
                "name": f"infnet_batch{batch_size}_run{run}",
                "model_type": "Inf-Net",
                "batch_size": batch_size,
                "run": run,
                "epoch": epoch,
                "enable_morphology": False,
                "enable_privacy": False,
                "enable_groupnorm": False,
                "morph_operation": None,
                "script": "MyTrain_LungInf_Morph.py",
            }
            configs.append(config)

    # Inf-Net with Morphology (no DP)
    for batch_size in batch_sizes:
        for morph_op in morph_operations:
            for run in range(1, 4):  # 3 runs per configuration
                config = {
                    "name": f"infnet_morph_{morph_op}_batch{batch_size}_run{run}",
                    "model_type": "Inf-Net_Morph",
                    "batch_size": batch_size,
                    "morph_operation": morph_op,
                    "epoch": epoch,
                    "run": run,
                    "enable_morphology": True,
                    "enable_privacy": False,
                    "enable_groupnorm": False,
                    "script": "MyTrain_LungInf_Morph.py",
                }
                configs.append(config)

    # Inf-Net with GroupNorm (no DP, no morphology)
    for batch_size in batch_sizes:
        for run in range(1, 4):  # 3 runs per configuration
            config = {
                "name": f"infnet_groupnorm_batch{batch_size}_run{run}",
                "model_type": "Inf-Net_GroupNorm",
                "batch_size": batch_size,
                "run": run,
                "epoch": epoch,
                "enable_morphology": False,
                "enable_privacy": False,
                "enable_groupnorm": True,
                "morph_operation": None,
                "script": "MyTrain_LungInf_GroupNorm.py",
            }
            configs.append(config)

    # Inf-Net with GroupNorm and Morphology (no DP)
    for batch_size in batch_sizes:
        for morph_op in morph_operations:
            for run in range(1, 4):  # 3 runs per configuration
                config = {
                    "name": f"infnet_groupnorm_morph_{morph_op}_batch{batch_size}_run{run}",
                    "model_type": "Inf-Net_Morph_GroupNorm",
                    "batch_size": batch_size,
                    "morph_operation": morph_op,
                    "epoch": epoch,
                    "run": run,
                    "enable_morphology": True,
                    "enable_privacy": False,
                    "enable_groupnorm": True,
                    "script": "MyTrain_LungInf_GroupNorm.py",
                }
                configs.append(config)

    # Inf-Net with DP (no morphology)
    for batch_size in batch_sizes:
        for nm in noise_multiplier:
            for run in range(1, 4):  # 3 runs per configuration
                config = {
                    "name": f"infnet_dp_nm{nm}_batch{batch_size}_run{run}",
                    "model_type": "Inf-Net_DP",
                    "batch_size": batch_size,
                    "run": run,
                    "epoch": epoch,
                    "enable_morphology": False,
                    "enable_privacy": True,
                    "enable_groupnorm": False,
                    "noise_multiplier": nm,
                    "max_grad_norm": max_grad_norm,
                    "morph_operation": None,
                    "script": "MyTrain_LungInfDP_Morph.py",
                }
                configs.append(config)

    # Inf-Net with DP and Morphology
    for batch_size in batch_sizes:
        for nm in noise_multiplier:
            for morph_op in morph_operations:
                for run in range(1, 4):  # 3 runs per configuration
                    config = {
                        "name": f"infnet_dpmorph_{morph_op}_nm{nm}_batch{batch_size}_run{run}",
                        "model_type": "Inf-Net_DP_Morph",
                        "batch_size": batch_size,
                        "morph_operation": morph_op,
                        "run": run,
                        "epoch": epoch,
                        "enable_morphology": True,
                        "enable_privacy": True,
                        "enable_groupnorm": False,
                        "noise_multiplier": nm,
                        "max_grad_norm": max_grad_norm,
                        "script": "MyTrain_LungInfDP_Morph.py",
                    }
                    configs.append(config)

    return configs


def build_python_args(config):
    """Build python command line arguments from config"""
    args = [
        f"--batchsize {config['batch_size']}",
        f"--run {config['run']}",
        f"--epoch {config['epoch']}",
    ]

    if config["enable_privacy"]:
        args.append("--enable_privacy")
        args.append(f"--noise_multiplier {config['noise_multiplier']}")
        args.append(f"--max_grad_norm {config['max_grad_norm']}")

    if config["enable_morphology"]:
        args.append("--enable_morphology")
        args.append(f"--morph_operation {config['morph_operation']}")

    return " ".join(args)


def main():
    parser = argparse.ArgumentParser(
        description="Submit Inf-Net training jobs to cluster"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts without submitting",
    )
    parser.add_argument(
        "--configs",
        type=str,
        choices=[
            "standard",
            "morph",
            "groupnorm",
            "groupnorm_morph",
            "dp",
            "dpmorph",
            "all",
        ],
        default="all",
        help="Which configurations to submit",
    )
    parser.add_argument(
        "--time-limit",
        type=str,
        default="00:15:00",
        help="SLURM time limit (format: HH:MM:SS)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[24, 48, 64, 72],
        help="Batch sizes to train with",
    )
    parser.add_argument(
        "--morph-operations",
        type=str,
        nargs="+",
        default=["open", "close", "both", "dilation", "erosion"],
        help="Morphological operations to apply",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--script-dir",
        type=str,
        default="./job_scripts",
        help="Directory to save job scripts",
    )

    args = parser.parse_args()

    # Create script directory
    os.makedirs(args.script_dir, exist_ok=True)
    os.makedirs("logs/train", exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Inf-Net Training Job Submission Script")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    # Generate all configs
    all_configs = generate_training_configs()

    # Filter configs based on user selection
    if args.configs == "standard":
        configs = [
            c
            for c in all_configs
            if not c["enable_morphology"]
            and not c["enable_privacy"]
            and not c["enable_groupnorm"]
        ]
    elif args.configs == "morph":
        configs = [
            c
            for c in all_configs
            if c["enable_morphology"]
            and not c["enable_privacy"]
            and not c["enable_groupnorm"]
        ]
    elif args.configs == "groupnorm":
        configs = [
            c
            for c in all_configs
            if c["enable_groupnorm"] and not c["enable_morphology"]
        ]
    elif args.configs == "groupnorm_morph":
        configs = [
            c for c in all_configs if c["enable_groupnorm"] and c["enable_morphology"]
        ]
    elif args.configs == "dp":
        configs = [
            c for c in all_configs if c["enable_privacy"] and not c["enable_morphology"]
        ]
    elif args.configs == "dpmorph":
        configs = [
            c for c in all_configs if c["enable_privacy"] and c["enable_morphology"]
        ]
    else:
        configs = all_configs

    # Update batch sizes if specified
    if args.batch_sizes != [24, 48, 64, 72]:
        configs = [c for c in configs if c["batch_size"] in args.batch_sizes]

    # Update morphology operations if specified
    if args.morph_operations != ["open", "close", "both", "dilation", "erosion"]:
        configs = [
            c
            for c in configs
            if not c["enable_morphology"]
            or c["morph_operation"] in args.morph_operations
        ]

    print(f"Total configurations to process: {len(configs)}\n")
    print("Configuration Summary:")
    print("-" * 70)

    job_ids = []

    for i, config in enumerate(configs, 1):
        model_type = config["model_type"]
        batch_size = config["batch_size"]
        run = config["run"]
        morph_info = (
            f" | Morph: {config['morph_operation']}"
            if config["enable_morphology"]
            else ""
        )
        privacy_info = (
            f" | NM={config['noise_multiplier']}" if config["enable_privacy"] else ""
        )

        print(
            f"{i:3d}. {model_type:20s} | Batch: {batch_size:3d} | Run: {run}{privacy_info}{morph_info}"
        )

        # Create job script
        python_args = build_python_args(config)
        job_name = config["name"]
        script_name = f"{job_name}.sh"
        script_path = os.path.join(args.script_dir, script_name)

        output_log = f"logs/train/{job_name}_%j.out"
        error_log = f"logs/train/{job_name}_%j.err"

        sbatch_script = create_sbatch_script(
            job_name=job_name,
            output_log=output_log,
            error_log=error_log,
            python_script=config["script"],
            python_args=python_args,
            time_limit=args.time_limit,
        )

        if args.dry_run:
            # Just save scripts without submitting
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            with open(script_path, "w") as f:
                f.write(sbatch_script)
            os.chmod(script_path, 0o755)
            print(f"   → Script created (dry-run): {script_path}")
        else:
            # Submit to cluster
            job_id = submit_job(sbatch_script, script_path)
            if job_id:
                job_ids.append((config["name"], job_id))

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print(f"✓ Dry-run completed! Scripts saved to: {args.script_dir}")
        print(f"  To submit jobs, run: sbatch {args.script_dir}/*.sh")
    else:
        print(f"✓ Job submission completed!")
        print(f"  Total jobs submitted: {len(job_ids)}")

    if job_ids:
        print(f"\n{'Submitted Jobs:':^70}")
        print("-" * 70)
        for name, job_id in job_ids:
            print(f"  {name:40s} → Job ID: {job_id}")

    print(f"{'=' * 70}\n")

    # Save job submission log
    log_file = "job_submission_log.txt"
    with open(log_file, "a") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"Submission: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configurations: {len(configs)}\n")
        for name, job_id in job_ids:
            f.write(f"  {name}: {job_id}\n")

    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
