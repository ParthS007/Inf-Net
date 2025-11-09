#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Cluster Job Submission Script for Inf-Net Training using SLURM Array Jobs
Submits jobs for all three networks with all configurations:
- Inf_Net GroupNorm, UNet_GroupNorm, NestedUNet_GroupNorm
- Base (no DP, no Morph)
- Base with/without Morph
- Base with/without DP (with clipping strategies: base, automatic, psac, nsgd)
- Batch sizes: 24, 48, 64
- Epsilon: 8, 200
- Morph operation: both only
- Epoch: 100
- Max grad norm: 1.2
- 3 runs per configuration
- Jobs scheduled evenly across partitions using array jobs
"""

import os
import argparse
import subprocess
from datetime import datetime
from collections import defaultdict


def generate_training_configs():
    """Generate all training configurations"""
    configs = []

    networks = ["Inf_Net", "UNet", "NestedUNet"]
    batch_sizes = [24, 48, 64]
    epsilon_values = [8, 200]
    clipping_strategies = ["base", "automatic", "psac", "nsgd"]
    num_runs = 3
    epoch = 100
    max_grad_norm = 1.2
    morph_operation = "both"  # Only "both" as specified

    for network in networks:
        # 1. Base (no DP, no Morph)
        for batch_size in batch_sizes:
            for run in range(1, num_runs + 1):
                config = {
                    "network": network,
                    "name": f"{network.lower()}_base_batch{batch_size}_run{run}",
                    "batch_size": batch_size,
                    "run": run,
                    "epoch": epoch,
                    "enable_morphology": False,
                    "enable_privacy": False,
                    "morph_operation": None,
                    "epsilon": None,
                    "clipping_strategy": None,
                    "max_grad_norm": None,
                }
                configs.append(config)

        # 2. Base with Morph (no DP)
        for batch_size in batch_sizes:
            for run in range(1, num_runs + 1):
                config = {
                    "network": network,
                    "name": f"{network.lower()}_morph_batch{batch_size}_run{run}",
                    "batch_size": batch_size,
                    "run": run,
                    "epoch": epoch,
                    "enable_morphology": True,
                    "enable_privacy": False,
                    "morph_operation": morph_operation,
                    "epsilon": None,
                    "clipping_strategy": None,
                    "max_grad_norm": None,
                }
                configs.append(config)

        # 3. Base with DP (no Morph) - all clipping strategies
        for batch_size in batch_sizes:
            for eps in epsilon_values:
                for clipping in clipping_strategies:
                    for run in range(1, num_runs + 1):
                        config = {
                            "network": network,
                            "name": f"{network.lower()}_dp_{clipping}_eps{eps}_batch{batch_size}_run{run}",
                            "batch_size": batch_size,
                            "run": run,
                            "epoch": epoch,
                            "enable_morphology": False,
                            "enable_privacy": True,
                            "morph_operation": None,
                            "epsilon": eps,
                            "clipping_strategy": clipping,
                            "max_grad_norm": max_grad_norm,
                        }
                        configs.append(config)

        # 4. Base with DP and Morph - all clipping strategies
        for batch_size in batch_sizes:
            for eps in epsilon_values:
                for clipping in clipping_strategies:
                    for run in range(1, num_runs + 1):
                        config = {
                            "network": network,
                            "name": f"{network.lower()}_dpmorph_{clipping}_eps{eps}_batch{batch_size}_run{run}",
                            "batch_size": batch_size,
                            "run": run,
                            "epoch": epoch,
                            "enable_morphology": True,
                            "enable_privacy": True,
                            "morph_operation": morph_operation,
                            "epsilon": eps,
                            "clipping_strategy": clipping,
                            "max_grad_norm": max_grad_norm,
                        }
                        configs.append(config)

    return configs


def build_python_command(config):
    """Build python command line from config"""
    args = [
        f"--network {config['network']}",
        f"--batchsize {config['batch_size']}",
        f"--run {config['run']}",
        f"--epoch {config['epoch']}",
    ]

    if config["enable_privacy"]:
        args.append("--enable_privacy")
        args.append(f"--epsilon {config['epsilon']}")
        args.append(f"--max_grad_norm {config['max_grad_norm']}")
        args.append(f"--clipping_strategy {config['clipping_strategy']}")

    if config["enable_morphology"]:
        args.append("--enable_morphology")
        args.append(f"--morph_operation {config['morph_operation']}")

    cmd = f"python MyTrain_LungInf_Unified.py {' '.join(args)}"
    return cmd


def create_array_job_script(
    job_name,
    commands_file,
    num_tasks,
    max_concurrent,
    output_log,
    error_log,
    time_limit="06:00:00",
    mem="64G",
    cpus=4,
    gpus=1,
    partition="a100-80g",
    qos="gpu6hours",
):
    """Create a SLURM array job script"""
    array_spec = f"1-{num_tasks}%{max_concurrent}" if max_concurrent > 0 else f"1-{num_tasks}"

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
#SBATCH --array={array_spec}

# Create logs directory if it doesn't exist
mkdir -p logs/train

# Navigate to project directory
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net code directory
cd code/inf-net

# Get the command for this array task
SEEDFILE={commands_file}
SEED=$(sed -n ${{SLURM_ARRAY_TASK_ID}}p $SEEDFILE)

# Execute the command
eval $SEED

echo "Training completed at $(date)"
"""
    return script


def get_available_partitions():
    """Get available partitions - adjust based on your cluster"""
    # Common partitions - adjust these based on your cluster setup
    partitions = [
        "a100-80g",
        "gpu",
        "gpu-a100",
        # Add more partitions as needed
    ]
    return partitions


def distribute_jobs_across_partitions(configs, partitions):
    """Distribute jobs evenly across available partitions"""
    partition_configs = defaultdict(list)

    # Simple round-robin distribution
    for i, config in enumerate(configs):
        partition = partitions[i % len(partitions)]
        partition_configs[partition].append(config)

    return partition_configs


def main():
    parser = argparse.ArgumentParser(
        description="Submit unified Inf-Net training jobs to cluster using array jobs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts without submitting",
    )
    parser.add_argument(
        "--networks",
        type=str,
        nargs="+",
        choices=["Inf_Net", "UNet", "NestedUNet"],
        default=["Inf_Net", "UNet", "NestedUNet"],
        help="Which networks to train",
    )
    parser.add_argument(
        "--time-limit",
        type=str,
        default="06:00:00",
        help="SLURM time limit (format: HH:MM:SS)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[24, 48, 64],
        help="Batch sizes to train with",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./slurm_jobs_unified",
        help="Directory to save array job scripts and commands files",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        nargs="+",
        default=None,
        help="Partitions to use (default: auto-detect or use a100-80g, gpu)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum number of concurrent array tasks per partition",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs/train", exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Unified Inf-Net Training Job Submission Script (Array Jobs)")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    # Generate all configs
    all_configs = generate_training_configs()

    # Filter by networks if specified
    if args.networks != ["Inf_Net", "UNet", "NestedUNet"]:
        all_configs = [c for c in all_configs if c["network"] in args.networks]

    # Filter by batch sizes if specified
    if args.batch_sizes != [24, 48, 64]:
        all_configs = [c for c in all_configs if c["batch_size"] in args.batch_sizes]

    # Adjust num_runs if specified
    if args.num_runs != 3:
        # Filter to only first N runs
        filtered_configs = []
        for config in all_configs:
            if config["run"] <= args.num_runs:
                filtered_configs.append(config)
        all_configs = filtered_configs

    # Get partitions
    if args.partitions:
        partitions = args.partitions
    else:
        partitions = get_available_partitions()

    # Distribute jobs across partitions
    partition_configs = distribute_jobs_across_partitions(all_configs, partitions)

    print(f"Total configurations to process: {len(all_configs)}\n")
    print("Partition Distribution:")
    print("-" * 70)
    for partition in sorted(partition_configs.keys()):
        count = len(partition_configs[partition])
        print(f"  {partition:20s}: {count:4d} jobs")
    print("-" * 70)

    # Generate commands files and array scripts for each partition
    job_ids = []
    summary_info = []

    for partition, configs in partition_configs.items():
        if not configs:
            continue

        # Create partition-specific directory
        partition_dir = os.path.join(args.output_dir, partition)
        os.makedirs(partition_dir, exist_ok=True)

        # Generate commands.cmd file
        commands_file = os.path.join(partition_dir, "commands.cmd")
        with open(commands_file, "w") as f:
            for config in configs:
                cmd = build_python_command(config)
                f.write(cmd + "\n")

        num_tasks = len(configs)
        job_name = f"infnet_{partition}"
        array_script_name = "array_job.sh"
        array_script_path = os.path.join(partition_dir, array_script_name)

        # Use absolute path for commands file in the script
        commands_file_abs = os.path.abspath(commands_file)

        output_log = f"logs/train/{job_name}_%A_%a.out"
        error_log = f"logs/train/{job_name}_%A_%a.err"

        # Create array job script
        array_script = create_array_job_script(
            job_name=job_name,
            commands_file=commands_file_abs,
            num_tasks=num_tasks,
            max_concurrent=args.max_concurrent,
            output_log=output_log,
            error_log=error_log,
            time_limit=args.time_limit,
            partition=partition,
        )

        # Write array script
        with open(array_script_path, "w") as f:
            f.write(array_script)
        os.chmod(array_script_path, 0o755)

        print(f"\nPartition: {partition}")
        print(f"  Commands file: {commands_file} ({num_tasks} commands)")
        print(f"  Array script: {array_script_path}")
        print(f"  Max concurrent tasks: {args.max_concurrent}")

        if not args.dry_run:
            # Submit array job (use absolute path)
            try:
                abs_script_path = os.path.abspath(array_script_path)
                result = subprocess.run(
                    ["sbatch", abs_script_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                job_id = result.stdout.strip().split()[-1]
                print(f"  ✓ Submitted array job: {job_id}")
                job_ids.append((partition, job_id, num_tasks))
                summary_info.append({
                    "partition": partition,
                    "job_id": job_id,
                    "num_tasks": num_tasks,
                    "commands_file": commands_file,
                    "script_path": array_script_path,
                })
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to submit array job")
                print(f"    Error: {e.stderr}")
        else:
            print(f"  → Array script created (dry-run)")

    # Create summary file
    summary_file = os.path.join(args.output_dir, "experiments_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Inf-Net Training Experiments Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total configurations: {len(all_configs)}\n")
        f.write(f"Max concurrent tasks per partition: {args.max_concurrent}\n\n")

        task_id = 1
        for partition, configs in sorted(partition_configs.items()):
            f.write(f"\nPartition: {partition}\n")
            f.write("-" * 70 + "\n")
            for config in configs:
                morph_info = (
                    f" | Morph: {config['morph_operation']}"
                    if config["enable_morphology"]
                    else ""
                )
                privacy_info = (
                    f" | ε={config['epsilon']}, clipping={config['clipping_strategy']}"
                    if config["enable_privacy"]
                    else ""
                )
                f.write(
                    f"Task {task_id:4d}: {config['network']:12s} | "
                    f"Batch: {config['batch_size']:3d} | Run: {config['run']}{privacy_info}{morph_info}\n"
                )
                task_id += 1

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print(f"✓ Dry-run completed!")
        print(f"  Scripts saved to: {args.output_dir}")
        print(f"  Summary file: {summary_file}")
        print(f"  To submit jobs, run: sbatch {args.output_dir}/*/array_job.sh")
    else:
        print(f"✓ Job submission completed!")
        print(f"  Total array jobs submitted: {len(job_ids)}")
        print(f"  Summary file: {summary_file}")

    if job_ids:
        print(f"\n{'Submitted Array Jobs:':^70}")
        print("-" * 70)
        for partition, job_id, num_tasks in job_ids:
            print(f"  {partition:20s}: Job ID {job_id:>10s} ({num_tasks:4d} tasks)")

    print(f"{'=' * 70}\n")

    # Save job submission log
    log_file = "job_submission_log_unified.txt"
    with open(log_file, "a") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"Submission: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total configurations: {len(all_configs)}\n")
        f.write(f"Partitions used: {', '.join(partitions)}\n")
        f.write(f"Max concurrent: {args.max_concurrent}\n")
        for partition, job_id, num_tasks in job_ids:
            f.write(f"  {partition}: Array job {job_id} ({num_tasks} tasks)\n")

    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
