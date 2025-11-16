#!/usr/bin/env python3
from pathlib import Path
from typing import List, Dict

# Experimental configuration
MODELS = ["Inf_Net", "UNet", "NestedUNet"]
EPSILON_VALUES = [8, 200]
MAX_GRAD_NORMS = [1.2, 1.5, 2.0]
CLIPPING_STRATEGIES = {
    "base": "base",
    "automatic": "automatic",
    "psac": "psac",
    "normalized_sgd": "normalized_sgd",
}

# SLURM configuration
SLURM_CONFIG = {
    "partition": "a100",
    "qos": "gpu30min",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 4,
    "gres": "gpu:1",
    "mem": "64G",
    "time": "00:30:00",
    "max_concurrent": 50,
}

# Training parameters
DEFAULT_TRAINING_PARAMS = {
    "epoch": 70,
    "lr": 1e-4,
    "batch_sizes": [24, 48, 64],
    "num_runs": 3,
    "operations": ["both", "close", "open"],
    "kernel_sizes": [3, 5, 7, 9],
    "max_grad_norms": [1.2, 1.5, 2.0],
}


def build_training_command(
    model: str,
    dpsgd: bool,
    clipping: str,
    epsilon: float,
    max_grad_norm: float,
    morphology: bool,
    epoch: int,
    batch_size: int,
    run_number: int,
    operation: str = None,
    kernel_size: int = None,
) -> str:
    """Build the Python training command for a single experiment."""
    python_args = [
        f"--network {model}",
        f"--epoch {epoch}",
        f"--lr {DEFAULT_TRAINING_PARAMS['lr']}",
        f"--batchsize {batch_size}",
        f"--run {run_number}",
    ]

    if morphology:
        python_args.append("--enable_morphology")
        if operation:
            python_args.append(f"--morph_operation {operation}")
        if kernel_size:
            python_args.append(f"--morph_kernel_size {kernel_size}")

    if dpsgd:
        python_args.append("--enable_privacy")
        python_args.append(f"--epsilon {epsilon}")
        python_args.append(f"--max_grad_norm {max_grad_norm}")
        python_args.append(f"--clipping_strategy {clipping}")

    python_cmd = " ".join(python_args)
    return f"python MyTrain_LungInf_Unified.py {python_cmd}"


def generate_experiment_name(
    model: str,
    dpsgd: bool,
    clipping: str,
    epsilon: float,
    max_grad_norm: float,
    morphology: bool,
    operation: str = None,
    kernel_size: int = None,
    batch_size: int = None,
    run_number: int = None,
) -> str:
    """Generate a descriptive name for the experiment."""
    parts = ["lung", model.lower()]

    if dpsgd:
        parts.append("dp")
    else:
        parts.append("nodp")

    if morphology:
        parts.append("morph")
        if operation:
            parts.append(operation)
        if kernel_size:
            parts.append(f"k{kernel_size}")
    else:
        parts.append("nomorph")

    if batch_size:
        parts.append(f"b{batch_size}")

    if run_number:
        parts.append(f"r{run_number}")

    if dpsgd:
        parts.append(f"eps{int(epsilon)}")
        parts.append(f"mg{max_grad_norm}")
        parts.append(clipping)

    return "_".join(parts)


def generate_all_experiments(
    test_run: bool = False,
    epoch: int = None,
    model_filter: str = None,
    clipping_filter: str = None,
    num_runs: int = None,
    batch_sizes: List[int] = None,
    operations: List[str] = None,
    kernel_sizes: List[int] = None,
    max_grad_norms: List[float] = None,
) -> List[Dict]:
    """
    Generate all experiment configurations.

    Args:
        test_run: Whether this is a test run
        epoch: Number of epochs
        model_filter: If provided, only generate experiments for this model
        clipping_filter: If provided, only generate experiments for this clipping strategy (DP only)
        batch_sizes: List of batch sizes to use
        operations: List of morphological operations to use (when morphology=True)
        kernel_sizes: List of kernel sizes to use (when morphology=True)
        max_grad_norms: List of max grad norms to use (when DP=True)

    Returns:
        List of experiment configuration dictionaries
    """
    if epoch is None:
        epoch = 10 if test_run else DEFAULT_TRAINING_PARAMS["epoch"]

    # Use provided parameters or defaults
    batch_sizes_to_use = (
        batch_sizes if batch_sizes else DEFAULT_TRAINING_PARAMS["batch_sizes"]
    )
    num_runs_to_use = num_runs if num_runs else DEFAULT_TRAINING_PARAMS["num_runs"]
    operations_to_use = (
        operations if operations else DEFAULT_TRAINING_PARAMS["operations"]
    )
    kernel_sizes_to_use = (
        kernel_sizes if kernel_sizes else DEFAULT_TRAINING_PARAMS["kernel_sizes"]
    )
    max_grad_norms_to_use = (
        max_grad_norms if max_grad_norms else DEFAULT_TRAINING_PARAMS["max_grad_norms"]
    )

    # Filter models if specified
    models_to_use = (
        [model_filter] if model_filter and model_filter in MODELS else MODELS
    )

    # Filter clipping strategies if specified
    clipping_to_use = (
        [clipping_filter]
        if clipping_filter and clipping_filter in CLIPPING_STRATEGIES.keys()
        else list(CLIPPING_STRATEGIES.keys())
    )

    experiments = []

    # Non-DP experiments
    if not clipping_filter:
        for model in models_to_use:
            for morphology in [False, True]:
                if morphology:
                    # With morphology: iterate over operations and kernel sizes
                    for operation in operations_to_use:
                        for kernel_size in kernel_sizes_to_use:
                            for batch_size in batch_sizes_to_use:
                                for run_number in range(1, num_runs_to_use + 1):
                                    config = {
                                        "model": model,
                                        "dpsgd": False,
                                        "clipping": "none",
                                        "epsilon": 0.0,
                                        "max_grad_norm": 0.0,
                                        "morphology": True,
                                        "operation": operation,
                                        "kernel_size": kernel_size,
                                        "epoch": epoch,
                                        "batch_size": batch_size,
                                        "run_number": run_number,
                                    }
                                    experiments.append(config)
                else:
                    # Without morphology
                    for batch_size in batch_sizes_to_use:
                        for run_number in range(1, num_runs_to_use + 1):
                            config = {
                                "model": model,
                                "dpsgd": False,
                                "clipping": "none",
                                "epsilon": 0.0,
                                "max_grad_norm": 0.0,
                                "morphology": False,
                                "operation": None,
                                "kernel_size": None,
                                "epoch": epoch,
                                "batch_size": batch_size,
                                "run_number": run_number,
                            }
                            experiments.append(config)

    # DP experiments
    for model in models_to_use:
        for clipping in clipping_to_use:
            for epsilon in EPSILON_VALUES:
                for max_grad_norm in max_grad_norms_to_use:
                    for morphology in [False, True]:
                        if morphology:
                            # With morphology: iterate over operations and kernel sizes
                            for operation in operations_to_use:
                                for kernel_size in kernel_sizes_to_use:
                                    for batch_size in batch_sizes_to_use:
                                        for run_number in range(1, num_runs_to_use + 1):
                                            config = {
                                                "model": model,
                                                "dpsgd": True,
                                                "clipping": clipping,
                                                "epsilon": epsilon,
                                                "max_grad_norm": max_grad_norm,
                                                "morphology": True,
                                                "operation": operation,
                                                "kernel_size": kernel_size,
                                                "epoch": epoch,
                                                "batch_size": batch_size,
                                                "run_number": run_number,
                                            }
                                            experiments.append(config)
                        else:
                            # Without morphology
                            for batch_size in batch_sizes_to_use:
                                for run_number in range(1, num_runs_to_use + 1):
                                    config = {
                                        "model": model,
                                        "dpsgd": True,
                                        "clipping": clipping,
                                        "epsilon": epsilon,
                                        "max_grad_norm": max_grad_norm,
                                        "morphology": False,
                                        "operation": None,
                                        "kernel_size": None,
                                        "epoch": epoch,
                                        "batch_size": batch_size,
                                        "run_number": run_number,
                                    }
                                    experiments.append(config)

    return experiments


def main():
    """Main function to generate SLURM array job scripts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SLURM array job scripts for COVID CT experiments"
    )
    parser.add_argument(
        "--test", action="store_true", help="Generate test run scripts (10 epochs)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Number of epochs (default: 10 for test, 70 for full)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="slurm_jobs",
        help="Output directory for SLURM scripts",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent array tasks (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=MODELS,
        help="Filter to specific model only (Inf_Net, UNet, NestedUNet)",
    )
    parser.add_argument(
        "--clipping",
        type=str,
        default=None,
        choices=list(CLIPPING_STRATEGIES.keys()),
        help="Filter to specific clipping strategy only (base, automatic, psac, nsgd) - DP only",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Number of runs per combination (default: 3)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to use (default: 24 48 64)",
    )
    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        default=None,
        help="Morphological operations to use (default: both close open)",
    )
    parser.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Morphological kernel sizes to use (default: 3 5 7 9)",
    )
    parser.add_argument(
        "--max-grad-norms",
        type=float,
        nargs="+",
        default=None,
        help="Max gradient norms to use for DP (default: 1.2 1.5 2.0)",
    )

    args = parser.parse_args()

    test_run = args.test
    epoch = args.epoch
    output_dir = Path(args.output_dir)
    max_concurrent = (
        args.max_concurrent if args.max_concurrent else SLURM_CONFIG["max_concurrent"]
    )

    # Determine subdirectory
    subdir = "test_run" if test_run else "full_run"
    output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    experiments = generate_all_experiments(
        test_run=test_run,
        epoch=epoch,
        model_filter=args.model,
        clipping_filter=args.clipping,
        num_runs=args.num_runs,
        batch_sizes=args.batch_sizes,
        operations=args.operations,
        kernel_sizes=args.kernel_sizes,
        max_grad_norms=args.max_grad_norms,
    )

    filter_info = []
    if args.model:
        filter_info.append(f"Model: {args.model}")
    if args.clipping:
        filter_info.append(f"Clipping: {args.clipping}")
    filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

    print(
        f"Generating SLURM array job for {len(experiments)} experiments{filter_str}..."
    )
    print(f"Test run: {test_run}, Epochs: {epoch}")
    print(f"Max concurrent tasks: {max_concurrent}")
    print(f"Output directory: {output_dir}")

    # Generate commands file
    commands_file = output_dir / "commands.cmd"
    experiment_names = []

    with open(commands_file, "w") as f:
        for i, config in enumerate(experiments, 1):
            command = build_training_command(
                model=config["model"],
                dpsgd=config["dpsgd"],
                clipping=config["clipping"],
                epsilon=config["epsilon"],
                max_grad_norm=config["max_grad_norm"],
                morphology=config["morphology"],
                epoch=config["epoch"],
                batch_size=config["batch_size"],
                run_number=config["run_number"],
                operation=config.get("operation"),
                kernel_size=config.get("kernel_size"),
            )
            f.write(command + "\n")

            # Generate experiment name for summary
            exp_name = generate_experiment_name(
                model=config["model"],
                dpsgd=config["dpsgd"],
                clipping=config["clipping"],
                epsilon=config["epsilon"],
                max_grad_norm=config["max_grad_norm"],
                morphology=config["morphology"],
                operation=config.get("operation"),
                kernel_size=config.get("kernel_size"),
                batch_size=config["batch_size"],
                run_number=config["run_number"],
            )
            experiment_names.append(exp_name)

    print(f"Commands file created: {commands_file}")

    # Generate array job script
    array_job_script = output_dir / "array_job.sh"
    array_job_content = f"""#!/bin/bash
#SBATCH --job-name=lung_inf_experiments
#SBATCH --output={output_dir}/logs/lung_inf_experiments_%A_%a.out
#SBATCH --error={output_dir}/logs/lung_inf_experiments_%A_%a.err
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus_per_task']}
#SBATCH --gres={SLURM_CONFIG['gres']}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --qos={SLURM_CONFIG['qos']}
#SBATCH --array=1-{len(experiments)}%{max_concurrent}

# COVID CT Segmentation Experimental Matrix
# Array job with {len(experiments)} tasks
# Test run: {test_run}
# Epochs: {epoch}

# Navigate to project root
cd /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/

# Activate conda environment
source .venv/bin/activate

# Navigate to inf-net directory
cd code/inf-net

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Get command for this array task
COMMAND=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {commands_file.absolute()})

echo "=========================================="
echo "Task ${{SLURM_ARRAY_TASK_ID}} of {len(experiments)}"
echo "Command: $COMMAND"
echo "=========================================="

# Execute the command
eval $COMMAND

echo "Task ${{SLURM_ARRAY_TASK_ID}} completed!"
"""

    array_job_script.write_text(array_job_content)
    array_job_script.chmod(0o755)
    print(f"Array job script created: {array_job_script}")

    # Generate summary file
    summary_file = output_dir / "experiments_summary.txt"
    with open(summary_file, "w") as f:
        f.write("COVID CT Segmentation Experimental Matrix\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Test Run: {test_run}\n")
        f.write(f"Epochs: {epoch}\n")
        f.write(f"Total Experiments: {len(experiments)}\n")
        f.write(f"Max Concurrent Tasks: {max_concurrent}\n\n")
        f.write(
            f"{'Task ID':<8} {'Experiment Name':<50} {'Model':<12} {'DP':<5} {'Clipping':<12} "
            f"{'Eps':<6} {'MaxGrad':<8} {'Morph':<6} {'Op':<10} {'Kernel':<7} {'Batch':<6} {'Run':<4}\n"
        )
        f.write("-" * 100 + "\n")

        for i, (config, exp_name) in enumerate(zip(experiments, experiment_names), 1):
            f.write(
                f"{i:<8} {exp_name:<50} {config['model']:<12} "
                f"{str(config['dpsgd']):<5} {config['clipping']:<12} "
                f"{str(config['epsilon']):<6} {str(config['max_grad_norm']):<8} "
                f"{str(config['morphology']):<6} {str(config.get('operation', 'N/A')):<10} "
                f"{str(config.get('kernel_size', 'N/A')):<7} {config['batch_size']:<6} {config['run_number']:<4}\n"
            )

    print(f"Summary written to {summary_file}")

    # Create submission helper script
    submit_script = output_dir / "submit_job.sh"
    submit_content = f"""#!/bin/bash
# Submit SLURM array job

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting SLURM array job with {len(experiments)} tasks..."
echo "Max concurrent tasks: {max_concurrent}"
echo ""

sbatch array_job.sh

echo ""
echo "Job submitted!"
echo "Check status with: squeue -u $USER"
echo "Monitor progress: squeue -j <JOB_ID>"
echo "View logs: ls -lh {output_dir}/logs/"
"""

    submit_script.write_text(submit_content)
    submit_script.chmod(0o755)

    print(f"Submission script created: {submit_script}")
    print(f"\nTo submit the array job, run:")
    print(f"  cd {output_dir}")
    print(f"  bash submit_job.sh")
    print(f"\nOr directly:")
    print(f"  cd {output_dir}")
    print(f"  sbatch array_job.sh")


if __name__ == "__main__":
    main()
