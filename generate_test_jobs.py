#!/usr/bin/env python3
"""
Generate SLURM array job scripts for Inf-Net testing phase.
This script generates testing commands that match the training experiment combinations.
"""

from pathlib import Path
from typing import List, Dict

# Experimental configuration (must match training)
MODELS = ["Inf_Net", "UNet", "NestedUNet"]
EPSILON_VALUES = [8, 200]
MAX_GRAD_NORMS = [1.2, 1.5, 2.0]
CLIPPING_STRATEGIES = {
    "base": "base",
    "automatic": "automatic",
    "psac": "psac",
    "normalized_sgd": "nsgd",  # Note: training uses "normalized_sgd", testing uses "nsgd"
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

# Testing parameters (must match training)
DEFAULT_TESTING_PARAMS = {
    "epoch": 70,
    "batch_sizes": [24, 48, 64],
    "num_runs": 3,
    "operations": ["both", "close", "open"],
    "kernel_sizes": [3, 5, 7, 9],
    "max_grad_norms": [1.2, 1.5, 2.0],
}


def map_model_to_test_type(model: str, dpsgd: bool, morphology: bool) -> str:
    """Map training model name to testing model_type argument.

    Training creates:
    - Non-DP: {Network}_GroupNorm or {Network}_Morph_GroupNorm
    - DP: {Network}_DP or {Network}_DP_Morph
    """
    # Map network names
    if model == "Inf_Net":
        network = "Inf-Net"
    elif model == "UNet":
        network = "UNet"
    elif model == "NestedUNet":
        network = "NestedUNet"
    else:
        network = "Inf-Net"

    # Build model type based on DP and morphology
    if dpsgd:
        if morphology:
            return f"{network}_DP_Morph"
        else:
            return f"{network}_DP"
    else:
        if morphology:
            return f"{network}_Morph_GroupNorm"
        else:
            return f"{network}_GroupNorm"


def build_testing_command(
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
    """Build the Python testing command for a single experiment."""
    model_type = map_model_to_test_type(model, dpsgd, morphology)

    python_args = [
        f"--model_type {model_type}",
        f"--epoch {epoch}",
        f"--batchsize {batch_size}",
        f"--run {run_number}",
    ]

    if morphology:
        python_args.append("--enable_morphology_test")
        if operation:
            python_args.append(f"--morph_operation {operation}")
        if kernel_size:
            python_args.append(f"--morph_kernel_size {kernel_size}")

    if dpsgd:
        python_args.append(f"--epsilon {epsilon}")
        python_args.append(f"--max_grad_norm {max_grad_norm}")
        # Map clipping strategy: training uses "normalized_sgd", testing uses "nsgd"
        test_clipping = CLIPPING_STRATEGIES.get(clipping, clipping)
        python_args.append(f"--clipping_strategy {test_clipping}")

    python_cmd = " ".join(python_args)
    return f"python MyTest_LungInf_All.py {python_cmd}"


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
    parts = ["test", model.lower()]

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

    if dpsgd:
        parts.append(f"eps{int(epsilon)}")
        parts.append(f"mg{max_grad_norm}")
        if clipping != "base":
            parts.append(clipping)

    if batch_size:
        parts.append(f"b{batch_size}")
    if run_number:
        parts.append(f"r{run_number}")

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
    Generate all experiment configurations for testing.
    Must match the training experiment combinations exactly.
    """
    if epoch is None:
        epoch = 70 if not test_run else 10

    # Use provided parameters or defaults
    batch_sizes_to_use = (
        batch_sizes if batch_sizes else DEFAULT_TESTING_PARAMS["batch_sizes"]
    )
    num_runs_to_use = num_runs if num_runs else DEFAULT_TESTING_PARAMS["num_runs"]
    operations_to_use = (
        operations if operations else DEFAULT_TESTING_PARAMS["operations"]
    )
    kernel_sizes_to_use = (
        kernel_sizes if kernel_sizes else DEFAULT_TESTING_PARAMS["kernel_sizes"]
    )
    max_grad_norms_to_use = (
        max_grad_norms if max_grad_norms else DEFAULT_TESTING_PARAMS["max_grad_norms"]
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


def create_array_job_script(
    job_name: str,
    commands_file: str,
    num_tasks: int,
    max_concurrent: int,
    output_log: str,
    error_log: str,
    work_dir: str,
) -> str:
    """Create the SLURM array job script."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_log}
#SBATCH --error={error_log}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --qos={SLURM_CONFIG['qos']}
#SBATCH --nodes={SLURM_CONFIG['nodes']}
#SBATCH --ntasks={SLURM_CONFIG['ntasks']}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus_per_task']}
#SBATCH --gres={SLURM_CONFIG['gres']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --array=1-{num_tasks}%{max_concurrent}

# Navigate to project directory
cd {work_dir}

# Activate virtual environment
source /scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/.venv/bin/activate

# Navigate to inf-net directory
cd code/inf-net

# Get the command for this array task
COMMAND=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {commands_file})

# Execute the command
echo "Task ${{SLURM_ARRAY_TASK_ID}}: $COMMAND"
eval $COMMAND
"""
    return script


def main():
    """Main function to generate SLURM array job scripts for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SLURM array job scripts for Inf-Net testing"
    )
    parser.add_argument(
        "--test", action="store_true", help="Generate test run scripts (10 epochs)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to test (default: 70 for full, 10 for test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="slurm_jobs_test",
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
        help="Filter to specific clipping strategy only (base, automatic, psac, normalized_sgd) - DP only",
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
        help="Batch sizes to test (default: 24 48 64)",
    )
    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        default=None,
        help="Morphological operations (default: both close open)",
    )
    parser.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Kernel sizes (default: 3 5 7 9)",
    )
    parser.add_argument(
        "--max-grad-norms",
        type=float,
        nargs="+",
        default=None,
        help="Max grad norms (default: 1.2 1.5 2.0)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.test:
        output_dir = Path(args.output_dir) / "test_run"
    else:
        output_dir = Path(args.output_dir) / "full_run"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all experiments
    experiments = generate_all_experiments(
        test_run=args.test,
        epoch=args.epoch,
        model_filter=args.model,
        clipping_filter=args.clipping,
        num_runs=args.num_runs,
        batch_sizes=args.batch_sizes,
        operations=args.operations,
        kernel_sizes=args.kernel_sizes,
        max_grad_norms=args.max_grad_norms,
    )

    print(f"Generated {len(experiments)} testing experiments")

    # Generate commands
    commands = []
    for exp in experiments:
        cmd = build_testing_command(
            model=exp["model"],
            dpsgd=exp["dpsgd"],
            clipping=exp["clipping"],
            epsilon=exp["epsilon"],
            max_grad_norm=exp["max_grad_norm"],
            morphology=exp["morphology"],
            epoch=exp["epoch"],
            batch_size=exp["batch_size"],
            run_number=exp["run_number"],
            operation=exp.get("operation"),
            kernel_size=exp.get("kernel_size"),
        )
        commands.append(cmd)

    # Write commands file
    commands_file = output_dir / "commands.txt"
    with open(commands_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    print(f"Written {len(commands)} commands to {commands_file}")

    # Create array job script
    max_concurrent = (
        args.max_concurrent if args.max_concurrent else SLURM_CONFIG["max_concurrent"]
    )

    job_name = "infnet_test" if not args.test else "infnet_test_test"
    # Use absolute paths for output/error logs (SLURM paths are relative to submission directory)
    output_log = str(output_dir.resolve() / "test_%A_%a.out")
    error_log = str(output_dir.resolve() / "test_%A_%a.err")
    work_dir = Path("/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya")
    # After cd code/inf-net, the path should be relative to that directory
    # commands_file is in output_dir, which is relative to code/inf-net
    infnet_dir = work_dir / "code" / "inf-net"
    commands_file_abs = commands_file.resolve()
    commands_file_rel = str(commands_file_abs.relative_to(infnet_dir.resolve()))

    array_script = create_array_job_script(
        job_name=job_name,
        commands_file=commands_file_rel,
        num_tasks=len(experiments),
        max_concurrent=max_concurrent,
        output_log=output_log,
        error_log=error_log,
        work_dir=work_dir,
    )

    # Write array job script
    array_script_file = output_dir / "array_job.sh"
    with open(array_script_file, "w") as f:
        f.write(array_script)

    print(f"Created array job script: {array_script_file}")
    print(f"To submit: cd {output_dir} && sbatch array_job.sh")

    # Generate summary
    summary_file = output_dir / "experiments_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Inf-Net Testing Experiments Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total experiments: {len(experiments)}\n")
        f.write(f"Epoch: {experiments[0]['epoch'] if experiments else 'N/A'}\n")
        f.write(f"Models: {', '.join(MODELS)}\n")
        f.write(
            f"Batch sizes: {', '.join(map(str, DEFAULT_TESTING_PARAMS['batch_sizes']))}\n"
        )
        f.write(f"Runs per combination: {DEFAULT_TESTING_PARAMS['num_runs']}\n")
        f.write(f"Operations: {', '.join(DEFAULT_TESTING_PARAMS['operations'])}\n")
        f.write(
            f"Kernel sizes: {', '.join(map(str, DEFAULT_TESTING_PARAMS['kernel_sizes']))}\n"
        )
        f.write(
            f"Max grad norms: {', '.join(map(str, DEFAULT_TESTING_PARAMS['max_grad_norms']))}\n"
        )
        f.write(f"Epsilon values: {', '.join(map(str, EPSILON_VALUES))}\n")
        f.write(f"Clipping strategies: {', '.join(CLIPPING_STRATEGIES.keys())}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Count by category
        non_dp = sum(1 for e in experiments if not e["dpsgd"])
        dp = sum(1 for e in experiments if e["dpsgd"])
        morph = sum(1 for e in experiments if e["morphology"])
        dp_morph = sum(1 for e in experiments if e["dpsgd"] and e["morphology"])

        f.write("Breakdown:\n")
        f.write(f"  Non-DP experiments: {non_dp}\n")
        f.write(f"  DP experiments: {dp}\n")
        f.write(f"  Morphology experiments: {morph}\n")
        f.write(f"  DP + Morphology experiments: {dp_morph}\n")

    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    main()
