#!/usr/bin/env python3
"""
Generate SLURM array job scripts for Inf-Net evaluation phase.
This script generates evaluation commands for all result directories found in the Results folder.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict

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


def find_result_directories(base_path: str) -> List[Dict]:
    """Find all result directories containing PNG files"""
    result_dirs = []

    if not os.path.exists(base_path):
        print(f"Results directory not found: {base_path}")
        return result_dirs

    # Find all subdirectories with PNG files
    for root, dirs, files in os.walk(base_path):
        png_files = [f for f in files if f.endswith(".png")]
        if png_files:
            rel_path = os.path.relpath(root, base_path)
            result_dirs.append(
                {
                    "path": root,
                    "relative_path": rel_path,
                    "num_images": len(png_files),
                }
            )

    return result_dirs


def build_evaluation_command(
    relative_path: str, gt_path: str = "../Dataset/TestingSet/LungInfection-Test/GT/"
) -> str:
    """Build the Python evaluation command for a single result directory"""
    # Use the --result_dir argument to evaluate a single directory
    return f"python main_all.py --gt_path {gt_path} --result_dir {relative_path}"


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

# Navigate to evaluation tool directory
cd code/inf-net/EvaluationToolPython

# Get the command for this array task (commands file is in code/inf-net/slurm_jobs_eval/full_run/)
COMMAND=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" ../{commands_file})

# Execute the command
echo "Task ${{SLURM_ARRAY_TASK_ID}}: $COMMAND"
eval $COMMAND
"""
    return script


def main():
    """Main function to generate SLURM array job scripts for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SLURM array job scripts for Inf-Net evaluation"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../Results/Lung_infection_segmentation",
        help="Path to results directory (relative to EvaluationToolPython)",
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        default="../Dataset/TestingSet/LungInfection-Test/GT/",
        help="Path to ground truth images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="slurm_jobs_eval",
        help="Output directory for SLURM scripts",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent array tasks (default: 50)",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Filter to specific model type only",
    )
    parser.add_argument(
        "--batch-filter",
        type=str,
        default=None,
        help="Filter to specific batch size only",
    )
    parser.add_argument(
        "--run-filter",
        type=str,
        default=None,
        help="Filter to specific run number only",
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = Path(args.output_dir) / "full_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result directories
    # The path should be relative to EvaluationToolPython directory
    work_dir = Path("/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya")
    eval_tool_dir = work_dir / "code/inf-net/EvaluationToolPython"

    # Resolve results path - args.results_dir is relative to EvaluationToolPython
    if args.results_dir.startswith("../"):
        # Remove "../" and build path from code/inf-net
        rel_path = args.results_dir.replace("../", "")
        results_path = work_dir / "code/inf-net" / rel_path
    elif os.path.isabs(args.results_dir):
        results_path = Path(args.results_dir)
    else:
        # Relative to EvaluationToolPython
        results_path = eval_tool_dir / args.results_dir

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        return

    print(f"Scanning for result directories in: {results_path}")
    result_dirs = find_result_directories(str(results_path))

    if not result_dirs:
        print("No result directories found!")
        return

    print(f"Found {len(result_dirs)} result directories")

    # Apply filters
    filtered_dirs = result_dirs
    if args.model_filter:
        filtered_dirs = [
            d for d in filtered_dirs if args.model_filter in d["relative_path"]
        ]
        print(f"After model filter: {len(filtered_dirs)} directories")

    if args.batch_filter:
        filtered_dirs = [
            d
            for d in filtered_dirs
            if f"batch_{args.batch_filter}" in d["relative_path"]
        ]
        print(f"After batch filter: {len(filtered_dirs)} directories")

    if args.run_filter:
        filtered_dirs = [
            d for d in filtered_dirs if f"run_{args.run_filter}" in d["relative_path"]
        ]
        print(f"After run filter: {len(filtered_dirs)} directories")

    if not filtered_dirs:
        print("No directories match the filters!")
        return

    print(f"Generating evaluation commands for {len(filtered_dirs)} directories...")

    # Generate commands for each result directory
    commands = []
    for result_dir in filtered_dirs:
        rel_path = result_dir["relative_path"]
        cmd = build_evaluation_command(rel_path, args.gt_path)
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

    job_name = "infnet_eval"
    output_log = str(output_dir.resolve() / "eval_%A_%a.out")
    error_log = str(output_dir.resolve() / "eval_%A_%a.err")
    work_dir = "/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya"
    # Commands file path relative to code/inf-net (EvaluationToolPython is code/inf-net/EvaluationToolPython)
    # From EvaluationToolPython, we go up one level (..) to code/inf-net, then the path is just slurm_jobs_eval/full_run/commands.txt
    infnet_dir = Path(work_dir) / "code/inf-net"
    commands_file_abs = commands_file.resolve()
    commands_file_rel = str(commands_file_abs.relative_to(infnet_dir.resolve()))

    array_script = create_array_job_script(
        job_name=job_name,
        commands_file=commands_file_rel,
        num_tasks=len(filtered_dirs),
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
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Inf-Net Evaluation Jobs Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total result directories: {len(filtered_dirs)}\n")
        f.write(f"Results path: {results_path}\n")
        f.write(f"Ground truth path: {args.gt_path}\n")
        if args.model_filter:
            f.write(f"Model filter: {args.model_filter}\n")
        if args.batch_filter:
            f.write(f"Batch filter: {args.batch_filter}\n")
        if args.run_filter:
            f.write(f"Run filter: {args.run_filter}\n")
        f.write(f"\nMax concurrent tasks: {max_concurrent}\n")
        f.write(f"Time limit per task: {SLURM_CONFIG['time']}\n")

    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    main()
