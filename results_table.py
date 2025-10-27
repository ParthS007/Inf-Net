#!/usr/bin/env python3
"""
Create comprehensive results table combining evaluation metrics and privacy budgets.
Combines data from:
1. Evaluation results (*.txt summary files)
2. Training logs (*.out files for epsilon values)
3. Job submission log (for job ID mapping)
"""

import os
import re
import pandas as pd
from pathlib import Path
import numpy as np

# Base directories
BASE_DIR = Path(__file__).parent
EVAL_DIR = BASE_DIR / "EvaluateResults" / "Lung_infection_segmentation"
LOG_DIR = BASE_DIR / "logs" / "train"
JOB_LOG = BASE_DIR / "job_submission_log.txt"


def parse_job_submission_log():
    """Parse job submission log to get mapping of config_name -> job_id"""
    config_to_job = {}

    with open(JOB_LOG, "r") as f:
        for line in f:
            # Don't strip - need to check for leading whitespace
            # Match lines like: "  infnet_batch24_run1: 59954611"
            match = re.match(r"\s+(\S+):\s+(\d+)", line)
            if match:
                config_name = match.group(1)
                job_id = match.group(2)
                config_to_job[config_name] = job_id

    return config_to_job


def extract_epsilon_from_log(log_file):
    """Extract the last epsilon value from a training log file"""
    if not os.path.exists(log_file):
        return None

    last_epsilon = None
    try:
        with open(log_file, "r") as f:
            for line in f:
                # Match: [Privacy Budget]: ε = inf (δ = 1e-05)
                # or:    [Privacy Budget]: ε = 123.45 (δ = 1e-05)
                match = re.search(r"\[Privacy Budget\]:\s*ε\s*=\s*(\S+)\s*\(", line)
                if match:
                    epsilon_str = match.group(1)
                    last_epsilon = epsilon_str
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None

    return last_epsilon


def parse_summary_file(summary_file):
    """Parse a summary.txt file to extract metrics"""
    metrics = {}

    try:
        with open(summary_file, "r") as f:
            content = f.read()

            # Extract metrics using regex
            patterns = {
                "mean_dice": r"Mean Dice:\s*([\d.]+)",
                "max_dice": r"Max Dice:\s*([\d.]+)",
                "mean_sensitivity": r"Mean Sensitivity:\s*([\d.]+)",
                "max_sensitivity": r"Max Sensitivity:\s*([\d.]+)",
                "mean_specificity": r"Mean Specificity:\s*([\d.]+)",
                "max_specificity": r"Max Specificity:\s*([\d.]+)",
                "s_measure": r"S-measure:\s*([\d.]+)",
                "mean_e_measure": r"Mean E-measure:\s*([\d.]+)",
                "max_e_measure": r"Max E-measure:\s*([\d.]+)",
                "mae": r"MAE:\s*([\d.]+)",
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metrics[key] = float(match.group(1))
                else:
                    metrics[key] = None

    except Exception as e:
        print(f"Error parsing {summary_file}: {e}")

    return metrics


def find_all_summary_files():
    """Find all summary files in the evaluation directory"""
    summary_files = []

    # Pattern: *_summary.txt but exclude the main evaluation_summary.txt
    for summary_file in EVAL_DIR.rglob("*_summary.txt"):
        # Skip the main evaluation summary
        if summary_file.name == "evaluation_summary.txt":
            continue
        summary_files.append(summary_file)

    return summary_files


def extract_config_from_path(file_path):
    """Extract configuration details from file path and name"""
    config = {}

    # Get relative path from EVAL_DIR
    rel_path = file_path.relative_to(EVAL_DIR)
    parts = rel_path.parts

    # Parse based on directory structure
    # Examples:
    # Inf-Net/batch_24/run_1/Inf-Net_batch24_run1_summary.txt
    # Inf-Net_DP/batch_24/run_1/noise_multiplier_0.3/Inf-Net_DP_batch24_run1_noise0.3_summary.txt
    # Inf-Net_DP_Morph/both/batch_24/run_1/noise_multiplier_0.3/Inf-Net_DP_Morph_batch24_run1_noise0.3_both_summary.txt
    # Inf-Net_Morph/both/batch_24/run_1/Inf-Net_Morph_batch24_run1_both_summary.txt

    model_type = parts[0]  # Inf-Net, Inf-Net_DP, Inf-Net_Morph, Inf-Net_DP_Morph
    config["model_type"] = model_type

    # Initialize defaults
    config["dp_enabled"] = "DP" in model_type
    config["morph_enabled"] = "Morph" in model_type
    config["morph_operation"] = None
    config["noise_multiplier"] = None
    config["batch_size"] = None
    config["run"] = None

    # Parse filename
    filename = file_path.name

    # Extract batch size
    batch_match = re.search(r"batch(\d+)", filename)
    if batch_match:
        config["batch_size"] = int(batch_match.group(1))

    # Extract run number
    run_match = re.search(r"run(\d+)", filename)
    if run_match:
        config["run"] = int(run_match.group(1))

    # Extract noise multiplier (for DP models)
    noise_match = re.search(r"noise([\d.]+)", filename)
    if noise_match:
        config["noise_multiplier"] = float(noise_match.group(1))

    # Extract morphology operation (for morph models)
    # Check both filename and directory structure
    morph_ops = ["both", "close", "open", "dilation", "erosion"]
    for op in morph_ops:
        if op in filename or op in str(rel_path):
            config["morph_operation"] = op
            break

    return config


def create_config_key_for_job_log(config):
    """Create a key to match against job submission log"""
    # Job log format examples:
    # infnet_batch24_run1
    # infnet_dp_nm0.3_batch24_run1
    # infnet_morph_both_batch24_run1
    # infnet_dpmorph_both_nm0.3_batch24_run1

    parts = ["infnet"]

    if config["dp_enabled"] and config["morph_enabled"]:
        parts.append("dpmorph")
        if config["morph_operation"]:
            parts.append(config["morph_operation"])
        if config["noise_multiplier"]:
            parts.append(f"nm{config['noise_multiplier']}")
    elif config["dp_enabled"]:
        parts.append("dp")
        if config["noise_multiplier"]:
            parts.append(f"nm{config['noise_multiplier']}")
    elif config["morph_enabled"]:
        parts.append("morph")
        if config["morph_operation"]:
            parts.append(config["morph_operation"])

    if config["batch_size"]:
        parts.append(f"batch{config['batch_size']}")
    if config["run"]:
        parts.append(f"run{config['run']}")

    return "_".join(parts)


def main():
    print("=" * 80)
    print("Creating Comprehensive Results Table")
    print("=" * 80)

    # Step 1: Parse job submission log
    print("\n[1/4] Parsing job submission log...")
    config_to_job = parse_job_submission_log()
    print(f"Found {len(config_to_job)} job configurations")

    # Step 2: Find all summary files
    print("\n[2/4] Finding evaluation summary files...")
    summary_files = find_all_summary_files()
    print(f"Found {len(summary_files)} summary files")

    # Step 3: Process all configurations
    print("\n[3/4] Processing configurations...")
    all_results = []

    for summary_file in summary_files:
        # Extract config from path
        config = extract_config_from_path(summary_file)

        # Parse metrics from summary file
        metrics = parse_summary_file(summary_file)

        # Create key for job log lookup
        job_key = create_config_key_for_job_log(config)
        job_id = config_to_job.get(job_key)

        # Extract epsilon from training log
        epsilon = None
        if job_id:
            log_file = LOG_DIR / f"{job_key}_{job_id}.out"
            epsilon = extract_epsilon_from_log(log_file)

        # If no DP, set epsilon to None
        if not config["dp_enabled"]:
            epsilon = None

        # Combine all data
        result = {
            "model": config["model_type"],
            "dp_enabled": config["dp_enabled"],
            "morph_enabled": config["morph_enabled"],
            "morph_operation": (
                config["morph_operation"] if config["morph_enabled"] else None
            ),
            "noise_multiplier": (
                config["noise_multiplier"] if config["dp_enabled"] else None
            ),
            "batch_size": config["batch_size"],
            "run": config["run"],
            "epsilon": epsilon,
            "job_id": job_id,
            **metrics,
        }

        all_results.append(result)

    print(f"Processed {len(all_results)} configurations")

    # Step 4: Create DataFrame and save
    print("\n[4/4] Creating and saving results...")
    df = pd.DataFrame(all_results)

    # Sort by model type, batch size, noise multiplier, morph operation, and run
    sort_columns = ["model", "batch_size", "noise_multiplier", "morph_operation", "run"]
    df = df.sort_values(by=[col for col in sort_columns if col in df.columns])

    # Reorder columns for better readability
    column_order = [
        "model",
        "batch_size",
        "run",
        "dp_enabled",
        "noise_multiplier",
        "epsilon",
        "morph_enabled",
        "morph_operation",
        "mean_dice",
        "max_dice",
        "s_measure",
        "mae",
        "mean_sensitivity",
        "max_sensitivity",
        "mean_specificity",
        "max_specificity",
        "mean_e_measure",
        "max_e_measure",
        "job_id",
    ]

    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Save to CSV
    output_csv = BASE_DIR / "comprehensive_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved comprehensive results to: {output_csv}")

    # Save to Excel for better formatting
    output_excel = BASE_DIR / "comprehensive_results.xlsx"
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"✓ Saved comprehensive results to: {output_excel}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total configurations: {len(df)}")
    print(f"Models: {df['model'].unique()}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"Runs per configuration: {sorted(df['run'].unique())}")

    if "noise_multiplier" in df.columns:
        print(
            f"Noise multipliers (DP): {sorted([x for x in df['noise_multiplier'].unique() if pd.notna(x)])}"
        )

    if "morph_operation" in df.columns:
        print(
            f"Morphology operations: {sorted([x for x in df['morph_operation'].unique() if pd.notna(x)])}"
        )

    # Count by model type
    print("\nConfigurations by model type:")
    print(df["model"].value_counts().to_string())

    # Check for missing epsilon values in DP models
    dp_models = df[df["dp_enabled"] == True]
    missing_epsilon = dp_models[dp_models["epsilon"].isna()]
    if len(missing_epsilon) > 0:
        print(
            f"\n⚠ Warning: {len(missing_epsilon)} DP configurations missing epsilon values"
        )

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
