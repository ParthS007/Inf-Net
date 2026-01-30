"""
Aggregate training and evaluation results into merged and mean±std CSVs.

Append-only / merge-with-existing: when saving filtered, merged, and aggregated
CSVs we merge with existing files (if present) and never drop rows from
previous runs. New data for the same row key replaces the old (keep='last').
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from collections import defaultdict

# Paths
results_dir = (
    "/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/results"
)
eval_dir = "/scicore/home/wagner0024/shandi0000/2025-msc-parth-shandilya/code/inf-net/EvaluateResults/Lung_infection_segmentation"
training_csv = os.path.join(results_dir, "all_results_training_global.csv")

# Create output directories
filtered_dir = os.path.join(results_dir, "filtered")
merged_dir = os.path.join(results_dir, "merged")
aggregated_dir = os.path.join(results_dir, "aggregated")
combined_dir = os.path.join(results_dir, "combined")

for dir_path in [filtered_dir, merged_dir, aggregated_dir, combined_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Columns that uniquely identify a row (for merge-with-existing / append-only behavior)
KEY_COLS_FILTERED_MERGED = [
    "Model_Name",
    "Dataset",
    "DPSGD",
    "Batch_Size",
    "Learning_Rate",
    "Run_Number",
    "Epsilon",
    "Max_Grad_Norm",
    "Clipping_Strategy",
    "Morphology",
    "Operation",
    "Kernel_Size",
]
KEY_COLS_AGGREGATED = [
    "Model_Name",
    "Dataset",
    "DPSGD",
    "Batch_Size",
    "Learning_Rate",
    "Epsilon",
    "Max_Grad_Norm",
    "Clipping_Strategy",
    "Morphology",
    "Operation",
    "Kernel_Size",
]


def _merge_with_existing(df_new, output_path, key_cols, keep="last"):
    """Merge new dataframe with existing file if present; never drop existing rows."""
    key_cols = [c for c in key_cols if c in df_new.columns]
    if not key_cols:
        return df_new
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            # Only keep columns that exist in both; key columns must exist in existing
            key_in_existing = [c for c in key_cols if c in df_existing.columns]
            if key_in_existing:
                # Align columns: union of columns, fill missing with NaN
                all_cols = list(
                    dict.fromkeys(list(df_existing.columns) + list(df_new.columns))
                )
                df_existing = df_existing.reindex(columns=all_cols)
                df_new = df_new.reindex(columns=all_cols)
                combined = pd.concat([df_existing, df_new], ignore_index=True)
                combined = combined.drop_duplicates(subset=key_in_existing, keep=keep)
                print(
                    f"  Merged with existing file: {len(df_existing)} + {len(df_new)} -> {len(combined)} rows (append-only)."
                )
                return combined
        except Exception as e:
            print(
                f"  Warning: could not merge with existing ({e}); writing new file only."
            )
    return df_new


def parse_evaluation_summary(summary_file):
    """Parse evaluation summary file and extract metrics"""
    metrics = {}
    try:
        with open(summary_file, "r") as f:
            content = f.read()
            # Extract metrics using regex
            patterns = {
                "Mean_Dice": r"Mean Dice:\s*([\d.]+)",
                "Max_Dice": r"Max Dice:\s*([\d.]+)",
                "Mean_Sensitivity": r"Mean Sensitivity:\s*([\d.]+)",
                "Max_Sensitivity": r"Max Sensitivity:\s*([\d.]+)",
                "Mean_Specificity": r"Mean Specificity:\s*([\d.]+)",
                "Max_Specificity": r"Max Specificity:\s*([\d.]+)",
                "S_measure": r"S-measure:\s*([\d.]+)",
                "Mean_E_measure": r"Mean E-measure:\s*([\d.]+)",
                "Max_E_measure": r"Max E-measure:\s*([\d.]+)",
                "MAE": r"MAE:\s*([\d.]+)",
            }
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metrics[key] = float(match.group(1))
    except Exception as e:
        print(f"Error parsing {summary_file}: {e}")
    return metrics


def find_evaluation_file(
    model_name,
    batch_size,
    run_number,
    epsilon=None,
    max_grad_norm=None,
    clipping_strategy=None,
    morph_operation=None,
    kernel_size=None,
):
    """Find the evaluation summary file for a given model configuration"""
    base_path = os.path.join(eval_dir, model_name)

    if not os.path.exists(base_path):
        return None

    # Determine model type based on name patterns
    has_dp = "_DP" in model_name
    has_morph = "_Morph" in model_name

    # Handle different model types
    if not has_dp and not has_morph:
        # GroupNorm models: ModelName/batch_XX/run_X/ModelName_batchXX_runX_summary.txt
        summary_file = os.path.join(
            base_path,
            f"batch_{batch_size}",
            f"run_{run_number}",
            f"{model_name}_batch{batch_size}_run{run_number}_summary.txt",
        )

    elif has_dp and not has_morph:
        # DP models: ModelName/batch_XX/run_X/epsilon_XXX/maxgrad_X.X/strategy/ModelName_batchXX_runX_epsXXX_mgX.X_strategy_summary.txt
        if epsilon is None or max_grad_norm is None or clipping_strategy is None:
            return None
        eps_str = (
            str(int(float(epsilon)))
            if float(epsilon) == int(float(epsilon))
            else str(epsilon)
        )
        mg_str = str(max_grad_norm)  # Keep dot, don't replace with underscore
        summary_file = os.path.join(
            base_path,
            f"batch_{batch_size}",
            f"run_{run_number}",
            f"epsilon_{eps_str}",
            f"maxgrad_{mg_str}",
            clipping_strategy,
            f"{model_name}_batch{batch_size}_run{run_number}_eps{eps_str}_mg{mg_str}_{clipping_strategy}_summary.txt",
        )

    elif not has_dp and has_morph:
        # Morph_GroupNorm: ModelName/operation/kernel_X/batch_XX/run_X/ModelName_batchXX_runX_operation_kX_summary.txt
        if morph_operation is None or kernel_size is None:
            return None
        summary_file = os.path.join(
            base_path,
            morph_operation,
            f"kernel_{kernel_size}",
            f"batch_{batch_size}",
            f"run_{run_number}",
            f"{model_name}_batch{batch_size}_run{run_number}_{morph_operation}_k{kernel_size}_summary.txt",
        )

    elif has_dp and has_morph:
        # DP_Morph: ModelName/operation/kernel_X/batch_XX/run_X/epsilon_XXX/maxgrad_X.X/strategy/ModelName_batchXX_runX_epsXXX_mgX.X_strategy_operation_kX_summary.txt
        if (
            epsilon is None
            or max_grad_norm is None
            or clipping_strategy is None
            or morph_operation is None
            or kernel_size is None
        ):
            return None
        eps_str = (
            str(int(float(epsilon)))
            if float(epsilon) == int(float(epsilon))
            else str(epsilon)
        )
        mg_str = str(max_grad_norm)  # Keep dot, don't replace with underscore
        summary_file = os.path.join(
            base_path,
            morph_operation,
            f"kernel_{kernel_size}",
            f"batch_{batch_size}",
            f"run_{run_number}",
            f"epsilon_{eps_str}",
            f"maxgrad_{mg_str}",
            clipping_strategy,
            f"{model_name}_batch{batch_size}_run{run_number}_eps{eps_str}_mg{mg_str}_{clipping_strategy}_{morph_operation}_k{kernel_size}_summary.txt",
        )
    else:
        return None

    if os.path.exists(summary_file):
        return summary_file
    return None


# Step 1: Read training CSV and filter models
print("Step 1: Reading training CSV and filtering models...")
df = pd.read_csv(training_csv)

# Filter for all models (Inf-Net, UNet, NestedUNet)
all_models = [
    "Inf-Net_GroupNorm",
    "Inf-Net_DP",
    "Inf-Net_Morph_GroupNorm",
    "Inf-Net_DP_Morph",
    "UNet_GroupNorm",
    "UNet_DP",
    "UNet_Morph_GroupNorm",
    "UNet_DP_Morph",
    "NestedUNet_GroupNorm",
    "NestedUNet_DP",
    "NestedUNet_Morph_GroupNorm",
    "NestedUNet_DP_Morph",
]
df_filtered_models = df[df["Model_Name"].isin(all_models)].copy()

print(f"Found {len(df_filtered_models)} training records for all models")

# Step 2: Filter to keep only combinations with exactly 2 runs
print("\nStep 2: Filtering to combinations with exactly 2 runs...")


# Define grouping columns based on model type
def get_grouping_key(row):
    """Get grouping key tuple for each model type"""
    base_key = (
        row["Model_Name"],
        row["Dataset"],
        row["DPSGD"],
        row["Batch_Size"],
        row["Learning_Rate"],
    )

    model_name = row["Model_Name"]
    has_dp = "_DP" in model_name
    has_morph = "_Morph" in model_name

    if not has_dp and not has_morph:
        # GroupNorm models
        return base_key
    elif has_dp and not has_morph:
        # DP models
        return base_key + (
            row["Epsilon"],
            row["Max_Grad_Norm"],
            row["Clipping_Strategy"],
        )
    elif not has_dp and has_morph:
        # Morph_GroupNorm models
        return base_key + (row["Morphology"], row["Operation"], row["Kernel_Size"])
    elif has_dp and has_morph:
        # DP_Morph models
        return base_key + (
            row["Epsilon"],
            row["Max_Grad_Norm"],
            row["Clipping_Strategy"],
            row["Morphology"],
            row["Operation"],
            row["Kernel_Size"],
        )
    return base_key


# Create grouping key column
df_filtered_models["_group_key"] = df_filtered_models.apply(get_grouping_key, axis=1)

# Group and filter
grouped = df_filtered_models.groupby("_group_key")
valid_combinations = []

for group_key, group_df in grouped:
    run_numbers = set(group_df["Run_Number"].values)
    # Accept if we have exactly 2 runs (1 and 2), or if we have at least runs 1 and 2 (may have duplicates)
    # Also accept if we have only 1 run (will use std=0.0)
    if {1, 2}.issubset(run_numbers):
        # If we have exactly 2 records with runs 1 and 2, use them
        if len(group_df) == 2 and run_numbers == {1, 2}:
            group_df_clean = group_df.drop(columns=["_group_key"])
            valid_combinations.append(group_df_clean)
        # If we have more than 2 records but include runs 1 and 2, take the first occurrence of each
        elif len(group_df) >= 2:
            # Get first occurrence of run 1 and run 2
            run_1 = group_df[group_df["Run_Number"] == 1]
            run_2 = group_df[group_df["Run_Number"] == 2]
            if len(run_1) > 0 and len(run_2) > 0:
                # Take first occurrence of each run
                selected = pd.concat([run_1.iloc[[0]], run_2.iloc[[0]]])
                group_df_clean = selected.drop(columns=["_group_key"])
                valid_combinations.append(group_df_clean)
    # Accept single run (will use std=0.0 in aggregation)
    elif len(run_numbers) == 1 and 1 in run_numbers:
        # Take first occurrence of run 1
        run_1 = group_df[group_df["Run_Number"] == 1]
        if len(run_1) > 0:
            selected = run_1.iloc[[0]]
            group_df_clean = selected.drop(columns=["_group_key"])
            valid_combinations.append(group_df_clean)

if not valid_combinations:
    print("No combinations found with 1 or 2 runs!")
    exit(1)

df_filtered = pd.concat(valid_combinations, ignore_index=True)
# Count combinations (some may have 1 run, some 2 runs)
unique_combinations = len(
    df_filtered.groupby(df_filtered.apply(get_grouping_key, axis=1))
)
print(
    f"Found {len(df_filtered)} records ({unique_combinations} unique combinations, some with 1 run, some with 2 runs)"
)

# Save filtered CSV (combined) — merge with existing so we never overwrite/drop rows
output_filtered = os.path.join(combined_dir, "filtered_two_runs.csv")
df_filtered = _merge_with_existing(
    df_filtered, output_filtered, KEY_COLS_FILTERED_MERGED, keep="last"
)
df_filtered.to_csv(output_filtered, index=False)
print(f"Saved filtered CSV to: {output_filtered}")

# Step 3: Map with evaluation results
print("\nStep 3: Mapping with evaluation results...")

evaluation_data = []
missing_eval_count = 0

for idx, row in df_filtered.iterrows():
    model_name = row["Model_Name"]
    batch_size = int(row["Batch_Size"])
    run_number = int(row["Run_Number"])

    # Extract parameters
    epsilon = row["Epsilon"] if pd.notna(row["Epsilon"]) else None
    max_grad_norm = row["Max_Grad_Norm"] if pd.notna(row["Max_Grad_Norm"]) else None
    clipping_strategy = (
        row["Clipping_Strategy"]
        if pd.notna(row["Clipping_Strategy"]) and row["Clipping_Strategy"] != "none"
        else None
    )
    morph_operation = (
        row["Operation"]
        if pd.notna(row["Operation"]) and row["Operation"] != "none"
        else None
    )
    kernel_size = (
        int(row["Kernel_Size"])
        if pd.notna(row["Kernel_Size"]) and row["Kernel_Size"] != 0
        else None
    )

    # Find evaluation file
    eval_file = find_evaluation_file(
        model_name,
        batch_size,
        run_number,
        epsilon,
        max_grad_norm,
        clipping_strategy,
        morph_operation,
        kernel_size,
    )

    if eval_file and os.path.exists(eval_file):
        metrics = parse_evaluation_summary(eval_file)
        # Combine training and evaluation data
        combined = row.to_dict()
        combined.update(metrics)
        evaluation_data.append(combined)
    else:
        missing_eval_count += 1
        if missing_eval_count <= 10:  # Only print first 10 warnings
            print(
                f"Warning: Evaluation file not found for {model_name}, batch={batch_size}, run={run_number}, eps={epsilon}, mg={max_grad_norm}, clip={clipping_strategy}, op={morph_operation}, k={kernel_size}"
            )
        # Still add the row but without evaluation metrics
        combined = row.to_dict()
        evaluation_data.append(combined)

if missing_eval_count > 10:
    print(f"... and {missing_eval_count - 10} more missing evaluation files")

df_merged = pd.DataFrame(evaluation_data)

# Save merged CSV (combined) — merge with existing so we never overwrite/drop rows
output_merged = os.path.join(combined_dir, "merged_training_eval.csv")
df_merged = _merge_with_existing(
    df_merged, output_merged, KEY_COLS_FILTERED_MERGED, keep="last"
)
df_merged.to_csv(output_merged, index=False)
print(f"Saved merged CSV to: {output_merged}")
print(f"Total merged records: {len(df_merged)}")
print(f"Missing evaluation files: {missing_eval_count}")

# Step 4: Calculate mean ± std deviation over runs (1 or 2 runs)
print("\nStep 4: Calculating mean ± std deviation over runs (1 or 2 runs)...")

# Define numeric columns for aggregation
numeric_cols = [
    "Training_Loss",
    "Training_Time_Seconds",
    "Mean_Dice",
    "Max_Dice",
    "Mean_Sensitivity",
    "Max_Sensitivity",
    "Mean_Specificity",
    "Max_Specificity",
    "S_measure",
    "Mean_E_measure",
    "Max_E_measure",
    "MAE",
]

# Filter to only columns that exist
numeric_cols = [col for col in numeric_cols if col in df_merged.columns]

# Create grouping key column for merged data
df_merged["_group_key"] = df_merged.apply(get_grouping_key, axis=1)

# Group by same grouping as before
grouped_merged = df_merged.groupby("_group_key")

aggregated_data = []
skipped_combinations = 0

for group_key, group_df in grouped_merged:
    # Accept combinations with 1 or 2 runs
    # For 1 run: use that value with std=0.0
    # For 2 runs: compute mean and std
    if len(group_df) not in [1, 2]:
        skipped_combinations += 1
        continue

    # Get the first row as base (non-numeric columns)
    base_row = group_df.iloc[0].copy()

    # Calculate mean and std for numeric columns
    for col in numeric_cols:
        values = group_df[col].dropna()
        if len(values) == 2:
            mean_val = values.mean()
            std_val = values.std()
            base_row[f"{col}_mean"] = mean_val
            base_row[f"{col}_std"] = std_val
            base_row[f"{col}_mean_std"] = f"{mean_val:.4f} ± {std_val:.4f}"
        elif len(values) == 1:
            # Single run: use that value with std=0.0
            base_row[f"{col}_mean"] = values.iloc[0]
            base_row[f"{col}_std"] = 0.0
            base_row[f"{col}_mean_std"] = f"{values.iloc[0]:.4f} ± 0.0000"
        else:
            base_row[f"{col}_mean"] = np.nan
            base_row[f"{col}_std"] = np.nan
            base_row[f"{col}_mean_std"] = "N/A"

    # Remove original numeric columns and run-specific columns
    cols_to_drop = numeric_cols + ["Run_Number", "Iterations", "_group_key"]
    for col in cols_to_drop:
        if col in base_row.index:
            base_row = base_row.drop(labels=[col])

    aggregated_data.append(base_row)

# Remove temporary column from merged dataframe
if "_group_key" in df_merged.columns:
    df_merged = df_merged.drop(columns=["_group_key"])

df_aggregated = pd.DataFrame(aggregated_data)

# Save aggregated CSV (combined) — merge with existing so we never overwrite/drop rows
output_aggregated = os.path.join(combined_dir, "aggregated_mean_std.csv")
key_cols_agg = [c for c in KEY_COLS_AGGREGATED if c in df_aggregated.columns]
df_aggregated = _merge_with_existing(
    df_aggregated, output_aggregated, key_cols_agg, keep="last"
)
df_aggregated.to_csv(output_aggregated, index=False)
print(f"Saved aggregated CSV to: {output_aggregated}")
print(f"Total aggregated combinations: {len(df_aggregated)}")
if skipped_combinations > 0:
    print(f"Skipped {skipped_combinations} combinations (not 1 or 2 runs)")

# Save separate CSV files for each model
print("\nSaving separate CSV files for each model...")

# For filtered data — merge with existing per-model file
for model_name in df_filtered["Model_Name"].unique():
    model_df = df_filtered[df_filtered["Model_Name"] == model_name]
    safe_name = model_name.replace("-", "_").replace(" ", "_")
    output_file = os.path.join(filtered_dir, f"{safe_name}_filtered_two_runs.csv")
    model_df = _merge_with_existing(
        model_df, output_file, KEY_COLS_FILTERED_MERGED, keep="last"
    )
    model_df.to_csv(output_file, index=False)
    print(f"  Saved {model_name} filtered: {len(model_df)} records")

# For merged data — merge with existing per-model file
for model_name in df_merged["Model_Name"].unique():
    model_df = df_merged[df_merged["Model_Name"] == model_name]
    safe_name = model_name.replace("-", "_").replace(" ", "_")
    output_file = os.path.join(merged_dir, f"{safe_name}_merged_training_eval.csv")
    model_df = _merge_with_existing(
        model_df, output_file, KEY_COLS_FILTERED_MERGED, keep="last"
    )
    model_df.to_csv(output_file, index=False)
    print(f"  Saved {model_name} merged: {len(model_df)} records")

# For aggregated data — merge with existing per-model file
for model_name in df_aggregated["Model_Name"].unique():
    model_df = df_aggregated[df_aggregated["Model_Name"] == model_name]
    safe_name = model_name.replace("-", "_").replace(" ", "_")
    output_file = os.path.join(aggregated_dir, f"{safe_name}_aggregated_mean_std.csv")
    key_cols_agg = [c for c in KEY_COLS_AGGREGATED if c in model_df.columns]
    model_df = _merge_with_existing(model_df, output_file, key_cols_agg, keep="last")
    model_df.to_csv(output_file, index=False)
    print(f"  Saved {model_name} aggregated: {len(model_df)} combinations")

print("\nDone! Summary:")
print(f"1. Filtered CSV (2 runs per combination): {output_filtered}")
print(f"2. Merged CSV (training + evaluation): {output_merged}")
print(f"3. Aggregated CSV (mean ± std): {output_aggregated}")
print(
    f"\nPlus separate CSV files for each model type ({len(df_aggregated['Model_Name'].unique())} models)"
)
