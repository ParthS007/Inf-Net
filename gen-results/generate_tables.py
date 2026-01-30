#!/usr/bin/env python3
"""
Generate LaTeX results tables for Lung CT Segmentation experiments.

Usage:
    python generate_tables.py --arch infnet
    python generate_tables.py --arch unet --output results_unet.tex
    python generate_tables.py --arch nestedunet --traceability
"""

import os
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

# Default results path
DEFAULT_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "aggregated")

# Architecture display name mapping
ARCH_DISPLAY_NAMES = {
    "infnet": "Inf-Net",
    "unet": "U-Net",
    "nestedunet": "U-Net++",
}

# Model name mapping (from architecture to CSV file names)
MODEL_NAME_MAP = {
    "infnet": "Inf_Net",  # Note: underscore in filename
    "unet": "UNet",
    "nestedunet": "NestedUNet",
}

# Global variables set by CLI args
RESULTS_PATH = None
ARCH = None
ARCH_DISPLAY = None

# Clipping strategy display names
STRATEGY_DISPLAY = {
    "automatic": "AUTO-S",
    "base": "Flat",
    "nsgd": "Normalized SGD",
    "psac": "PSAC",
}

STRATEGY_ORDER = ["automatic", "base", "nsgd", "psac"]

# Max grad norm values (for reference; ablation is skipped)
MAX_GRAD_NORM_VALUES = [1.2, 1.5, 2.0]

# Max grad norm is selected from global ablation (see generate_ablation_combined.py / ablation_combined.tex)

# Metrics to report (6 total)
# Standard metrics (4)
STANDARD_METRICS = [
    ("Mean_Dice", "Dice", "↑"),
    ("Mean_Sensitivity", "Sens", "↑"),
    ("Mean_Specificity", "Spec", "↑"),
    ("MAE", "MAE", "↓"),
]

# Structure-aware metrics (2)
STRUCTURE_AWARE_METRICS = [
    ("S_measure", "S-meas", "↑"),
    ("Mean_E_measure", "E-meas", "↑"),
]

# All metrics (for backward compatibility)
METRICS = STANDARD_METRICS + STRUCTURE_AWARE_METRICS


def load_aggregated_csv(filename, path=None):
    """Load aggregated CSV file. If path is given, use it; else use RESULTS_PATH."""
    base = path if path is not None else RESULTS_PATH
    filepath = os.path.join(base, filename)
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def format_metric(mean, std, precision=4):
    """Format metric as mean ± std."""
    if pd.isna(mean) or mean is None:
        return "---"
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"


def format_metric_bold(mean, std, is_best, precision=4):
    """Format metric as mean ± std, with bold if best."""
    if pd.isna(mean) or mean is None:
        return "---"
    fmt = f"{{:.{precision}f}}"
    val_str = f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"
    if is_best:
        return f"\\textbf{{{val_str}}}"
    return val_str


def collect_nonprivate_results():
    """Collect non-private baseline and morphology results."""
    results = defaultdict(dict)

    model_name = MODEL_NAME_MAP[ARCH]

    # Load baseline CSV
    baseline_file = f"{model_name}_GroupNorm_aggregated_mean_std.csv"
    baseline_df = load_aggregated_csv(baseline_file)

    if baseline_df is not None:
        for idx, row in baseline_df.iterrows():
            bs = int(row["Batch_Size"])
            key = ("baseline", bs)
            trace_info = {
                "csv_file": baseline_file,
                "row_index": idx + 2,  # +2 for header and 0-index
            }
            results[key] = {
                "dice": (row["Mean_Dice_mean"], row["Mean_Dice_std"]),
                "sens": (row["Mean_Sensitivity_mean"], row["Mean_Sensitivity_std"]),
                "spec": (row["Mean_Specificity_mean"], row["Mean_Specificity_std"]),
                "s_meas": (row["S_measure_mean"], row["S_measure_std"]),
                "e_meas": (row["Mean_E_measure_mean"], row["Mean_E_measure_std"]),
                "mae": (row["MAE_mean"], row["MAE_std"]),
                "trace": trace_info,
            }

    # Load morphology CSV
    morph_file = f"{model_name}_Morph_GroupNorm_aggregated_mean_std.csv"
    morph_df = load_aggregated_csv(morph_file)

    if morph_df is not None:
        for idx, row in morph_df.iterrows():
            if row["Operation"] in ["open", "close", "both"] and row["Kernel_Size"] in [
                3,
                5,
            ]:
                op = row["Operation"]
                k = int(row["Kernel_Size"])
                bs = int(row["Batch_Size"])
                key = ("morph", op, k, bs)
                trace_info = {
                    "csv_file": morph_file,
                    "row_index": idx + 2,
                    "operation": op,
                    "kernel_size": k,
                }
                results[key] = {
                    "dice": (row["Mean_Dice_mean"], row["Mean_Dice_std"]),
                    "sens": (row["Mean_Sensitivity_mean"], row["Mean_Sensitivity_std"]),
                    "spec": (row["Mean_Specificity_mean"], row["Mean_Specificity_std"]),
                    "s_meas": (row["S_measure_mean"], row["S_measure_std"]),
                    "e_meas": (row["Mean_E_measure_mean"], row["Mean_E_measure_std"]),
                    "mae": (row["MAE_mean"], row["MAE_std"]),
                    "trace": trace_info,
                }

    return results


def collect_dp_ablation_results():
    """Collect DP baseline results for max grad norm ablation (no morphology)."""
    results = defaultdict(dict)

    model_name = MODEL_NAME_MAP[ARCH]
    dp_file = f"{model_name}_DP_aggregated_mean_std.csv"
    dp_df = load_aggregated_csv(dp_file)

    if dp_df is not None:
        # Filter for baseline only (no morphology)
        baseline_df = dp_df[
            (dp_df["Morphology"] == False) | (dp_df["Operation"] == "none")
        ]

        for idx, row in baseline_df.iterrows():
            eps = float(row["Epsilon"])
            strategy = row["Clipping_Strategy"]
            max_grad = float(row["Max_Grad_Norm"])
            bs = int(row["Batch_Size"])

            key = (eps, strategy, max_grad, bs)
            trace_info = {
                "csv_file": dp_file,
                "row_index": idx + 2,
                "epsilon": eps,
                "strategy": strategy,
                "max_grad_norm": max_grad,
            }
            results[key] = {
                "dice": (row["Mean_Dice_mean"], row["Mean_Dice_std"]),
                "sens": (row["Mean_Sensitivity_mean"], row["Mean_Sensitivity_std"]),
                "spec": (row["Mean_Specificity_mean"], row["Mean_Specificity_std"]),
                "s_meas": (row["S_measure_mean"], row["S_measure_std"]),
                "e_meas": (row["Mean_E_measure_mean"], row["Mean_E_measure_std"]),
                "mae": (row["MAE_mean"], row["MAE_std"]),
                "trace": trace_info,
            }

    return results


def collect_dp_ablation_results_for_arch(arch, results_path):
    """Collect DP baseline results for max grad norm ablation for a given architecture."""
    results = defaultdict(dict)
    model_name = MODEL_NAME_MAP[arch]
    dp_file = f"{model_name}_DP_aggregated_mean_std.csv"
    dp_df = load_aggregated_csv(dp_file, path=results_path)
    if dp_df is None:
        return results
    baseline_df = dp_df[(dp_df["Morphology"] == False) | (dp_df["Operation"] == "none")]
    for idx, row in baseline_df.iterrows():
        eps = float(row["Epsilon"])
        strategy = row["Clipping_Strategy"]
        max_grad = float(row["Max_Grad_Norm"])
        bs = int(row["Batch_Size"])
        key = (eps, strategy, max_grad, bs)
        trace_info = {
            "csv_file": dp_file,
            "row_index": idx + 2,
            "epsilon": eps,
            "strategy": strategy,
            "max_grad_norm": max_grad,
        }
        results[key] = {
            "dice": (row["Mean_Dice_mean"], row["Mean_Dice_std"]),
            "sens": (row["Mean_Sensitivity_mean"], row["Mean_Sensitivity_std"]),
            "spec": (row["Mean_Specificity_mean"], row["Mean_Specificity_std"]),
            "s_meas": (row["S_measure_mean"], row["S_measure_std"]),
            "e_meas": (row["Mean_E_measure_mean"], row["Mean_E_measure_std"]),
            "mae": (row["MAE_mean"], row["MAE_std"]),
            "trace": trace_info,
        }
    return results


def build_ablation_metrics_list(ablation_results):
    """Build the list of row dicts (maxgrad_* keys) from raw ablation results dict."""
    all_metrics_data = []
    for eps in [8.0, 200.0]:
        eps_metrics = []
        for strategy in STRATEGY_ORDER:
            for bs in [24, 48]:
                row_data = {"strategy": strategy, "bs": bs, "eps": eps}
                for max_grad in MAX_GRAD_NORM_VALUES:
                    key = (eps, strategy, max_grad, bs)
                    if key in ablation_results:
                        data = ablation_results[key]
                        row_data[f"maxgrad_{max_grad}_dice"] = data["dice"]
                        row_data[f"maxgrad_{max_grad}_sens"] = data["sens"]
                        row_data[f"maxgrad_{max_grad}_spec"] = data["spec"]
                        row_data[f"maxgrad_{max_grad}_s_meas"] = data["s_meas"]
                        row_data[f"maxgrad_{max_grad}_e_meas"] = data["e_meas"]
                        row_data[f"maxgrad_{max_grad}_mae"] = data["mae"]
                        row_data[f"maxgrad_{max_grad}_trace"] = data.get("trace", {})
                    else:
                        row_data[f"maxgrad_{max_grad}_dice"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_sens"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_spec"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_s_meas"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_e_meas"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_mae"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_trace"] = {}
                eps_metrics.append(row_data)
        all_metrics_data.extend(eps_metrics)
    return all_metrics_data


def select_global_best_max_grad_norm(results_path):
    """Select one best max grad norm from ablation data of all three architectures."""
    combined_ablation_metrics = []
    for arch in ["infnet", "unet", "nestedunet"]:
        ablation_results = collect_dp_ablation_results_for_arch(arch, results_path)
        metrics_list = build_ablation_metrics_list(ablation_results)
        combined_ablation_metrics.extend(metrics_list)
    return select_best_max_grad_norm(combined_ablation_metrics)


def collect_dp_results(chosen_max_grad_norm):
    """Collect DP results with chosen max grad norm (all configurations)."""
    results = defaultdict(dict)

    model_name = MODEL_NAME_MAP[ARCH]

    # Load DP baseline
    dp_file = f"{model_name}_DP_aggregated_mean_std.csv"
    dp_df = load_aggregated_csv(dp_file)

    if dp_df is not None:
        baseline_df = dp_df[
            ((dp_df["Morphology"] == False) | (dp_df["Operation"] == "none"))
            & (dp_df["Max_Grad_Norm"] == chosen_max_grad_norm)
        ]

        for idx, row in baseline_df.iterrows():
            eps = float(row["Epsilon"])
            strategy = row["Clipping_Strategy"]
            bs = int(row["Batch_Size"])

            key = (eps, "baseline", strategy, bs)
            trace_info = {
                "csv_file": dp_file,
                "row_index": idx + 2,
                "epsilon": eps,
                "strategy": strategy,
                "max_grad_norm": chosen_max_grad_norm,
            }
            results[key] = {
                "dice": (row["Mean_Dice_mean"], row["Mean_Dice_std"]),
                "sens": (row["Mean_Sensitivity_mean"], row["Mean_Sensitivity_std"]),
                "spec": (row["Mean_Specificity_mean"], row["Mean_Specificity_std"]),
                "s_meas": (row["S_measure_mean"], row["S_measure_std"]),
                "e_meas": (row["Mean_E_measure_mean"], row["Mean_E_measure_std"]),
                "mae": (row["MAE_mean"], row["MAE_std"]),
                "trace": trace_info,
            }

    # Load DP Morph
    dp_morph_file = f"{model_name}_DP_Morph_aggregated_mean_std.csv"
    dp_morph_df = load_aggregated_csv(dp_morph_file)

    if dp_morph_df is not None:
        morph_df = dp_morph_df[
            (dp_morph_df["Morphology"] == True)
            & (dp_morph_df["Operation"].isin(["open", "close", "both"]))
            & (dp_morph_df["Kernel_Size"].isin([3, 5]))
            & (dp_morph_df["Max_Grad_Norm"] == chosen_max_grad_norm)
        ]

        for idx, row in morph_df.iterrows():
            eps = float(row["Epsilon"])
            strategy = row["Clipping_Strategy"]
            op = row["Operation"]
            k = int(row["Kernel_Size"])
            bs = int(row["Batch_Size"])

            key = (eps, "morph", op, k, strategy, bs)
            trace_info = {
                "csv_file": dp_morph_file,
                "row_index": idx + 2,
                "epsilon": eps,
                "strategy": strategy,
                "max_grad_norm": chosen_max_grad_norm,
                "operation": op,
                "kernel_size": k,
            }
            results[key] = {
                "dice": (row["Mean_Dice_mean"], row["Mean_Dice_std"]),
                "sens": (row["Mean_Sensitivity_mean"], row["Mean_Sensitivity_std"]),
                "spec": (row["Mean_Specificity_mean"], row["Mean_Specificity_std"]),
                "s_meas": (row["S_measure_mean"], row["S_measure_std"]),
                "e_meas": (row["Mean_E_measure_mean"], row["Mean_E_measure_std"]),
                "mae": (row["MAE_mean"], row["MAE_std"]),
                "trace": trace_info,
            }

    return results


# Only kernel size 5 in tables (k=3 omitted to keep tables shorter)
KERNEL_SIZES_IN_TABLE = [5]


def generate_nonprivate_table(results):
    """Generate Table 1: Non-Private Results."""
    configs = [
        ("Baseline", "baseline", None, None),
        ("Morph-Open $k$=5", "morph", "open", 5),
        ("Morph-Close $k$=5", "morph", "close", 5),
        ("Morph-Both $k$=5", "morph", "both", 5),
    ]

    all_metrics = []

    for config_name, config_type, op, k in configs:
        for bs in [24, 48]:
            if config_type == "baseline":
                key = ("baseline", bs)
            else:
                key = ("morph", op, k, bs)

            if key in results:
                data = results[key]
                all_metrics.append(
                    {
                        "config": config_name,
                        "bs": bs,
                        "dice": data["dice"],
                        "sens": data["sens"],
                        "spec": data["spec"],
                        "s_meas": data["s_meas"],
                        "e_meas": data["e_meas"],
                        "mae": data["mae"],
                        "trace": data.get("trace", {}),
                    }
                )
            else:
                all_metrics.append(
                    {
                        "config": config_name,
                        "bs": bs,
                        "dice": (None, None),
                        "sens": (None, None),
                        "spec": (None, None),
                        "s_meas": (None, None),
                        "e_meas": (None, None),
                        "mae": (None, None),
                        "trace": {},
                    }
                )

    # Find best values per metric
    best_vals = {}
    metric_map = {
        "Mean_Dice": "dice",
        "Mean_Sensitivity": "sens",
        "Mean_Specificity": "spec",
        "S_measure": "s_meas",
        "Mean_E_measure": "e_meas",
        "MAE": "mae",
    }

    for metric_key, metric_name, direction in METRICS:
        metric_accessor = metric_map[metric_key]
        if metric_key == "MAE":
            # Lower is better
            vals = [
                m[metric_accessor][0]
                for m in all_metrics
                if m[metric_accessor][0] is not None
            ]
            best_vals[metric_key] = min(vals) if vals else None
        else:
            # Higher is better
            vals = [
                m[metric_accessor][0]
                for m in all_metrics
                if m[metric_accessor][0] is not None
            ]
            best_vals[metric_key] = max(vals) if vals else None

    # Generate table rows
    latex_rows = []
    prev_config = None

    for m in all_metrics:
        if prev_config is not None and prev_config != m["config"].split()[0]:
            latex_rows.append("\\midrule")
        prev_config = m["config"].split()[0]

        # Standard metrics columns
        standard_cols = []
        metric_map = {
            "Mean_Dice": "dice",
            "Mean_Sensitivity": "sens",
            "Mean_Specificity": "spec",
            "MAE": "mae",
        }
        for metric_key, metric_name, direction in STANDARD_METRICS:
            metric_accessor = metric_map[metric_key]
            metric_data = m[metric_accessor]
            mean, std = metric_data
            is_best = mean == best_vals[metric_key] and mean is not None
            standard_cols.append(format_metric_bold(mean, std, is_best))

        # Structure-aware metrics columns
        structure_cols = []
        metric_map = {
            "S_measure": "s_meas",
            "Mean_E_measure": "e_meas",
        }
        for metric_key, metric_name, direction in STRUCTURE_AWARE_METRICS:
            metric_accessor = metric_map[metric_key]
            metric_data = m[metric_accessor]
            mean, std = metric_data
            is_best = mean == best_vals[metric_key] and mean is not None
            structure_cols.append(format_metric_bold(mean, std, is_best))

        latex_rows.append(
            f"{m['config']} & {m['bs']} & {' & '.join(standard_cols)} & {' & '.join(structure_cols)} \\\\"
        )

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} non-private baseline results on COVID-19 Lung Infection dataset. 
$\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. 
Morphology applied to all classes (binary segmentation).
Best values per metric are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-nonprivate}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll|cccc|cc}}
\\toprule
\\multirow{{2}}{{*}}{{Configuration}} & \\multirow{{2}}{{*}}{{Batch Size}} & 
\\multicolumn{{4}}{{c}}{{Standard Metrics}} & \\multicolumn{{2}}{{c}}{{Structure-aware Metrics}} \\\\
\\cmidrule(lr){{3-6}} \\cmidrule(lr){{7-8}}
& & Dice $\\uparrow$ & Sens $\\uparrow$ & Spec $\\uparrow$ & MAE $\\downarrow$ & 
S-meas $\\uparrow$ & E-meas $\\uparrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, all_metrics


def generate_maxgrad_ablation_table(results):
    """Generate Table 2: DP Max Grad Norm Ablation."""
    latex_rows = []
    all_metrics_data = []

    # Process each epsilon section
    for eps in [8.0, 200.0]:
        eps_metrics = []

        for strategy in STRATEGY_ORDER:
            for bs in [24, 48]:
                row_data = {
                    "strategy": strategy,
                    "bs": bs,
                    "eps": eps,
                }

                # Collect metrics for each max grad norm
                for max_grad in MAX_GRAD_NORM_VALUES:
                    key = (eps, strategy, max_grad, bs)
                    if key in results:
                        data = results[key]
                        row_data[f"maxgrad_{max_grad}_dice"] = data["dice"]
                        row_data[f"maxgrad_{max_grad}_sens"] = data["sens"]
                        row_data[f"maxgrad_{max_grad}_spec"] = data["spec"]
                        row_data[f"maxgrad_{max_grad}_s_meas"] = data["s_meas"]
                        row_data[f"maxgrad_{max_grad}_e_meas"] = data["e_meas"]
                        row_data[f"maxgrad_{max_grad}_mae"] = data["mae"]
                        row_data[f"maxgrad_{max_grad}_trace"] = data.get("trace", {})
                    else:
                        row_data[f"maxgrad_{max_grad}_dice"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_sens"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_spec"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_s_meas"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_e_meas"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_mae"] = (None, None)
                        row_data[f"maxgrad_{max_grad}_trace"] = {}

                eps_metrics.append(row_data)

        all_metrics_data.extend(eps_metrics)

        # Find best values per metric within this epsilon section
        best_vals = {}
        metric_map = {
            "Mean_Dice": "dice",
            "Mean_Sensitivity": "sens",
            "Mean_Specificity": "spec",
            "S_measure": "s_meas",
            "Mean_E_measure": "e_meas",
            "MAE": "mae",
        }
        for metric_key, metric_name, direction in METRICS:
            metric_accessor = metric_map[metric_key]
            if metric_key == "MAE":
                # Lower is better
                vals = []
                for m in eps_metrics:
                    for max_grad in MAX_GRAD_NORM_VALUES:
                        val = m.get(
                            f"maxgrad_{max_grad}_{metric_accessor}", (None, None)
                        )[0]
                        if val is not None:
                            vals.append(val)
                best_vals[metric_key] = min(vals) if vals else None
            else:
                # Higher is better
                vals = []
                for m in eps_metrics:
                    for max_grad in MAX_GRAD_NORM_VALUES:
                        val = m.get(
                            f"maxgrad_{max_grad}_{metric_accessor}", (None, None)
                        )[0]
                        if val is not None:
                            vals.append(val)
                best_vals[metric_key] = max(vals) if vals else None

        # Generate rows for this epsilon
        for i, m in enumerate(eps_metrics):
            if i == 0:
                eps_cell = f"\\multirow{{8}}{{*}}{{$\\varepsilon = {int(eps)}$}}"
            else:
                eps_cell = ""

            # Build columns for each max grad norm value
            all_cols = []
            for max_grad in MAX_GRAD_NORM_VALUES:
                # Standard metrics for this max grad norm
                standard_cols = []
                metric_map = {
                    "Mean_Dice": "dice",
                    "Mean_Sensitivity": "sens",
                    "Mean_Specificity": "spec",
                    "MAE": "mae",
                }
                for metric_key, metric_name, direction in STANDARD_METRICS:
                    metric_accessor = metric_map[metric_key]
                    metric_data = m.get(
                        f"maxgrad_{max_grad}_{metric_accessor}", (None, None)
                    )
                    mean, std = metric_data
                    is_best = mean == best_vals[metric_key] and mean is not None
                    standard_cols.append(format_metric_bold(mean, std, is_best))

                # Structure-aware metrics for this max grad norm
                structure_cols = []
                metric_map = {
                    "S_measure": "s_meas",
                    "Mean_E_measure": "e_meas",
                }
                for metric_key, metric_name, direction in STRUCTURE_AWARE_METRICS:
                    metric_accessor = metric_map[metric_key]
                    metric_data = m.get(
                        f"maxgrad_{max_grad}_{metric_accessor}", (None, None)
                    )
                    mean, std = metric_data
                    is_best = mean == best_vals[metric_key] and mean is not None
                    structure_cols.append(format_metric_bold(mean, std, is_best))

                all_cols.extend(standard_cols)
                all_cols.extend(structure_cols)

            latex_rows.append(
                f"{eps_cell} & {STRATEGY_DISPLAY[m['strategy']]} & {m['bs']} & {' & '.join(all_cols)} \\\\"
            )

        if eps != 200.0:  # Add midrule between epsilon sections
            latex_rows.append("\\midrule")

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} DP max gradient norm ablation study on COVID-19 Lung Infection dataset comparing 
max grad norm values of 1.2, 1.5, and 2.0 under $\\varepsilon = 8$ (strong privacy) and 
$\\varepsilon = 200$ (weak privacy). $\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. 
Only baseline configurations (no morphology) are shown to isolate max grad norm effects.
Best values per metric within each privacy level are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-dp-maxgrad-ablation}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lll|cccc|cc|cccc|cc|cccc|cc}}
\\toprule
\\multirow{{2}}{{*}}{{$\\varepsilon$}} & \\multirow{{2}}{{*}}{{Clipping Strategy}} & \\multirow{{2}}{{*}}{{Batch Size}} & 
\\multicolumn{{6}}{{c}}{{Max Grad Norm = 1.2}} & 
\\multicolumn{{6}}{{c}}{{Max Grad Norm = 1.5}} & 
\\multicolumn{{6}}{{c}}{{Max Grad Norm = 2.0}} \\\\
\\cmidrule(lr){{4-9}} \\cmidrule(lr){{10-15}} \\cmidrule(lr){{16-21}}
& & & \\multicolumn{{4}}{{c}}{{Standard}} & \\multicolumn{{2}}{{c}}{{Structure-aware}} & 
\\multicolumn{{4}}{{c}}{{Standard}} & \\multicolumn{{2}}{{c}}{{Structure-aware}} & 
\\multicolumn{{4}}{{c}}{{Standard}} & \\multicolumn{{2}}{{c}}{{Structure-aware}} \\\\
\\cmidrule(lr){{4-7}} \\cmidrule(lr){{8-9}} \\cmidrule(lr){{10-13}} \\cmidrule(lr){{14-15}} \\cmidrule(lr){{16-19}} \\cmidrule(lr){{20-21}}
& & & Dice $\\uparrow$ & Sens $\\uparrow$ & Spec $\\uparrow$ & MAE $\\downarrow$ & 
S-meas $\\uparrow$ & E-meas $\\uparrow$ & 
Dice $\\uparrow$ & Sens $\\uparrow$ & Spec $\\uparrow$ & MAE $\\downarrow$ & 
S-meas $\\uparrow$ & E-meas $\\uparrow$ & 
Dice $\\uparrow$ & Sens $\\uparrow$ & Spec $\\uparrow$ & MAE $\\downarrow$ & 
S-meas $\\uparrow$ & E-meas $\\uparrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, all_metrics_data


def select_best_max_grad_norm(ablation_metrics):
    """Select best max grad norm based on all 6 metrics (weighted/majority vote)."""
    max_grad_scores = {1.2: 0, 1.5: 0, 2.0: 0}

    # Score each max grad norm based on best performance across metrics
    metric_map = {
        "Mean_Dice": "dice",
        "Mean_Sensitivity": "sens",
        "Mean_Specificity": "spec",
        "S_measure": "s_meas",
        "Mean_E_measure": "e_meas",
        "MAE": "mae",
    }
    for metric_key, metric_name, direction in METRICS:
        metric_accessor = metric_map[metric_key]
        is_lower_better = metric_key == "MAE"

        best_val = None
        best_max_grad = None

        for m in ablation_metrics:
            for max_grad in MAX_GRAD_NORM_VALUES:
                metric_data = m.get(
                    f"maxgrad_{max_grad}_{metric_accessor}", (None, None)
                )
                mean = metric_data[0]
                if mean is not None:
                    if best_val is None:
                        best_val = mean
                        best_max_grad = max_grad
                    elif is_lower_better:
                        if mean < best_val:
                            best_val = mean
                            best_max_grad = max_grad
                    else:
                        if mean > best_val:
                            best_val = mean
                            best_max_grad = max_grad

        if best_max_grad:
            max_grad_scores[best_max_grad] += 1

    # Return max grad norm with highest score
    best_max_grad = max(max_grad_scores, key=max_grad_scores.get)
    return best_max_grad


def generate_dp_table(results, chosen_max_grad_norm):
    """Generate Table 3: DP Results with chosen max grad norm."""
    configs = [
        ("Standard No Morph", "baseline", None, None),
        ("Morph-Open $k$=5", "morph", "open", 5),
        ("Morph-Close $k$=5", "morph", "close", 5),
        ("Morph-Both $k$=5", "morph", "both", 5),
    ]

    latex_rows = []
    all_metrics_data = []

    # Process each epsilon section
    for eps in [8.0, 200.0]:
        eps_metrics = []

        for strategy in STRATEGY_ORDER:
            for config_name, config_type, op, k in configs:
                for bs in [24, 48]:
                    if config_type == "baseline":
                        key = (eps, "baseline", strategy, bs)
                    else:
                        key = (eps, "morph", op, k, strategy, bs)

                    row_data = {
                        "eps": eps,
                        "strategy": strategy,
                        "config": config_name,
                        "bs": bs,
                    }

                    if key in results:
                        data = results[key]
                        row_data["dice"] = data["dice"]
                        row_data["sens"] = data["sens"]
                        row_data["spec"] = data["spec"]
                        row_data["s_meas"] = data["s_meas"]
                        row_data["e_meas"] = data["e_meas"]
                        row_data["mae"] = data["mae"]
                        row_data["trace"] = data.get("trace", {})
                    else:
                        row_data["dice"] = (None, None)
                        row_data["sens"] = (None, None)
                        row_data["spec"] = (None, None)
                        row_data["s_meas"] = (None, None)
                        row_data["e_meas"] = (None, None)
                        row_data["mae"] = (None, None)
                        row_data["trace"] = {}

                    eps_metrics.append(row_data)

        all_metrics_data.extend(eps_metrics)

        # Find best values per metric within this epsilon section
        best_vals = {}
        metric_map = {
            "Mean_Dice": "dice",
            "Mean_Sensitivity": "sens",
            "Mean_Specificity": "spec",
            "S_measure": "s_meas",
            "Mean_E_measure": "e_meas",
            "MAE": "mae",
        }
        for metric_key, metric_name, direction in METRICS:
            metric_accessor = metric_map[metric_key]
            if metric_key == "MAE":
                vals = [
                    m[metric_accessor][0]
                    for m in eps_metrics
                    if m[metric_accessor][0] is not None
                ]
                best_vals[metric_key] = min(vals) if vals else None
            else:
                vals = [
                    m[metric_accessor][0]
                    for m in eps_metrics
                    if m[metric_accessor][0] is not None
                ]
                best_vals[metric_key] = max(vals) if vals else None

        # Generate rows for this epsilon
        for i, m in enumerate(eps_metrics):
            if i == 0:
                # 4 configs × 4 strategies × 2 bs = 32 rows per epsilon
                eps_cell = f"\\multirow{{32}}{{*}}{{$\\varepsilon = {int(eps)}$}}"
            else:
                eps_cell = ""

            # Standard metrics columns
            standard_cols = []
            metric_map = {
                "Mean_Dice": "dice",
                "Mean_Sensitivity": "sens",
                "Mean_Specificity": "spec",
                "MAE": "mae",
            }
            for metric_key, metric_name, direction in STANDARD_METRICS:
                metric_accessor = metric_map[metric_key]
                mean, std = m[metric_accessor]
                is_best = mean == best_vals[metric_key] and mean is not None
                standard_cols.append(format_metric_bold(mean, std, is_best))

            # Structure-aware metrics columns
            structure_cols = []
            metric_map = {
                "S_measure": "s_meas",
                "Mean_E_measure": "e_meas",
            }
            for metric_key, metric_name, direction in STRUCTURE_AWARE_METRICS:
                metric_accessor = metric_map[metric_key]
                mean, std = m[metric_accessor]
                is_best = mean == best_vals[metric_key] and mean is not None
                structure_cols.append(format_metric_bold(mean, std, is_best))

            latex_rows.append(
                f"{eps_cell} & {m['config']} & {STRATEGY_DISPLAY[m['strategy']]} & {m['bs']} & {' & '.join(standard_cols)} & {' & '.join(structure_cols)} \\\\"
            )

        if eps != 200.0:  # Add midrule between epsilon sections
            latex_rows.append("\\midrule")

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{ARCH_DISPLAY} DP results on COVID-19 Lung Infection dataset comparing $\\varepsilon = 8$ (strong privacy) 
and $\\varepsilon = 200$ (weak privacy) using max grad norm = {chosen_max_grad_norm} (selected from ablation study, Table~\\ref{{tab:lung-dp-maxgrad-ablation-combined}}). 
$\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. 
Morphology applied to all classes (binary segmentation).
Best values per metric within each privacy level are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs.}}
\\label{{tab:{ARCH}-dp}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llll|cccc|cc}}
\\toprule
\\multirow{{2}}{{*}}{{$\\varepsilon$}} & \\multirow{{2}}{{*}}{{Configuration}} & \\multirow{{2}}{{*}}{{Clipping}} & \\multirow{{2}}{{*}}{{BS}} & 
\\multicolumn{{4}}{{c}}{{Standard Metrics}} & \\multicolumn{{2}}{{c}}{{Structure-aware Metrics}} \\\\
\\cmidrule(lr){{5-8}} \\cmidrule(lr){{9-10}}
& & & & Dice $\\uparrow$ & Sens $\\uparrow$ & Spec $\\uparrow$ & MAE $\\downarrow$ & 
S-meas $\\uparrow$ & E-meas $\\uparrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table, all_metrics_data


def generate_traceability_report(nonprivate_metrics, dp_metrics):
    """Generate traceability report showing CSV file paths and row numbers."""
    report_lines = []
    report_lines.append("\\section*{Traceability Report}")

    # Non-private results
    report_lines.append("\\subsection*{Non-Private Results}")
    for m in nonprivate_metrics:
        if m.get("trace"):
            config = m["config"]
            bs = m["bs"]
            trace = m["trace"]
            csv_file = trace.get("csv_file", "").replace("_", "\\_")
            row_idx = trace.get("row_index", "N/A")
            report_lines.append(f"\\paragraph{{{config}, BS={bs}}}")
            report_lines.append(f"CSV: {csv_file}, Row: {row_idx}")

    # DP results
    report_lines.append("\\subsection*{DP Results}")
    for m in dp_metrics:
        if m.get("trace"):
            config = m["config"]
            strategy = m["strategy"]
            bs = m["bs"]
            eps = m["eps"]
            trace = m["trace"]
            csv_file = trace.get("csv_file", "").replace("_", "\\_")
            row_idx = trace.get("row_index", "N/A")
            report_lines.append(
                f"\\paragraph{{ε={eps}, {config}, {STRATEGY_DISPLAY[strategy]}, BS={bs}}}"
            )
            report_lines.append(f"CSV: {csv_file}, Row: {row_idx}")

    return "\n".join(report_lines)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX results tables for Lung CT Segmentation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_tables.py --arch infnet
    python generate_tables.py --arch unet --output results_unet.tex
    python generate_tables.py --arch nestedunet --traceability
        """,
    )
    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        required=True,
        help="Architecture name (e.g., infnet, unet, nestedunet)",
    )
    parser.add_argument(
        "--arch-display",
        type=str,
        default=None,
        help="Display name for architecture (default: auto-detect from ARCH_DISPLAY_NAMES)",
    )
    parser.add_argument(
        "--results-path",
        "-r",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to aggregated results directory (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for LaTeX tables (default: print to stdout)",
    )
    parser.add_argument(
        "--traceability",
        "-t",
        action="store_true",
        help="Include traceability report",
    )
    return parser.parse_args()


def main():
    global RESULTS_PATH, ARCH, ARCH_DISPLAY

    args = parse_args()

    # Set global variables from args
    ARCH = args.arch.lower()
    RESULTS_PATH = args.results_path

    # Set display name
    if args.arch_display:
        ARCH_DISPLAY = args.arch_display
    elif ARCH in ARCH_DISPLAY_NAMES:
        ARCH_DISPLAY = ARCH_DISPLAY_NAMES[ARCH]
    else:
        ARCH_DISPLAY = ARCH.upper()

    # Output handling
    output_lines = []

    def log(msg=""):
        output_lines.append(msg)
        print(msg)

    log("=" * 80)
    log(f"Generating results tables for {ARCH_DISPLAY}")
    log(f"Results path: {RESULTS_PATH}")
    log("=" * 80)

    # Collect non-private results
    log("\nCollecting non-private results...")
    nonprivate_results = collect_nonprivate_results()
    log(f"Found {len(nonprivate_results)} non-private configurations")

    # Generate Table 1: Non-Private
    log("\n" + "=" * 80)
    log("TABLE 1: Non-Private Results")
    log("=" * 80)
    table1, nonprivate_metrics = generate_nonprivate_table(nonprivate_results)
    log(table1)

    # Select best max grad norm from global ablation (all three architectures) for thesis justification
    log("\n" + "=" * 80)
    log("SELECTING BEST MAX GRAD NORM (GLOBAL ABLATION)")
    log("=" * 80)
    chosen_max_grad_norm = select_global_best_max_grad_norm(RESULTS_PATH)
    log(
        f"Selected max grad norm: {chosen_max_grad_norm} (for thesis: report this and reference ablation table)."
    )

    # Collect DP results with chosen max grad norm
    log(f"\nCollecting DP results with max grad norm = {chosen_max_grad_norm}...")
    dp_results = collect_dp_results(chosen_max_grad_norm)
    log(f"Found {len(dp_results)} DP configurations")

    # Generate Table 3: DP Results
    log("\n" + "=" * 80)
    log("TABLE 3: DP Results (with chosen max grad norm)")
    log("=" * 80)
    table3, dp_metrics = generate_dp_table(dp_results, chosen_max_grad_norm)
    log(table3)

    # Generate traceability report if requested
    if args.traceability:
        log("\n" + "=" * 80)
        log("TRACEABILITY REPORT")
        log("=" * 80)
        trace_report = generate_traceability_report(nonprivate_metrics, dp_metrics)
        log(trace_report)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
