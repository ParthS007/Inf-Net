#!/usr/bin/env python3
"""
Generate a combined LaTeX ablation table for kernel size (k=3 vs k=5) across
Inf-Net, U-Net++, and U-Net (Flat clipping strategy, chosen max grad norm).

Usage:
    python generate_kernel_ablation.py
    python generate_kernel_ablation.py --output kernel_ablation.tex
"""

import os
import argparse
from collections import defaultdict

# Reuse constants and helpers from generate_tables
from generate_tables import (
    DEFAULT_RESULTS_PATH,
    ARCH_DISPLAY_NAMES,
    MODEL_NAME_MAP,
    STANDARD_METRICS,
    STRUCTURE_AWARE_METRICS,
    load_aggregated_csv,
    format_metric_bold,
    select_global_best_max_grad_norm,
)

# Model order: Inf-Net, U-Net++, U-Net
ARCH_ORDER = ["infnet", "nestedunet", "unet"]

# Only Flat clipping strategy for kernel ablation
FLAT_STRATEGY = "base"

# Kernel sizes to compare
KERNEL_SIZES = [3, 5]

# Morphology operations
MORPH_OPERATIONS = ["open", "close", "both"]


def collect_kernel_ablation_for_arch(arch, results_path, chosen_max_grad_norm):
    """Collect DP Morph results for kernel size ablation (Flat strategy, chosen max grad norm)."""
    results = defaultdict(dict)

    model_name = MODEL_NAME_MAP[arch]
    dp_morph_file = f"{model_name}_DP_Morph_aggregated_mean_std.csv"
    dp_morph_df = load_aggregated_csv(dp_morph_file, path=results_path)

    if dp_morph_df is not None:
        morph_df = dp_morph_df[
            (dp_morph_df["Morphology"] == True)
            & (dp_morph_df["Operation"].isin(MORPH_OPERATIONS))
            & (dp_morph_df["Kernel_Size"].isin(KERNEL_SIZES))
            & (dp_morph_df["Max_Grad_Norm"] == chosen_max_grad_norm)
            & (dp_morph_df["Clipping_Strategy"] == FLAT_STRATEGY)
        ]

        for idx, row in morph_df.iterrows():
            eps = float(row["Epsilon"])
            op = row["Operation"]
            k = int(row["Kernel_Size"])
            bs = int(row["Batch_Size"])

            key = (eps, op, k, bs)
            results[key] = {
                "dice": (row["Mean_Dice_mean"], row["Mean_Dice_std"]),
                "sens": (row["Mean_Sensitivity_mean"], row["Mean_Sensitivity_std"]),
                "spec": (row["Mean_Specificity_mean"], row["Mean_Specificity_std"]),
                "s_meas": (row["S_measure_mean"], row["S_measure_std"]),
                "e_meas": (row["Mean_E_measure_mean"], row["Mean_E_measure_std"]),
                "mae": (row["MAE_mean"], row["MAE_std"]),
            }

    return results


def build_kernel_ablation_rows(results_path, chosen_max_grad_norm):
    """
    Build list of row dicts for kernel size ablation table.
    Each row: arch_display, eps, bs, op, then k3_* and k5_* for each metric.
    Order: Inf-Net (12 rows), U-Net++ (12 rows), U-Net (12 rows).
    """
    all_rows = []
    for arch in ARCH_ORDER:
        kernel_results = collect_kernel_ablation_for_arch(
            arch, results_path, chosen_max_grad_norm
        )
        arch_display = ARCH_DISPLAY_NAMES[arch]

        for eps in [8.0, 200.0]:
            for bs in [24, 48]:
                for op in MORPH_OPERATIONS:
                    row = {
                        "arch": arch,
                        "arch_display": arch_display,
                        "eps": eps,
                        "bs": bs,
                        "op": op,
                    }

                    for k in KERNEL_SIZES:
                        key = (eps, op, k, bs)
                        if key in kernel_results:
                            data = kernel_results[key]
                            row[f"k{k}_dice"] = data["dice"]
                            row[f"k{k}_sens"] = data["sens"]
                            row[f"k{k}_spec"] = data["spec"]
                            row[f"k{k}_s_meas"] = data["s_meas"]
                            row[f"k{k}_e_meas"] = data["e_meas"]
                            row[f"k{k}_mae"] = data["mae"]
                        else:
                            row[f"k{k}_dice"] = (None, None)
                            row[f"k{k}_sens"] = (None, None)
                            row[f"k{k}_spec"] = (None, None)
                            row[f"k{k}_s_meas"] = (None, None)
                            row[f"k{k}_e_meas"] = (None, None)
                            row[f"k{k}_mae"] = (None, None)

                    all_rows.append(row)

    return all_rows


def best_per_metric_within_model(rows_per_arch):
    """
    For each arch, compute best value per metric across all its rows and all kernel sizes.
    Returns dict: arch -> metric_key -> best_value (for bold).
    """
    metric_map = {
        "Mean_Dice": "dice",
        "Mean_Sensitivity": "sens",
        "Mean_Specificity": "spec",
        "S_measure": "s_meas",
        "Mean_E_measure": "e_meas",
        "MAE": "mae",
    }
    best_per_arch = {}

    all_metrics = STANDARD_METRICS + STRUCTURE_AWARE_METRICS

    for arch, rows in rows_per_arch.items():
        best_per_arch[arch] = {}
        for metric_key, metric_name, direction in all_metrics:
            accessor = metric_map[metric_key]
            vals = []
            for r in rows:
                for k in KERNEL_SIZES:
                    t = r.get(f"k{k}_{accessor}", (None, None))
                    if t[0] is not None:
                        vals.append(t[0])
            if metric_key == "MAE":
                best_per_arch[arch][metric_key] = min(vals) if vals else None
            else:
                best_per_arch[arch][metric_key] = max(vals) if vals else None
    return best_per_arch


def generate_kernel_ablation_table(results_path, chosen_max_grad_norm):
    """Generate the kernel size ablation LaTeX table (Flat only, all three models)."""
    all_rows = build_kernel_ablation_rows(results_path, chosen_max_grad_norm)
    rows_per_arch = defaultdict(list)
    for r in all_rows:
        rows_per_arch[r["arch"]].append(r)

    best_per_arch = best_per_metric_within_model(rows_per_arch)

    metric_map = {
        "Mean_Dice": "dice",
        "Mean_Sensitivity": "sens",
        "Mean_Specificity": "spec",
        "MAE": "mae",
        "S_measure": "s_meas",
        "Mean_E_measure": "e_meas",
    }

    op_display = {
        "open": "Open",
        "close": "Close",
        "both": "Both",
    }

    latex_rows = []
    for arch in ARCH_ORDER:
        rows = rows_per_arch.get(arch, [])
        best_vals = best_per_arch.get(arch, {})
        n_rows = len(rows)

        for i, r in enumerate(rows):
            if i == 0:
                model_cell = f"\\multirow{{{n_rows}}}{{*}}{{{r['arch_display']}}}"
            else:
                model_cell = ""

            eps_cell = f"$\\varepsilon = {int(r['eps'])}$"
            bs_cell = str(r["bs"])
            op_cell = op_display[r["op"]]

            # Standard metrics columns (k=3 then k=5)
            standard_cols = []
            for metric_key, metric_name, direction in STANDARD_METRICS:
                accessor = metric_map[metric_key]
                for k in KERNEL_SIZES:
                    mean, std = r.get(f"k{k}_{accessor}", (None, None))
                    is_best = (
                        mean is not None
                        and best_vals.get(metric_key) is not None
                        and mean == best_vals[metric_key]
                    )
                    standard_cols.append(format_metric_bold(mean, std, is_best))

            # Structure-aware metrics columns (k=3 then k=5)
            structure_cols = []
            for metric_key, metric_name, direction in STRUCTURE_AWARE_METRICS:
                accessor = metric_map[metric_key]
                for k in KERNEL_SIZES:
                    mean, std = r.get(f"k{k}_{accessor}", (None, None))
                    is_best = (
                        mean is not None
                        and best_vals.get(metric_key) is not None
                        and mean == best_vals[metric_key]
                    )
                    structure_cols.append(format_metric_bold(mean, std, is_best))

            latex_rows.append(
                f"{model_cell} & {eps_cell} & {bs_cell} & {op_cell} & {' & '.join(standard_cols)} & {' & '.join(structure_cols)} \\\\"
            )

        if arch != ARCH_ORDER[-1]:
            latex_rows.append("\\midrule")

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Kernel size ablation (Flat clipping, max grad norm = {chosen_max_grad_norm}) for Inf-Net, U-Net++, and U-Net on COVID-19 Lung Infection dataset. 
Comparing kernel sizes $k = 3$ and $k = 5$ for morphological operations (Open, Close, Both) under $\\varepsilon = 8$ (strong privacy) and $\\varepsilon = 200$ (weak privacy). 
$\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. 
Best values per metric within each model are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs (or single run with std=0.0). 
Entries marked ``---'' indicate that configuration was not run or not present in the aggregated data. 
Kernel size $k = 5$ was selected for the main DP results (Tables~\\ref{{tab:infnet-dp}}, \\ref{{tab:nestedunet-dp}}, \\ref{{tab:unet-dp}}) based on performance across standard and structure-aware metrics.}}
\\label{{tab:lung-kernel-ablation-combined}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llll|cccc|cccc|cc|cc}}
\\toprule
\\multirow{{2}}{{*}}{{Model}} & \\multirow{{2}}{{*}}{{$\\varepsilon$}} & \\multirow{{2}}{{*}}{{BS}} & \\multirow{{2}}{{*}}{{Operation}} & 
\\multicolumn{{8}}{{c}}{{Standard Metrics}} & \\multicolumn{{4}}{{c}}{{Structure-aware Metrics}} \\\\
\\cmidrule(lr){{5-12}} \\cmidrule(lr){{13-16}}
& & & & \\multicolumn{{4}}{{c}}{{Kernel Size $k = 3$}} & \\multicolumn{{4}}{{c}}{{Kernel Size $k = 5$}} & 
\\multicolumn{{2}}{{c}}{{Kernel Size $k = 3$}} & \\multicolumn{{2}}{{c}}{{Kernel Size $k = 5$}} \\\\
\\cmidrule(lr){{5-8}} \\cmidrule(lr){{9-12}} \\cmidrule(lr){{13-14}} \\cmidrule(lr){{15-16}}
& & & & Dice $\\uparrow$ & Sens $\\uparrow$ & Spec $\\uparrow$ & MAE $\\downarrow$ & 
Dice $\\uparrow$ & Sens $\\uparrow$ & Spec $\\uparrow$ & MAE $\\downarrow$ & 
S-meas $\\uparrow$ & E-meas $\\uparrow$ & 
S-meas $\\uparrow$ & E-meas $\\uparrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    return table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate combined kernel size ablation table (Flat only) for Inf-Net, U-Net++, U-Net.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        default="kernel_ablation.tex",
        help="Output LaTeX file (default: kernel_ablation.tex)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = args.results_path

    # Select best max grad norm (same as in generate_tables.py)
    chosen_max_grad_norm = select_global_best_max_grad_norm(results_path)
    print(
        f"Using max grad norm = {chosen_max_grad_norm} (selected from global ablation)"
    )

    table = generate_kernel_ablation_table(results_path, chosen_max_grad_norm)
    output_lines = [
        "=" * 80,
        "Combined kernel size ablation (Flat only, max grad norm = {}) for Inf-Net, U-Net++, and U-Net".format(
            chosen_max_grad_norm
        ),
        f"Results path: {results_path}",
        "=" * 80,
        "",
        table,
    ]
    with open(args.output, "w") as f:
        f.write("\n".join(output_lines))
    print("\n".join(output_lines))
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
