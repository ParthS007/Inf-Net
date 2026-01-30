#!/usr/bin/env python3
"""
Generate a single combined LaTeX ablation table for max grad norm across
Inf-Net, U-Net++, and U-Net (Flat clipping strategy only).

Usage:
    python generate_ablation_combined.py
    python generate_ablation_combined.py --output ablation_combined.tex
"""

import os
import argparse
from collections import defaultdict

# Reuse constants and helpers from generate_tables
from generate_tables import (
    DEFAULT_RESULTS_PATH,
    ARCH_DISPLAY_NAMES,
    MAX_GRAD_NORM_VALUES,
    collect_dp_ablation_results_for_arch,
    format_metric_bold,
)

# Only Dice and MAE for ablation combined
ABLATION_METRICS = [
    ("Mean_Dice", "Dice", "↑"),
    ("MAE", "MAE", "↓"),
]

# Model order: Inf-Net, U-Net++, U-Net
ARCH_ORDER = ["infnet", "nestedunet", "unet"]

# Only Flat clipping strategy for ablation combined
FLAT_STRATEGY = "base"


def collect_flat_ablation_for_arch(arch, results_path):
    """Collect DP baseline ablation for one arch, Flat strategy only."""
    raw = collect_dp_ablation_results_for_arch(arch, results_path)
    # Filter for Flat (base) only
    flat = {}
    for key, data in raw.items():
        eps, strategy, max_grad, bs = key
        if strategy == FLAT_STRATEGY:
            flat[key] = data
    return flat


def build_combined_rows(results_path):
    """
    Build list of row dicts for combined table.
    Each row: arch_display, eps, bs, then maxgrad_1.2_*, maxgrad_1.5_*, maxgrad_2.0_* for each metric.
    Order: Inf-Net (4 rows), U-Net++ (4 rows), U-Net (4 rows).
    """
    all_rows = []
    for arch in ARCH_ORDER:
        flat_results = collect_flat_ablation_for_arch(arch, results_path)
        arch_display = ARCH_DISPLAY_NAMES[arch]
        for eps in [8.0, 200.0]:
            for bs in [24, 48]:
                row = {
                    "arch": arch,
                    "arch_display": arch_display,
                    "eps": eps,
                    "bs": bs,
                }
                for max_grad in MAX_GRAD_NORM_VALUES:
                    key = (eps, FLAT_STRATEGY, max_grad, bs)
                    if key in flat_results:
                        data = flat_results[key]
                        row[f"maxgrad_{max_grad}_dice"] = data["dice"]
                        row[f"maxgrad_{max_grad}_mae"] = data["mae"]
                    else:
                        row[f"maxgrad_{max_grad}_dice"] = (None, None)
                        row[f"maxgrad_{max_grad}_mae"] = (None, None)
                all_rows.append(row)
    return all_rows


def best_per_metric_within_model(rows_per_arch):
    """
    For each arch, compute best value per metric across all its rows and all max_grad values.
    Returns dict: arch -> metric_key -> best_value (for bold).
    """
    metric_map = {
        "Mean_Dice": "dice",
        "MAE": "mae",
    }
    best_per_arch = {}
    for arch, rows in rows_per_arch.items():
        best_per_arch[arch] = {}
        for metric_key, metric_name, direction in ABLATION_METRICS:
            accessor = metric_map[metric_key]
            vals = []
            for r in rows:
                for max_grad in MAX_GRAD_NORM_VALUES:
                    t = r.get(f"maxgrad_{max_grad}_{accessor}", (None, None))
                    if t[0] is not None:
                        vals.append(t[0])
            if metric_key == "MAE":
                best_per_arch[arch][metric_key] = min(vals) if vals else None
            else:
                best_per_arch[arch][metric_key] = max(vals) if vals else None
    return best_per_arch


def generate_combined_table(results_path):
    """Generate the combined ablation LaTeX table (Flat only, all three models)."""
    all_rows = build_combined_rows(results_path)
    rows_per_arch = defaultdict(list)
    for r in all_rows:
        rows_per_arch[r["arch"]].append(r)
    best_per_arch = best_per_metric_within_model(rows_per_arch)
    metric_map = {
        "Mean_Dice": "dice",
        "MAE": "mae",
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
            cols = []
            for max_grad in MAX_GRAD_NORM_VALUES:
                for metric_key, metric_name, direction in ABLATION_METRICS:
                    accessor = metric_map[metric_key]
                    mean, std = r.get(f"maxgrad_{max_grad}_{accessor}", (None, None))
                    is_best = (
                        mean is not None
                        and best_vals.get(metric_key) is not None
                        and mean == best_vals[metric_key]
                    )
                    cols.append(format_metric_bold(mean, std, is_best))
            latex_rows.append(
                f"{model_cell} & {eps_cell} & {bs_cell} & {' & '.join(cols)} \\\\"
            )
        if arch != ARCH_ORDER[-1]:
            latex_rows.append("\\midrule")
    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{DP max gradient norm ablation (Flat clipping only) for Inf-Net, U-Net++, and U-Net on COVID-19 Lung Infection dataset. 
Max grad norm values 1.2, 1.5, and 2.0 under $\\varepsilon = 8$ (strong privacy) and $\\varepsilon = 200$ (weak privacy). 
$\\uparrow$ indicates higher is better, $\\downarrow$ indicates lower is better. 
Best values per metric within each model are in \\textbf{{bold}}. 
Results show mean $\\pm$ std over two independent runs (or single run with std=0.0). 
Entries marked ``---'' indicate that configuration was not run or not present in the aggregated data. 
The max grad norm for the main DP results (Tables~\\ref{{tab:infnet-dp}}, \\ref{{tab:nestedunet-dp}}, \\ref{{tab:unet-dp}}) was selected by majority vote across Dice and MAE metrics over all three architectures.}}
\\label{{tab:lung-dp-maxgrad-ablation-combined}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lll|cc|cc|cc}}
\\toprule
Model & $\\varepsilon$ & Batch Size & 
\\multicolumn{{2}}{{c}}{{Max Grad Norm = 1.2}} & 
\\multicolumn{{2}}{{c}}{{Max Grad Norm = 1.5}} & 
\\multicolumn{{2}}{{c}}{{Max Grad Norm = 2.0}} \\\\
\\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}} \\cmidrule(lr){{8-9}}
& & & Dice $\\uparrow$ & MAE $\\downarrow$ & 
Dice $\\uparrow$ & MAE $\\downarrow$ & 
Dice $\\uparrow$ & MAE $\\downarrow$ \\\\
\\midrule
{chr(10).join(latex_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""
    return table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate combined max grad norm ablation table (Flat only) for Inf-Net, U-Net++, U-Net.",
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
        default="ablation_combined.tex",
        help="Output LaTeX file (default: ablation_combined.tex)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = args.results_path
    table = generate_combined_table(results_path)
    output_lines = [
        "=" * 80,
        "Combined DP max grad norm ablation (Flat only) for Inf-Net, U-Net++, and U-Net",
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
