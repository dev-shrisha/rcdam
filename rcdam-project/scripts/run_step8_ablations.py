"""Step 8: Comprehensive Ablation Study Runner (A0 - A6, B1).

Runs multi-seed evaluations of:
- A0: Baselines (ARQ-only static, Fixed-margin ACM, Predictive ACM)
- A1: Calibrated forecast + MODCOD only
- A2: A1 + Power step
- A3: A2 + Packet-level erasure coding
- A4: A3 + Interleaver depth chosen under deadline
- A5: A4 + Deferral
- A6: Full RC-DAM + Infeasibility signal
- B1: RC-DAM with compressed headers

Outputs:
- outputs/step8_ablations.csv
- outputs/step8_pareto_curves.png
- outputs/step8_infeasibility_pr.png
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

from rcdam.forecaster import QuantileForecaster
from rcdam.ablations import run_ablation_seed, ABLATION_SYSTEMS

OUT = os.path.join(PROJECT_ROOT, "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hours", type=float, default=12.0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print(f"=== Step 8: Comprehensive Ablations ({args.seeds} seeds x {args.hours:.0f} h) ===\n")

    # Load forecaster
    forecaster = QuantileForecaster(window_steps=30, horizon_steps=20)
    weights_path = os.path.join(OUT, "step5_forecaster_weights.pt")
    if os.path.exists(weights_path):
        forecaster.model.load_state_dict(torch.load(weights_path))

    t0 = time.time()
    all_rows = []
    for s in range(1, args.seeds + 1):
        s_t0 = time.time()
        print(f"Running seed {s}/{args.seeds}...", flush=True)
        rows = run_ablation_seed(seed=s, hours=args.hours, forecaster=forecaster)
        all_rows.extend(rows)
        print(f"  Seed {s} completed in {time.time() - s_t0:.1f}s", flush=True)

    print(f"\nCompleted {args.seeds} seeds in {time.time() - t0:.1f}s\n")

    # Aggregate stats across seeds
    grouped = defaultdict(list)
    for r in all_rows:
        grouped[(r["system"], r["system_id"])].append(r)

    metrics = ["miss_rate", "avoidable_miss_rate", "infeasible_frac", "overhead",
               "infeas_precision", "infeas_recall", "infeas_f1"]

    agg_results = []
    tcrit = stats.t.ppf(0.975, args.seeds - 1) if args.seeds > 1 else 1.96

    for (sys_name, sys_id), rs in grouped.items():
        k = len(rs)
        rec = {"system": sys_name, "system_id": sys_id, "n_seeds": k}
        for m in metrics:
            vals = np.array([r[m] for r in rs], dtype=float)
            rec[m + "_mean"] = float(np.mean(vals))
            rec[m + "_ci"] = float(tcrit * np.std(vals, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
        agg_results.append(rec)

    # Sort in order of ABLATION_SYSTEMS
    order = [sid for _, sid in ABLATION_SYSTEMS]
    agg_results.sort(key=lambda x: order.index(x["system_id"]))

    # Save to CSV
    csv_path = os.path.join(OUT, "step8_ablations.csv")
    fieldnames = ["system", "system_id", "n_seeds"] + \
                 [f"{m}_{suffix}" for m in metrics for suffix in ("mean", "ci")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(agg_results)
    print(f"Saved: {csv_path}\n")

    # Print summary table
    print(f"{'Ablation System':35s} {'Miss Rate (95% CI)':>22s} {'Avoidable Miss':>18s} "
          f"{'Overhead O(a)':>15s} {'Infeas F1':>10s}")
    print("-" * 105)
    for r in agg_results:
        print(f"{r['system']:35s} "
              f"{r['miss_rate_mean'] * 100:6.3f}% +/- {r['miss_rate_ci'] * 100:5.3f}% "
              f"{r['avoidable_miss_rate_mean'] * 100:6.3f}% +/- {r['avoidable_miss_rate_ci'] * 100:5.3f}% "
              f"{r['overhead_mean']:7.3f} +/- {r['overhead_ci']:5.3f} "
              f"{r['infeas_f1_mean']:8.3f}")

    # Plot 1: Pareto Curve (Overhead vs On-Time Delivery Reliability)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for r in agg_results:
        # Style baselines differently
        is_baseline = r["system_id"].startswith("A0")
        color = "gray" if is_baseline else None
        marker = "s" if is_baseline else "o"
        ax1.errorbar(
            r["overhead_mean"], r["miss_rate_mean"],
            xerr=r["overhead_ci"], yerr=r["miss_rate_ci"],
            fmt=marker, capsize=3, label=r["system"],
            markersize=8, alpha=0.9
        )
        ax1.annotate(r["system_id"], (r["overhead_mean"], r["miss_rate_mean"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_yscale("symlog", linthresh=1e-4)
    ax1.set_xlabel("Airtime Overhead O(a)")
    ax1.set_ylabel("Deadline-Miss Rate (95% CI)")
    ax1.set_title("Pareto Tradeoff: Overhead vs Deadline Miss Rate")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc="upper right")

    # Plot 2: Infeasibility Detection Precision and Recall
    rcdam_systems = [r for r in agg_results if not r["system_id"].startswith("A0")]
    names = [r["system_id"] for r in rcdam_systems]
    precisions = [r["infeas_precision_mean"] for r in rcdam_systems]
    recalls = [r["infeas_recall_mean"] for r in rcdam_systems]
    f1s = [r["infeas_f1_mean"] for r in rcdam_systems]

    x = np.arange(len(names))
    width = 0.25
    ax2.bar(x - width, precisions, width, label="Precision", color="royalblue")
    ax2.bar(x, recalls, width, label="Recall", color="forestgreen")
    ax2.bar(x + width, f1s, width, label="F1 Score", color="darkorange")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Score [0, 1]")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Infeasibility Detection Performance across Ablations")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    plt.suptitle(f"Step 8: Comprehensive Ablation Study ({args.seeds} seeds x {args.hours:.0f} h)")
    plt.tight_layout()
    plot_path = os.path.join(OUT, "step8_pareto_curves.png")
    plt.savefig(plot_path, dpi=130)
    print(f"Saved: {plot_path}")
    print("=== Step 8 Completed Successfully ===")


if __name__ == "__main__":
    main()
