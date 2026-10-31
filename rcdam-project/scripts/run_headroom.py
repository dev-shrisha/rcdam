"""M0 Gate: Headroom analysis across deadlines.

Evaluates unrecoverable and avoidable deadline misses for baselines vs oracle
across deadlines (2s, 10s, 60s, 300s).
Saves outputs/headroom.csv and outputs/headroom.png.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcdam.experiment import run_seed, METHODS
from rcdam.stats import aggregate, save_csv

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def main():
    parser = argparse.ArgumentParser(description="M0 Headroom Analysis across deadlines")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument("--first-seed", type=int, default=1, help="First seed index")
    parser.add_argument("--hours", type=float, default=24.0, help="Duration in hours per seed")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--deadlines", nargs="+", type=float, default=[2.0, 10.0, 60.0, 300.0],
                        help="List of deadlines in seconds")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("==================================================")
    print("   MILESTONE M0: HEADROOM ANALYSIS ACROSS DEADLINES")
    print("==================================================")
    print(f"Config: {args.seeds} seeds (from {args.first_seed}), {args.hours}h each, workers={args.workers}")
    print(f"Deadlines to evaluate: {args.deadlines} seconds\n")

    all_headroom_rows = []
    t_start = time.time()

    for deadline in args.deadlines:
        print(f"--> Running deadline = {deadline}s ...")
        jobs = [(s, args.hours, deadline) for s in range(args.first_seed, args.first_seed + args.seeds)]
        rows = []
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for res in executor.map(run_seed, jobs):
                    rows.extend(res)
        else:
            for job in jobs:
                rows.extend(run_seed(job))

        agg = aggregate(rows)
        # Find best baseline avoidable miss rate (excluding oracle)
        baseline_rows = [r for r in agg if "Oracle" not in r["method"]]
        best_baseline = min(baseline_rows, key=lambda x: x["avoidable_miss_rate"]) if baseline_rows else None

        for r in agg:
            r["deadline_s"] = deadline
            all_headroom_rows.append(r)

        oracle_r = next(r for r in agg if "Oracle" in r["method"])
        print(f"    Infeasible fraction (oracle miss): {oracle_r['miss_rate']*100:.3f}% (+/- {oracle_r['miss_rate_ci']*100:.3f}%)")
        if best_baseline:
            print(f"    Best baseline: {best_baseline['method']} ({best_baseline['param_db']}dB) -> avoidable miss = "
                  f"{best_baseline['avoidable_miss_rate']*100:.3f}% (+/- {best_baseline['avoidable_miss_rate_ci']*100:.3f}%)")
            headroom = best_baseline["avoidable_miss_rate"]
            print(f"    Demonstrated headroom for deadline {deadline}s: {headroom*100:.3f}%\n")

    # Save CSV
    csv_path = os.path.join(OUT, "headroom.csv")
    save_csv(all_headroom_rows, csv_path)
    print(f"Saved headroom metrics to: {csv_path}")

    # Plot headroom curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    deadlines = sorted(list(set(args.deadlines)))
    # Pick representative config for each method
    methods_to_plot = [
        ("Oracle ACM (zero delay, bound)", 0, "black", "-", "Oracle (unrecoverable)"),
        ("Fixed-margin ACM + ARQ", 3, "blue", "--", "Fixed-margin (3dB)"),
        ("Predictive ACM (linear) + ARQ", 2, "green", "-.", "Predictive ACM (2dB)"),
        ("ARQ-only (static MODCOD)", 8, "red", ":", "Static MODCOD (8dB)"),
    ]

    for name, prm, color, ls, label in methods_to_plot:
        pts = [next((r for r in all_headroom_rows if r["deadline_s"] == d and r["method"] == name and r["param_db"] == prm), None)
               for d in deadlines]
        pts = [p for p in pts if p is not None]
        if pts:
            x = [p["deadline_s"] for p in pts]
            y_total = [p["miss_rate"] * 100 for p in pts]
            y_avoid = [p["avoidable_miss_rate"] * 100 for p in pts]
            y_avoid_err = [p["avoidable_miss_rate_ci"] * 100 for p in pts]

            ax1.plot(x, y_total, color=color, linestyle=ls, marker="o", label=label)
            if "Oracle" not in name:
                ax2.errorbar(x, y_avoid, yerr=y_avoid_err, color=color, linestyle=ls, marker="s", capsize=4, label=label)

    ax1.set_xscale("log")
    ax1.set_xlabel("Deadline (seconds)")
    ax1.set_ylabel("Total Deadline Miss Rate (%)")
    ax1.set_title("Total Miss Rate vs Deadline")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("Deadline (seconds)")
    ax2.set_ylabel("Avoidable Miss Rate (%) (95% CI)")
    ax2.set_title("Avoidable Miss Rate (Headroom) vs Deadline")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    png_path = os.path.join(OUT, "headroom.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved headroom plot to: {png_path}")
    print(f"Elapsed time: {time.time() - t_start:.1f}s")
    print("==================================================\n")


if __name__ == "__main__":
    main()
