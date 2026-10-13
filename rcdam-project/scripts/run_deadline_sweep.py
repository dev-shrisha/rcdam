"""Step 3: Deadline sweep experiment.

Evaluates how the deadline (2s, 5s, 10s, 30s, 60s, 300s) impacts the fraction
of messages that are unrecoverable by MODCOD adaptation alone, and how baselines
perform as deadline expands.
"""
import argparse
import csv
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcdam.config import Scenario
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.experiment import step2_scenario, evaluate
from rcdam import policies

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def run_deadline_seed(seed: int, deadline_s: float, hours: float = 24.0):
    sc = replace(step2_scenario(), seed=seed, duration_h=hours, deadline_s=deadline_s)
    table = ModcodTable()
    att = generate_attenuation_db(sc)
    esn0 = sc.clear_sky_esn0_db - att
    n = len(esn0)
    win = max(1, int(round(sc.slope_window_s / sc.dt_s)))

    # Oracle bound:
    idx_oracle = policies.oracle_policy(table, esn0, 0.0)
    res_oracle, miss_oracle = evaluate(idx_oracle, esn0, sc, table)
    oracle_miss_rate = res_oracle["miss_rate"]

    # Fixed margin ACM (margin = 2 dB)
    idx_margin2 = policies.margin_policy(table, esn0, 2.0, sc.feedback_delay_steps)
    res_margin2, miss_margin2 = evaluate(idx_margin2, esn0, sc, table)

    # Predictive ACM (margin = 2 dB)
    idx_pred2 = policies.predictive_policy(table, esn0, 2.0, sc.feedback_delay_steps, win)
    res_pred2, miss_pred2 = evaluate(idx_pred2, esn0, sc, table)

    # Static MODCOD (margin = 10 dB)
    idx_stat10 = policies.static_policy(table, sc.clear_sky_esn0_db, 10.0, n)
    res_stat10, miss_stat10 = evaluate(idx_stat10, esn0, sc, table)

    feasible = ~miss_oracle
    avoidable_margin2 = float((miss_margin2 & feasible).sum() / max(feasible.sum(), 1))

    return {
        "seed": seed,
        "deadline_s": deadline_s,
        "oracle_miss_rate": oracle_miss_rate,
        "margin2_miss_rate": res_margin2["miss_rate"],
        "margin2_avoidable_rate": avoidable_margin2,
        "margin2_overhead": res_margin2["overhead"],
        "pred2_miss_rate": res_pred2["miss_rate"],
        "stat10_miss_rate": res_stat10["miss_rate"],
        "stat10_overhead": res_stat10["overhead"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    deadlines = [2.0, 5.0, 10.0, 30.0, 60.0, 300.0]

    print(f"Running deadline sweep across {deadlines} with {args.seeds} seeds x {args.hours:.0f} h...")
    t0 = time.time()
    all_results = []

    for d in deadlines:
        d_results = []
        for s in range(1, args.seeds + 1):
            r = run_deadline_seed(s, d, args.hours)
            d_results.append(r)
        all_results.extend(d_results)

        # Print summary for this deadline:
        mean_oracle = np.mean([r["oracle_miss_rate"] for r in d_results])
        mean_margin = np.mean([r["margin2_miss_rate"] for r in d_results])
        mean_avoid = np.mean([r["margin2_avoidable_rate"] for r in d_results])
        print(f"Deadline {d:5.1f} s -> Oracle unrecoverable: {100*mean_oracle:6.3f}%, "
              f"ACM 2dB total miss: {100*mean_margin:6.3f}%, avoidable: {100*mean_avoid:6.3f}%")

    # Save to CSV
    csv_path = os.path.join(OUT, "step3_deadline_sweep.csv")
    fields = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"Saved: {csv_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregate means across seeds
    d_means = []
    for d in deadlines:
        sub = [r for r in all_results if r["deadline_s"] == d]
        d_means.append({
            "deadline_s": d,
            "oracle_miss": np.mean([r["oracle_miss_rate"] for r in sub]),
            "margin_miss": np.mean([r["margin2_miss_rate"] for r in sub]),
            "pred_miss": np.mean([r["pred2_miss_rate"] for r in sub]),
            "stat_miss": np.mean([r["stat10_miss_rate"] for r in sub]),
            "margin_ovh": np.mean([r["margin2_overhead"] for r in sub]),
            "stat_ovh": np.mean([r["stat10_overhead"] for r in sub]),
        })

    ds = [m["deadline_s"] for m in d_means]
    ax1.plot(ds, [100 * m["oracle_miss"] for m in d_means], "k--o", label="Oracle bound (Unrecoverable)", linewidth=2)
    ax1.plot(ds, [100 * m["margin_miss"] for m in d_means], "b-s", label="Fixed-margin ACM (2 dB)")
    ax1.plot(ds, [100 * m["pred_miss"] for m in d_means], "g-^", label="Predictive ACM (2 dB)")
    ax1.plot(ds, [100 * m["stat_miss"] for m in d_means], "r-d", label="Static MODCOD (10 dB)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Deadline D (seconds)")
    ax1.set_ylabel("Deadline-miss rate (%)")
    ax1.set_title("Delivery Reliability vs Deadline")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ds, [m["margin_ovh"] for m in d_means], "b-s", label="Fixed-margin ACM (2 dB)")
    ax2.plot(ds, [m["stat_ovh"] for m in d_means], "r-d", label="Static MODCOD (10 dB)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Deadline D (seconds)")
    ax2.set_ylabel("Airtime Overhead O(a)")
    ax2.set_title("Overhead vs Deadline")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Step 3: Deadline Sweep (Impact of Horizon on Recoverability)")
    plt.tight_layout()
    plot_path = os.path.join(OUT, "step3_deadline_sweep.png")
    plt.savefig(plot_path, dpi=130)
    print(f"Saved: {plot_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
