"""Step 2: multi-seed baseline comparison with ARQ round-trip delay and smooth block errors.

Run from the project root, e.g.:
    python scripts/run_step2.py --seeds 20 --workers 4
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rcdam.experiment import run_seed, METHODS
from rcdam.stats import aggregate, save_csv

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    jobs = [(s, a.hours) for s in range(a.first_seed, a.first_seed + a.seeds)]
    t0 = time.time()
    rows = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(run_seed, jobs):
                rows.extend(r)
    else:
        for j in jobs:
            rows.extend(run_seed(j))
            print(f"  seed {j[0]} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nFinished {a.seeds} seeds x {a.hours:.0f} h in {time.time() - t0:.0f}s\n")

    agg = aggregate(rows)
    save_csv(agg, os.path.join(OUT, "step2_results.csv"))
    any_row = agg[0]
    print(f"Messages per method: {any_row['n_messages']:,}; "
          f"unrecoverable by MODCOD adaptation (oracle also misses): "
          f"{100 * any_row['infeasible_frac']:.2f}% (+/- {100 * any_row['infeasible_frac_ci']:.2f})\n")

    print(f"{'method':32s} {'dB':>3} {'miss rate':>9} {'avoidable miss (95% CI)':>26} "
          f"{'avoid/feasible':>15} {'overhead':>9} {'goodput':>8} {'latency':>8}")
    for name, _, params in METHODS:
        for prm in params:
            r = next(x for x in agg if x["method"] == name and x["param_db"] == prm)
            print(f"{name:32s} {prm:3g} {r['miss_rate']:9.4f} "
                  f"{r['avoidable_miss_rate']:12.5f} +/- {r['avoidable_miss_rate_ci']:8.5f} "
                  f"{r['n_avoidable_miss']:6d}/{r['n_feasible']:<8d} {r['overhead']:9.3f} "
                  f"{r['goodput_mbps']:8.2f} {r['mean_latency_s']:8.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, (key, title) in enumerate([("miss_rate", "All messages"),
                                      ("avoidable_miss_rate", "Avoidable misses only (oracle could deliver)")]):
        for name, _, params in METHODS:
            rs = [next(x for x in agg if x["method"] == name and x["param_db"] == p) for p in params]
            ax[k].errorbar([r["overhead"] for r in rs], [r[key] for r in rs],
                           yerr=[r[key + "_ci"] for r in rs], marker="o", capsize=3, label=name)
            for r in rs:
                ax[k].annotate(f"{r['param_db']:g}", (r["overhead"], r[key]),
                               textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax[k].set_yscale("symlog", linthresh=1e-4)
        ax[k].set_xlabel("overhead (airtime per message / clear-sky no-redundancy airtime)")
        ax[k].set_ylabel("deadline-miss rate (95% CI over seeds)")
        ax[k].set_title(title); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    plt.suptitle(f"Step 2 baselines ({a.seeds} seeds x {a.hours:.0f} h); labels = margin in dB")
    plt.tight_layout()
    path = os.path.join(OUT, "step2_baselines.png")
    plt.savefig(path, dpi=130)
    print(f"\nSaved: {os.path.normpath(path)} and step2_results.csv")


if __name__ == "__main__":
    main()
