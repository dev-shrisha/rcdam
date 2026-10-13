"""Aggregate per-seed results into mean and 95% confidence interval (t-interval across seeds)."""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

METRICS = ["miss_rate", "avoidable_miss_rate", "infeasible_frac", "overhead",
           "goodput_mbps", "mean_latency_s"]


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["param_db"])].append(r)
    out = []
    for (method, prm), rs in groups.items():
        k = len(rs)
        rec = {"method": method, "param_db": prm, "n_seeds": k,
               "n_messages": sum(r["n_messages"] for r in rs),
               "n_miss": sum(r["n_miss"] for r in rs),
               "n_avoidable_miss": sum(r["n_avoidable_miss"] for r in rs),
               "n_feasible": sum(r["n_feasible"] for r in rs)}
        tcrit = stats.t.ppf(0.975, k - 1) if k > 1 else float("nan")
        for m in METRICS:
            v = np.array([r[m] for r in rs], dtype=float)
            rec[m] = float(np.nanmean(v))
            rec[m + "_ci"] = float(tcrit * np.nanstd(v, ddof=1) / np.sqrt(k)) if k > 1 else float("nan")
        out.append(rec)
    return out


def save_csv(agg, path):
    cols = ["method", "param_db", "n_seeds", "n_messages", "n_miss", "n_avoidable_miss", "n_feasible"] + \
           [c for m in METRICS for c in (m, m + "_ci")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in agg:
            w.writerow({c: r[c] for c in cols})
