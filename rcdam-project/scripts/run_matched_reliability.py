"""Task 4 / Phase 2: Matched-Reliability Comparison.

Sweeps safety settings for each method:
- V0_Static: static_modcod in [0, 2, 4, 6]
- V1_Margin: margin_db in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
- V2_Predictive: margin_db in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
- V3_Raw: quantile prediction without conformal calibration
- V4_Conformal_MODCOD: safety offset lambda_db in [0.0, 1.0, 2.0, 3.0]
- V5_All_Levers: safety offset lambda_db in [0.0, 1.0, 2.0, 3.0]
- V6_RC_DAM_Full: safety offset lambda_db in [0.0, 1.0, 2.0, 3.0]

Reports overhead at fixed target miss rates (eps = 0.1, 0.01, and 0.001 if n >= 3000)
using Wilson upper bound via rcdam.stats_utils.overhead_at_target.
Saves outputs/matched_reliability.csv and outputs/matched_reliability.png.
"""
import argparse
import csv
import os
import sys
from dataclasses import replace
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcdam.config import Scenario, get_preset
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.experiment import step2_scenario
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.calibration import load_calibration_json, fade_state
from rcdam.optimizer import Action, Message
from rcdam.groundtruth import simulate_true_delivery
from rcdam.stats_utils import wilson_interval, overhead_at_target
from rcdam.variants import (
    PolicyContext,
    policy_v0_static,
    policy_v1_fixed_margin,
    policy_v2_linear_predictive,
    policy_v3_quantile_no_conformal,
    policy_v4_conformal_modcod,
    policy_v5_stage_a_levers,
    policy_v6_full_rcdam,
    policy_oracle,
)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def run_seed_eval(
    seed: int,
    sc: Scenario,
    table: ModcodTable,
    forecaster: Any,
    cal_corr: dict,
    policy_name: str,
    param: float,
    msg_interval_s: float = 15.0,
) -> Dict[str, Any]:
    sc = replace(sc, seed=seed)
    rng = np.random.default_rng(seed)
    att = generate_attenuation_db(sc)
    n_steps = len(att)
    dt = sc.dt_s
    delay_steps = sc.feedback_delay_steps
    hist_steps = int(round(60.0 / dt))

    arrival_indices = []
    cur_t = 60.0
    end_t = sc.duration_h * 3600.0 - 5.0 * sc.deadline_s
    while cur_t < end_t:
        jitter = rng.uniform(-2.0, 2.0)
        cur_t += msg_interval_s + jitter
        if cur_t < end_t:
            arrival_indices.append(int(round(cur_t / dt)))

    priorities = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]

    # If policy uses safety offset lambda, adjust cal_corr
    adjusted_corr = {}
    if policy_name in ("V4", "V5", "V6"):
        for h, t_dict in cal_corr.items():
            adjusted_corr[h] = {}
            for tau, st_dict in t_dict.items():
                adjusted_corr[h][tau] = {st: v + param for st, v in st_dict.items()}
    else:
        adjusted_corr = cal_corr

    delivered_list = []
    overhead_list = []

    for i, arr_idx in enumerate(arrival_indices):
        p_name, eps = priorities[i % len(priorities)]
        msg = Message(
            size_bits=sc.msg_bits,
            priority=p_name,
            deadline_s=sc.deadline_s,
            eps=eps,
            arrival_idx=arr_idx,
        )

        last_m = max(hist_steps, arr_idx - delay_steps)
        cur_att = float(att[last_m])
        s5 = int(round(5.0 / dt))
        s30 = int(round(30.0 / dt))
        feats = {
            "cur": np.array([cur_att]),
            "slope5": np.array([(cur_att - att[max(0, last_m - s5)]) / 5.0]),
            "slope30": np.array([(cur_att - att[max(0, last_m - s30)]) / 30.0]),
            "max60": np.array([np.max(att[max(0, last_m - hist_steps): last_m + 1])]),
        }
        H_window = int(round((sc.deadline_s + sc.one_way_delay_s) / dt)) + 5
        true_window = att[arr_idx: min(n_steps, arr_idx + H_window)]

        ctx = PolicyContext(
            arrival_idx=arr_idx,
            last_meas_idx=last_m,
            cur_att_db=cur_att,
            features=feats,
            true_future_att=true_window,
            msg=msg,
            sc=sc,
            table=table,
            forecaster=forecaster,
            calibration_corr=adjusted_corr,
        )

        if policy_name == "V0_Static":
            dec = policy_v0_static(ctx, static_modcod=int(param))
        elif policy_name == "V1_Margin":
            dec = policy_v1_fixed_margin(ctx, margin_db=param)
        elif policy_name == "V2_Predictive":
            dec = policy_v2_linear_predictive(ctx, margin_db=param)
        elif policy_name == "V3_Raw":
            dec = policy_v3_quantile_no_conformal(ctx)
        elif policy_name == "V4":
            dec = policy_v4_conformal_modcod(ctx)
        elif policy_name == "V5":
            dec = policy_v5_stage_a_levers(ctx)
        elif policy_name == "V6":
            dec = policy_v6_full_rcdam(ctx)
        else:
            raise ValueError(f"Unknown policy {policy_name}")

        sim_rng = np.random.default_rng(seed * 10000 + i)
        res = simulate_true_delivery(dec.action, ctx.arrival_idx, att, msg, table, sc, sim_rng)
        delivered_list.append(res.delivered)
        overhead_list.append(res.overhead)

    n_tot = len(delivered_list)
    n_miss = int(np.sum(~np.array(delivered_list)))
    return {
        "n_total": n_tot,
        "n_miss": n_miss,
        "mean_overhead": float(np.mean(overhead_list)) if overhead_list else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Matched-Reliability Comparison")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=101)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sc0 = step2_scenario()
    sc0 = replace(sc0, duration_h=args.hours)
    table = ModcodTable()

    cal_json = os.path.join(OUT, "calibration.json")
    fc_json = os.path.join(OUT, "forecaster_baseline.json")

    if os.path.exists(cal_json):
        _, cal_corr = load_calibration_json(cal_json)
    else:
        cal_corr = {}

    if os.path.exists(fc_json):
        forecaster = EmpiricalQuantileForecaster.load(fc_json)
    else:
        train_cur = np.linspace(0.0, 30.0, 500)
        train_Z = np.column_stack([np.linspace(0.1, 35.0, 500) for _ in range(8)])
        forecaster = EmpiricalQuantileForecaster().fit(train_cur, train_Z)

    policies = {
        "V0_Static": [2, 4, 6],
        "V1_Margin": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "V2_Predictive": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "V4": [0.0, 1.0, 2.0, 3.0],
        "V5": [0.0, 1.0, 2.0, 3.0],
        "V6": [0.0, 1.0, 2.0, 3.0],
    }

    results = []
    print("==================================================")
    print("      MATCHED-RELIABILITY COMPARISON SWEEPS       ")
    print(f" Seeds: {args.seeds} ({args.first_seed}..) | Hours: {args.hours} | Interval: {args.interval}s")
    print("==================================================")

    for pol, params in policies.items():
        for p in params:
            tot_n = 0
            tot_m = 0
            ovs = []
            for s in range(args.first_seed, args.first_seed + args.seeds):
                res = run_seed_eval(s, sc0, table, forecaster, cal_corr, pol, p, msg_interval_s=args.interval)
                tot_n += res["n_total"]
                tot_m += res["n_miss"]
                ovs.append(res["mean_overhead"])
            m_rate = tot_m / max(1, tot_n)
            w_lo, w_hi = wilson_interval(tot_m, tot_n)
            mean_ov = float(np.mean(ovs))
            results.append({
                "policy": pol,
                "param": p,
                "n_total": tot_n,
                "n_miss": tot_m,
                "miss_rate": m_rate,
                "wilson_lo": w_lo,
                "wilson_hi": w_hi,
                "overhead": mean_ov,
            })
            print(f"{pol:15s} param={p:4.1f} | miss={tot_m:3d}/{tot_n:<5d} ({m_rate*100:6.3f}%) | Wilson [{w_lo*100:.3f}%, {w_hi*100:.3f}%] | Overhead={mean_ov:.3f}")

    # Compute overhead at target for 0.1, 0.01 (and 0.001 if n >= 3000)
    total_msgs_sample = max(r["n_total"] for r in results) if results else 0
    targets = [0.10, 0.01]
    if total_msgs_sample >= 3000:
        targets.append(0.001)

    summary = []
    print("\n--- Matched-Reliability Summary (Cheapest Certified Config) ---")
    for t in targets:
        print(f"\nTarget Miss Rate: {t*100:.3f}% (Wilson Upper Bound <= {t*100:.3f}%)")
        for pol in policies:
            subset = [r for r in results if r["policy"] == pol]
            k_list = [r["n_miss"] for r in subset]
            n_list = [r["n_total"] for r in subset]
            ov_list = [r["overhead"] for r in subset]
            best_idx = overhead_at_target(k_list, n_list, ov_list, t)
            if best_idx is not None:
                b = subset[best_idx]
                print(f"  {pol:15s}: param={b['param']} | Overhead={b['overhead']:.3f} | Miss={b['n_miss']}/{b['n_total']} ({b['miss_rate']*100:.3f}%) | WilsonHi={b['wilson_hi']*100:.3f}%")
                summary.append({
                    "target": t,
                    "policy": pol,
                    "param": b["param"],
                    "overhead": b["overhead"],
                    "n_miss": b["n_miss"],
                    "n_total": b["n_total"],
                    "miss_rate": b["miss_rate"],
                    "wilson_hi": b["wilson_hi"],
                })
            else:
                print(f"  {pol:15s}: No setting certified below {t*100:.3f}% (95% Wilson)")

    csv_path = os.path.join(OUT, "matched_reliability.csv")
    if summary:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        print(f"\nSaved matched-reliability table to {csv_path}")

    # Generate Matched-Reliability Tradeoff Figure
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        markers = {"V0_Static": "x", "V1_Margin": "s", "V2_Predictive": "^", "V4": "d", "V5": "v", "V6": "o"}
        colors = {"V0_Static": "gray", "V1_Margin": "blue", "V2_Predictive": "orange", "V4": "green", "V5": "purple", "V6": "red"}

        for pol in policies:
            sub = [r for r in results if r["policy"] == pol]
            if not sub:
                continue
            x_vals = [r["wilson_hi"] * 100 for r in sub]
            y_vals = [r["overhead"] for r in sub]
            order = np.argsort(x_vals)
            ax.plot(
                np.array(x_vals)[order], np.array(y_vals)[order],
                marker=markers.get(pol, "o"),
                color=colors.get(pol, "black"),
                label=pol,
                linewidth=1.8,
                markersize=6,
            )

        ax.axvline(1.0, color="k", linestyle="--", alpha=0.5, label="Target 1.0%")
        ax.axvline(10.0, color="gray", linestyle=":", alpha=0.5, label="Target 10.0%")
        ax.set_xlabel("95% Wilson Upper Bound Miss Rate (%)")
        ax.set_ylabel("Airtime Overhead")
        ax.set_title("Matched-Reliability Frontier: Overhead vs Certified Miss Rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig_path = os.path.join(OUT, "matched_reliability.png")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved matched-reliability figure to {fig_path}")
    except Exception as e:
        print(f"Warning: Could not save figure: {e}")


if __name__ == "__main__":
    main()
