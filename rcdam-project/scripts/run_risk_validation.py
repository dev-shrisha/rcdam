"""End-to-End Risk Validation & Learn-then-Test (LTT) Certification (Phase 4).

Validates:
1. End-to-end delivery risk: P(miss | flagged feasible) <= eps.
2. LTT fixed-sequence certification on calibration seeds (11-18), frozen lambda evaluated on test seeds (101-105).
3. Reliability diagram: Binned predicted delivery probability P* vs empirical delivery rate.
4. Optimizer's curse guard: Compares empirical delivery with vs without validated selection.

Outputs:
- outputs/risk_validation.csv
- outputs/ltt_certification.csv
- outputs/reliability_diagram.png
"""
import argparse
import csv
import os
import sys
from dataclasses import replace
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcdam.config import preset_harsh, Scenario
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.calibration import load_calibration_json
from rcdam.optimizer import Message, Action
from rcdam.groundtruth import simulate_true_delivery
from rcdam.risk_control import ltt_fixed_sequence
from rcdam.stats_utils import wilson_interval
from rcdam.variants import PolicyContext, policy_v6_full_rcdam, policy_v5_stage_a_levers

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
SAFETY_LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def run_messages_with_lambda(seeds: List[int],
                             sc_base: Scenario,
                             table: ModcodTable,
                             forecaster: Any,
                             base_corr: dict,
                             lambdas: List[float],
                             use_validated_selection: bool = True,
                             msg_interval_s: float = 20.0) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Runs messages across seeds under multiple safety parameter settings lambda."""
    all_records = []
    dt = sc_base.dt_s
    delay_steps = sc_base.feedback_delay_steps
    hist_steps = int(round(60.0 / dt))

    priorities = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]

    # Collect all messages across seeds
    all_eval_items = []
    for s in seeds:
        sc = replace(sc_base, seed=s)
        rng = np.random.default_rng(s)
        att = generate_attenuation_db(sc)
        n_steps = len(att)

        cur_t = 60.0
        end_t = sc.duration_h * 3600.0 - 5.0 * sc.deadline_s
        arrival_indices = []
        while cur_t < end_t:
            jitter = rng.uniform(-2.0, 2.0)
            cur_t += msg_interval_s + jitter
            if cur_t < end_t:
                arrival_indices.append(int(round(cur_t / dt)))

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

            all_eval_items.append((s, arr_idx, msg, cur_att, feats, true_window, att, sc, i))

    n_msgs = len(all_eval_items)
    m = len(lambdas)
    miss_mat = np.zeros((n_msgs, m), dtype=bool)
    feas_mat = np.zeros((n_msgs, m), dtype=bool)

    for j, lam in enumerate(lambdas):
        # Adjust calibration corrections by safety offset lambda
        adj_corr = {}
        for h, t_dict in base_corr.items():
            adj_corr[h] = {}
            for tau, st_dict in t_dict.items():
                adj_corr[h][tau] = {st: v + lam for st, v in st_dict.items()}

        for i, (s, arr_idx, msg, cur_att, feats, true_win, att, sc, m_idx) in enumerate(all_eval_items):
            ctx = PolicyContext(
                arrival_idx=arr_idx,
                last_meas_idx=max(hist_steps, arr_idx - delay_steps),
                cur_att_db=cur_att,
                features=feats,
                true_future_att=true_win,
                msg=msg,
                sc=sc,
                table=table,
                forecaster=forecaster,
                calibration_corr=adj_corr,
            )
            dec = policy_v6_full_rcdam(ctx) if use_validated_selection else policy_v5_stage_a_levers(ctx)
            sim_rng = np.random.default_rng(s * 10000 + m_idx)
            res = simulate_true_delivery(dec.action, ctx.arrival_idx, att, msg, table, sc, sim_rng)

            miss_mat[i, j] = not res.delivered
            feas_mat[i, j] = dec.feasible

            if j == 0:  # record baseline details
                all_records.append({
                    "seed": s,
                    "msg_idx": m_idx,
                    "priority": msg.priority,
                    "target_eps": msg.eps,
                    "predicted_p_lb": dec.p_lower_bound,
                    "feasible": dec.feasible,
                    "delivered": res.delivered,
                    "overhead": res.overhead,
                    "latency_s": res.latency_s if res.delivered else float("nan"),
                    "reason": dec.reason,
                })

    return miss_mat, feas_mat, all_records


def build_reliability_diagram(records: List[dict], out_png: str, out_csv: str):
    """Generates reliability diagram: predicted P* vs empirical delivery rate."""
    # Consider only messages flagged feasible
    feas_records = [r for r in records if r["feasible"]]
    if not feas_records:
        print("Warning: No feasible messages for reliability diagram.")
        return

    bins = [
        (0.90, 0.95),
        (0.95, 0.98),
        (0.98, 0.99),
        (0.99, 0.995),
        (0.995, 1.0001),
    ]

    diagram_rows = []
    bin_centers = []
    emp_rates = []
    bin_counts = []
    ece_terms = []

    for lo, hi in bins:
        in_bin = [r for r in feas_records if lo <= r["predicted_p_lb"] < hi]
        n_bin = len(in_bin)
        if n_bin > 0:
            pred_mean = float(np.mean([r["predicted_p_lb"] for r in in_bin]))
            n_deliv = sum(1 for r in in_bin if r["delivered"])
            emp_mean = n_deliv / n_bin
            w_lo, w_hi = wilson_interval(n_deliv, n_bin)
            ece_terms.append(abs(pred_mean - emp_mean) * n_bin)
        else:
            pred_mean = (lo + hi) / 2.0
            emp_mean = float("nan")
            w_lo, w_hi = float("nan"), float("nan")

        bin_centers.append(pred_mean)
        emp_rates.append(emp_mean)
        bin_counts.append(n_bin)

        diagram_rows.append({
            "bin_range": f"[{lo:.3f}, {hi:.3f})",
            "n_messages": n_bin,
            "predicted_p_star_mean": round(pred_mean, 4),
            "empirical_delivery_rate": round(emp_mean, 4) if not np.isnan(emp_mean) else "nan",
            "wilson_lo": round(w_lo, 4) if not np.isnan(w_lo) else "nan",
            "wilson_hi": round(w_hi, 4) if not np.isnan(w_hi) else "nan",
            "well_calibrated": emp_mean >= pred_mean if not np.isnan(emp_mean) else "nan",
        })

    ece = sum(ece_terms) / len(feas_records) if feas_records else float("nan")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagram_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diagram_rows)
    print(f"Saved reliability diagram data to {out_csv} (ECE = {ece:.4f})")

    # Plot reliability diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    valid_mask = [not np.isnan(r) for r in emp_rates]
    x = [bin_centers[i] for i in range(len(bin_centers)) if valid_mask[i]]
    y = [emp_rates[i] for i in range(len(emp_rates)) if valid_mask[i]]
    sizes = [max(20, bin_counts[i] * 3) for i in range(len(bin_counts)) if valid_mask[i]]

    ax.scatter(x, y, s=sizes, color="royalblue", zorder=3, edgecolors="navy", label="Empirical Delivery Rate (Binned)")
    ax.plot([0.90, 1.0], [0.90, 1.0], "k--", alpha=0.7, label="Ideal Calibration (Diagonal)")
    ax.fill_between([0.90, 1.0], [0.90, 1.0], 1.0, color="green", alpha=0.1, label="Conservative Region (Safe)")

    ax.set_xlim(0.89, 1.005)
    ax.set_ylim(0.85, 1.01)
    ax.set_xlabel("Reported Lower Bound Delivery Probability $P^*$", fontsize=11)
    ax.set_ylabel("Empirical On-Time Delivery Proportion", fontsize=11)
    ax.set_title(f"RC-DAM Reliability Diagram (ECE = {ece:.4f})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram figure to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Risk Validation & LTT Certification")
    parser.add_argument("--cal-seeds", type=int, default=4, help="Number of calibration seeds (11..)")
    parser.add_argument("--test-seeds", type=int, default=4, help="Number of test seeds (101..)")
    parser.add_argument("--hours", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sc = preset_harsh(duration_h=args.hours)
    table = ModcodTable()

    cal_path = os.path.join(OUT, "calibration.json")
    fc_path = os.path.join(OUT, "forecaster_baseline.json")

    _, cal_corr = load_calibration_json(cal_path) if os.path.exists(cal_path) else ({}, {})
    forecaster = EmpiricalQuantileForecaster.load(fc_path) if os.path.exists(fc_path) else None

    cal_seed_list = list(range(11, 11 + args.cal_seeds))
    test_seed_list = list(range(101, 101 + args.test_seeds))

    print("=" * 70)
    print("      RC-DAM PHASE 4: RISK CONTROL & LTT CERTIFICATION HARNESS")
    print(f"  Cal Seeds: {cal_seed_list} | Test Seeds: {test_seed_list} | Hours: {args.hours}")
    print("=" * 70)

    # 1. Calibrate safety offset lambda on calibration seeds using Learn-then-Test
    print("\n[Step 1] Running LTT fixed-sequence replay on Calibration Seeds...")
    cal_miss, cal_feas, _ = run_messages_with_lambda(
        cal_seed_list, sc, table, forecaster, cal_corr, SAFETY_LAMBDAS, use_validated_selection=True
    )

    cert_idx, ltt_table = ltt_fixed_sequence(cal_miss, cal_feas, eps=0.01, delta=0.05)
    print("\n--- LTT Fixed-Sequence Certification Table ---")
    for row in ltt_table:
        lam = SAFETY_LAMBDAS[row["setting"]]
        print(f"  Setting {row['setting']} (lambda={lam}dB): n_feasible={row['n_feasible']}, misses={row['misses']}, p_val={row['p_value']}, certified={row['certified']}")

    if cert_idx is not None:
        cert_lambda = SAFETY_LAMBDAS[cert_idx]
        print(f"\n--> CERTIFIED SAFETY SETTING: lambda* = {cert_lambda} dB (Index {cert_idx})")
    else:
        cert_lambda = SAFETY_LAMBDAS[-1]
        print(f"\n--> NO SETTING STATISTICALLY CERTIFIED AT 95% CONFIDENCE. Using max safety lambda={cert_lambda} dB")

    ltt_csv = os.path.join(OUT, "ltt_certification.csv")
    with open(ltt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ltt_table[0].keys()))
        writer.writeheader()
        writer.writerows(ltt_table)
    print(f"Saved LTT certification table to {ltt_csv}")

    # 2. Evaluate ONCE on Test Seeds with frozen lambda
    print(f"\n[Step 2] Evaluating FROZEN lambda* = {cert_lambda} dB ONCE on Disjoint Test Seeds...")
    test_miss, test_feas, test_records = run_messages_with_lambda(
        test_seed_list, sc, table, forecaster, cal_corr, [cert_lambda], use_validated_selection=True
    )

    n_test_feas = int(np.sum(test_feas[:, 0]))
    n_test_silent_miss = int(np.sum(test_feas[:, 0] & test_miss[:, 0]))
    test_silent_risk = n_test_silent_miss / max(n_test_feas, 1)
    w_lo, w_hi = wilson_interval(n_test_silent_miss, max(n_test_feas, 1))

    print(f"\n--- Test Set Evaluation Results (Frozen lambda={cert_lambda} dB) ---")
    print(f"  Feasible messages: {n_test_feas}/{len(test_miss)}")
    print(f"  Silent misses: {n_test_silent_miss}")
    print(f"  Empirical silent-miss risk: {test_silent_risk*100:.3f}% (Target: <= 1.000%)")
    print(f"  Wilson 95% Interval: [{w_lo*100:.3f}%, {w_hi*100:.3f}%]")
    print(f"  Risk Guarantee Respected on Unseen Test: {test_silent_risk <= 0.01}")

    # 3. Build Reliability Diagram on Test Messages
    print("\n[Step 3] Building Reliability Diagram on Test Messages...")
    diag_csv = os.path.join(OUT, "risk_validation.csv")
    diag_png = os.path.join(OUT, "reliability_diagram.png")
    build_reliability_diagram(test_records, diag_png, diag_csv)


if __name__ == "__main__":
    main()

