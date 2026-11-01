"""Evaluation of INFEASIBLE decision handling rules (R1 - R4) in RC-DAM (Phase 3).

Rules evaluated:
- R1: Send anyway and flag (best-effort transmission, sender warned)
- R2: Defer transmission (wait up to 30% of deadline slack; if feasible, send deferred)
- R3: Downgrade reliability (relax eps by 2x with explicit permission; retry optimizer)
- R4: Reject message (do not transmit; return Action=None with zero overhead)

Metrics evaluated:
- Flag recall vs Oracle: TP / (TP + FN)
- Flag precision vs Oracle: TP / (TP + FP)
- False infeasible rate: FP / oracle_feasible
- Silent miss rate: P(miss | flagged feasible) with Wilson 95% interval
- Flag lift: P(miss | flagged) / P(miss | not flagged)
- Mean airtime overhead
"""
import argparse
import csv
import os
import sys
from dataclasses import replace
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from rcdam.config import preset_harsh, Scenario
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.calibration import load_calibration_json
from rcdam.optimizer import Message
from rcdam.groundtruth import simulate_true_delivery
from rcdam.stats_utils import wilson_interval
from rcdam.variants import PolicyContext, policy_oracle, policy_v6_infeasible_rule

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def evaluate_rules_on_seed(seed: int,
                           sc_base: Scenario,
                           table: ModcodTable,
                           forecaster: Any,
                           cal_corr: dict,
                           msg_interval_s: float = 20.0) -> List[Dict[str, Any]]:
    sc = replace(sc_base, seed=seed)
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

    # Pre-build contexts and evaluate the idealized oracle reference.
    # Oracle labels are comparator labels, not directly observed physical truth.
    contexts = []
    oracle_infeasible = []

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
            calibration_corr=cal_corr,
        )
        contexts.append(ctx)

        dec_orc = policy_oracle(ctx)
        oracle_infeasible.append(not dec_orc.feasible)

    oracle_infeas_arr = np.array(oracle_infeasible, dtype=bool)
    n_msgs = len(contexts)

    seed_results = []

    for rule in ["R1", "R2", "R3", "R4"]:
        flagged_infeasible = []
        actual_miss = []
        overheads = []
        transmitted_flags = []
        outcomes = []

        for i, ctx in enumerate(contexts):
            dec = policy_v6_infeasible_rule(ctx, rule=rule)
            sim_rng = np.random.default_rng(seed * 10000 + i)
            res = simulate_true_delivery(dec.action, ctx.arrival_idx, att, ctx.msg, table, sc, sim_rng)

            # A message is flagged infeasible if dec.feasible is False
            flagged = not dec.feasible
            flagged_infeasible.append(flagged)
            actual_miss.append(not delivered)
            overheads.append(message_overhead)
            transmitted_flags.append(was_transmitted)
            outcomes.append(dec.decision_log.get("outcome", "unknown") if dec.decision_log else dec.reason)

        flag_arr = np.array(flagged_infeasible, dtype=bool)
        miss_arr = np.array(actual_miss, dtype=bool)

        # Confusion against oracle
        tp = int(np.sum(flag_arr & oracle_infeas_arr))
        fp = int(np.sum(flag_arr & ~oracle_infeas_arr))
        fn = int(np.sum(~flag_arr & oracle_infeas_arr))
        tn = int(np.sum(~flag_arr & ~oracle_infeas_arr))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        # False-positive rate among oracle-feasible messages: FP / (FP + TN).
        false_infeas_rate = fp / max(fp + tn, 1)

        # Silent misses: missed while flagged feasible
        n_feasible_decisions = int(np.sum(~flag_arr))
        n_silent_misses = int(np.sum(~flag_arr & miss_arr))
        silent_miss_rate = n_silent_misses / max(n_feasible_decisions, 1)
        w_lo, w_hi = wilson_interval(n_silent_misses, max(n_feasible_decisions, 1))

        # Flag lift = P(miss | flagged) / P(miss | not flagged)
        p_miss_flagged = float(np.mean(miss_arr[flag_arr])) if np.sum(flag_arr) > 0 else 0.0
        p_miss_unflagged = float(np.mean(miss_arr[~flag_arr])) if np.sum(~flag_arr) > 0 else 0.0
        flag_lift = (p_miss_flagged / p_miss_unflagged) if p_miss_unflagged > 0 else float("nan")

        seed_results.append({
            "seed": seed,
            "rule": rule,
            "n_total": n_msgs,
            "n_oracle_infeasible": int(np.sum(oracle_infeas_arr)),
            "n_flagged": int(np.sum(flag_arr)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "false_infeas_rate": false_infeas_rate,
            "n_silent_miss": n_silent_misses,
            "n_feasible": n_feasible_decisions,
            "silent_miss_rate": silent_miss_rate,
            "wilson_lo": w_lo,
            "wilson_hi": w_hi,
            "flag_lift": flag_lift,
            "mean_overhead": float(np.mean(overheads)),
            "n_transmitted": int(np.sum(transmitted_flags)),
            "mean_overhead_per_transmitted": (
                float(np.mean([o for o, sent in zip(overheads, transmitted_flags) if sent]))
                if any(transmitted_flags) else 0.0
            ),
            "n_deferred": sum(1 for o in outcomes if "deferred" in o),
            "n_downgraded": sum(1 for o in outcomes if "downgraded" in o),
            "n_rejected": sum(1 for o in outcomes if "rejected" in o),
        })

    return seed_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate INFEASIBLE decision rules (R1-R4)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=101)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--interval", type=float, default=20.0)
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sc = preset_harsh(duration_h=args.hours)
    table = ModcodTable()

    cal_path = os.path.join(OUT, "calibration.json")
    fc_path = os.path.join(OUT, "forecaster_baseline.json")

    _, cal_corr = load_calibration_json(cal_path) if os.path.exists(cal_path) else ({}, {})
    forecaster = EmpiricalQuantileForecaster.load(fc_path) if os.path.exists(fc_path) else None

    print("=" * 70)
    print("      RC-DAM PHASE 3: INFEASIBLE DECISION RULES EVALUATION (R1-R4)")
    print(f"  Seeds: {args.seeds} ({args.first_seed}..) | Hours: {args.hours} | Preset: harsh")
    print("=" * 70)

    all_seed_results = []
    for s in range(args.first_seed, args.first_seed + args.seeds):
        print(f"--> Running Seed {s}...")
        res = evaluate_rules_on_seed(s, sc, table, forecaster, cal_corr, msg_interval_s=args.interval)
        all_seed_results.extend(res)

    # Aggregate by rule
    print("\n--- Aggregated INFEASIBLE Decision Rules Performance ---")
    summary = []
    for rule in ["R1", "R2", "R3", "R4"]:
        sub = [r for r in all_seed_results if r["rule"] == rule]
        tot_msgs = sum(r["n_total"] for r in sub)
        tot_orc_infeas = sum(r["n_oracle_infeasible"] for r in sub)
        tot_flag = sum(r["n_flagged"] for r in sub)
        tot_tp = sum(r["tp"] for r in sub)
        tot_fp = sum(r["fp"] for r in sub)
        tot_fn = sum(r["fn"] for r in sub)
        tot_tn = sum(r["tn"] for r in sub)
        tot_silent = sum(r["n_silent_miss"] for r in sub)
        tot_feasible = sum(r["n_feasible"] for r in sub)

        prec = tot_tp / max(tot_tp + tot_fp, 1)
        rec = tot_tp / max(tot_tp + tot_fn, 1)
        false_infeas = tot_fp / max(tot_fp + tot_tn, 1)
        silent_rate = tot_silent / max(tot_feasible, 1)
        w_lo, w_hi = wilson_interval(tot_silent, max(tot_feasible, 1))
        ov = (
            sum(r["mean_overhead"] * r["n_total"] for r in sub)
            / max(tot_msgs, 1)
        )

        tot_deferred = sum(r["n_deferred"] for r in sub)
        tot_downgraded = sum(r["n_downgraded"] for r in sub)
        tot_rejected = sum(r["n_rejected"] for r in sub)
        tot_transmitted = sum(r["n_transmitted"] for r in sub)
        ov_per_tx = (
            sum(r["mean_overhead_per_transmitted"] * r["n_transmitted"] for r in sub)
            / max(tot_transmitted, 1)
        )

        row = {
            "rule": rule,
            "n_total": tot_msgs,
            "n_oracle_infeasible": tot_orc_infeas,
            "n_flagged": tot_flag,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "false_infeasible_rate": round(false_infeas, 4),
            "silent_miss_rate": round(silent_rate, 4),
            "silent_miss_count": f"{tot_silent}/{tot_feasible}",
            "wilson_ci": f"[{w_lo*100:.2f}%, {w_hi*100:.2f}%]",
            "mean_overhead_per_offered": round(ov, 3),
            "mean_overhead_per_transmitted": round(ov_per_tx, 3),
            "n_transmitted": tot_transmitted,
            "n_deferred": tot_deferred,
            "n_downgraded": tot_downgraded,
            "n_rejected": tot_rejected,
        }
        summary.append(row)
        print(f"Rule {rule:2s}: Recall={rec*100:5.2f}% | Precision={prec*100:5.2f}% | FalseInfeas={false_infeas*100:5.2f}% | SilentMiss={tot_silent}/{tot_feasible} ({silent_rate*100:5.2f}%) Wilson {row['wilson_ci']} | Overhead/offered={ov:.3f} | Overhead/transmitted={ov_per_tx:.3f}")

    out_csv = os.path.join(OUT, "infeasible_rules.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved INFEASIBLE rules evaluation results to {out_csv}")


if __name__ == "__main__":
    main()

