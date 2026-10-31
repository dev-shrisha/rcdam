"""M5 Closed-Loop Evaluation Harness for Hypotheses H1-H4 (Phase 2).

Key features:
- 20 independent seeds x thousands of messages per variant
- Deterministic oracle feasibility (evaluated under true window max with alpha=0)
- Per-message diagnostics logged to outputs/h1_h4_messages.csv.gz for ALL variants
- Miss taxonomy (ok, flagged_unrecoverable, flagged_avoidable, silent_unrecoverable,
  bound_violation, model_mismatch) logged to outputs/h1_h4_miss_categories.csv
- Exact paired McNemar tests for V5/V6 vs V1, V2, V4 saved to outputs/h1_h4_mcnemar.csv
- Cluster bootstrap over seeds + Wilson intervals for all metrics
- Channel stress presets ensuring >=100 oracle-infeasible messages
"""
import argparse
import csv
import gzip
import io
import os
import sys
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats

from rcdam.config import Scenario, get_preset, preset_harsh
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.calibration import load_calibration_json, fade_state
from rcdam.optimizer import Action, Message, FRAME_BITS
from rcdam.groundtruth import simulate_true_delivery
from rcdam.stats_utils import wilson_interval, mcnemar_exact
from rcdam.diagnostics import LOG_COLUMNS, classify_misses, infeasibility_report
from rcdam.variants import (
    HORIZONS,
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

VARIANTS = [
    ("Oracle", policy_oracle),
    ("V0_Static", lambda ctx: policy_v0_static(ctx, static_modcod=6)),
    ("V1_Margin", policy_v1_fixed_margin),
    ("V2_Predictive", policy_v2_linear_predictive),
    ("V3_Quantile_Raw", policy_v3_quantile_no_conformal),
    ("V4_Conformal_MODCOD", policy_v4_conformal_modcod),
    ("V5_All_Levers", policy_v5_stage_a_levers),
    ("V6_RC_DAM_Full", policy_v6_full_rcdam),
]


def cluster_bootstrap_mean(values: List[float], n_boot: int = 3000, seed: int = 0, conf: float = 0.95):
    """Cluster bootstrap over seed-level aggregates."""
    vals = np.asarray(values, float)
    if len(vals) <= 1:
        m = float(vals.mean()) if len(vals) == 1 else 0.0
        return m, m, m
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot_means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boot_means, [(1.0 - conf) / 2.0, 1.0 - (1.0 - conf) / 2.0])
    return float(vals.mean()), float(lo), float(hi)


def run_single_seed(
    seed: int,
    sc_base: Scenario,
    table: ModcodTable,
    forecaster: Any,
    calibration_corr: dict,
    msg_interval_s: float = 10.0,
) -> Tuple[List[dict], List[dict], Dict[str, np.ndarray]]:
    """Runs all variants on a single seed realization.

    Returns:
      (seed_summary_records, per_message_log_rows, delivered_arrays_by_variant)
    """
    sc = replace(sc_base, seed=seed)
    rng = np.random.default_rng(seed)
    att = generate_attenuation_db(sc)
    n_steps = len(att)
    dt = sc.dt_s
    delay_steps = sc.feedback_delay_steps
    hist_steps = int(round(60.0 / dt))

    # Generate message arrivals with jitter
    arrival_indices = []
    cur_t = 60.0
    end_t = sc.duration_h * 3600.0 - 5.0 * sc.deadline_s
    while cur_t < end_t:
        jitter = rng.uniform(-2.0, 2.0)
        cur_t += msg_interval_s + jitter
        if cur_t < end_t:
            arrival_indices.append(int(round(cur_t / dt)))

    priorities = [
        ("critical", 1e-3),
        ("normal", 1e-2),
        ("best_effort", 1e-1),
    ]

    messages = []
    contexts_by_msg = []

    for i, arr_idx in enumerate(arrival_indices):
        p_name, eps = priorities[i % len(priorities)]
        msg = Message(
            size_bits=sc.msg_bits,
            priority=p_name,
            deadline_s=sc.deadline_s,
            eps=eps,
            arrival_idx=arr_idx,
        )
        messages.append(msg)

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
        true_window_max = float(np.max(true_window)) if len(true_window) > 0 else 0.0

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
            calibration_corr=calibration_corr,
        )
        contexts_by_msg.append((ctx, true_window_max, H_window))

    # Evaluate Oracle first with deterministic feasibility
    oracle_feasible = []
    oracle_airtimes = []
    for i, msg in enumerate(messages):
        ctx, true_win_max, _ = contexts_by_msg[i]
        dec_orc = policy_oracle(ctx)
        oracle_feasible.append(dec_orc.feasible)

        if dec_orc.feasible and dec_orc.action is not None:
            sim_rng = np.random.default_rng(seed * 10000 + i)
            res_orc = simulate_true_delivery(dec_orc.action, ctx.arrival_idx, att, msg, table, sc, sim_rng)
            oracle_airtimes.append(res_orc.airtime_s)
        else:
            oracle_airtimes.append(0.0)

    oracle_feas_arr = np.array(oracle_feasible, dtype=bool)
    n_total = len(messages)
    oracle_infeasible_count = int(np.sum(~oracle_feas_arr))

    seed_records = []
    all_msg_rows = []
    delivered_by_variant = {}

    for v_name, policy_fn in VARIANTS:
        v_delivered = []
        v_overheads = []
        v_latencies = []
        v_feasibles = []
        v_p_lower_bounds = []
        v_bound_dbs = []

        for i, msg in enumerate(messages):
            ctx, true_win_max, H_window = contexts_by_msg[i]
            dec = policy_fn(ctx)
            v_feasibles.append(dec.feasible)
            v_p_lower_bounds.append(dec.p_lower_bound)

            # Determine calibrated bound dB for diagnostic logging
            st = fade_state(ctx.cur_att_db)
            tau = 1.0 - dec.alpha
            corr = calibration_corr.get(H_window, {}).get(tau, {}).get(st, 0.0) if calibration_corr else 0.0
            bound_db = ctx.cur_att_db + corr if dec.alpha < 1.0 else ctx.cur_att_db + 3.0
            v_bound_dbs.append(bound_db)

            sim_rng = np.random.default_rng(seed * 10000 + i)
            res = simulate_true_delivery(dec.action, ctx.arrival_idx, att, msg, table, sc, sim_rng)
            v_delivered.append(res.delivered)
            v_overheads.append(res.overhead)
            if res.delivered:
                v_latencies.append(res.latency_s)

            # Per-message log row (matching LOG_COLUMNS)
            msg_log_row = {
                "seed": seed,
                "variant": v_name,
                "msg_id": f"{seed}_{i}",
                "arrival_idx": ctx.arrival_idx,
                "priority": msg.priority,
                "eps": msg.eps,
                "size_bits": msg.size_bits,
                "deadline_s": msg.deadline_s,
                "fade_state": st,
                "modcod": dec.action.modcod if dec.action else -1,
                "power_db": dec.action.power_db if dec.action else 0.0,
                "parity_frac": dec.action.parity_frac if dec.action else 0.0,
                "defer_s": dec.action.defer_s if dec.action else 0.0,
                "alpha": dec.alpha,
                "p_lb": dec.p_lower_bound,
                "flagged_infeasible": not dec.feasible,
                "bound_db": bound_db,
                "bound_horizon_steps": H_window,
                "true_window_max_db": true_win_max,
                "delivered": res.delivered,
                "completion_s": res.latency_s if res.delivered else float("nan"),
                "airtime_s": res.airtime_s,
                "oracle_delivered": oracle_feas_arr[i],
                "oracle_airtime_s": oracle_airtimes[i],
            }
            all_msg_rows.append(msg_log_row)

        v_del_arr = np.array(v_delivered, dtype=bool)
        v_feas_arr = np.array(v_feasibles, dtype=bool)
        delivered_by_variant[v_name] = v_del_arr

        n_miss = int(np.sum(~v_del_arr))
        avoidable_misses = oracle_feas_arr & (~v_del_arr)
        unrecoverable = ~oracle_feas_arr

        true_infeasible = unrecoverable
        pred_infeasible = ~v_feas_arr

        false_infeasible = (~true_infeasible) & pred_infeasible
        false_feasible = true_infeasible & (~pred_infeasible)
        silent_miss = (~v_del_arr) & (~pred_infeasible)

        w_lo, w_hi = wilson_interval(n_miss, n_total) if n_total > 0 else (0.0, 0.0)

        rec = {
            "seed": seed,
            "variant": v_name,
            "n_messages": n_total,
            "n_miss": n_miss,
            "n_oracle_miss": oracle_infeasible_count,
            "miss_rate": float(n_miss / max(1, n_total)),
            "miss_rate_wilson_lo": float(w_lo),
            "miss_rate_wilson_hi": float(w_hi),
            "n_avoidable_miss": int(np.sum(avoidable_misses)),
            "avoidable_miss_rate": float(np.sum(avoidable_misses) / max(1, np.sum(oracle_feas_arr))),
            "unrecoverable_frac": float(np.mean(unrecoverable)),
            "mean_overhead": float(np.mean(v_overheads)) if v_overheads else float("nan"),
            "mean_latency_s": float(np.mean(v_latencies)) if v_latencies else float("nan"),
            "p99_latency_s": float(np.percentile(v_latencies, 99)) if v_latencies else float("nan"),
            "false_infeasible_rate": float(np.mean(false_infeasible)),
            "false_feasible_rate": float(np.mean(false_feasible)),
            "silent_miss_rate": float(np.mean(silent_miss)),
        }

        for p_name_p, _ in priorities:
            mask_p = np.array([m.priority == p_name_p for m in messages])
            if np.any(mask_p):
                n_p = int(np.sum(mask_p))
                k_p = int(np.sum(~v_del_arr[mask_p]))
                rec[f"miss_rate_{p_name_p}"] = float(k_p / max(1, n_p))
                rec[f"n_miss_{p_name_p}"] = k_p
                rec[f"n_total_{p_name_p}"] = n_p
                rec[f"avoidable_miss_{p_name_p}"] = float(
                    np.sum(avoidable_misses & mask_p) / max(1, np.sum(oracle_feas_arr & mask_p))
                )
                rec[f"overhead_{p_name_p}"] = float(
                    np.mean(np.array(v_overheads)[mask_p])
                )

        seed_records.append(rec)

    return seed_records, all_msg_rows, delivered_by_variant


def main():
    parser = argparse.ArgumentParser(
        description="M5 Closed-Loop Evaluation Harness (H1-H4) — Phase 2 Fixed Benchmark"
    )
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default 20)")
    parser.add_argument("--first-seed", type=int, default=101, help="First seed (default 101)")
    parser.add_argument("--hours", type=float, default=6.0, help="Hours per seed (default 6.0)")
    parser.add_argument("--interval", type=float, default=10.0, help="Message arrival interval in seconds")
    parser.add_argument("--harsh", action="store_true", default=True, help="Use harsh stress channel preset")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sc0 = preset_harsh(duration_h=args.hours, msg_interval_s=args.interval) if args.harsh else get_preset("base", duration_h=args.hours, msg_interval_s=args.interval)
    table = ModcodTable()

    print("================================================================================")
    print("      RC-DAM PHASE 2: CLOSED-LOOP BENCHMARK & EVALUATION HARNESS                ")
    print(f"  Seeds: {args.seeds} ({args.first_seed}..) | Hours: {args.hours} | Interval: {args.interval}s | Harsh: {args.harsh}")
    print("================================================================================")

    cal_json = os.path.join(OUT, "calibration.json")
    fc_json = os.path.join(OUT, "forecaster_baseline.json")

    if not os.path.exists(cal_json) or not os.path.exists(fc_json):
        raise RuntimeError(
            f"Missing required calibration/forecaster artifacts ({cal_json}, {fc_json}). "
            f"Run 'python scripts/calibrate.py' first."
        )

    cal_metadata, cal_corr = load_calibration_json(cal_json)
    forecaster = EmpiricalQuantileForecaster.load(fc_json)
    print("Loaded calibration matrix from outputs/calibration.json")
    print(f"Loaded trained baseline forecaster from {fc_json}")

    # Verify disjoint data seeds
    eval_seeds = set(range(args.first_seed, args.first_seed + args.seeds))
    train_seeds = set(cal_metadata.get("train_seeds", []))
    cal_seeds = set(cal_metadata.get("cal_seeds", []))
    assert not (eval_seeds & train_seeds), f"Seed overlap between eval and train: {eval_seeds & train_seeds}"
    assert not (eval_seeds & cal_seeds), f"Seed overlap between eval and cal: {eval_seeds & cal_seeds}"
    print(f"Verified disjoint seeds: Train={sorted(train_seeds)}, Cal={sorted(cal_seeds)}, Eval={args.first_seed}..{args.first_seed + args.seeds - 1}")

    all_seed_records = []
    all_msg_rows = []
    delivered_series_by_variant = {v[0]: [] for v in VARIANTS}
    t0 = time.time()

    for s_idx in range(args.first_seed, args.first_seed + args.seeds):
        t_s = time.time()
        print(f"--> Seed {s_idx:3d} ... ", end="", flush=True)
        recs, msg_rows, del_map = run_single_seed(
            s_idx, sc0, table, forecaster, cal_corr, msg_interval_s=args.interval
        )
        all_seed_records.extend(recs)
        all_msg_rows.extend(msg_rows)
        for v_name, d_arr in del_map.items():
            delivered_series_by_variant[v_name].append(d_arr)

        orc_miss_this = next(r["n_oracle_miss"] for r in recs if r["variant"] == "Oracle")
        n_msg = next(r["n_messages"] for r in recs if r["variant"] == "Oracle")
        print(f"done in {time.time()-t_s:5.1f}s  [{n_msg:4d} msgs | Oracle Infeasible: {orc_miss_this:3d}]")

    print(f"\nAll {args.seeds} seeds completed in {time.time()-t0:.1f}s\n")

    # 1. Save per-seed CSV
    per_seed_path = os.path.join(OUT, "h1_h4_per_seed.csv")
    if all_seed_records:
        fieldnames = list(all_seed_records[0].keys())
        with open(per_seed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_seed_records)
        print(f"Saved per-seed summary: {per_seed_path}")

    # 2. Save compressed per-message logs (outputs/h1_h4_messages.csv.gz)
    msg_gz_path = os.path.join(OUT, "h1_h4_messages.csv.gz")
    msg_df = pd.DataFrame(all_msg_rows)
    msg_df.to_csv(msg_gz_path, index=False, compression="gzip")
    print(f"Saved per-message logs: {msg_gz_path} ({len(msg_df)} rows)")

    # 3. Compute miss taxonomy per variant (outputs/h1_h4_miss_categories.csv)
    var_names = [v[0] for v in VARIANTS]
    miss_category_rows = []
    for v in var_names:
        sub_df = msg_df[msg_df["variant"] == v]
        classified = classify_misses(sub_df)
        cat_counts = classified["category"].value_counts().to_dict()
        miss_category_rows.append({
            "variant": v,
            "total_messages": len(sub_df),
            "ok": cat_counts.get("ok", 0),
            "flagged_unrecoverable": cat_counts.get("flagged_unrecoverable", 0),
            "flagged_avoidable": cat_counts.get("flagged_avoidable", 0),
            "silent_unrecoverable": cat_counts.get("silent_unrecoverable", 0),
            "bound_violation": cat_counts.get("bound_violation", 0),
            "model_mismatch": cat_counts.get("model_mismatch", 0),
        })

    cat_path = os.path.join(OUT, "h1_h4_miss_categories.csv")
    with open(cat_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(miss_category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(miss_category_rows)
    print(f"Saved miss taxonomy: {cat_path}")

    # 4. Paired McNemar Tests: V5 and V6 vs V1, V2, V4
    mcnemar_rows = []
    comparisons = [
        ("V6_RC_DAM_Full", "V1_Margin"),
        ("V6_RC_DAM_Full", "V2_Predictive"),
        ("V6_RC_DAM_Full", "V4_Conformal_MODCOD"),
        ("V5_All_Levers", "V1_Margin"),
        ("V5_All_Levers", "V2_Predictive"),
        ("V5_All_Levers", "V4_Conformal_MODCOD"),
        ("V6_RC_DAM_Full", "V5_All_Levers"),
    ]
    for va, vb in comparisons:
        arr_a = np.concatenate(delivered_series_by_variant[va])
        arr_b = np.concatenate(delivered_series_by_variant[vb])
        # b = A missed and B delivered; c = A delivered and B missed
        miss_a = ~arr_a
        miss_b = ~arr_b
        b_count = int(np.sum(miss_a & (~miss_b)))
        c_count = int(np.sum((~miss_a) & miss_b))
        p_val = mcnemar_exact(b_count, c_count)
        mcnemar_rows.append({
            "variant_A": va,
            "variant_B": vb,
            "miss_A_only": b_count,
            "miss_B_only": c_count,
            "mcnemar_p_value": p_val,
            "significant_0_01": p_val < 0.01,
        })

    mcnemar_path = os.path.join(OUT, "h1_h4_mcnemar.csv")
    with open(mcnemar_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mcnemar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mcnemar_rows)
    print(f"Saved paired McNemar tests: {mcnemar_path}")

    # 5. Aggregate Across Seeds with Wilson & Cluster Bootstrap Intervals
    summary_rows = []
    total_oracle_infeasible = sum(r["n_oracle_miss"] for r in all_seed_records if r["variant"] == "Oracle")

    print(
        f"\n{'Variant':22s} {'n_miss/n_total':>18s} {'MissRate%':>10s}"
        f" {'Wilson95CI%':>22s} {'Bootstrap95CI%':>22s} {'Overhead':>10s} {'SilentMiss%':>12s}"
    )
    print("-" * 125)

    for v in var_names:
        recs_v = [r for r in all_seed_records if r["variant"] == v]
        n_s = len(recs_v)
        n_total_agg = sum(r["n_messages"] for r in recs_v)
        n_miss_agg = sum(r["n_miss"] for r in recs_v)

        w_lo, w_hi = wilson_interval(n_miss_agg, n_total_agg) if n_total_agg > 0 else (0.0, 0.0)

        seed_miss_rates = [r["miss_rate"] for r in recs_v]
        _, b_lo, b_hi = cluster_bootstrap_mean(seed_miss_rates, seed=101)

        overheads = [r["mean_overhead"] for r in recs_v if np.isfinite(r["mean_overhead"])]
        mean_ov, ov_lo, ov_hi = cluster_bootstrap_mean(overheads, seed=102) if overheads else (float("nan"), float("nan"), float("nan"))

        lats = [r["mean_latency_s"] for r in recs_v if np.isfinite(r.get("mean_latency_s", float("nan")))]
        mean_lat = float(np.mean(lats)) if lats else float("nan")

        avoid_rates = [r["avoidable_miss_rate"] for r in recs_v if np.isfinite(r["avoidable_miss_rate"])]
        m_avoid = float(np.mean(avoid_rates)) if avoid_rates else float("nan")
        ci_avoid = float(stats.t.ppf(0.975, df=max(n_s - 1, 1)) * stats.sem(avoid_rates)) if len(avoid_rates) > 1 else 0.0

        silent_rates = [r["silent_miss_rate"] for r in recs_v]
        m_silent = float(np.mean(silent_rates)) if silent_rates else float("nan")

        false_inf_rates = [r["false_infeasible_rate"] for r in recs_v]
        m_false_inf = float(np.mean(false_inf_rates)) if false_inf_rates else float("nan")

        miss_rate = n_miss_agg / max(1, n_total_agg)

        print(
            f"{v:22s} {n_miss_agg:6d}/{n_total_agg:<9d}"
            f" {miss_rate*100:9.3f}%"
            f"  [{w_lo*100:.3f}%, {w_hi*100:.3f}%]"
            f"  [{b_lo*100:.3f}%, {b_hi*100:.3f}%]"
            f"  {mean_ov:9.3f}  {m_silent*100:11.3f}%"
        )

        summary_rows.append({
            "variant": v,
            "n_seeds": n_s,
            "n_messages": n_total_agg,
            "n_miss": n_miss_agg,
            "n_oracle_infeasible": total_oracle_infeasible,
            "miss_rate": miss_rate,
            "miss_rate_wilson_lo": w_lo,
            "miss_rate_wilson_hi": w_hi,
            "miss_rate_boot_lo": b_lo,
            "miss_rate_boot_hi": b_hi,
            "avoidable_miss_rate": m_avoid,
            "avoidable_miss_rate_ci": ci_avoid,
            "overhead": mean_ov,
            "overhead_boot_lo": ov_lo,
            "overhead_boot_hi": ov_hi,
            "mean_latency_s": mean_lat,
            "silent_miss_rate": m_silent,
            "false_infeasible_rate": m_false_inf,
        })

    # Save summary results CSV
    csv_path = os.path.join(OUT, "h1_h4_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary results: {csv_path}")

    # Acceptance Checks
    print("\n--- Phase 2 Acceptance Criteria Verification ---")
    print(f"Total Oracle Infeasible count: {total_oracle_infeasible}")
    if total_oracle_infeasible >= 100:
        print("  -> Criteria: >=100 oracle-infeasible messages across test set: PASSED [OK]")
    else:
        print(f"  -> Criteria: >=100 oracle-infeasible messages: FAILED (got {total_oracle_infeasible}) [WARNING]")

    # Check zero-width intervals
    zero_width_wilson = any((r["miss_rate_wilson_hi"] - r["miss_rate_wilson_lo"]) == 0.0 for r in summary_rows if r["n_messages"] > 0)
    print(f"  -> Criteria: No zero-width intervals: {'FAILED' if zero_width_wilson else 'PASSED [OK]'}")

    # Print Hypotheses Status Review
    print("\n--- Hypotheses Evaluation Status ---")
    v6_row = next((r for r in summary_rows if r["variant"] == "V6_RC_DAM_Full"), None)
    v5_row = next((r for r in summary_rows if r["variant"] == "V5_All_Levers"), None)
    v1_row = next((r for r in summary_rows if r["variant"] == "V1_Margin"), None)

    if v6_row and v1_row:
        print(f"H1': Overhead at matched reliability (see outputs/matched_reliability.csv)")
        print(f"H3': V6 Silent miss rate = {v6_row['silent_miss_rate']*100:.3f}%, False infeasible rate = {v6_row['false_infeasible_rate']*100:.3f}%")
        print(f"     V5 Silent miss rate = {v5_row['silent_miss_rate']*100:.3f}%, False infeasible rate = {v5_row['false_infeasible_rate']*100:.3f}%")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
