"""Stress test scenarios ensuring >= 100 oracle-infeasible messages (Phase 2 & Item 4).

Sweeps stress presets:
- 'harsh': High Poisson arrival of rain events with deep log-normal fades
- 'deadline_1s': 1.0s strict deadline (tight timing stress)
- 'size_50mb': 50 Mbit large payloads (capacity stress)

Evaluates deterministic Oracle feasibility and verifies >= 100 oracle-infeasible events.
Saves outputs/stress_test_results.csv.
"""
import argparse
import csv
import os
import sys
from dataclasses import replace
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from rcdam.config import get_preset, PRESET_REGISTRY, Scenario
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.optimizer import Message
from rcdam.stats_utils import wilson_interval
from rcdam.variants import PolicyContext, policy_oracle

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def evaluate_preset(preset_name: str, seeds: List[int], hours: float = 3.0, interval_s: float = 20.0) -> Dict[str, Any]:
    table = ModcodTable()
    sc_base = get_preset(preset_name, duration_h=hours, msg_interval_s=interval_s)

    tot_msgs = 0
    tot_infeas = 0

    for s in seeds:
        sc = replace(sc_base, seed=s)
        rng = np.random.default_rng(s)
        att = generate_attenuation_db(sc)
        n_steps = len(att)
        dt = sc.dt_s
        delay_steps = sc.feedback_delay_steps
        hist_steps = int(round(60.0 / dt))

        cur_t = 60.0
        end_t = sc.duration_h * 3600.0 - 5.0 * sc.deadline_s
        arrival_indices = []
        while cur_t < end_t:
            jitter = rng.uniform(-2.0, 2.0)
            cur_t += interval_s + jitter
            if cur_t < end_t:
                arrival_indices.append(int(round(cur_t / dt)))

        priorities = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]

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
                forecaster=None,
                calibration_corr={},
            )
            dec_orc = policy_oracle(ctx)
            tot_msgs += 1
            if not dec_orc.feasible:
                tot_infeas += 1

    infeas_rate = tot_infeas / max(tot_msgs, 1)
    w_lo, w_hi = wilson_interval(tot_infeas, max(tot_msgs, 1))

    return {
        "preset": preset_name,
        "n_seeds": len(seeds),
        "hours_per_seed": hours,
        "n_messages": tot_msgs,
        "n_oracle_infeasible": tot_infeas,
        "infeasible_rate": round(infeas_rate, 4),
        "wilson_lo": round(w_lo, 4),
        "wilson_hi": round(w_hi, 4),
        "meets_100_infeasible_goal": tot_infeas >= 100,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = [101, 102, 103, 104, 105]

    print("=" * 70)
    print("      RC-DAM STRESS-TEST SCENARIOS EVALUATION")
    print(f"  Seeds: {seeds} | Verifying >= 100 Oracle-Infeasible Events")
    print("=" * 70)

    presets = ["harsh", "deadline_1s", "size_50mb"]
    results = []
    for p in presets:
        print(f"--> Evaluating preset '{p}'...")
        res = evaluate_preset(p, seeds=seeds, hours=3.0, interval_s=20.0)
        results.append(res)
        print(f"  Preset {p:12s}: {res['n_oracle_infeasible']:4d}/{res['n_messages']} infeasible ({res['infeasible_rate']*100:.2f}%) | Meets >=100 Goal: {res['meets_100_infeasible_goal']}")

    out_csv = os.path.join(OUT, "stress_test_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved stress-test results to {out_csv}")


if __name__ == "__main__":
    main()

