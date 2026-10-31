"""Script to check stress presets and report oracle-infeasible message counts.

Implements Phase 2, Task 3:
Stress presets (base, harsh, deadline sweeps 1, 2, 5, 10s, size sweeps)
Reports the count of oracle-infeasible messages per preset.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from rcdam.config import get_preset, PRESET_REGISTRY
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.optimizer import Message
from rcdam.variants import PolicyContext, policy_oracle


def evaluate_preset(preset_name: str, seeds: list, hours: float = 3.0, interval_s: float = 15.0):
    table = ModcodTable()
    total_msgs = 0
    total_infeasible = 0
    priorities = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]

    t0 = time.time()
    for s in seeds:
        sc = get_preset(preset_name, seed=s, duration_h=hours, msg_interval_s=interval_s)
        att = generate_attenuation_db(sc)
        dt = sc.dt_s
        delay_steps = sc.feedback_delay_steps
        hist_steps = int(round(60.0 / dt))
        n_steps = len(att)

        rng = np.random.default_rng(s)
        cur_t = 60.0
        end_t = sc.duration_h * 3600.0 - 5.0 * sc.deadline_s
        arr_indices = []
        while cur_t < end_t:
            cur_t += interval_s + rng.uniform(-2.0, 2.0)
            if cur_t < end_t:
                arr_indices.append(int(round(cur_t / dt)))

        for i, arr_idx in enumerate(arr_indices):
            p_name, eps = priorities[i % len(priorities)]
            msg = Message(
                size_bits=sc.msg_bits,
                priority=p_name,
                deadline_s=sc.deadline_s,
                eps=eps,
                arrival_idx=arr_idx,
            )
            last_m = max(hist_steps, arr_idx - delay_steps)
            H_window = int(round((sc.deadline_s + sc.one_way_delay_s) / dt)) + 5
            true_window = att[arr_idx: min(n_steps, arr_idx + H_window)]

            ctx = PolicyContext(
                arrival_idx=arr_idx,
                last_meas_idx=last_m,
                cur_att_db=float(att[last_m]),
                features={},
                true_future_att=true_window,
                msg=msg,
                sc=sc,
                table=table,
                forecaster=None,
                calibration_corr={},
            )
            dec = policy_oracle(ctx)
            total_msgs += 1
            if not dec.feasible:
                total_infeasible += 1

    elapsed = time.time() - t0
    pct = (100.0 * total_infeasible / max(1, total_msgs))
    return {
        "preset": preset_name,
        "n_seeds": len(seeds),
        "hours_per_seed": hours,
        "n_messages": total_msgs,
        "n_oracle_infeasible": total_infeasible,
        "infeasible_pct": pct,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Check Oracle-Infeasible Counts Across Stress Presets")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to test (default 5)")
    parser.add_argument("--first-seed", type=int, default=101, help="First seed (default 101)")
    parser.add_argument("--hours", type=float, default=2.0, help="Duration in hours per seed")
    parser.add_argument("--interval", type=float, default=20.0, help="Message interval in seconds")
    args = parser.parse_args()

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    presets_to_test = [
        "base",
        "harsh",
        "deadline_1s",
        "deadline_2s",
        "deadline_5s",
        "deadline_10s",
        "size_5mb",
        "size_20mb",
        "size_50mb",
    ]

    print("================================================================================")
    print("           RC-DAM STRESS PRESET ORACLE INFEASIBILITY REPORT                    ")
    print(f"  Seeds: {args.seeds} (from {args.first_seed}) | Hours: {args.hours} | Interval: {args.interval}s")
    print("================================================================================")
    print(f"{'Preset':18s} {'Seeds':>6s} {'Messages':>10s} {'Infeasible':>12s} {'Infeasible%':>14s} {'Time(s)':>10s}")
    print("-" * 80)

    for p in presets_to_test:
        res = evaluate_preset(p, seeds, hours=args.hours, interval_s=args.interval)
        print(
            f"{res['preset']:18s} {res['n_seeds']:6d} {res['n_messages']:10d} "
            f"{res['n_oracle_infeasible']:12d} {res['infeasible_pct']:13.2f}% {res['elapsed_s']:10.1f}s"
        )

    print("================================================================================")


if __name__ == "__main__":
    main()

