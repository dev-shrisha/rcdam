"""Module 8: Evaluation Harness and Ablation Framework (A0 - A6, B1).

Implements ARCHITECTURE.md Section 11:
- A0: Baselines (Static ARQ, Fixed-margin ACM, Predictive ACM point forecast)
- A1: Calibrated forecast + MODCOD only (does calibration alone beat point-forecast ACM?)
- A2: A1 + Power step (value of power control)
- A3: A2 + Packet-level erasure coding (value of fast-fade protection)
- A4: A3 + Interleaver depth chosen under deadline (joint depth/deadline optimization)
- A5: A4 + Deferral (value of timing)
- A6: Full RC-DAM + Infeasibility signal (precision/recall of infeasibility detection)
- B1: A6 with header reduction (header bytes impact)
"""
from dataclasses import dataclass, replace
from typing import Dict, List, Any, Tuple
import numpy as np

from .config import Scenario
from .link import ModcodTable
from .cost import Action, CostModel
from .erasure import ErasureModel
from .forecaster import QuantileForecaster
from .conformal import StratifiedConformalCalibrator
from .optimizer import RedundancyOptimizer, Message
from .attenuation import generate_attenuation_db
from .experiment import step2_scenario, evaluate
from . import policies

ABLATION_SYSTEMS = [
    ("A0: Fixed-margin ACM (2dB)", "A0_margin"),
    ("A0: Predictive ACM (2dB)", "A0_predictive"),
    ("A0: Static MODCOD (10dB)", "A0_static"),
    ("A1: Calibrated + MODCOD only", "A1_modcod"),
    ("A2: A1 + Power step", "A2_power"),
    ("A3: A2 + Erasure coding", "A3_erasure"),
    ("A4: A3 + Interleaver depth", "A4_interleaver"),
    ("A5: A4 + Deferral", "A5_defer"),
    ("A6: Full RC-DAM + Infeasibility", "A6_full"),
    ("B1: RC-DAM (compressed headers)", "B1_comp"),
]


def create_optimizer_for_ablation(system_id: str,
                                  table: ModcodTable,
                                  cost_model: CostModel,
                                  erasure_model: ErasureModel) -> RedundancyOptimizer:
    """Instantiate optimizer with exactly the active levers for the given ablation ID."""
    if system_id == "A1_modcod":
        return RedundancyOptimizer(
            table, cost_model, erasure_model,
            allow_power_control=False,
            allow_erasure_coding=False,
            allow_interleaving=False,
            allow_deferral=False,
            allow_duplication=False,
        )
    elif system_id == "A2_power":
        return RedundancyOptimizer(
            table, cost_model, erasure_model,
            allow_power_control=True,
            allow_erasure_coding=False,
            allow_interleaving=False,
            allow_deferral=False,
            allow_duplication=False,
        )
    elif system_id == "A3_erasure":
        return RedundancyOptimizer(
            table, cost_model, erasure_model,
            allow_power_control=True,
            allow_erasure_coding=True,
            allow_interleaving=False,
            allow_deferral=False,
            allow_duplication=False,
        )
    elif system_id == "A4_interleaver":
        return RedundancyOptimizer(
            table, cost_model, erasure_model,
            allow_power_control=True,
            allow_erasure_coding=True,
            allow_interleaving=True,
            allow_deferral=False,
            allow_duplication=False,
        )
    elif system_id == "A5_defer":
        return RedundancyOptimizer(
            table, cost_model, erasure_model,
            allow_power_control=True,
            allow_erasure_coding=True,
            allow_interleaving=True,
            allow_deferral=True,
            allow_duplication=False,
        )
    elif system_id in ("A6_full", "B1_comp"):
        return RedundancyOptimizer(
            table, cost_model, erasure_model,
            allow_power_control=True,
            allow_erasure_coding=True,
            allow_interleaving=True,
            allow_deferral=True,
            allow_duplication=True,
        )
    else:
        raise ValueError(f"Unknown ablation system_id: {system_id}")


def run_ablation_seed(seed: int,
                      hours: float = 24.0,
                      forecaster: QuantileForecaster = None) -> List[Dict[str, Any]]:
    """Run all ablation systems A0-A6 and B1 on a single simulated channel seed."""
    sc = replace(step2_scenario(), seed=seed, duration_h=hours)
    table = ModcodTable()
    att = generate_attenuation_db(sc)
    esn0 = sc.clear_sky_esn0_db - att
    n = len(esn0)
    dt = sc.dt_s
    delay = sc.feedback_delay_steps

    # 1. Oracle benchmark for unrecoverable messages
    idx_oracle = policies.oracle_policy(table, esn0, 0.0)
    res_oracle, miss_oracle = evaluate(idx_oracle, esn0, sc, table)
    oracle_unrecoverable = miss_oracle

    # Horizon steps
    w = 30
    h = 20
    from numpy.lib.stride_tricks import sliding_window_view

    # Message arrival schedule
    interval_steps = int(round(sc.msg_interval_s / dt))
    horizon_guard = int(5 * sc.deadline_s / dt)
    arrival_steps = np.arange(w + delay, n - horizon_guard, interval_steps)
    m_count = len(arrival_steps)

    # Telemetry windows for arrivals
    telemetry_ends = arrival_steps - delay
    windows = sliding_window_view(att, w)[telemetry_ends - w]

    # Pre-predict quantiles using forecaster if provided
    if forecaster is not None:
        raw_quantiles = forecaster.predict_quantiles(windows)
        q95_arr = raw_quantiles[:, 6]
    else:
        # Fallback to persistence + margin
        last_obs = windows[:, -1]
        q95_arr = last_obs + 1.0

    # True worst attenuation over deadline horizon for each message arrival
    deadline_steps = int(round(sc.deadline_s / dt))
    true_horizon_peaks = np.array([
        np.max(att[t:t + deadline_steps]) if t + deadline_steps <= n else att[-1]
        for t in arrival_steps
    ])

    rows = []

    # --- Run A0 Baselines ---
    # A0 Margin
    win = max(1, int(round(sc.slope_window_s / dt)))
    idx_margin2 = policies.margin_policy(table, esn0, 2.0, delay)
    res_m2, miss_m2 = evaluate(idx_margin2, esn0, sc, table)
    feasible = ~oracle_unrecoverable
    rows.append({
        "seed": seed, "system": "A0: Fixed-margin ACM (2dB)", "system_id": "A0_margin",
        "miss_rate": res_m2["miss_rate"],
        "avoidable_miss_rate": float((miss_m2 & feasible).sum() / max(feasible.sum(), 1)),
        "infeasible_frac": float(oracle_unrecoverable.mean()),
        "overhead": res_m2["overhead"],
        "latency_s": res_m2["mean_latency_s"],
        "infeas_precision": 0.0, "infeas_recall": 0.0, "infeas_f1": 0.0,
    })

    # A0 Predictive
    idx_pred2 = policies.predictive_policy(table, esn0, 2.0, delay, win)
    res_p2, miss_p2 = evaluate(idx_pred2, esn0, sc, table)
    rows.append({
        "seed": seed, "system": "A0: Predictive ACM (2dB)", "system_id": "A0_predictive",
        "miss_rate": res_p2["miss_rate"],
        "avoidable_miss_rate": float((miss_p2 & feasible).sum() / max(feasible.sum(), 1)),
        "infeasible_frac": float(oracle_unrecoverable.mean()),
        "overhead": res_p2["overhead"],
        "latency_s": res_p2["mean_latency_s"],
        "infeas_precision": 0.0, "infeas_recall": 0.0, "infeas_f1": 0.0,
    })

    # A0 Static
    idx_stat10 = policies.static_policy(table, sc.clear_sky_esn0_db, 10.0, n)
    res_s10, miss_s10 = evaluate(idx_stat10, esn0, sc, table)
    rows.append({
        "seed": seed, "system": "A0: Static MODCOD (10dB)", "system_id": "A0_static",
        "miss_rate": res_s10["miss_rate"],
        "avoidable_miss_rate": float((miss_s10 & feasible).sum() / max(feasible.sum(), 1)),
        "infeasible_frac": float(oracle_unrecoverable.mean()),
        "overhead": res_s10["overhead"],
        "latency_s": res_s10["mean_latency_s"],
        "infeas_precision": 0.0, "infeas_recall": 0.0, "infeas_f1": 0.0,
    })

    # --- Run A1 through A6 and B1 ---
    for sys_name, sys_id in [
        ("A1: Calibrated + MODCOD only", "A1_modcod"),
        ("A2: A1 + Power step", "A2_power"),
        ("A3: A2 + Erasure coding", "A3_erasure"),
        ("A4: A3 + Interleaver depth", "A4_interleaver"),
        ("A5: A4 + Deferral", "A5_defer"),
        ("A6: Full RC-DAM + Infeasibility", "A6_full"),
        ("B1: RC-DAM (compressed headers)", "B1_comp"),
    ]:
        cost_m = CostModel(table=table)
        erasure_m = ErasureModel(table=table)
        if sys_id == "B1_comp":
            erasure_m = replace(erasure_m, header_bytes=10)

        opt = create_optimizer_for_ablation(sys_id, table, cost_m, erasure_m)
        calibrator = StratifiedConformalCalibrator(target_miss_rate=0.05, feedback_delay_steps=delay)

        overheads = []
        misses = []
        infeasible_flags = []

        for i, t_arr in enumerate(arrival_steps):
            obs_step = t_arr - delay
            true_h_peak = float(np.max(att[obs_step:obs_step + h]))
            calibrator.receive_feedback(current_step=t_arr, observed_att_db=true_h_peak)

            recent_obs = float(windows[i, -1])
            calibrated_att = calibrator.predict_envelope(
                step_t=t_arr,
                raw_upper_quantile=float(q95_arr[i]),
                recent_att_db=recent_obs,
            )

            msg = Message(msg_id=i, bits=sc.msg_bits, priority=2, deadline_s=sc.deadline_s)
            res = opt.optimize(msg, calibrated_att_db=calibrated_att, clear_sky_esn0_db=sc.clear_sky_esn0_db)

            # Evaluate real outcome
            actual_esn0 = sc.clear_sky_esn0_db - true_horizon_peaks[i]
            p_actual = erasure_m.delivery_probability(
                msg_bits=msg.bits,
                action=res.action,
                esn0_db=actual_esn0,
                deadline_s=msg.deadline_s,
            )

            is_miss = bool(p_actual < 0.5)
            misses.append(is_miss)
            overheads.append(res.overhead)
            infeasible_flags.append(not res.is_feasible)

        miss_arr = np.array(misses)
        infeas_arr = np.array(infeasible_flags)
        oracle_slice = oracle_unrecoverable[:m_count]

        # Precision/recall of infeasibility signal against true unrecoverable events
        tp = np.sum(infeas_arr & oracle_slice)
        fp = np.sum(infeas_arr & ~oracle_slice)
        fn = np.sum(~infeas_arr & oracle_slice)

        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-6))

        avoidable_miss = float(np.sum(miss_arr & ~oracle_slice) / max(np.sum(~oracle_slice), 1))

        rows.append({
            "seed": seed, "system": sys_name, "system_id": sys_id,
            "miss_rate": float(np.mean(miss_arr)),
            "avoidable_miss_rate": avoidable_miss,
            "infeasible_frac": float(np.mean(oracle_slice)),
            "overhead": float(np.mean(overheads)),
            "latency_s": float(sc.rtt_s / 2.0),
            "infeas_precision": precision,
            "infeas_recall": recall,
            "infeas_f1": f1,
        })

    return rows
