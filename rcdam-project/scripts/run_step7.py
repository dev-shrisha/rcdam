"""Step 7: Full RC-DAM Redundancy Optimizer End-to-End Evaluation.

Demonstrates the full RC-DAM decision pipeline:
1. Multi-quantile fade forecasting.
2. Stratified conformal calibration with delayed feedback.
3. Joint redundancy sizing (MODCOD, power step, erasure coding, interleaving, duplication, deferral).
4. Early infeasibility signalling when channel capacity is insufficient.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rcdam.config import Scenario
from rcdam.link import ModcodTable
from rcdam.cost import CostModel
from rcdam.erasure import ErasureModel
from rcdam.forecaster import QuantileForecaster
from rcdam.conformal import StratifiedConformalCalibrator
from rcdam.optimizer import RedundancyOptimizer, Message
from rcdam.attenuation import generate_attenuation_db

OUT = os.path.join(PROJECT_ROOT, "outputs")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Step 7: Full RC-DAM End-to-End Optimizer Evaluation ===\n")

    table = ModcodTable()
    cost_model = CostModel(table=table)
    erasure = ErasureModel(table=table)
    optimizer = RedundancyOptimizer(table=table, cost_model=cost_model, erasure_model=erasure)

    # 1. Load forecaster
    forecaster = QuantileForecaster(window_steps=30, horizon_steps=20)
    weights_path = os.path.join(OUT, "step5_forecaster_weights.pt")
    if os.path.exists(weights_path):
        forecaster.model.load_state_dict(torch.load(weights_path))

    # 2. Setup simulation scenario
    sc = Scenario(seed=404, duration_h=12.0)
    att = generate_attenuation_db(sc)
    n = len(att)
    w = forecaster.window_steps
    h = forecaster.horizon_steps
    delay = sc.feedback_delay_steps

    calibrator = StratifiedConformalCalibrator(target_miss_rate=0.05, feedback_delay_steps=delay)

    # Messages arrive every 30 seconds
    msg_interval_steps = int(round(sc.msg_interval_s / sc.dt_s))
    horizon_guard = int(5 * sc.deadline_s / sc.dt_s)
    arrival_steps = np.arange(w + delay, n - horizon_guard, msg_interval_steps)

    print(f"Simulating {len(arrival_steps)} message arrivals over {sc.duration_h:.0f}h run...")

    results = []
    # Pre-extract telemetry windows for message arrivals
    from numpy.lib.stride_tricks import sliding_window_view
    telemetry_ends = arrival_steps - delay
    windows = sliding_window_view(att, w)[telemetry_ends - w]
    quantiles_arr = forecaster.predict_quantiles(windows)
    q95_arr = quantiles_arr[:, 6]  # 95th percentile

    # Simulate streaming timeline
    for i, t_arr in enumerate(arrival_steps):
        # Update delayed feedback
        obs_step = t_arr - delay
        true_h_peak = float(np.max(att[obs_step:obs_step + h]))
        calibrator.receive_feedback(current_step=t_arr, observed_att_db=true_h_peak)

        # Calibrated attenuation envelope
        recent_obs = float(windows[i, -1])
        calibrated_att = calibrator.predict_envelope(
            step_t=t_arr,
            raw_upper_quantile=float(q95_arr[i]),
            recent_att_db=recent_obs,
        )

        # Message with priority (cycle through 1, 2, 3)
        priority = (i % 3) + 1
        msg = Message(
            msg_id=i,
            bits=sc.msg_bits,
            priority=priority,
            deadline_s=sc.deadline_s,
            arrival_s=t_arr * sc.dt_s,
        )

        # Joint optimization
        opt_res = optimizer.optimize(msg, calibrated_att_db=calibrated_att, clear_sky_esn0_db=sc.clear_sky_esn0_db)

        # True outcome evaluation
        # Message transmission over actual channel
        actual_slice = att[t_arr:t_arr + int(round(msg.deadline_s / sc.dt_s))]
        actual_min_esn0 = sc.clear_sky_esn0_db - np.max(actual_slice) if len(actual_slice) > 0 else 18.0
        p_actual = erasure.delivery_probability(
            msg_bits=msg.bits,
            action=opt_res.action,
            esn0_db=actual_min_esn0,
            deadline_s=msg.deadline_s,
        )

        results.append({
            "arrival_s": t_arr * sc.dt_s,
            "priority": priority,
            "calibrated_att": calibrated_att,
            "actual_att": float(np.max(actual_slice)) if len(actual_slice) else 0.0,
            "is_feasible": opt_res.is_feasible,
            "modcod_idx": opt_res.action.modcod_idx,
            "modcod_name": table.names[opt_res.action.modcod_idx],
            "modcod_eff": table.eff[opt_res.action.modcod_idx],
            "power_step_db": opt_res.action.power_step_db,
            "code_rate": opt_res.action.code_rate,
            "interleaver_depth": opt_res.action.interleaver_depth,
            "overhead": opt_res.overhead,
            "predicted_p": opt_res.delivery_prob,
            "actual_p": p_actual,
            "delivered": (p_actual >= 0.5),  # delivered on time
        })

    # Summary statistics
    n_total = len(results)
    n_infeasible = sum(1 for r in results if not r["is_feasible"])
    mean_overhead = np.mean([r["overhead"] for r in results])
    miss_rate = np.mean([1.0 if not r["delivered"] else 0.0 for r in results])

    print(f"\nTotal messages: {n_total}")
    print(f"Infeasible flagged messages: {n_infeasible} ({100 * n_infeasible / n_total:.2f}%)")
    print(f"Mean action overhead O(a): {mean_overhead:.3f}")
    print(f"Overall deadline-miss rate: {100 * miss_rate:.2f}%\n")

    # Plot decisions
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    t_hrs = np.array([r["arrival_s"] for r in results]) / 3600.0

    # 1. Attenuation profile and infeasibility alerts
    axes[0].plot(t_hrs, [r["actual_att"] for r in results], "r-", label="Actual Peak Attenuation (dB)")
    axes[0].plot(t_hrs, [r["calibrated_att"] for r in results], "k--", alpha=0.7, label="Calibrated Upper Bound")
    infeas_t = [t_hrs[i] for i, r in enumerate(results) if not r["is_feasible"]]
    infeas_att = [results[i]["actual_att"] for i, r in enumerate(results) if not r["is_feasible"]]
    if infeas_t:
        axes[0].scatter(infeas_t, infeas_att, color="red", marker="x", s=80, zorder=5, label="INFEASIBLE Alert")
    axes[0].set_ylabel("Attenuation (dB)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Channel Fade & Early Infeasibility Alerts")
    axes[0].grid(True, alpha=0.3)

    # 2. Chosen MODCOD Spectral Efficiency
    axes[1].step(t_hrs, [r["modcod_eff"] for r in results], where="post", color="navy", label="MODCOD Efficiency (b/s/Hz)")
    axes[1].set_ylabel("Spectral Eff. (b/s/Hz)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Physical Layer MODCOD Selection")
    axes[1].grid(True, alpha=0.3)

    # 3. Packet Erasure Code Rate & Power Step
    axes[2].step(t_hrs, [r["code_rate"] for r in results], where="post", color="green", label="Erasure Code Rate (r = K/N)")
    ax2_twin = axes[2].twinx()
    ax2_twin.step(t_hrs, [r["power_step_db"] for r in results], where="post", color="orange", label="Power Step (dB)")
    axes[2].set_ylabel("Code Rate (K/N)", color="green")
    ax2_twin.set_ylabel("Power Step (dB)", color="orange")
    axes[2].set_title("Packet-Level Erasure Parity and RF Power Levers")
    axes[2].grid(True, alpha=0.3)

    # 4. Airtime Overhead O(a)
    axes[3].plot(t_hrs, [r["overhead"] for r in results], color="purple", label="Airtime Overhead O(a)")
    axes[3].set_ylabel("Overhead O(a)")
    axes[3].set_xlabel("Simulation Time (hours)")
    axes[3].set_title("Measurable Overhead O(a)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.suptitle("Step 7: Full RC-DAM Decision Dynamics & Joint Sizing under Fades")
    plt.tight_layout()
    plot_path = os.path.join(OUT, "step7_rcdam_decisions.png")
    plt.savefig(plot_path, dpi=130)
    print(f"Saved: {plot_path}")
    print("=== Step 7 Completed Successfully ===")


if __name__ == "__main__":
    main()
