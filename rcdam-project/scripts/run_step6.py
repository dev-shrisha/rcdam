"""Step 6: Stratified Conformal Calibrator Evaluation with Delayed Feedback.

Evaluates the online conformal calibrator combined with the trained quantile forecaster
on a held-out simulated channel with 0.54s RTT delayed feedback.
Outputs:
- Per-fade-state coverage table
- outputs/step6_conformal_coverage.png
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
from rcdam.attenuation import generate_attenuation_db
from rcdam.forecaster import QuantileForecaster
from rcdam.conformal import StratifiedConformalCalibrator, STATE_NAMES

OUT = os.path.join(PROJECT_ROOT, "outputs")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Step 6: Stratified Conformal Calibrator Evaluation ===\n")

    # 1. Load trained forecaster
    forecaster = QuantileForecaster(window_steps=30, horizon_steps=20)
    weights_path = os.path.join(OUT, "step5_forecaster_weights.pt")
    if os.path.exists(weights_path):
        forecaster.model.load_state_dict(torch.load(weights_path))
        print(f"Loaded trained forecaster weights from: {weights_path}")
    else:
        print("Trained weights not found; training lightweight model...")
        sc_tr = Scenario(seed=999, duration_h=6.0)
        att_tr = generate_attenuation_db(sc_tr)
        x_tr, y_tr = forecaster.prepare_dataset(att_tr, stride=10)
        forecaster.fit(x_tr, y_tr, epochs=8, batch_size=256)

    # 2. Setup held-out test scenario
    sc = Scenario(seed=303, duration_h=12.0)
    att = generate_attenuation_db(sc)
    n = len(att)
    w = forecaster.window_steps
    h = forecaster.horizon_steps
    delay_steps = sc.feedback_delay_steps  # round(0.54 / 0.1) = 5 steps

    target_epsilon = 0.05  # 95% nominal coverage
    calibrator = StratifiedConformalCalibrator(
        target_miss_rate=target_epsilon,
        feedback_delay_steps=delay_steps,
        aci_gamma=0.005,
    )

    print(f"Running online streaming simulation over {sc.duration_h:.0f}h test trace...")
    print(f"Delayed feedback arrives {delay_steps} steps ({delay_steps * sc.dt_s:.2f} s) late.\n")

    # Quantile index 6 corresponds to 0.95 quantile (DEFAULT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    q95_idx = 6

    # Batch compute raw forecasts for lightning-fast execution
    from numpy.lib.stride_tricks import sliding_window_view
    stride = 5
    t_vals = np.arange(w + delay_steps, n - h, stride)
    telemetry_ends = t_vals - delay_steps
    
    # Extract telemetry windows
    window_indices = telemetry_ends - w
    full_windows = sliding_window_view(att, w)[window_indices]
    print(f"Batch-predicting {len(t_vals)} forecast windows...")
    all_quantiles = forecaster.predict_quantiles(full_windows)
    raw_q95_arr = all_quantiles[:, q95_idx]
    recent_obs_arr = full_windows[:, -1]

    # Streaming evaluation with delayed feedback
    for i, t in enumerate(t_vals):
        # 1. Receiver feedback arrives for measurement at t - delay_steps
        obs_step = t - delay_steps
        true_horizon_peak = float(np.max(att[obs_step:obs_step + h]))
        calibrator.receive_feedback(current_step=t, observed_att_db=true_horizon_peak)

        # 2. Conformal calibrator updates envelope
        calibrator.predict_envelope(
            step_t=t,
            raw_upper_quantile=float(raw_q95_arr[i]),
            recent_att_db=float(recent_obs_arr[i]),
        )

    # 3. Report per-fade-state coverage
    report = calibrator.get_coverage_report()
    print(f"{'Fade Stratum':20s} {'Samples':>9s} {'Coverage':>12s} {'Miss Rate':>12s} {'Target Cov.':>12s}")
    print("-" * 68)
    for state_name, stats in report.items():
        cov = stats["empirical_coverage"]
        miss = stats["empirical_miss_rate"]
        print(f"{state_name:20s} {stats['n_samples']:9d} {cov * 100:11.2f}% {miss * 100:11.2f}% "
              f"{stats['target_coverage'] * 100:11.1f}%")

    # 4. Plot empirical coverage per fade stratum vs target
    fig, ax = plt.subplots(figsize=(8, 5))
    states = list(report.keys())
    coverages = [report[s]["empirical_coverage"] * 100 for s in states]
    targets = [report[s]["target_coverage"] * 100 for s in states]

    bars = ax.bar(states, coverages, color="steelblue", width=0.5, label="Empirical Coverage")
    ax.axhline(100.0 * (1.0 - target_epsilon), color="red", linestyle="--", linewidth=2,
               label=f"Nominal Target ({100*(1-target_epsilon):.0f}%)")

    for bar, cov in zip(bars, coverages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{cov:.1f}%",
                ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_title("Step 6: Empirical Coverage Stratified by Fade State (Held-Out Data)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(OUT, "step6_conformal_coverage.png")
    plt.savefig(plot_path, dpi=130)
    print(f"\nSaved coverage plot: {plot_path}")
    print("=== Step 6 Completed Successfully ===")


if __name__ == "__main__":
    main()
