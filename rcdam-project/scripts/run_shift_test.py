"""M3 Distribution Shift Experiment: Stratified Conformal Calibration under Channel Shift.

Implements M3 requirements:
- Base scenario for training and calibration
- Shifted scenario for testing (heavier and more frequent rain events)
- Comparison across:
  1. Raw empirical quantile forecast (uncalibrated)
  2. Pooled conformal calibration
  3. Stratified conformal calibration
  4. Adaptive Conformal Inference (ACI) tracking
- Reports coverage per fade state with sample counts and Clopper-Pearson binomial CIs
- Saves outputs/shift_test.csv and outputs/shift_test.png
"""
import argparse
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcdam.config import Scenario
from rcdam.attenuation import generate_attenuation_db
from rcdam.experiment import step2_scenario
from rcdam.windows import build_windows
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.calibration import (
    fade_state,
    cqr_correction,
    fit_stratified_corrections,
    apply_corrections,
    aci_step,
    binomial_confidence_interval,
    STATE_NAMES,
)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def make_dataset(seeds, scenario, horizons, stride_steps):
    all_dicts = []
    for s in seeds:
        sc = replace(scenario, seed=s)
        att = generate_attenuation_db(sc)
        w = build_windows(att, sc.dt_s, horizons, stride_steps=stride_steps)
        all_dicts.append(w)
    return {k: np.concatenate([d[k] for d in all_dicts]) for k in all_dicts[0]}


def main():
    parser = argparse.ArgumentParser(description="M3 Distribution Shift Experiment")
    parser.add_argument("--hours", type=float, default=24.0, help="Hours per seed")
    parser.add_argument("--target-tau", type=float, default=0.99, help="Target coverage level")
    parser.add_argument("--horizon-s", type=float, default=30.0, help="Forecast horizon in seconds")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dt = 0.1
    H_steps = int(round(args.horizon_s / dt))
    horizons = [H_steps]
    tau = args.target_tau

    print("==================================================")
    print("   MILESTONE M3: CONFORMAL CALIBRATION UNDER SHIFT")
    print("==================================================")
    print(f"Target coverage = {tau*100:.1f}%, Horizon = {args.horizon_s:.1f}s ({H_steps} steps)")

    # 1. Base channel scenario (Train & Calibration)
    base_sc = replace(step2_scenario(), duration_h=args.hours)
    # 2. Shifted channel scenario (Test): deeper rain events and higher frequency
    shift_sc = replace(step2_scenario(), duration_h=args.hours, event_peak_ln_mean=2.0, events_per_day=6.0)

    print("\n[1] Generating base train/cal windows...")
    train_seeds = list(range(1, 11))
    cal_seeds = list(range(11, 19))
    test_seeds = list(range(19, 27))

    t0 = time.time()
    train_data = make_dataset(train_seeds, base_sc, horizons, stride_steps=150)
    cal_data = make_dataset(cal_seeds, base_sc, horizons, stride_steps=300)
    print(f"    Base data generated in {time.time()-t0:.1f}s: Train={len(train_data['cur'])}, Cal={len(cal_data['cur'])}")

    print("\n[2] Generating shifted test windows...")
    t1 = time.time()
    test_data = make_dataset(test_seeds, shift_sc, horizons, stride_steps=150)
    print(f"    Shifted test data generated in {time.time()-t1:.1f}s: Test={len(test_data['cur'])}")

    # 3. Fit empirical forecaster on base train data
    fc = EmpiricalQuantileForecaster(taus=(tau,)).fit(train_data["cur"], train_data["Z"])

    # 4. Calibration on base calibration data
    q_cal = fc.predict(cal_data["cur"], 0, tau)
    # Pooled correction
    pooled_corr = cqr_correction(cal_data["Z"][:, 0] - q_cal, tau)
    # Stratified corrections
    strat_corr = fit_stratified_corrections(cal_data["cur"], cal_data["Z"][:, 0], q_cal, tau, min_n=100)

    print(f"\n[3] Fitted corrections on Base Calibration Set:")
    print(f"    Pooled correction: +{pooled_corr:.2f} dB")
    for s, c in strat_corr.items():
        print(f"    {STATE_NAMES[s]}: +{c:.2f} dB")

    # 5. Evaluate on Shifted Test Set
    cur_te = test_data["cur"]
    z_true_te = test_data["Z"][:, 0]
    st_te = fade_state(cur_te)

    # 5a. Raw uncalibrated
    raw_bound = fc.predict(cur_te, 0, tau)
    # 5b. Pooled conformal
    pooled_bound = raw_bound + pooled_corr
    # 5c. Stratified conformal
    strat_bound = apply_corrections(cur_te, raw_bound, strat_corr)

    # 5d. Online ACI tracking (chronological processing)
    alpha_target = 1.0 - tau
    alpha_aci = {s: alpha_target for s in range(4)}
    aci_bound = np.empty_like(raw_bound)
    # Chronological pass
    for i in range(len(cur_te)):
        s = st_te[i]
        # Current correction using adaptive alpha
        cur_alpha = alpha_aci[s]
        # Look up nearest quantile correction for this alpha
        eff_tau = float(np.clip(1.0 - cur_alpha, 0.5, 0.999))
        c_i = cqr_correction(cal_data["Z"][fade_state(cal_data["cur"]) == s, 0] - q_cal[fade_state(cal_data["cur"]) == s], eff_tau)
        if not np.isfinite(c_i):
            c_i = strat_corr[s]
        bound_i = raw_bound[i] + c_i
        aci_bound[i] = bound_i
        # Outcome observation (after delay): update ACI
        err = 1 if z_true_te[i] > bound_i else 0
        alpha_aci[s] = aci_step(alpha_aci[s], err, alpha_target, gamma=0.005)

    methods = [
        ("Raw Empirical Forecaster", raw_bound),
        ("Pooled Conformal", pooled_bound),
        ("Stratified Conformal", strat_bound),
        ("Adaptive Conformal (ACI)", aci_bound),
    ]

    print("\n[4] Empirical Coverage on Shifted Test Channel:")
    print(f"{'Method':28s} {'Clear (<1dB)':>14s} {'Light (1-5dB)':>14s} {'Moderate (5-10dB)':>17s} {'Deep (>=10dB)':>14s} {'Overall':>12s}")

    results_table = []
    for name, b_arr in methods:
        cov_overall = float((z_true_te <= b_arr).mean())
        row = {"method": name, "overall_cov": cov_overall, "target_tau": tau}
        state_strs = []
        for s in range(4):
            mask = st_te == s
            n_s = int(mask.sum())
            k_s = int((z_true_te[mask] <= b_arr[mask]).sum())
            cov_s = k_s / n_s if n_s > 0 else float("nan")
            ci_low, ci_high = binomial_confidence_interval(k_s, n_s)
            row[f"cov_state_{s}"] = cov_s
            row[f"ci_low_state_{s}"] = ci_low
            row[f"ci_high_state_{s}"] = ci_high
            row[f"n_state_{s}"] = n_s
            state_strs.append(f"{cov_s*100:.1f}% (n={n_s})")
        results_table.append(row)
        print(f"{name:28s} {state_strs[0]:>14s} {state_strs[1]:>14s} {state_strs[2]:>17s} {state_strs[3]:>14s} {cov_overall*100:>11.1f}%")

    # Save CSV
    csv_path = os.path.join(OUT, "shift_test.csv")
    import csv
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_table[0].keys())
        writer.writeheader()
        writer.writerows(results_table)
    print(f"\nSaved shift test results to: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_indices = np.arange(4)
    bar_width = 0.2
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]

    for idx, (name, b_arr) in enumerate(methods):
        r = results_table[idx]
        y_vals = [r[f"cov_state_{s}"] * 100 for s in range(4)]
        y_errs = [
            [(r[f"cov_state_{s}"] - r[f"ci_low_state_{s}"]) * 100 for s in range(4)],
            [(r[f"ci_high_state_{s}"] - r[f"cov_state_{s}"]) * 100 for s in range(4)],
        ]
        ax.bar(x_indices + (idx - 1.5) * bar_width, y_vals, bar_width, yerr=y_errs,
               capsize=3, label=name, color=colors[idx], alpha=0.85)

    ax.axhline(tau * 100, color="black", linestyle="--", linewidth=1.5, label=f"Target ({tau*100:.0f}%)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel("Empirical Coverage (%) with 95% Binomial CI")
    ax.set_title(f"Conformal Risk Control under Channel Shift (Target = {tau*100:.1f}%, H = {args.horizon_s:.1f}s)")
    ax.set_ylim(85.0, 101.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    png_path = os.path.join(OUT, "shift_test.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved shift test figure to: {png_path}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
