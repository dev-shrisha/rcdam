"""M2 Forecaster Training and Evaluation Runner.

Implements M2 requirements:
- Horizon grid [10, 20, 30, 50, 100, 200, 300, 600]
- Independent seeds for train, calibration, test
- Fit empirical baseline (EmpiricalQuantileForecaster)
- Fit QuantileMLP neural network
- Compare against persistence baseline
- Pinball loss and raw coverage tables
- Saves outputs/forecaster.pt, outputs/forecaster_eval.csv, outputs/forecaster_eval.png
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
import torch

from rcdam.config import Scenario
from rcdam.attenuation import generate_attenuation_db
from rcdam.experiment import step2_scenario
from rcdam.windows import build_windows
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.forecaster_nn import NeuralQuantileForecaster

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
HORIZONS = [10, 20, 30, 50, 100, 200, 300, 600]
TAUS = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


def make_dataset(seeds, scenario, horizons, stride_steps=200):
    all_dicts = []
    for s in seeds:
        sc = replace(scenario, seed=s)
        att = generate_attenuation_db(sc)
        w = build_windows(att, sc.dt_s, horizons, stride_steps=stride_steps)
        all_dicts.append(w)
    return {k: np.concatenate([d[k] for d in all_dicts]) for k in all_dicts[0]}


def main():
    parser = argparse.ArgumentParser(description="M2 Forecaster Training & Benchmark")
    parser.add_argument("--hours", type=float, default=12.0, help="Hours per seed")
    parser.add_argument("--stride", type=int, default=150, help="Window stride in steps")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sc0 = replace(step2_scenario(), duration_h=args.hours)

    print("==================================================")
    print("   MILESTONE M2: MULTI-QUANTILE FORECASTER TRAINING")
    print("==================================================")
    print(f"Generating datasets with {args.hours}h per seed, stride={args.stride}...")

    # Independent seeds for train, cal, test
    train_seeds = list(range(1, 7))
    cal_seeds = list(range(7, 10))
    test_seeds = list(range(10, 14))

    t0 = time.time()
    train_data = make_dataset(train_seeds, sc0, HORIZONS, stride_steps=args.stride)
    cal_data = make_dataset(cal_seeds, sc0, HORIZONS, stride_steps=args.stride)
    test_data = make_dataset(test_seeds, sc0, HORIZONS, stride_steps=args.stride)
    print(f"Dataset extracted in {time.time()-t0:.1f}s: Train={len(train_data['cur'])}, Cal={len(cal_data['cur'])}, Test={len(test_data['cur'])}")

    # 1. Fit Empirical Quantile Forecaster
    print("\n[1] Fitting Empirical Quantile Forecaster baseline...")
    emp_fc = EmpiricalQuantileForecaster(taus=TAUS).fit(train_data["cur"], train_data["Z"])

    # 2. Fit Neural Quantile Forecaster (QuantileMLP)
    print("\n[2] Training Neural Quantile Forecaster (QuantileMLP)...")
    nn_fc = NeuralQuantileForecaster(horizons_steps=HORIZONS, taus=TAUS, lr=1e-3)
    nn_fc.fit(train_data, val_feats=cal_data, epochs=args.epochs, batch_size=256)

    # 3. Evaluate Pinball Losses on Held-Out Test Set
    persistence_loss = nn_fc.evaluate_persistence_pinball_loss(test_data)
    nn_loss = nn_fc.evaluate_pinball_loss(test_data)

    print("\n[3] Held-out Test Set Loss Comparison:")
    print(f"    Persistence Baseline Loss: {persistence_loss:.4f}")
    print(f"    Neural Model (QuantileMLP) Loss: {nn_loss:.4f}")

    assert nn_loss < persistence_loss, f"Forecaster ({nn_loss:.4f}) must beat persistence ({persistence_loss:.4f})"
    print("    -> Acceptance passed: Neural model beats persistence!")

    # 4. Save Weights
    # NOTE: The canonical weights file used by the M-track evaluation harness is
    #       outputs/forecaster.pt (this file). The legacy file outputs/step5_forecaster_weights.pt
    #       is from the old step-based track and is no longer actively used.
    weights_path = os.path.join(OUT, "forecaster.pt")
    torch.save(nn_fc.model.state_dict(), weights_path)
    print(f"    Saved weights to: {weights_path}")

    # 5. Raw Coverage Analysis per fade state and tau
    # Edge states: Clear < 1, Light 1-5, Moderate 5-10, Deep >= 10
    state_edges = (1.0, 5.0, 10.0)
    st_test = np.digitize(test_data["cur"], state_edges)
    state_names = ["Clear (<1dB)", "Light (1-5dB)", "Moderate (5-10dB)", "Deep (>=10dB)"]

    eval_rows = []
    # Test for horizon k=6 (30s horizon)
    k_eval = 6
    H_eval_s = HORIZONS[k_eval] * sc0.dt_s

    print(f"\n[4] Raw Coverage at Horizon {H_eval_s:.1f}s (k={k_eval}) on Test Set:")
    for tau in [0.5, 0.8, 0.9, 0.95, 0.99]:
        q_pred_emp = emp_fc.predict(test_data["cur"], k_eval, tau)
        q_pred_nn = nn_fc.predict(test_data, k_eval, tau)

        emp_cov_total = float((test_data["Z"][:, k_eval] <= q_pred_emp).mean())
        nn_cov_total = float((test_data["Z"][:, k_eval] <= q_pred_nn).mean())

        row = {
            "horizon_s": H_eval_s,
            "target_tau": tau,
            "emp_cov_overall": emp_cov_total,
            "nn_cov_overall": nn_cov_total,
        }
        for s in range(4):
            mask_s = st_test == s
            n_s = int(mask_s.sum())
            if n_s > 0:
                row[f"emp_cov_state_{s}"] = float((test_data["Z"][mask_s, k_eval] <= q_pred_emp[mask_s]).mean())
                row[f"nn_cov_state_{s}"] = float((test_data["Z"][mask_s, k_eval] <= q_pred_nn[mask_s]).mean())
                row[f"count_state_{s}"] = n_s
            else:
                row[f"emp_cov_state_{s}"] = float("nan")
                row[f"nn_cov_state_{s}"] = float("nan")
                row[f"count_state_{s}"] = 0
        eval_rows.append(row)
        print(f"    Target tau={tau:.2f} -> Overall: Emp={emp_cov_total:.3f}, NN={nn_cov_total:.3f} | Deep State: NN={row['nn_cov_state_3']:.3f} (n={row['count_state_3']})")

    # Save CSV
    csv_path = os.path.join(OUT, "forecaster_eval.csv")
    import csv
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"\nSaved forecaster evaluation to: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    target_taus = [r["target_tau"] for r in eval_rows]
    ax.plot(target_taus, target_taus, "k--", label="Ideal Diagonal")
    ax.plot(target_taus, [r["nn_cov_overall"] for r in eval_rows], "b-o", label="Neural Forecaster (Overall)")
    ax.plot(target_taus, [r["emp_cov_overall"] for r in eval_rows], "g-s", label="Empirical Forecaster (Overall)")
    ax.plot(target_taus, [r["nn_cov_state_3"] for r in eval_rows], "r-^", label="Neural Forecaster (Deep State)")
    ax.set_xlabel("Target Quantile (tau)")
    ax.set_ylabel("Empirical Coverage P(Z <= q_hat)")
    ax.set_title(f"Multi-Quantile Forecaster Calibration Curve (H = {H_eval_s:.1f}s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(OUT, "forecaster_eval.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved evaluation plot to: {png_path}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
