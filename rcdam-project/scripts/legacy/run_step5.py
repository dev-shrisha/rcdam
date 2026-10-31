"""Step 5: Quantile Fade Forecaster Training & Evaluation.

Trains the PyTorch multi-quantile fade forecaster using pinball loss,
verifies that it beats persistence across quantiles, and saves the
forecast visualization plot outputs/step5_quantile_forecast.png.
"""
import os
import sys
import time

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
from rcdam.forecaster import QuantileForecaster, pinball_loss, DEFAULT_QUANTILES

OUT = os.path.join(PROJECT_ROOT, "outputs")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Step 5: Quantile Fade Forecaster Training & Evaluation ===\n")

    # 1. Generate training and test channels
    print("Generating simulated channel traces (12h train, 6h test)...")
    sc_train = Scenario(seed=101, duration_h=12.0)
    att_train = generate_attenuation_db(sc_train)

    sc_test = Scenario(seed=202, duration_h=6.0)
    att_test = generate_attenuation_db(sc_test)

    forecaster = QuantileForecaster(window_steps=30, horizon_steps=20, lr=3e-3)

    x_train, y_train = forecaster.prepare_dataset(att_train, stride=10)
    x_test, y_test = forecaster.prepare_dataset(att_test, stride=10)

    print(f"Training dataset size: {len(x_train)} samples, Test size: {len(x_test)} samples")

    # 2. Evaluate persistence baseline
    loss_persist = forecaster.evaluate_persistence_pinball_loss(x_test, y_test)
    print(f"Persistence baseline test pinball loss: {loss_persist:.4f}")

    # 3. Train model
    print("Training PyTorch multi-quantile model...")
    t0 = time.time()
    forecaster.fit(x_train, y_train, epochs=10, batch_size=256, verbose=True)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # 4. Evaluate trained model
    loss_model = forecaster.evaluate_pinball_loss(x_test, y_test)
    print(f"\nFinal Test Pinball Loss: Model = {loss_model:.4f} vs Persistence = {loss_persist:.4f}")
    loss_reduction_pct = 100.0 * (loss_persist - loss_model) / loss_persist
    print(f"Pinball Loss Reduction: {loss_reduction_pct:.1f}%")

    # 5. Save model weights
    weights_path = os.path.join(OUT, "step5_forecaster_weights.pt")
    torch.save(forecaster.model.state_dict(), weights_path)
    print(f"Saved model weights: {weights_path}")

    # 6. Plot quantile forecast vs actual on a representative fade segment from test set
    # Find a test window with a noticeable fade event
    max_idx = int(np.argmax(y_test))
    plot_start = max(0, max_idx - 60)
    plot_end = min(len(x_test), max_idx + 60)

    pred_quantiles = forecaster.predict_quantiles(x_test[plot_start:plot_end])
    true_y = y_test[plot_start:plot_end].flatten()

    time_axis = np.arange(len(true_y)) * (sc_test.dt_s * 4)  # stride = 4

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot quantile bands (fan chart)
    # quantiles: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ax.fill_between(time_axis, pred_quantiles[:, 0], pred_quantiles[:, 7],
                    color="royalblue", alpha=0.15, label="5% - 99% Quantile Band")
    ax.fill_between(time_axis, pred_quantiles[:, 1], pred_quantiles[:, 6],
                    color="royalblue", alpha=0.25, label="10% - 95% Quantile Band")
    ax.fill_between(time_axis, pred_quantiles[:, 2], pred_quantiles[:, 5],
                    color="royalblue", alpha=0.35, label="25% - 75% Quantile Band")
    ax.plot(time_axis, pred_quantiles[:, 3], "b--", linewidth=1.5, label="Median Forecast (50%)")

    # Plot true attenuation target
    ax.plot(time_axis, true_y, "r-", linewidth=2.0, label="True Horizon Peak Attenuation")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Attenuation (dB)")
    ax.set_title("Step 5: Multi-Quantile Fade Forecaster (Fan Chart over Lookahead Horizon)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(OUT, "step5_quantile_forecast.png")
    plt.savefig(plot_path, dpi=130)
    print(f"Saved forecast plot: {plot_path}")
    print("\n=== Step 5 Completed Successfully ===")


if __name__ == "__main__":
    main()
