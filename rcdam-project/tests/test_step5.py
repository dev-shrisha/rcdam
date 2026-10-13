"""Unit tests for Step 5: Quantile Forecaster."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcdam.forecaster import QuantileForecaster, MonotonicQuantileNet, pinball_loss, DEFAULT_QUANTILES
from rcdam.config import Scenario
from rcdam.attenuation import generate_attenuation_db


def test_pinball_loss():
    q = torch.tensor([0.1, 0.5, 0.9])
    y_true = torch.tensor([[10.0]])

    # When y_pred = y_true, loss must be exactly 0
    y_pred_exact = torch.tensor([[10.0, 10.0, 10.0]])
    loss_exact = pinball_loss(y_pred_exact, y_true, q)
    assert abs(loss_exact.item()) < 1e-6, f"Expected 0 loss, got {loss_exact.item()}"

    # Asymmetry test: at tau=0.9, underpredicting by 1 (pred=9) has loss 0.9 * 1 = 0.9.
    # Overpredicting by 1 (pred=11) has loss (1 - 0.9) * 1 = 0.1.
    y_under = torch.tensor([[9.0]])
    y_over = torch.tensor([[11.0]])
    loss_under = pinball_loss(y_under, y_true, torch.tensor([0.9]))
    loss_over = pinball_loss(y_over, y_true, torch.tensor([0.9]))
    assert abs(loss_under.item() - 0.9) < 1e-5
    assert abs(loss_over.item() - 0.1) < 1e-5
    print("Pinball loss asymmetry verified.")


def test_monotonic_quantiles():
    net = MonotonicQuantileNet(in_features=25, num_quantiles=len(DEFAULT_QUANTILES))
    x = torch.randn(32, 25)
    out = net(x).detach().numpy()

    # Assert shape
    assert out.shape == (32, len(DEFAULT_QUANTILES))

    # Assert strict monotonicity across quantile columns
    diffs = np.diff(out, axis=1)
    assert np.all(diffs >= -1e-6), "Quantiles must never decrease across tau"
    print("Monotonic non-crossing quantiles verified.")


def test_forecaster_training_and_persistence_comparison():
    sc = Scenario(seed=42, duration_h=12.0)
    att = generate_attenuation_db(sc)

    forecaster = QuantileForecaster(window_steps=20, horizon_steps=15, lr=2e-3)
    X, y = forecaster.prepare_dataset(att, stride=5)

    n_train = int(len(X) * 0.7)
    x_train, y_train = X[:n_train], y[:n_train]
    x_test, y_test = X[n_train:], y[n_train:]

    persistence_loss = forecaster.evaluate_persistence_pinball_loss(x_test, y_test)
    forecaster.fit(x_train, y_train, epochs=12, batch_size=128)
    model_loss = forecaster.evaluate_pinball_loss(x_test, y_test)

    print(f"Test Pinball Loss: Model = {model_loss:.4f} vs Persistence Baseline = {persistence_loss:.4f}")
    assert model_loss < persistence_loss, f"Forecaster ({model_loss:.4f}) must beat persistence ({persistence_loss:.4f})"
    print("Forecaster successfully beats persistence baseline.")


if __name__ == "__main__":
    test_pinball_loss()
    test_monotonic_quantiles()
    test_forecaster_training_and_persistence_comparison()
    print("All Step 5 unit tests passed.")
