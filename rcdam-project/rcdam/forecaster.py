"""Module 5: PyTorch Multi-Quantile Fade Forecaster.

Predicts quantiles of future channel attenuation over horizon H >= RTT + T_tx
using pinball (quantile) loss. Enforces non-crossing quantiles via monotonic
parameterization (base + softplus steps).
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Compute pinball (quantile) loss across all quantiles.

    y_pred: shape (batch_size, num_quantiles)
    y_true: shape (batch_size, 1) or (batch_size,)
    quantiles: shape (num_quantiles,)
    """
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(1)
    err = y_true - y_pred  # (batch_size, num_quantiles)
    loss = torch.maximum(quantiles * err, (quantiles - 1.0) * err)
    return torch.mean(loss)


class MonotonicQuantileNet(nn.Module):
    """Deep neural network predicting strictly monotonic quantiles."""
    def __init__(self, in_features: int, num_quantiles: int = len(DEFAULT_QUANTILES), hidden_dim: int = 64):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # First output is base (lowest quantile), remaining are non-negative increments
        self.head = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        raw_out = self.head(h)
        base = raw_out[:, 0:1]
        increments = torch.nn.functional.softplus(raw_out[:, 1:])
        quantiles = torch.cat([base, increments], dim=1)
        # Cumulative sum guarantees monotonic increase: q_0 <= q_1 <= ... <= q_K
        return torch.cumsum(quantiles, dim=1)


class QuantileForecaster:
    """Wrapper for training and running the multi-quantile forecaster."""
    def __init__(self,
                 window_steps: int = 30,
                 horizon_steps: int = 20,
                 quantiles: List[float] = None,
                 hidden_dim: int = 64,
                 lr: float = 1e-3):
        self.window_steps = window_steps
        self.horizon_steps = horizon_steps
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.q_tensor = torch.tensor(self.quantiles, dtype=torch.float32)

        # Features: raw window values (window_steps) + summary stats (mean, std, slope, min, max) = window_steps + 5
        self.in_features = window_steps + 5
        self.model = MonotonicQuantileNet(self.in_features, len(self.quantiles), hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Compute engineered features from a telemetry history window."""
        # window: shape (N, window_steps) or (window_steps,)
        if window.ndim == 1:
            window = window.reshape(1, -1)
        mean = np.mean(window, axis=1, keepdims=True)
        std = np.std(window, axis=1, keepdims=True)
        slope = (window[:, -1:] - window[:, 0:1]) / max(window.shape[1], 1)
        w_min = np.min(window, axis=1, keepdims=True)
        w_max = np.max(window, axis=1, keepdims=True)
        return np.concatenate([window, mean, std, slope, w_min, w_max], axis=1)

    def prepare_dataset(self, attenuation_series: np.ndarray,
                        stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Extract (X, y) pairs from continuous attenuation time series using vectorized sliding windows."""
        from numpy.lib.stride_tricks import sliding_window_view
        w = self.window_steps
        h = self.horizon_steps
        total_len = w + h

        all_windows = sliding_window_view(attenuation_series, total_len)[::stride]
        x_raw = np.ascontiguousarray(all_windows[:, :w], dtype=np.float32)
        y_targets = np.ascontiguousarray(np.max(all_windows[:, w:], axis=1, keepdims=True), dtype=np.float32)
        features = self.extract_features(x_raw)
        return features, y_targets

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 15, batch_size: int = 128, verbose: bool = False):
        """Train model with pinball loss."""
        self.model.train()
        n = len(x_train)
        x_t = torch.tensor(x_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        q_t = self.q_tensor

        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            num_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                bx, by = x_t[idx], y_t[idx]

                self.optimizer.zero_grad()
                pred = self.model(bx)
                loss = pinball_loss(pred, by, q_t)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Pinball Loss: {total_loss / max(num_batches, 1):.4f}")

    def predict_quantiles(self, x: np.ndarray) -> np.ndarray:
        """Predict quantiles for raw window(s) or pre-extracted feature matrix. Returns (N, num_quantiles)."""
        self.model.eval()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] == self.in_features:
            feat = x
        elif x.shape[1] == self.window_steps:
            feat = self.extract_features(x)
        else:
            raise ValueError(f"Expected input with {self.window_steps} or {self.in_features} features, got {x.shape[1]}")

        with torch.no_grad():
            feat_t = torch.tensor(feat, dtype=torch.float32)
            out = self.model(feat_t)
        return out.cpu().numpy()

    def evaluate_pinball_loss(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute average pinball loss on test set."""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(x_test, dtype=torch.float32))
            loss = pinball_loss(pred, torch.tensor(y_test, dtype=torch.float32), self.q_tensor)
        return float(loss.item())

    def evaluate_persistence_pinball_loss(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute pinball loss for persistence baseline (predicting last observed telemetry)."""
        # Last observed raw attenuation is at index window_steps - 1
        last_obs = x_test[:, self.window_steps - 1:self.window_steps]
        # Repeat across all quantiles
        pred = np.repeat(last_obs, len(self.quantiles), axis=1)
        loss = pinball_loss(torch.tensor(pred, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32),
                            self.q_tensor)
        return float(loss.item())
