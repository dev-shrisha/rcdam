"""Deep Multi-Quantile Forecaster with non-crossing heads.

Implements RCDAM_Build_Guide.md Section 4.7 & M2:
- pinball_loss(pred, target, taus)
- QuantileMLP with monotonic non-crossing quantiles via cumsum(softplus)
- Training, validation, and evaluation against persistence baseline
"""
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """Mean pinball (quantile) loss across batch, horizons, and quantiles.

    Args:
        pred: Predicted quantiles, shape [B, H, Q] or [B, Q]
        target: Ground truth target, shape [B, H, 1] or [B, 1]
        taus: Quantile levels, shape [Q] (float tensor)
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(1)
    if target.ndim == 2:
        target = target.unsqueeze(1)
    diff = target - pred  # [B, H, Q]
    loss = torch.maximum(taus * diff, (taus - 1.0) * diff)
    return loss.mean()


class QuantileMLP(nn.Module):
    """Predicts non-crossing quantiles of (Z_H - current attenuation) for every horizon.

    Architecture guarantees monotonicity:
    base = head output for tau_0
    increments = softplus(head output for tau_k) >= 0
    quantiles = base + cumsum(increments)
    """
    def __init__(self, d_in: int, n_h: int, n_q: int, hidden: int = 64):
        super().__init__()
        self.d_in = d_in
        self.n_h = n_h
        self.n_q = n_q
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_h * n_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor [B, d_in]
        Returns:
            Quantiles of delta Z_H [B, n_h, n_q]
        """
        z = self.head(self.body(x)).view(-1, self.n_h, self.n_q)
        base = z[..., :1]
        inc = F.softplus(z[..., 1:])
        return torch.cat([base, base + inc.cumsum(-1)], dim=-1)  # strictly non-crossing in quantile axis


class NeuralQuantileForecaster:
    """Convenience wrapper for training, predicting, and evaluating QuantileMLP."""

    def __init__(self,
                 horizons_steps: list,
                 taus: tuple = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999),
                 hidden: int = 64,
                 lr: float = 1e-3,
                 seed: int = 42):
        self.horizons_steps = list(horizons_steps)
        self.n_h = len(self.horizons_steps)
        self.taus = tuple(taus)
        self.n_q = len(self.taus)
        self.hidden = hidden
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)

        # Standard features: [cur, slope5, slope30, max60 - cur] (d_in = 4)
        self.d_in = 4
        self.model = QuantileMLP(self.d_in, self.n_h, self.n_q, hidden=hidden)
        self.tau_tensor = torch.tensor(self.taus, dtype=torch.float32)

    def _extract_x(self, feats: dict) -> np.ndarray:
        cur = feats["cur"]
        s5 = feats["slope5"]
        s30 = feats["slope30"]
        m60 = feats["max60"] - cur
        return np.column_stack([cur, s5, s30, m60])

    def fit(self,
            train_feats: dict,
            val_feats: Optional[dict] = None,
            epochs: int = 15,
            batch_size: int = 256,
            patience: int = 4):
        """Fit model using Adam optimizer and pinball loss."""
        X_tr = self._extract_x(train_feats)
        cur_tr = train_feats["cur"][:, None, None]
        Z_tr = train_feats["Z"][:, :, None]
        # Target is residual delta: Z_H - cur
        y_tr = Z_tr - cur_tr

        tX_tr = torch.tensor(X_tr, dtype=torch.float32)
        ty_tr = torch.tensor(y_tr, dtype=torch.float32)
        dataset = TensorDataset(tX_tr, ty_tr)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if val_feats is not None:
            X_val = self._extract_x(val_feats)
            cur_val = val_feats["cur"][:, None, None]
            Z_val = val_feats["Z"][:, :, None]
            y_val = Z_val - cur_val
            tX_val = torch.tensor(X_val, dtype=torch.float32)
            ty_val = torch.tensor(y_val, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_loss = float("inf")
        best_weights = None
        no_improve = 0

        self.model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = pinball_loss(pred, by, self.tau_tensor)
                loss.backward()
                optimizer.step()

            if val_feats is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(tX_val)
                    val_loss = float(pinball_loss(val_pred, ty_val, self.tau_tensor).item())
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        self.model.eval()
        return self

    def predict_delta(self, feats: dict) -> np.ndarray:
        """Predict quantile increments for all horizons: returns shape [N, n_h, n_q]."""
        self.model.eval()
        X = self._extract_x(feats)
        tX = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(tX).cpu().numpy()
        return out

    def predict(self, feats: dict, k: int, tau: float) -> np.ndarray:
        """Predict calibrated absolute attenuation bound: cur + delta_Z_H(k, tau)."""
        deltas = self.predict_delta(feats)
        q_idx = self.taus.index(tau)
        delta_q = deltas[:, k, q_idx]
        return feats["cur"] + delta_q

    def evaluate_pinball_loss(self, feats: dict) -> float:
        """Compute average pinball loss on the dataset."""
        self.model.eval()
        X = self._extract_x(feats)
        cur = feats["cur"][:, None, None]
        Z = feats["Z"][:, :, None]
        y = Z - cur
        tX = torch.tensor(X, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(tX)
            loss = pinball_loss(pred, ty, self.tau_tensor)
        return float(loss.item())

    def evaluate_persistence_pinball_loss(self, feats: dict) -> float:
        """Persistence baseline: predicts future delta = 0 for all quantiles and horizons."""
        cur = feats["cur"][:, None, None]
        Z = feats["Z"][:, :, None]
        y = Z - cur
        ty = torch.tensor(y, dtype=torch.float32)
        pred_zero = torch.zeros(len(feats["cur"]), self.n_h, self.n_q, dtype=torch.float32)
        loss = pinball_loss(pred_zero, ty, self.tau_tensor)
        return float(loss.item())
