"""Empirical Quantile Forecaster baseline.

Implements RCDAM_Build_Guide.md Section 4.2:
- Empirical quantiles of residual (Z_H - current_att) binned by current attenuation
- Fallback logic for sparse bins
- Fit on TRAIN seeds only
"""
import numpy as np


class EmpiricalQuantileForecaster:
    """Non-neural baseline: quantiles of (Z_H - current attenuation) within bins of the current attenuation.
    Fit on TRAIN seeds only. Build this first; only add a neural model if it beats it."""

    def __init__(self,
                 edges=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0),
                 taus=(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999),
                 min_n=20):
        self.edges = np.asarray(edges, dtype=float)
        self.taus = tuple(taus)
        self.min_n = min_n
        self.q = {}

    def fit(self, cur: np.ndarray, Z: np.ndarray):
        cur = np.asarray(cur, dtype=float)
        Z = np.asarray(Z, dtype=float)
        b = np.digitize(cur, self.edges)
        for bi in range(len(self.edges) + 1):
            sel = b == bi
            n_sel = int(sel.sum())
            for k in range(Z.shape[1]):
                if n_sel >= self.min_n:
                    r = Z[sel, k] - cur[sel]
                    for t in self.taus:
                        self.q[(bi, k, t)] = float(np.quantile(r, t))
                else:
                    for t in self.taus:
                        self.q[(bi, k, t)] = None
        return self

    def predict(self, cur: np.ndarray, k: int, tau: float) -> np.ndarray:
        """Upper-quantile forecast of Z_H(k) for each row; sparse bins fall back to the nearest lower bin."""
        cur = np.asarray(cur, dtype=float)
        b = np.digitize(cur, self.edges)
        out = np.empty(len(cur))
        for i, bi in enumerate(b):
            j = bi
            while self.q.get((j, k, tau)) is None and j > 0:
                j -= 1
            val = self.q.get((j, k, tau))
            out[i] = cur[i] + (val if val is not None else 0.0)
        return out

    def save(self, path: str):
        """Persist forecaster parameters and binned quantiles to JSON."""
        import json
        data = {
            "edges": self.edges.tolist(),
            "taus": list(self.taus),
            "min_n": self.min_n,
            "q": {f"{bi}_{k}_{t}": v for (bi, k, t), v in self.q.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load persisted forecaster parameters and binned quantiles from JSON."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(edges=data["edges"], taus=data["taus"], min_n=data["min_n"])
        obj.q = {}
        for key_str, v in data["q"].items():
            bi, k, t = key_str.split("_")
            obj.q[(int(bi), int(k), float(t))] = v
        return obj
