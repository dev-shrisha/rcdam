"""Stratified Split-Conformal Risk Control and Adaptive Conformal Inference.

Implements RCDAM_Build_Guide.md Section 4.3 & M3:
- STATE_EDGES_DB = (1.0, 5.0, 10.0)
- fade_state(cur_db)
- cqr_correction(scores, tau)
- fit_stratified_corrections(cur_cal, z_cal, qhat_cal, tau, min_n=300)
- apply_corrections(cur, qhat, corr)
- aci_step(alpha_t, err, alpha_target, gamma=0.005)
- Save / load calibration tables to/from JSON
- Binomial confidence intervals for coverage validation
"""
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

STATE_EDGES_DB = (1.0, 5.0, 10.0)  # clear < 1 <= light < 5 <= moderate < 10 <= deep
STATE_NAMES = ["Clear (<1dB)", "Light (1-5dB)", "Moderate (5-10dB)", "Deep (>=10dB)"]


def fade_state(cur_db: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """Return stratum index for given attenuation value(s)."""
    return np.digitize(cur_db, STATE_EDGES_DB)


def cqr_correction(scores: np.ndarray, tau: float) -> float:
    """Finite-sample split-conformal quantile of the scores (Z - q_hat).

    Returns +inf if too few points, which must be treated as 'no valid bound'.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * tau))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def fit_stratified_corrections(cur_cal: np.ndarray,
                               z_cal: np.ndarray,
                               qhat_cal: np.ndarray,
                               tau: float,
                               min_n: int = 300) -> Dict[int, float]:
    """One correction per fade state. A state with fewer than min_n calibration windows
    borrows the next lower state's pool. Thin calibration windows beforehand (stride >= horizon)."""
    cur_cal = np.asarray(cur_cal, dtype=float)
    z_cal = np.asarray(z_cal, dtype=float)
    qhat_cal = np.asarray(qhat_cal, dtype=float)

    st = fade_state(cur_cal)
    scores = z_cal - qhat_cal
    corr = {}
    for s in range(len(STATE_EDGES_DB) + 1):
        pool = scores[st == s]
        j = s
        while len(pool) < min_n and j > 0:
            j -= 1
            pool = scores[(st <= s) & (st >= j)]
        corr[s] = cqr_correction(pool, tau)
    return corr


def apply_corrections(cur: np.ndarray, qhat: np.ndarray, corr: Dict[int, float]) -> np.ndarray:
    """Apply stratified conformal shift: returns bound = qhat + corr[stratum]."""
    st = fade_state(cur)
    offsets = np.array([corr[s] for s in st], dtype=float)
    return qhat + offsets


def aci_step(alpha_t: float, err: int, alpha_target: float, gamma: float = 0.005) -> float:
    """Adaptive conformal inference update.

    err = 1 if window maximum exceeded the bound issued at level alpha_t, else 0.
    Call when outcome becomes available (approx H + RTT after issue).
    """
    updated = alpha_t + gamma * (alpha_target - err)
    return float(np.clip(updated, 0.0001, 0.9999))


def binomial_confidence_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval for empirical coverage."""
    if n == 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else stats.beta.ppf(alpha / 2.0, k, n - k + 1)
    high = 1.0 if k == n else stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return float(low), float(high)


def save_calibration_json(path: str,
                          metadata: dict,
                          corrections_by_horizon_and_tau: dict):
    """Persist calibration corrections and metadata to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Convert integer keys to strings for JSON serialization
    serializable = {}
    for h_key, tau_dict in corrections_by_horizon_and_tau.items():
        serializable[str(h_key)] = {}
        for tau_key, corr_dict in tau_dict.items():
            serializable[str(h_key)][str(tau_key)] = {
                str(s): (float(v) if np.isfinite(v) else "inf")
                for s, v in corr_dict.items()
            }

    data = {
        "metadata": metadata,
        "corrections": serializable,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_calibration_json(path: str) -> Tuple[dict, dict]:
    """Load calibration JSON; returns (metadata, corrections)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    corrections = {}
    for h_str, tau_dict in data["corrections"].items():
        h = int(h_str)
        corrections[h] = {}
        for tau_str, corr_dict in tau_dict.items():
            tau = float(tau_str)
            corrections[h][tau] = {
                int(s): (float("inf") if v == "inf" else float(v))
                for s, v in corr_dict.items()
            }
    return data["metadata"], corrections
