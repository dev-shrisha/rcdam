"""Window statistic and feature windows for RC-DAM.

Implements RCDAM_Build_Guide.md Section 4.1:
- forward_window_max(a, H): forward maximum attenuation in the next H steps
- build_windows(att, dt, horizons_steps, stride_steps, hist_s=60.0):
    extracts non-leaking features from att[:m+1] and forward targets Z_H
"""
import numpy as np
from scipy.ndimage import maximum_filter1d


def forward_window_max(a: np.ndarray, H: int) -> np.ndarray:
    """Z[m] = max(a[m+1 : m+H+1]) for every index m (window of H steps AFTER m).
    The last H entries use the shorter available tail."""
    H = int(H)
    z = maximum_filter1d(a, size=H, origin=-(H // 2), mode="nearest")
    z = np.roll(z, -1)
    z[-1] = a[-1]
    return z


def build_windows(att: np.ndarray,
                  dt: float,
                  horizons_steps: list,
                  stride_steps: int,
                  hist_s: float = 60.0) -> dict:
    """One row per decision point m = index of the LAST MEASUREMENT.
    Features use only att[:m+1] (no leakage); targets Z[:, k] = max attenuation over (m, m + H_k].
    """
    n = len(att)
    L = int(round(hist_s / dt))
    max_h = max(horizons_steps)
    if n <= L + max_h + 2:
        raise ValueError(f"Series length {n} too short for hist_s={hist_s} and max_h={max_h}")

    idx = np.arange(L, n - max_h - 2, stride_steps)
    s5, s30 = int(round(5.0 / dt)), int(round(30.0 / dt))
    cur = att[idx]
    feats = {
        "idx": idx,
        "cur": cur,
        "slope5": (cur - att[idx - s5]) / 5.0,
        "slope30": (cur - att[idx - s30]) / 30.0,
        "max60": np.array([att[i - L:i + 1].max() for i in idx]),
    }
    feats["Z"] = np.stack([forward_window_max(att, H)[idx] for H in horizons_steps], axis=1)
    return feats
