"""M1 Data module: real and synthetic attenuation loading, splitting, and statistics.

Implements M1 requirements from RCDAM_Build_Guide.md:
- load_attenuation(path, dt_s=0.1, clear_sky_offset_db=None):
    resample to dt, mark gaps/NaNs, convert to differential attenuation.
- split_by_time(att, fractions=(0.5, 0.25, 0.25)):
    chronological train/calibration/test split (never random windows).
- compute_fade_statistics(att, dt_s=0.1):
    time above 1, 5, 10, 15, 20 dB; fade-slope percentiles.
"""
import os
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.interpolate import interp1d


def load_attenuation(path: str,
                     dt_s: float = 0.1,
                     clear_sky_offset_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load attenuation time series from file (CSV or whitespace-delimited).

    Args:
        path: Filepath containing [time_s, att_db] or [att_db].
        dt_s: Target simulation time step in seconds (default 0.1s).
        clear_sky_offset_db: If provided, subtracts clear sky offset to give
            differential rain attenuation. If None, estimated from the lower 5th percentile.

    Returns:
        (t_grid, att_grid): Uniformly sampled time and attenuation in dB.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Attenuation file not found: {path}")

    # Support CSV or plain text, skipping header if present
    try:
        data = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            try:
                data = np.loadtxt(path, skiprows=1)
            except Exception:
                data = np.loadtxt(path)

    if data.ndim == 1:
        # Single column: assume already at target dt_s
        att = data
        t = np.arange(len(att)) * dt_s
    elif data.shape[1] >= 2:
        t = data[:, 0]
        att = data[:, 1]
    else:
        raise ValueError(f"Unrecognized data shape in {path}: {data.shape}")

    # Remove NaNs / mark gaps
    valid = np.isfinite(att) & np.isfinite(t)
    if not np.all(valid):
        t_valid = t[valid]
        att_valid = att[valid]
        if len(t_valid) < 2:
            raise ValueError("Too few valid samples in attenuation file.")
        # Linear interpolation over gaps
        interpolator = interp1d(t_valid, att_valid, kind="linear", fill_value="extrapolate")
        att = interpolator(t)

    # Resample to uniform dt_s grid
    t_start = t[0]
    t_end = t[-1]
    # Check if already uniformly sampled with step dt_s
    expected_n = int(round((t_end - t_start) / dt_s)) + 1
    if len(t) == expected_n and np.allclose(np.diff(t), dt_s, atol=1e-5):
        t_grid = t
        att_grid = att
    else:
        t_grid = np.linspace(t_start, t_end, expected_n)
        f = interp1d(t, att, kind="linear", fill_value="extrapolate")
        att_grid = f(t_grid)

    # Differential attenuation: remove clear-sky baseline
    if clear_sky_offset_db is None:
        # Estimate clear-sky baseline from 5th percentile of quietest samples
        clear_sky_offset_db = float(np.percentile(att_grid, 5.0))

    diff_att = np.maximum(att_grid - clear_sky_offset_db, 0.0)
    return t_grid - t_grid[0], diff_att


def split_by_time(att: np.ndarray,
                  fractions: Tuple[float, float, float] = (0.5, 0.25, 0.25)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split attenuation into train, calibration, and test portions.

    Strictly chronological: never uses random window shuffling to avoid data leakage.
    """
    assert len(fractions) == 3, "fractions must be (train, cal, test)"
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1.0"

    n = len(att)
    n_train = int(n * fractions[0])
    n_cal = int(n * (fractions[0] + fractions[1]))

    train = att[:n_train]
    cal = att[n_train:n_cal]
    test = att[n_cal:]
    return train, cal, test


def compute_fade_statistics(att: np.ndarray, dt_s: float = 0.1) -> Dict[str, float]:
    """Compute standard propagation statistics: time above thresholds and fade slopes.

    Acceptance criteria for M1:
    - Time percentage above 1, 5, 10, 15, 20 dB.
    - Fade-slope (dB/s) percentiles (50th, 90th, 99th).
    """
    n = len(att)
    total_time_h = (n * dt_s) / 3600.0

    thresholds = [1.0, 5.0, 10.0, 15.0, 20.0]
    stats_dict = {
        "duration_h": float(total_time_h),
        "total_samples": int(n),
    }

    for thr in thresholds:
        frac = float(np.mean(att >= thr))
        stats_dict[f"pct_above_{int(thr)}db"] = float(frac * 100.0)
        stats_dict[f"hours_above_{int(thr)}db"] = float(frac * total_time_h)

    # Fade slope: 1-second central difference
    step1s = max(1, int(round(1.0 / dt_s)))
    if n > 2 * step1s:
        slope = np.abs(att[step1s:] - att[:-step1s]) / 1.0
        # Condition on rain events (att > 1.0 dB)
        rain_mask = att[step1s:] > 1.0
        rain_slope = slope[rain_mask] if np.any(rain_mask) else slope
        stats_dict["slope_p50_db_per_s"] = float(np.percentile(rain_slope, 50.0))
        stats_dict["slope_p90_db_per_s"] = float(np.percentile(rain_slope, 90.0))
        stats_dict["slope_p99_db_per_s"] = float(np.percentile(rain_slope, 99.0))
    else:
        stats_dict["slope_p50_db_per_s"] = 0.0
        stats_dict["slope_p90_db_per_s"] = 0.0
        stats_dict["slope_p99_db_per_s"] = 0.0

    return stats_dict


def save_synthetic_dataset(path: str, att: np.ndarray, dt_s: float = 0.1):
    """Save an attenuation array as a CSV dataset."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    t = np.arange(len(att)) * dt_s
    data = np.column_stack([t, att])
    np.savetxt(path, data, delimiter=",", header="time_s,att_db", comments="")
