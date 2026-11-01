"""Loader for real beacon attenuation data (CSV) behind the same interface as
the synthetic attenuation generator.

Format: CSV file with at least two columns:
    - time_s (or timestamp_s): elapsed seconds since first measurement
    - att_db (or attenuation_db): measured attenuation in dB (positive = fade)

Includes:
- Time series gap detection and warning
- Outlier filtering and clipping
- BeaconDataset class with seamless Scenario integration
- Strict documentation and warnings that results without real data are SIMULATION-ONLY.
"""
import os
import warnings
from dataclasses import replace
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


SIMULATION_ONLY_WARNING = (
    "WARNING: All RC-DAM results are SIMULATION-ONLY. "
    "No real beacon attenuation data has been loaded or used."
)


def load_beacon_csv(path: str, dt_s: float = 0.1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a beacon attenuation CSV, validate, clean, and resample to dt_s resolution.

    Args:
        path: path to CSV file with columns 'time_s' (or 'timestamp_s') and
              'att_db' (or 'attenuation_db').
        dt_s: target time step in seconds for resampling.

    Returns:
        (attenuation_array, metadata_dict)
        - attenuation_array: 1-D numpy array of attenuation values [dB], uniformly spaced at dt_s.
        - metadata_dict: statistics on duration, gaps, outliers, and fade events.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Beacon data file not found: {path}\n"
            f"{SIMULATION_ONLY_WARNING}"
        )

    df = pd.read_csv(path)

    # Resolve column names
    time_col = None
    for c in ["time_s", "timestamp_s", "time", "t"]:
        if c in df.columns:
            time_col = c
            break
    att_col = None
    for c in ["att_db", "attenuation_db", "attenuation", "att"]:
        if c in df.columns:
            att_col = c
            break

    if time_col is None or att_col is None:
        raise ValueError(
            f"Beacon CSV must have columns 'time_s' and 'att_db'. "
            f"Found: {list(df.columns)}"
        )

    t = df[time_col].values.astype(float)
    a = df[att_col].values.astype(float)

    # Sort chronologically
    order = np.argsort(t)
    t, a = t[order], a[order]

    # Quality check 1: Detect gaps in telemetry
    t_diff = np.diff(t)
    gap_threshold = 10.0 * dt_s
    n_gaps = int(np.sum(t_diff > gap_threshold))
    max_gap_s = float(np.max(t_diff)) if len(t_diff) > 0 else 0.0
    if n_gaps > 0:
        warnings.warn(f"Beacon CSV contains {n_gaps} gaps > {gap_threshold:.1f}s. Max gap: {max_gap_s:.2f}s.")

    # Quality check 2: Outliers
    n_negative = int(np.sum(a < -0.5))
    n_extreme = int(np.sum(a > 50.0))
    if n_negative > 0:
        warnings.warn(f"Beacon CSV contains {n_negative} negative attenuation values (clipped to 0 dB).")
    if n_extreme > 0:
        warnings.warn(f"Beacon CSV contains {n_extreme} extreme attenuation values > 50 dB.")

    # Clip physically impossible negative attenuation
    a = np.clip(a, 0.0, 60.0)

    # Resample to uniform dt_s grid
    t_start, t_end = t[0], t[-1]
    t_uniform = np.arange(t_start, t_end, dt_s)
    a_uniform = np.interp(t_uniform, t, a)

    metadata = {
        "path": path,
        "raw_points": len(t),
        "duration_h": (t_end - t_start) / 3600.0,
        "dt_s": dt_s,
        "n_samples": len(a_uniform),
        "mean_att_db": float(np.mean(a_uniform)),
        "max_att_db": float(np.max(a_uniform)),
        "frac_above_3db": float(np.mean(a_uniform >= 3.0)),
        "frac_above_8db": float(np.mean(a_uniform >= 8.0)),
        "n_gaps": n_gaps,
        "max_gap_s": max_gap_s,
    }

    return a_uniform, metadata


class BeaconDataset:
    """Wrapper that exposes real beacon data behind the same interface as Scenario attenuation generator."""
    def __init__(self, path: str, dt_s: float = 0.1):
        self.att, self.metadata = load_beacon_csv(path, dt_s=dt_s)
        self.dt_s = dt_s
        self.duration_h = self.metadata["duration_h"]

    def to_scenario(self, **overrides):
        """Builds a Scenario configuration matching this beacon dataset's duration and dt."""
        from rcdam.config import Scenario
        sc = Scenario(
            dt_s=self.dt_s,
            duration_h=self.duration_h,
            rain_model="beacon_real",
        )
        return replace(sc, **overrides) if overrides else sc

    def generate_attenuation_db(self, sc=None) -> np.ndarray:
        """Returns the attenuation trajectory matching the requested scenario steps."""
        if sc is not None and sc.n_steps < len(self.att):
            return self.att[:sc.n_steps]
        return self.att


def check_real_data_available(data_dir: str = "data") -> bool:
    """Check if any real beacon data CSV files are present."""
    if not os.path.isdir(data_dir):
        return False
    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "beacon" in f.lower() and "sample" not in f.lower()]
    return len(csvs) > 0


def print_simulation_only_notice():
    """Print a clear notice that results are simulation-only."""
    print()
    print("=" * 60)
    print(SIMULATION_ONLY_WARNING)
    print("Real beacon data loader: rcdam/beacon_loader.py")
    print("To use real data: load_beacon_csv('path/to/beacon.csv')")
    print("=" * 60)
    print()
