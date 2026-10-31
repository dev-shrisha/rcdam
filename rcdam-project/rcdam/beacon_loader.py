"""Loader for real beacon attenuation data (CSV) behind the same interface as
the synthetic attenuation generator.

Format: CSV file with at least two columns:
    - time_s (or timestamp_s): elapsed seconds since first measurement
    - att_db (or attenuation_db): measured attenuation in dB (positive = fade)

If no real data is available, the loader raises FileNotFoundError and documents
that all RC-DAM results are simulation-only.

NOTE: All current RC-DAM results are SIMULATION-ONLY. No real beacon
attenuation data has been used. This loader is provided for future use.
"""
import os
import numpy as np
import pandas as pd


SIMULATION_ONLY_WARNING = (
    "WARNING: All RC-DAM results are SIMULATION-ONLY. "
    "No real beacon attenuation data has been loaded or used."
)


def load_beacon_csv(path: str, dt_s: float = 0.1) -> np.ndarray:
    """Load a beacon attenuation CSV and resample to dt_s resolution.

    Args:
        path: path to CSV file with columns 'time_s' (or 'timestamp_s') and
              'att_db' (or 'attenuation_db').
        dt_s: target time step in seconds for resampling.

    Returns:
        1-D numpy array of attenuation values [dB], uniformly spaced at dt_s.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if required columns are missing.

    Usage (same interface as generate_attenuation_db):
        att = load_beacon_csv("data/beacon_2023.csv", dt_s=0.1)
        # Now pass att to run_single_seed() etc.
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

    # Sort by time
    order = np.argsort(t)
    t, a = t[order], a[order]

    # Clip negative attenuations (should not occur physically)
    a = np.maximum(a, 0.0)

    # Resample to uniform dt_s grid
    t_start, t_end = t[0], t[-1]
    t_uniform = np.arange(t_start, t_end, dt_s)
    a_uniform = np.interp(t_uniform, t, a)

    return a_uniform


def check_real_data_available(data_dir: str = "data") -> bool:
    """Check if any real beacon data CSV files are present."""
    if not os.path.isdir(data_dir):
        return False
    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "beacon" in f.lower()]
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
