"""MODCOD selection policies. Each returns an index array (one MODCOD index per time step).

If no MODCOD is decodable, the policy falls back to the most robust one (index 0), so the
link keeps transmitting and ARQ takes care of the losses.
"""
import numpy as np

from .link import ModcodTable


def _delayed(x, d):
    out = np.empty_like(x)
    out[d:] = x[: len(x) - d]
    out[:d] = x[0]
    return out


def static_policy(table: ModcodTable, clear_sky_esn0_db, design_margin_db, n):
    """No adaptation: one MODCOD chosen for clear sky minus a design margin."""
    k = int(table.select(np.array([clear_sky_esn0_db - design_margin_db]))[0])
    return np.full(n, max(k, 0), dtype=np.int64)


def margin_policy(table: ModcodTable, esn0_db, margin_db, delay_steps):
    """Fixed-margin ACM: use the Es/N0 measured one RTT ago (persistence), minus a margin."""
    m = _delayed(esn0_db, delay_steps)
    return np.maximum(table.select(m - margin_db), 0)


def predictive_policy(table: ModcodTable, esn0_db, margin_db, delay_steps, window_steps):
    """Predictive ACM with a POINT forecast: linearly extrapolate a falling Es/N0 trend over
    the feedback age (rising trends are not extrapolated), then subtract a margin."""
    m = _delayed(esn0_db, delay_steps)
    slope = np.zeros_like(m)
    slope[window_steps:] = (m[window_steps:] - m[:-window_steps]) / window_steps   # dB per step
    pred = m + np.minimum(slope, 0.0) * delay_steps
    return np.maximum(table.select(pred - margin_db), 0)


def oracle_policy(table: ModcodTable, esn0_db, margin_db):
    """Upper bound for MODCOD adaptation: knows the true Es/N0 at transmission time (no delay)."""
    return np.maximum(table.select(esn0_db - margin_db), 0)
