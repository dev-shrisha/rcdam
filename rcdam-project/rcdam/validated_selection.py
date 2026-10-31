"""Horizon helper and Optimizer‑curse guard (validated selection)."""
import numpy as np
from .stats_utils import wilson_interval


def horizon_steps_needed(d_steps, defer_s, span_s, dt):
    """
    Forecast horizon an action needs: feedback age + deferral + the time span
    the action occupies (transmission, interleaving, ARQ rounds you must cover).
    The calibrated bound must be taken at THIS horizon, not at a shorter one.
    """
    return int(d_steps + np.ceil((defer_s + span_s) / dt))


def pick_horizon(grid_steps, needed):
    """Smallest grid horizon >= needed, or None (infeasible)."""
    ok = [h for h in grid_steps if h >= needed]
    return min(ok) if ok else None


def select_validated(candidates, p_select, p_verify, target, conf=0.95):
    """
    Optimizer's‑curse guard.
    `candidates` are sorted by increasing overhead.
    p_select(c) -> estimated success probability from the selection draws.
    p_verify(c) -> (k_success, n) from INDEPENDENT draws.

    Walk up the overhead order; accept the first candidate that passes the
    selection test AND whose verification success proportion has a Wilson LOWER
    bound >= target.
    """
    for c in candidates:
        if p_select(c) < target:
            continue
        k, n = p_verify(c)
        if wilson_interval(k, n, conf)[0] >= target:
            return c
    return None
