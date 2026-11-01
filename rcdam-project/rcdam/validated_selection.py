"""Horizon helper and optimizer-selection validation utilities."""
import math
from .stats_utils import wilson_lower_bound


def horizon_steps_needed(d_steps, defer_s, span_s, dt):
    """Return the forecast horizon needed for feedback age, deferral and action span."""
    if dt <= 0:
        raise ValueError("dt must be greater than zero")
    if d_steps < 0 or defer_s < 0 or span_s < 0:
        raise ValueError("horizon inputs must be non-negative")
    return int(d_steps + math.ceil((defer_s + span_s) / dt))


def pick_horizon(grid_steps, needed):
    """Return the smallest available horizon >= needed, or None."""
    ok = [h for h in grid_steps if h >= needed]
    return min(ok) if ok else None


def select_validated(
    candidates, p_select, p_verify, target, conf=0.95,
    target_for_candidate=None,
):
    """Select the lowest-overhead candidate independently certified to meet target.

    Candidates must be sorted by increasing overhead.
    p_select(candidate) returns the selection-stage lower probability estimate.
    p_verify(candidate) returns (successes, trials) from independent verification.
    target_for_candidate may raise the verification target for candidate-specific
    risk allocation (for example, target + conformal alpha).
    """
    if not 0.0 < target < 1.0:
        raise ValueError("target must be strictly between 0 and 1")
    if not 0.0 < conf < 1.0:
        raise ValueError("conf must be strictly between 0 and 1")

    for candidate in candidates:
        if p_select(candidate) < target:
            continue
        k, n = p_verify(candidate)
        verify_target = (
            target_for_candidate(candidate)
            if target_for_candidate is not None else target
        )
        if not 0.0 < verify_target <= 1.0:
            raise ValueError("candidate verification target must be in (0, 1]")
        if wilson_lower_bound(k, n, conf) >= verify_target:
            return candidate
    return None
