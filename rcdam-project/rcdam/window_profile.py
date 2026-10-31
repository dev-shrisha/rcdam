"""Window-profile statistics for Stage B (RC-DAM Advancement Plan Part 6).

Calibrate a small profile:
  1. Max Z (peak attenuation over the window)
  2. Time (fraction) spent above threshold T1 (e.g., 3 dB)
  3. Time (fraction) spent above threshold T2 (e.g., 8 dB)

Split the miscoverage budget across these statistics with a union bound.
Build a worst-case profile and evaluate actions across it.

This enables interleaving and deferral to show measurable benefit:
  - Bursty fades: interleaving disperses frame losses; deferral can skip a short outage.
  - Constant fades: interleaving gives no benefit (all frames see the same channel).

Use after Tasks 1-6 pass.
"""
import numpy as np
from typing import List


def compute_window_profile(
    att: np.ndarray,
    dt_s: float,
    threshold1_db: float = 3.0,
    threshold2_db: float = 8.0,
) -> dict:
    """Compute the three-statistic profile for a window of attenuation values.

    Args:
        att: 1-D array of attenuation [dB] over the window
        dt_s: simulation time step [s]
        threshold1_db: light-fade threshold [dB]
        threshold2_db: deep-fade threshold [dB]

    Returns:
        dict with keys: max_db, frac_above_t1, frac_above_t2,
                        time_above_t1_s, time_above_t2_s
    """
    att = np.asarray(att, dtype=float)
    return {
        "max_db": float(np.max(att)),
        "frac_above_t1": float(np.mean(att >= threshold1_db)),
        "frac_above_t2": float(np.mean(att >= threshold2_db)),
        "time_above_t1_s": float(np.sum(att >= threshold1_db) * dt_s),
        "time_above_t2_s": float(np.sum(att >= threshold2_db) * dt_s),
    }


def calibrate_profile_bounds(
    profiles: List[dict],
    alpha_budget: float = 0.05,
) -> dict:
    """Calibrate upper bounds on each profile statistic using a union bound.

    Splits the miscoverage budget alpha_budget evenly across 3 statistics.
    Returns the (1 - alpha_i) empirical quantile for each statistic,
    where alpha_i = alpha_budget / 3.

    Args:
        profiles: list of profile dicts from compute_window_profile
        alpha_budget: total miscoverage budget (default 5%)

    Returns:
        dict with keys: max_db_bound, frac_t1_bound, frac_t2_bound,
                        n_profiles, alpha_budget, alpha_per_stat
    """
    alpha_i = alpha_budget / 3.0
    q = 1.0 - alpha_i

    max_dbs = np.array([p["max_db"] for p in profiles])
    frac_t1s = np.array([p["frac_above_t1"] for p in profiles])
    frac_t2s = np.array([p["frac_above_t2"] for p in profiles])

    return {
        "max_db_bound": float(np.quantile(max_dbs, q)),
        "frac_t1_bound": float(np.quantile(frac_t1s, q)),
        "frac_t2_bound": float(np.quantile(frac_t2s, q)),
        "n_profiles": len(profiles),
        "alpha_budget": alpha_budget,
        "alpha_per_stat": alpha_i,
    }


def build_worst_case_profile(
    max_db_bound: float,
    frac_t1_bound: float,
    frac_t2_bound: float,
    window_steps: int,
    dt_s: float,
    threshold1_db: float = 3.0,
    threshold2_db: float = 8.0,
) -> np.ndarray:
    """Build a worst-case attenuation profile from the calibrated bounds.

    The profile is a step outage: the channel is at max_db for the fraction
    of time above T2, then at threshold1_db for the remaining fraction above T1,
    then 0. All bad time is concentrated at the start (pessimistic).

    Interleaving spreads frames across this window, reducing concentrated losses.
    Deferral can skip a short outage if the deadline allows.

    Returns:
        1-D numpy array of length window_steps
    """
    att = np.zeros(window_steps)
    n_deep = min(window_steps, int(np.ceil(frac_t2_bound * window_steps)))
    n_light = min(window_steps, int(np.ceil(frac_t1_bound * window_steps)))

    att[:n_deep] = max_db_bound
    att[n_deep:n_light] = threshold1_db
    return att


def evaluate_interleaving_benefit(
    worst_case_att: np.ndarray,
    frame_att_threshold_db: float,
    n_frames: int,
    interleave_depth: int = 1,
) -> dict:
    """Evaluate how interleaving changes frame loss on a worst-case profile.

    interleave_depth=1: frames sent sequentially (no interleaving; packed at start)
    interleave_depth=k: frames spread evenly across the full window

    Args:
        worst_case_att: worst-case attenuation profile [dB]
        frame_att_threshold_db: attenuation above which a frame is lost
        n_frames: number of frames to transmit
        interleave_depth: 1 = no interleaving, k > 1 = spread over k windows

    Returns:
        dict with frames_lost, frames_total, loss_rate, interleave_depth
    """
    n_steps = len(worst_case_att)
    frames_lost = 0
    for i in range(n_frames):
        if interleave_depth > 1:
            # Spread frames evenly across the full window
            frac = i / max(1, n_frames - 1)
            step = int(round(frac * (n_steps - 1)))
        else:
            # Sequential: pack frames tightly at the start
            step = min(i * max(1, n_steps // n_frames), n_steps - 1)
        if worst_case_att[step] >= frame_att_threshold_db:
            frames_lost += 1

    return {
        "frames_lost": frames_lost,
        "frames_total": n_frames,
        "loss_rate": frames_lost / max(1, n_frames),
        "interleave_depth": interleave_depth,
    }

