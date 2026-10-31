"""Tests for window-profile statistics (Task 8)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from rcdam.window_profile import (
    compute_window_profile,
    calibrate_profile_bounds,
    build_worst_case_profile,
    evaluate_interleaving_benefit,
)


def test_compute_window_profile_basic():
    att = np.array([0.0, 1.0, 5.0, 10.0, 3.0, 1.0])
    p = compute_window_profile(att, dt_s=0.1)
    assert p["max_db"] == 10.0, f"max_db={p['max_db']}"
    # Values >= 3.0: indices 2 (5.0), 3 (10.0), 4 (3.0) -> 3/6
    assert abs(p["frac_above_t1"] - 3 / 6) < 0.01, f"frac_t1={p['frac_above_t1']}"
    # Values >= 8.0: index 3 (10.0) -> 1/6
    assert abs(p["frac_above_t2"] - 1 / 6) < 0.01, f"frac_t2={p['frac_above_t2']}"
    print(f"  window_profile: {p}  OK")


def test_calibrate_profile_bounds():
    rng = np.random.default_rng(0)
    profiles = [compute_window_profile(rng.exponential(5, 100), 0.1) for _ in range(200)]
    bounds = calibrate_profile_bounds(profiles, alpha_budget=0.05)
    assert "max_db_bound" in bounds
    assert bounds["n_profiles"] == 200
    assert abs(bounds["alpha_per_stat"] - 0.05 / 3) < 1e-10
    print(f"  profile_bounds: max_db={bounds['max_db_bound']:.2f}  OK")


def test_build_worst_case_profile():
    att = build_worst_case_profile(
        max_db_bound=15.0, frac_t1_bound=0.4, frac_t2_bound=0.2,
        window_steps=100, dt_s=0.1,
    )
    assert len(att) == 100
    assert att[0] == 15.0  # deep fade at start
    assert att[-1] == 0.0  # clear at end
    print(f"  worst_case_profile: max={att.max()}, mean={att.mean():.2f}  OK")


def test_interleaving_helps_on_bursty_fade():
    """Vacuous test note (Defect 1.6): Under a single global MDS erasure code across the window,
    frame loss count is invariant to frame order, so interleaving alone cannot change total lost frames.
    A true interleaving benefit requires limited-span block codes where dispersing burst losses across
    separate codewords prevents any single codeword from exceeding its recovery capability (Phase 5).
    Here we verify evaluation interface parity across interleaver depths without asserting false gains.
    """
    n_steps = 100
    att = np.zeros(n_steps)
    att[:30] = 15.0
    threshold = 10.0
    n_frames = 50

    r1 = evaluate_interleaving_benefit(att, threshold, n_frames, interleave_depth=1)
    r4 = evaluate_interleaving_benefit(att, threshold, n_frames, interleave_depth=4)
    print(f"  bursty: no_il={r1['loss_rate']:.3f}, il4={r4['loss_rate']:.3f} (MDS global count invariant)")
    assert r1["loss_rate"] == r4["loss_rate"], "Under global MDS count, loss count must match"
    print("  interleaving_helps_on_bursty_fade (verified global MDS count invariance; limited-span block codes in Phase 5) OK")


def test_interleaving_neutral_on_constant_fade():
    n_steps = 100
    att = np.full(n_steps, 12.0)
    threshold = 10.0
    n_frames = 50

    r1 = evaluate_interleaving_benefit(att, threshold, n_frames, interleave_depth=1)
    r4 = evaluate_interleaving_benefit(att, threshold, n_frames, interleave_depth=4)
    print(f"  constant: no_il={r1['loss_rate']:.3f}, il4={r4['loss_rate']:.3f}")
    assert abs(r4["loss_rate"] - r1["loss_rate"]) <= 0.02, (
        "Interleaving changed loss on constant fade"
    )
    print("  interleaving_neutral_on_constant_fade  OK")


if __name__ == "__main__":
    print("=== test_window_profile.py ===")
    test_compute_window_profile_basic()
    test_calibrate_profile_bounds()
    test_build_worst_case_profile()
    test_interleaving_helps_on_bursty_fade()
    test_interleaving_neutral_on_constant_fade()
    print("=== All window profile tests passed ===")

