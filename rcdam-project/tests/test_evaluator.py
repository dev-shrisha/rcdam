"""Unit tests for evaluator.py (Section 4.4 and M4 acceptance)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rcdam.evaluator import p_deliver_by_deadline


def test_evaluator_common_random_numbers_monotonicity():
    """Acceptance criterion: With common random numbers (fixed seed),
    success probability is non-decreasing in parity at fixed other settings."""
    K = 309
    p = 0.05
    frame_s = 0.0015
    rtt_s = 0.54
    one_way_s = 0.27
    defer_s = 0.0
    deadline_s = 2.0
    seed = 42

    parities = [0, 5, 15, 30, 50]
    p_successes = []
    for par in parities:
        p_ok, _ = p_deliver_by_deadline(
            K, par, p, frame_s, rtt_s, one_way_s, defer_s, deadline_s,
            n_mc=3000, max_rounds=8, seed=seed
        )
        p_successes.append(p_ok)

    diffs = np.diff(p_successes)
    assert np.all(diffs >= -1e-6), f"P(success) decreased with more parity: {p_successes}"
    print(f"evaluator.py: Monotonicity in parity verified: {p_successes}")


def test_evaluator_deadline_exceeded_by_transmission_time():
    """If defer_s + K*frame_s + one_way_s > deadline_s, must return p=0.0."""
    K = 100
    frame_s = 0.01  # 1.0 s needed
    rtt_s = 0.54
    one_way_s = 0.27
    defer_s = 0.0
    deadline_s = 0.5  # impossible to meet

    p_ok, sent = p_deliver_by_deadline(
        K, 0, 0.01, frame_s, rtt_s, one_way_s, defer_s, deadline_s, n_mc=1000
    )
    assert p_ok == 0.0
    print("evaluator.py: Deadline impossibility check verified.")


def test_evaluator_clean_channel():
    """At p=0, delivery probability must be 1.0 if time permits."""
    p_ok, sent = p_deliver_by_deadline(
        K=10, parity=0, p=0.0, frame_s=0.001, rtt_s=0.5, one_way_s=0.25,
        defer_s=0.0, deadline_s=1.0, n_mc=100
    )
    assert p_ok == 1.0
    assert sent == 10.0
    print("evaluator.py: Clean channel test verified.")


if __name__ == "__main__":
    test_evaluator_common_random_numbers_monotonicity()
    test_evaluator_deadline_exceeded_by_transmission_time()
    test_evaluator_clean_channel()
    print("All evaluator tests passed.")
