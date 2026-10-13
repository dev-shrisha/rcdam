"""Unit tests for Step 6: Stratified Conformal Calibrator."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcdam.conformal import (
    StratifiedConformalCalibrator,
    classify_fade_state,
    STATE_CLEAR,
    STATE_LIGHT,
    STATE_MODERATE,
    STATE_DEEP,
)


def test_classify_fade_state():
    assert classify_fade_state(0.4) == STATE_CLEAR
    assert classify_fade_state(1.5) == STATE_LIGHT
    assert classify_fade_state(4.2) == STATE_MODERATE
    assert classify_fade_state(12.0) == STATE_DEEP
    print("Fade state classification verified.")


def test_delayed_feedback_fifo():
    calib = StratifiedConformalCalibrator(target_miss_rate=0.05, feedback_delay_steps=5)

    # Step 0 to 4: predict envelope
    for t in range(5):
        calib.predict_envelope(step_t=t, raw_upper_quantile=2.0 + t, recent_att_db=0.5)

    assert len(calib.pending_queue) == 5, f"Expected 5 pending, got {len(calib.pending_queue)}"

    # Receive feedback at step 4: step 4 - 0 = 4 < 5, nothing should pop
    calib.receive_feedback(current_step=4, observed_att_db=2.2)
    assert len(calib.pending_queue) == 5

    # Receive feedback at step 5: step 5 - 0 = 5 >= 5, step 0 pops!
    calib.receive_feedback(current_step=5, observed_att_db=2.3)
    assert len(calib.pending_queue) == 4
    assert len(calib.history_covered[STATE_CLEAR]) == 1

    print("Delayed feedback FIFO timing verified.")


def test_stratified_coverage_property():
    """Verify that conformal calibrator provides coverage close to target 1 - eps."""
    target_eps = 0.05  # 95% coverage target
    calib = StratifiedConformalCalibrator(
        target_miss_rate=target_eps,
        feedback_delay_steps=3,
        aci_gamma=0.01,
        initial_margin_db=1.0,
    )

    rng = np.random.default_rng(42)
    n_steps = 1500

    # Simulate noisy stream with distinct strata
    for t in range(n_steps):
        # Rotate through strata
        s = (t // 300) % 4
        base_att = [0.5, 2.0, 5.0, 10.0][s]
        # True channel has random noise
        y_true = base_att + rng.normal(0.0, 0.5)
        # Raw forecaster slightly underestimates (needs conformal correction)
        raw_pred = base_att - 0.2

        calib.predict_envelope(step_t=t, raw_upper_quantile=raw_pred, recent_att_db=base_att)
        calib.receive_feedback(current_step=t, observed_att_db=y_true)

    report = calib.get_coverage_report()
    for state_name, stats in report.items():
        if stats["n_samples"] > 100:
            cov = stats["empirical_coverage"]
            print(f"{state_name}: coverage = {cov * 100:.2f}% (target = {stats['target_coverage'] * 100:.1f}%)")
            # Coverage should be within reasonable margin of 95% (e.g. >= 90%)
            assert cov >= 0.88, f"Coverage too low for {state_name}: {cov}"

    print("Stratified coverage property verified.")


if __name__ == "__main__":
    test_classify_fade_state()
    test_delayed_feedback_fifo()
    test_stratified_coverage_property()
    print("All Step 6 unit tests passed.")
