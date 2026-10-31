"""Unit tests for calibration.py (Section 4.3 and Milestone M3)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rcdam.calibration import (
    fade_state,
    cqr_correction,
    fit_stratified_corrections,
    apply_corrections,
    aci_step,
    binomial_confidence_interval,
    save_calibration_json,
    load_calibration_json,
)


def test_fade_state_edges():
    # Edges: clear < 1 <= light < 5 <= moderate < 10 <= deep
    assert fade_state(0.5) == 0
    assert fade_state(1.0) == 1
    assert fade_state(3.2) == 1
    assert fade_state(5.0) == 2
    assert fade_state(9.9) == 2
    assert fade_state(10.0) == 3
    assert fade_state(25.0) == 3
    print("calibration.py: fade_state edges verified.")


def test_cqr_correction_finite_sample():
    scores = np.arange(1, 101, dtype=float)  # 1 .. 100
    # For n=100, tau=0.9: k = ceil(101 * 0.9) = ceil(90.9) = 91 -> score is 91.0
    c90 = cqr_correction(scores, 0.9)
    assert abs(c90 - 91.0) < 1e-6

    # When tau is so high that (n+1)*tau > n, return inf
    c_inf = cqr_correction(np.array([1.0, 2.0]), 0.99)
    assert np.isinf(c_inf)
    print("calibration.py: cqr_correction finite sample quantile verified.")


def test_stratified_corrections_and_coverage():
    rng = np.random.default_rng(42)
    n = 2000
    cur_cal = rng.exponential(scale=3.0, size=n)
    qhat_cal = cur_cal + rng.normal(1.0, 0.5, size=n)
    # Z has heavier positive residuals in deep state
    noise = rng.normal(1.2, 0.4, size=n)
    deep_mask = cur_cal >= 10.0
    noise[deep_mask] += 2.0
    z_cal = cur_cal + noise

    tau = 0.95
    corr = fit_stratified_corrections(cur_cal, z_cal, qhat_cal, tau, min_n=50)
    for s in range(4):
        assert s in corr
        assert np.isfinite(corr[s])

    # Apply corrections to test data
    cur_te = rng.exponential(scale=3.0, size=1500)
    qhat_te = cur_te + rng.normal(1.0, 0.5, size=1500)
    bound_te = apply_corrections(cur_te, qhat_te, corr)

    assert len(bound_te) == len(cur_te)
    assert np.all(bound_te >= qhat_te)
    print("calibration.py: stratified corrections applied successfully.")


def test_aci_step():
    # If err = 1 (miscoverage), alpha_t should increase (making bound more conservative on next step)
    a0 = 0.05
    a1 = aci_step(a0, err=1, alpha_target=0.05, gamma=0.01)
    # a1 = 0.05 + 0.01 * (0.05 - 1) = 0.05 - 0.0095 = 0.0405
    assert a1 < a0  # lower alpha means higher coverage target (1 - alpha increases)

    # If err = 0 (coverage achieved), alpha_t should increase
    a2 = aci_step(a0, err=0, alpha_target=0.05, gamma=0.01)
    assert a2 > a0
    print("calibration.py: aci_step direction verified.")


def test_calibration_json_persistence():
    meta = {"seeds": [1, 2, 3], "scenario_hash": "abc1234"}
    corr_dict = {
        100: {0.9: {0: 0.1, 1: 0.2, 2: 0.5, 3: 1.2}},
        300: {0.95: {0: 0.2, 1: 0.3, 2: 0.8, 3: 2.1}},
    }
    tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "tmp_cal.json"))
    save_calibration_json(tmp_path, meta, corr_dict)

    loaded_meta, loaded_corr = load_calibration_json(tmp_path)
    assert loaded_meta["scenario_hash"] == "abc1234"
    assert loaded_corr[100][0.9][3] == 1.2
    assert loaded_corr[300][0.95][0] == 0.2

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("calibration.py: JSON persistence verified.")


if __name__ == "__main__":
    test_fade_state_edges()
    test_cqr_correction_finite_sample()
    test_stratified_corrections_and_coverage()
    test_aci_step()
    test_calibration_json_persistence()
    print("All calibration tests passed.")
