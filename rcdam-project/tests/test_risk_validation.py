"""Unit tests for Phase 4: Risk Control, LTT Certification, and Reliability Diagnostics."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from rcdam.risk_control import ltt_fixed_sequence, crc_lambda
from rcdam.optimizer import Decision, Action
from rcdam.variants import policy_v6_infeasible_rule, PolicyContext, Message
from rcdam.config import Scenario
from rcdam.link import ModcodTable


def test_ltt_rejects_overconfident():
    """LTT fixed-sequence test rejects candidate settings when empirical risk exceeds target epsilon."""
    n = 1000
    m = 5
    # Simulate a setting with 4% miss rate (above target eps=0.01)
    rng = np.random.default_rng(42)
    miss = rng.random((n, m)) < 0.04
    feasible = np.ones((n, m), dtype=bool)

    cert_idx, table = ltt_fixed_sequence(miss, feasible, eps=0.01, delta=0.05)
    assert cert_idx is None, f"Overconfident setting (4% > 1%) must NOT be certified, got cert_idx={cert_idx}"
    assert all(not r["certified"] for r in table), "Every row in table must fail certification"
    print("test_ltt_rejects_overconfident PASSED")


def test_ltt_certifies_safe():
    """LTT fixed-sequence test certifies settings when empirical risk is safely below epsilon."""
    n = 1000
    m = 4
    # Simulate very safe setting: 0.1% miss rate (far below target eps=0.01)
    rng = np.random.default_rng(42)
    miss = rng.random((n, m)) < 0.001
    feasible = np.ones((n, m), dtype=bool)

    cert_idx, table = ltt_fixed_sequence(miss, feasible, eps=0.01, delta=0.05)
    assert cert_idx is not None, "Safe setting must be certified"
    assert table[-1]["certified"], "Least conservative qualified setting must be certified"
    print(f"test_ltt_certifies_safe PASSED: cert_idx={cert_idx}")


def test_crc_lambda_bound():
    """Conformal Risk Control returns a valid index meeting the risk upper bound."""
    n = 200
    rng = np.random.default_rng(42)
    losses = np.column_stack([
        rng.uniform(0.04, 0.08, n),
        rng.uniform(0.01, 0.02, n),
        rng.uniform(0.00, 0.005, n),
    ])
    idx = crc_lambda(losses, alpha=0.03, B=1.0)
    assert idx is not None, "CRC should certify settings with risk below alpha"
    assert idx in [1, 2], f"Setting 1 or 2 should be selected, got {idx}"
    print("test_crc_lambda_bound PASSED")


def test_decision_log_present():
    """Verifies that policy_v6_infeasible_rule populates decision_log on all decisions."""
    sc = Scenario(dt_s=0.1, duration_h=1.0)
    table = ModcodTable()
    msg = Message(size_bits=20e6, priority="normal", deadline_s=2.0, eps=0.01)

    from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
    fc_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "forecaster_baseline.json")
    if os.path.exists(fc_path):
        fc = EmpiricalQuantileForecaster.load(fc_path)
    else:
        fc = EmpiricalQuantileForecaster().fit(np.linspace(0, 30, 100), np.column_stack([np.linspace(0.1, 35, 100) for _ in range(8)]))

    ctx = PolicyContext(
        arrival_idx=100,
        last_meas_idx=95,
        cur_att_db=22.0,  # heavy attenuation ensures infeasible
        features={"cur": np.array([22.0]), "slope5": np.array([0.0]), "slope30": np.array([0.0]), "max60": np.array([22.0])},
        true_future_att=np.full(30, 22.0),
        msg=msg,
        sc=sc,
        table=table,
        forecaster=fc,
        calibration_corr={},
    )

    for rule in ["R1", "R2", "R3", "R4"]:
        dec = policy_v6_infeasible_rule(ctx, rule=rule)
        assert dec.decision_log is not None, f"Rule {rule} must set decision_log"
        assert "rule" in dec.decision_log
        assert "outcome" in dec.decision_log
        assert dec.decision_log["rule"] == rule

    print("test_decision_log_present PASSED")


if __name__ == "__main__":
    test_ltt_rejects_overconfident()
    test_ltt_certifies_safe()
    test_crc_lambda_bound()
    test_decision_log_present()
    print("All risk validation tests passed!")
