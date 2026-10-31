"""Unit tests for groundtruth.py and variants.py."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rcdam.config import Scenario
from rcdam.link import ModcodTable
from rcdam.optimizer import Action, Message, ALPHAS
from rcdam.groundtruth import simulate_true_delivery
from rcdam.variants import (
    PolicyContext,
    policy_v0_static,
    policy_v1_fixed_margin,
    policy_v2_linear_predictive,
    policy_v6_full_rcdam,
    policy_oracle,
)


def test_simulate_true_delivery_clean_channel():
    sc = Scenario()
    table = ModcodTable()
    rng = np.random.default_rng(42)

    # Clean channel: 0 dB attenuation for 1000 steps
    att = np.zeros(1000)
    msg = Message(size_bits=10e6, priority="normal", deadline_s=2.0)
    # 32APSK 9/10
    a = Action(modcod=len(table.eff) - 1, power_db=0.0, parity_frac=0.0, defer_s=0.0)

    res = simulate_true_delivery(a, t_start_idx=10, att=att, msg=msg, table=table, sc=sc, rng=rng)
    assert res.delivered is True
    assert res.rounds_used == 1
    assert res.overhead >= 1.0
    assert res.latency_s < msg.deadline_s
    print(f"groundtruth.py: Clean channel test passed (Latency={res.latency_s:.3f}s, Overhead={res.overhead:.3f})")


def test_simulate_true_delivery_extreme_fade_fails():
    sc = Scenario()
    table = ModcodTable()
    rng = np.random.default_rng(42)

    # Extreme fade: 35 dB attenuation (complete outage)
    att = np.full(1000, 35.0)
    msg = Message(size_bits=10e6, priority="critical", deadline_s=2.0)
    a = Action(modcod=0, power_db=0.0, parity_frac=0.0, defer_s=0.0)

    res = simulate_true_delivery(a, t_start_idx=10, att=att, msg=msg, table=table, sc=sc, rng=rng)
    assert res.delivered is False
    assert res.reason in ("exhausted", "deadline_miss")
    print(f"groundtruth.py: Outage detection passed (Reason={res.reason})")


def test_policy_variants_execution():
    sc = Scenario()
    table = ModcodTable()
    msg = Message(size_bits=10e6, priority="normal", deadline_s=2.0, eps=0.01)

    class DummyForecaster:
        def predict(self, cur, k, tau):
            return cur + 1.0

    dummy_fc = DummyForecaster()
    dummy_cal = {30: {0.99: {0: 0.1, 1: 0.2, 2: 0.5, 3: 1.0}}}

    ctx = PolicyContext(
        arrival_idx=100,
        last_meas_idx=95,
        cur_att_db=2.5,
        features={"slope5": np.array([0.1])},
        true_future_att=np.array([2.5, 2.7, 3.0]),
        msg=msg,
        sc=sc,
        table=table,
        forecaster=dummy_fc,
        calibration_corr=dummy_cal,
    )

    d0 = policy_v0_static(ctx)
    d1 = policy_v1_fixed_margin(ctx)
    d2 = policy_v2_linear_predictive(ctx)
    d6 = policy_v6_full_rcdam(ctx)
    dor = policy_oracle(ctx)

    for name, d in [("V0", d0), ("V1", d1), ("V2", d2), ("V6", d6), ("Oracle", dor)]:
        assert d.action is not None
        assert d.overhead > 0
        assert d.reason in ("ok", "capacity", "deadline")
    print("variants.py: All policy variants executed cleanly.")


if __name__ == "__main__":
    test_simulate_true_delivery_clean_channel()
    test_simulate_true_delivery_extreme_fade_fails()
    test_policy_variants_execution()
    print("All groundtruth and variant tests passed.")
