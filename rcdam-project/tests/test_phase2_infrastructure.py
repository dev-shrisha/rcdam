"""Unit and regression tests for Phase 2: Evaluation Infrastructure.

Verifies:
1. Deterministic oracle feasibility
2. Config stress presets
3. Per-message log schema and column matching
4. Diagnostics miss taxonomy
5. Exact paired McNemar calculations
6. Cluster bootstrap interval calculations
"""
import pytest
import numpy as np
import pandas as pd

from rcdam.config import Scenario, get_preset, PRESET_REGISTRY
from rcdam.link import ModcodTable
from rcdam.optimizer import Message, Action
from rcdam.variants import PolicyContext, policy_oracle
from rcdam.diagnostics import LOG_COLUMNS, classify_misses, infeasibility_report
from rcdam.stats_utils import mcnemar_exact, wilson_interval
from scripts.run_h1_h4 import cluster_bootstrap_mean


def test_2_1_deterministic_oracle_feasibility():
    """Oracle feasibility must be deterministic: identical inputs yield identical decisions."""
    table = ModcodTable()
    sc = Scenario(clear_sky_esn0_db=18.0)
    msg = Message(size_bits=20e6, priority="normal", deadline_s=2.0, eps=0.01)

    # 1. Clear channel: true peak = 0.5 dB -> feasible
    ctx_clear = PolicyContext(
        arrival_idx=100, last_meas_idx=95, cur_att_db=0.5, features={},
        true_future_att=np.array([0.2, 0.5, 0.4]), msg=msg, sc=sc, table=table,
        forecaster=None, calibration_corr={}
    )
    dec1 = policy_oracle(ctx_clear)
    dec2 = policy_oracle(ctx_clear)
    assert dec1.feasible is True
    assert dec2.feasible is True
    assert dec1.action.modcod == dec2.action.modcod

    # 2. Severe unrecoverable fade: true peak = 50.0 dB -> infeasible
    ctx_severe = PolicyContext(
        arrival_idx=100, last_meas_idx=95, cur_att_db=50.0, features={},
        true_future_att=np.array([45.0, 50.0, 48.0]), msg=msg, sc=sc, table=table,
        forecaster=None, calibration_corr={}
    )
    dec_sev = policy_oracle(ctx_severe)
    assert dec_sev.feasible is False
    assert dec_sev.p_lower_bound < 0.99


def test_2_2_stress_presets_configuration():
    """Stress presets must exist and properly configure Scenario parameters."""
    for name in ["base", "harsh", "deadline_1s", "deadline_2s", "deadline_5s", "deadline_10s", "size_5mb", "size_20mb", "size_50mb"]:
        sc = get_preset(name)
        assert isinstance(sc, Scenario)

    sc_harsh = get_preset("harsh")
    assert sc_harsh.events_per_day >= 8.0
    assert sc_harsh.event_peak_ln_mean >= 2.0

    sc_d1 = get_preset("deadline_1s")
    assert sc_d1.deadline_s == 1.0

    sc_sz50 = get_preset("size_50mb")
    assert sc_sz50.msg_bits == 50e6


def test_2_3_diagnostics_log_columns_and_taxonomy():
    """Per-message schema must contain all LOG_COLUMNS and classify_misses must categorize correctly."""
    df = pd.DataFrame({
        "seed": [101, 101, 101, 101, 101, 101],
        "msg_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
        "arrival_idx": [10, 20, 30, 40, 50, 60],
        "priority": ["normal"] * 6,
        "eps": [0.01] * 6,
        "size_bits": [20e6] * 6,
        "deadline_s": [2.0] * 6,
        "fade_state": ["clear", "clear", "moderate", "moderate", "deep", "deep"],
        "modcod": [20, 20, 10, 10, 2, 2],
        "power_db": [0.0] * 6,
        "parity_frac": [0.0] * 6,
        "defer_s": [0.0] * 6,
        "alpha": [0.01] * 6,
        "p_lb": [0.99] * 6,
        "flagged_infeasible": [False, True, True, False, False, False],
        "bound_db": [4.0, 4.0, 6.0, 6.0, 12.0, 12.0],
        "bound_horizon_steps": [25] * 6,
        "true_window_max_db": [2.0, 2.0, 8.0, 8.0, 15.0, 8.0],
        "delivered": [True, False, False, False, False, False],
        "completion_s": [0.8, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
        "airtime_s": [0.8, 0.8, 1.2, 1.2, 2.0, 2.0],
        "oracle_delivered": [True, True, False, False, True, True],
        "oracle_airtime_s": [0.8, 0.8, 0.0, 0.0, 1.8, 1.8],
    })

    # All LOG_COLUMNS must be present
    for col in LOG_COLUMNS:
        assert col in df.columns

    classified = classify_misses(df)
    cats = classified["category"].tolist()
    assert cats[0] == "ok"
    assert cats[1] == "flagged_avoidable"
    assert cats[2] == "flagged_unrecoverable"
    assert cats[3] == "silent_unrecoverable"
    assert cats[4] == "bound_violation"  # true max 15.0 > bound 12.0
    assert cats[5] == "model_mismatch"   # true max 8.0 <= bound 12.0


def test_2_4_mcnemar_and_bootstrap_statistics():
    """Verify statistical functions: exact McNemar and non-zero cluster bootstrap interval."""
    # McNemar symmetry
    assert mcnemar_exact(0, 0) == 1.0
    p1 = mcnemar_exact(10, 2)
    assert p1 < 0.05
    assert mcnemar_exact(2, 10) == p1

    # Bootstrap interval over varying seed aggregates
    seed_rates = [0.02, 0.025, 0.018, 0.03, 0.022, 0.019, 0.028, 0.021]
    mean_val, lo, hi = cluster_bootstrap_mean(seed_rates, n_boot=2000, seed=42)
    assert 0.015 < lo < mean_val < hi < 0.035
    assert (hi - lo) > 0.002  # interval must be strictly positive width


if __name__ == "__main__":
    pytest.main([__file__])

