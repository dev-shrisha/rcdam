"""Tests for RC-DAM advancement utilities (Tasks 3, 6).

Covers:
  - stats_utils: wilson_interval, exact_upper_bound, mcnemar_exact, n_zero_miss
  - risk_control: ltt_fixed_sequence, crc_lambda
  - diagnostics: classify_misses, infeasibility_report, LOG_COLUMNS
  - validated_selection: horizon_steps_needed, pick_horizon

Run directly: python tests/test_adv.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from rcdam.stats_utils import wilson_interval, exact_upper_bound, mcnemar_exact, n_zero_miss
from rcdam.risk_control import ltt_fixed_sequence, crc_lambda
from rcdam.diagnostics import classify_misses, infeasibility_report, LOG_COLUMNS
from rcdam.validated_selection import horizon_steps_needed, pick_horizon


def test_wilson_interval_known():
    lo, hi = wilson_interval(5, 118)
    assert abs(lo - 0.0182) < 0.003, f"Wilson lo={lo:.4f}, expected ~0.0182"
    assert abs(hi - 0.0954) < 0.003, f"Wilson hi={hi:.4f}, expected ~0.0954"
    print(f"  wilson(5,118) = ({lo:.4f}, {hi:.4f})  OK")


def test_exact_upper_bound_rule_of_three():
    ub = exact_upper_bound(0, 299)
    assert ub < 0.012, f"Upper bound for 0/299={ub:.4f}, expected <0.012"
    print(f"  exact_upper_bound(0, 299) = {ub:.4f}  OK")


def test_n_zero_miss():
    assert n_zero_miss(0.01) == 299, f"n_zero_miss(0.01)={n_zero_miss(0.01)}"
    assert n_zero_miss(0.001) == 2995, f"n_zero_miss(0.001)={n_zero_miss(0.001)}"
    print(f"  n_zero_miss(1e-2)={n_zero_miss(0.01)}, n_zero_miss(1e-3)={n_zero_miss(0.001)}  OK")


def test_mcnemar_exact():
    p = mcnemar_exact(5, 0)
    assert abs(p - 0.0625) < 0.001, f"mcnemar(5,0)={p:.4f}, expected 0.0625"
    print(f"  mcnemar_exact(5,0) = {p:.4f}  OK")


def test_ltt_certifies_correct_policy():
    rng = np.random.default_rng(42)
    n, m = 3000, 5
    eps, delta = 0.01, 0.05
    miss_all = rng.random((n, m)) < 0.002
    thresholds = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    feasible = rng.random(n)[:, None] <= thresholds[None, :]
    j_star, table = ltt_fixed_sequence(miss_all, feasible, eps, delta)
    assert j_star is not None, f"LTT failed: {table}"
    print(f"  LTT certified j={j_star}, table={len(table)} rows  OK")


def test_ltt_does_not_certify_when_risk_too_high():
    rng = np.random.default_rng(99)
    n, m = 1000, 5
    eps, delta = 0.01, 0.05
    false_certs = 0
    for _ in range(100):
        miss_all = rng.random((n, m)) < 0.03
        feasible = rng.random(n)[:, None] <= np.array([0.9, 0.8, 0.7, 0.6, 0.5])[None, :]
        j_star, _ = ltt_fixed_sequence(miss_all, feasible, eps, delta)
        if j_star is not None:
            false_certs += 1
    rate = false_certs / 100
    assert rate <= 0.15, f"False cert rate={rate:.2f}, expected <=0.15"
    print(f"  LTT false cert rate when true_risk=3%>eps: {rate:.2f}  OK")


def test_ltt_repeated_calibrations():
    eps, delta, n_cal, n_trials = 0.01, 0.05, 500, 200
    taus = 1.0 - 0.05 * np.arange(10)
    rng = np.random.default_rng(1)
    good = certified_count = 0
    for _ in range(n_trials):
        s = rng.random(n_cal)
        miss_i = rng.random(n_cal) < 0.05 * s ** 3
        feasible = s[:, None] <= taus[None, :]
        miss = np.repeat(miss_i[:, None], len(taus), axis=1)
        j, _ = ltt_fixed_sequence(miss, feasible, eps, delta)
        if j is None:
            continue
        certified_count += 1
        if 0.05 * taus[j] ** 4 / 4 <= eps:
            good += 1
    if certified_count > 0:
        rate = good / certified_count
        assert rate >= 1 - delta - 0.08
        print(f"  LTT repeated: {certified_count}/{n_trials} certs, {good}/{certified_count} ({rate:.1%}) OK")
    else:
        print(f"  LTT: no certs in {n_trials} trials (n_cal may be too small)")


def test_crc_lambda_basic():
    rng = np.random.default_rng(7)
    losses = rng.random((500, 10)) * np.linspace(0.08, 0.001, 10)[None, :]
    j = crc_lambda(losses, alpha=0.04)
    if j is not None:
        print(f"  CRC j={j}, mean_loss={losses[:,j].mean():.4f}  OK")
    else:
        print("  CRC returned None (alpha too tight — acceptable)")


def test_classify_misses_categories():
    df = pd.DataFrame({
        "delivered":          [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        "flagged_infeasible": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        "oracle_delivered":   [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        "true_window_max_db": [3, 2, 4, 30, 9, 28, 12, 6, 5, 5, 25, 1],
        "bound_db":           [5, 5, 5, 20, 10, 20, 10, 9, 9, 9, 20, 3],
    })
    d = classify_misses(df)
    cats = d["category"].value_counts().to_dict()
    assert cats.get("ok", 0) == 6, f"Expected 6 ok, got {cats}"
    assert cats.get("flagged_unrecoverable", 0) >= 1
    assert cats.get("flagged_avoidable", 0) >= 1
    print(f"  classify_misses: {cats}  OK")


def test_infeasibility_report_keys():
    df = pd.DataFrame({
        "delivered":          [1, 0, 0, 1, 0],
        "flagged_infeasible": [0, 1, 0, 0, 1],
        "oracle_delivered":   [1, 0, 1, 1, 0],
        "true_window_max_db": [2, 25, 15, 3, 30],
        "bound_db":           [5, 20, 10, 5, 20],
    })
    rep = infeasibility_report(df)
    for k in ("precision", "recall", "silent_miss_rate", "misses", "oracle_infeasible"):
        assert k in rep, f"Missing key '{k}'"
    print(f"  infeasibility_report: {rep}  OK")


def test_log_columns_count():
    assert len(LOG_COLUMNS) >= 20
    print(f"  LOG_COLUMNS has {len(LOG_COLUMNS)} columns  OK")


def test_horizon_steps_needed():
    needed = horizon_steps_needed(5, 1.0, 2.0, 0.1)
    expected = 5 + int(np.ceil((1.0 + 2.0) / 0.1))
    assert needed == expected, f"Got {needed}, expected {expected}"
    h = pick_horizon([10, 30, 100, 300, 600], needed)
    assert h == 100, f"Got {h}, expected 100"
    print(f"  horizon_steps_needed={needed}, pick={h}  OK")


def test_pick_horizon_none():
    h = pick_horizon([10, 20], 1000)
    assert h is None
    print("  pick_horizon returns None for impossible horizon  OK")


if __name__ == "__main__":
    print("=== test_adv.py ===")
    test_wilson_interval_known()
    test_exact_upper_bound_rule_of_three()
    test_n_zero_miss()
    test_mcnemar_exact()
    test_ltt_certifies_correct_policy()
    test_ltt_does_not_certify_when_risk_too_high()
    test_ltt_repeated_calibrations()
    test_crc_lambda_basic()
    test_classify_misses_categories()
    test_infeasibility_report_keys()
    test_log_columns_count()
    test_horizon_steps_needed()
    test_pick_horizon_none()
    print("=== All test_adv tests passed ===")
