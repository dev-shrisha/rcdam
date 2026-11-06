"""Tests for the window-conformal engine, infeasibility diagnosis and reliability accounting."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
import numpy as np
import pandas as pd

from rcdam.config import preset_harsh
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.optimizer import Message
from rcdam.variants import PolicyContext
from rcdam.window_conformal import WindowConformal
from rcdam.engine import decide, EngineParams
from rcdam.reliability import reliability_report, infeasibility_report, cp_lower

_SC = preset_harsh()
_WC = None


def _wc():
    global _WC
    if _WC is None:
        _WC = WindowConformal()
        for s in (201, 202):
            sc = replace(_SC, seed=s, duration_h=3.0)
            _WC.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps)
    return _WC


def _ctx(cur, eps=0.01, size=20e6, deadline=2.0, slope=0.0, prio="normal"):
    msg = Message(size_bits=size, priority=prio, deadline_s=deadline, eps=eps, arrival_idx=1000)
    f = {"cur": np.array([cur]), "slope5": np.array([slope]), "slope30": np.array([0.0]), "max60": np.array([cur])}
    return PolicyContext(1000, 1000, cur, f, None, msg, _SC, ModcodTable(), None, None)


def test_window_conformal_coverage_on_heldout_trace():
    wc, alpha, cov, n = _wc(), 0.1, 0, 0
    sc = replace(_SC, seed=777, duration_h=3.0); att = generate_attenuation_db(sc)
    d, w = sc.feedback_delay_steps, int(round(2.0 / sc.dt_s))
    for t in range(700, len(att) - w - 1, 37):
        cov += att[t:t + w].max() <= wc.bound(att[t - d], 2.0, alpha); n += 1
    assert cov / n >= 1 - alpha - 0.05


def test_clear_sky_feasible_and_cheap():
    dec = decide(_ctx(0.3), _wc())
    assert dec.feasible and dec.reason == "ok" and dec.p_lower_bound >= 0.99
    assert dec.decision_log["meets_original_requirement"] is True


def test_infeasible_has_structured_explanation():
    dec = decide(_ctx(25.0, eps=0.001), _wc())
    inf = dec.decision_log["infeasibility"]
    assert not dec.feasible and inf["primary_reason"] in ("CHANNEL_ATTENUATION_TOO_HIGH", "POWER_INSUFFICIENT")
    assert any(r["code"] == "RELIABILITY_UNACHIEVABLE" for r in inf["reasons"])
    assert inf["best_achievable"] is not None


def test_deadline_and_size_reasons():
    dec = decide(_ctx(0.3, size=500e6, deadline=1.0), _wc())
    codes = [r["code"] for r in dec.decision_log["infeasibility"]["reasons"]]
    assert "DEADLINE_TOO_SHORT" in codes and "MESSAGE_TOO_LARGE" in codes


def test_r3_never_claims_original_requirement_met():
    for cur in (10.1, 12.0):
        dec = decide(_ctx(cur, eps=0.001), _wc(), rule="R3")
        if dec.reason == "relaxed_feasible":
            assert dec.feasible is False
            assert dec.decision_log["meets_original_requirement"] is False
            assert dec.decision_log["relaxed_eps"] > 0.001 and "NOT satisfied" in dec.decision_log["warning"]


def test_r4_rejects_without_action():
    dec = decide(_ctx(25.0, eps=0.001), _wc(), rule="R4")
    assert dec.action is None and dec.reason == "rejected" and dec.overhead == 0.0


def test_reliability_report_false_feasible():
    df = pd.DataFrame(dict(variant="X", priority="normal", eps=0.01,
                           delivered=[True] * 98 + [False] * 2, flagged=[False] * 100,
                           oracle_feasible=[True] * 100, pred_p=[0.999] * 100, pred_lb=[0.99] * 100))
    r = reliability_report(df).iloc[0]
    assert r.false_feasible == 2 and abs(r.false_feasible_rate - 0.02) < 1e-9
    assert r.validated_lb < r.required and not r.validated_meets_requirement
    assert infeasibility_report(df).iloc[0].FN_missed_infeasible == 0
    assert 0.0 < cp_lower(98, 100) < 0.98
