"""Tests for Options 1-3: adaptive window calibration, slack-aware decisions, infeasibility alternatives."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
import numpy as np

from rcdam.config import preset_harsh
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.optimizer import Message
from rcdam.variants import PolicyContext
from rcdam.window_conformal import WindowConformal
from rcdam.adaptive_window import RollingWindowConformal, AdaptiveWindowConformal
from rcdam.time_to_fade import TimeToFade
from rcdam.ttf_policy import TimeToRecovery, decide_with_slack
from rcdam.engine import EngineParams, decide, effective_params, load_priority_params, search
from rcdam.alternatives import best_alternatives

_SC = preset_harsh(duration_h=3.0)
_W = {}


def _models():
    if not _W:
        wc, ttf, ttr = WindowConformal(), TimeToFade(), TimeToRecovery()
        for s in (201, 202, 203):
            sc = replace(_SC, seed=s, duration_h=4.0); a = generate_attenuation_db(sc)
            wc.fit_from_trace(a, sc.dt_s, sc.feedback_delay_steps)
            ttf.fit_from_trace(a, sc.dt_s, sc.feedback_delay_steps); ttr.fit_from_trace(a, sc.dt_s, sc.feedback_delay_steps)
        _W.update(wc=wc, ttf=ttf, ttr=ttr)
    return _W


def _ctx(cur, eps=0.01, slope=0.0, prio="normal", size=20e6, deadline=2.0):
    msg = Message(size_bits=size, priority=prio, deadline_s=deadline, eps=eps, arrival_idx=1000)
    f = {"cur": np.array([cur]), "slope5": np.array([slope]), "slope30": np.array([0.0]), "max60": np.array([cur])}
    return PolicyContext(1000, 1000, cur, f, None, msg, _SC, ModcodTable(), None, None)


def test_rolling_cap_and_update_shift_the_bound():
    base = _models()["wc"]; r = RollingWindowConformal(base, cap=500)
    assert all(len(v) <= 500 for v in r.scores.values())
    b0 = r.bound(0.5, 1.7, 0.05)
    for _ in range(600):                                   # a much harsher regime keeps arriving
        r.update(0.5, 20.0, 1.7)
    assert r.bound(0.5, 1.7, 0.05) > b0 + 5.0
    assert all(len(v) <= 500 for v in r.scores.values())


def test_aci_moves_alpha_after_miss_and_hit():
    aci = AdaptiveWindowConformal(_models()["wc"], gamma=0.01)
    a0 = aci._a(0.05)
    aci.update(0.5, 50.0, 1.7, 0.05)                       # window max far above the bound -> alpha_t shrinks
    assert aci.state[round(0.05, 9)] < a0
    for _ in range(50):
        aci.update(0.5, 0.0, 1.7, 0.05)                    # all covered -> alpha_t grows back
    assert aci.state[round(0.05, 9)] > 0.0
    assert aci.bound(0.5, 1.7, 0.05) >= 0.0


def test_per_priority_parameters_override():
    p = EngineParams(by_priority={"normal": dict(burst_s=0.5, alpha_split=0.7)})
    e = effective_params(p, "normal"); assert (e.burst_s, e.alpha_split) == (0.5, 0.7)
    e2 = effective_params(p, "critical"); assert (e2.burst_s, e2.alpha_split) == (1.0, 0.5)


def test_recovery_bound_is_inf_when_not_certifiable_and_zero_when_already_ok():
    ttr = _models()["ttr"]
    assert ttr.t_upper(1.0, 0.0, 2.0, 0.01) == 0.0
    assert ttr.t_upper(10.0, 0.0, 2.0, 1e-6) == float("inf")


def test_slack_decision_clear_sky_sends_now_and_deep_fade_reports():
    m = _models(); ep = EngineParams()
    out = decide_with_slack(_ctx(0.2), m["wc"], m["ttf"], m["ttr"], ep, 60.0)
    assert out["decision"] == "SEND_NOW" and out["chosen"]["guarantee"]
    out2 = decide_with_slack(_ctx(25.0, eps=0.001), m["wc"], m["ttf"], m["ttr"], ep, 30.0)
    assert out2["decision"] == "REPORT_INFEASIBLE" and "No option can be guaranteed" in out2["explanation"]


def test_infeasibility_class_and_alternatives():
    m = _models(); dec = decide(_ctx(25.0, eps=0.001), m["wc"], EngineParams(), "R1", explain=True)
    lg = dec.decision_log
    assert not dec.feasible and lg["infeasibility_class"] in ("PHYSICALLY_INFEASIBLE", "UNCERTAINTY_LIMITED")
    alt = lg["alternatives"]
    assert set(["achievable_eps", "min_deadline_s", "min_power_db", "max_payload_bits", "limiting_constraints"]) <= set(alt)
    if alt["achievable_eps"]:
        assert alt["achievable_eps"]["eps"] > 0.001 and "NOT met" in alt["achievable_eps"]["note"]


def test_exhaustive_matches_pruned_on_small_grid():
    m = _models(); small = dict(powers=(0.0, 2.0), parities=(0.0, 0.15), defers=(0.0,), interleaves=(0.0,), dups=(1,))
    c = _ctx(4.0)
    a = search(c, m["wc"], 0.01, EngineParams(**small)); b = search(c, m["wc"], 0.01, EngineParams(exhaustive=True, **small))
    assert a["feasible"] == b["feasible"] and (not a["feasible"] or abs(a["cost"] - b["cost"]) < 1e-9)
