"""Tests for conformal time-to-fade prediction and the proactive scheduler."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
import numpy as np

from rcdam.config import preset_harsh
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.time_to_fade import TimeToFade, protection_tolerance_db, snap_down
from rcdam.scheduler import plan, QueuedMessage, SchedulerParams

_SC = preset_harsh(duration_h=3.0)
_TTF = None


def _ttf():
    global _TTF
    if _TTF is None:
        _TTF = TimeToFade()
        for s in (201, 202, 203):
            sc = replace(_SC, seed=s, duration_h=4.0)
            _TTF.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps)
    return _TTF


def test_zero_when_already_above_threshold():
    assert _ttf().t_lower(5.0, 0.0, 4.0, 0.05) == 0.0


def test_too_few_scores_refuses_to_claim_window():
    t = TimeToFade(); t.scores[(3, 1)] = [10.0] * 60
    assert t.t_lower(0.2, 0.0, 3.0, 1e-4) == 0.0          # k = floor(alpha (n+1)) < 1


def test_lower_bound_coverage_on_heldout_traces():
    ttf, alpha, ok, n = _ttf(), 0.05, 0, 0
    sc = replace(_SC, seed=777); att = generate_attenuation_db(sc); d = sc.feedback_delay_steps
    H = int(600 / sc.dt_s)
    for t0 in range(1000, len(att) - H - 1, 150):
        last = float(att[t0 - d]); sl = (last - att[t0 - d - 50]) / 5.0
        for th in (3.0, 6.0, 10.0):
            if th <= last:
                continue
            i = np.searchsorted(np.maximum.accumulate(att[t0:t0 + H]), th, side="right")
            T = i * sc.dt_s if i < H else 600.0
            ok += T >= ttf.t_lower(last, sl, th, alpha); n += 1
    assert n > 100 and ok / n >= 1 - alpha - 0.02


def test_snap_down_is_conservative():
    assert snap_down(3.7) == 3.5 and np.isnan(snap_down(0.2))


def test_plan_respects_usable_window_and_deadline():
    t = ModcodTable(); p = SchedulerParams()
    q = [QueuedMessage(0, 20e6, "normal", 1e-2, 0.0, 120.0), QueuedMessage(1, 20e6, "critical", 1e-3, 0.0, 120.0),
         QueuedMessage(2, 20e6, "best_effort", 1e-1, 0.0, 6.0)]
    items, left = plan(0.0, 0.2, 0.0, q, _ttf(), t, _SC, p)
    assert items and items[0].msg_id == 2                       # least slack first
    for it in items:
        assert it.start_offset_s + it.airtime_s + p.safety_s <= it.usable_lb_s + 1e-9
        assert protection_tolerance_db(t, _SC, it.action.modcod, it.action.power_db, p.margin_db) == it.theta_db


def test_deep_fade_leaves_everything_to_reactive_engine():
    items, left = plan(0.0, 25.0, 0.0, [QueuedMessage(0, 20e6, "normal", 1e-2, 0.0, 60.0)],
                       _ttf(), ModcodTable(), _SC)
    assert items == [] and len(left) == 1
