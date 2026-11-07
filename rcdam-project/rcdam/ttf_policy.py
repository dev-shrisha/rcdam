"""Time-to-fade / time-to-recovery aware decision for messages WITH SLACK (Option 2).

Options compared for one message that still has `remaining_s` before its deadline:
  SEND_NOW            window-conformal engine (V7) says a feasible action exists now
  SEND_STRONGER       minimum-cost protection whose usable window (conformal time-to-fade lower bound,
                      alpha_t = eta*eps) covers the whole transmission  [guaranteed before the fade]
  WAIT                channel is bad now, but the conformal time-to-recovery UPPER bound says it will
                      be good enough to send a cheap protection before the deadline
  REPORT_INFEASIBLE   none of the above can be guaranteed within the deadline
Only options with an explicit (conformal) guarantee are candidates; the cheapest total cost
    cost = overhead + lat_weight * expected_latency / remaining
wins. Everything returned carries the quantities behind the choice, so it can be audited.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .engine import EngineParams, search, effective_params, cost_breakdown
from .optimizer import Action, FRAME_BITS
from .scheduler import plan, QueuedMessage, SchedulerParams
from .time_to_fade import TimeToFade, THR_GRID, H_MAX_S, gap_bucket, trend_bucket, snap_down, protection_tolerance_db
from .infeasibility import action_to_dict

EXCESS_EDGES = (1.0, 3.0, 6.0, 10.0)           # dB above the target threshold


def excess_bucket(x: float) -> int:
    return int(np.searchsorted(EXCESS_EDGES, x, side="left"))


@dataclass
class TimeToRecovery:
    """Conformal UPPER bound on the time until attenuation first falls to <= theta."""
    min_stratum: int = 50
    scores: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    _sorted: Dict = field(default_factory=dict, repr=False)

    def fit_from_trace(self, att, dt_s, delay_steps, stride=20, thr_grid=THR_GRID, horizon_s=H_MAX_S):
        H = int(round(horizon_s / dt_s)); n = len(att); thr = np.asarray(thr_grid)
        for t in range(max(delay_steps + 50, 600), n - H - 1, stride):
            last = float(att[t - delay_steps]); slope = (last - att[t - delay_steps - 50]) / 5.0
            sel = thr[thr < last]
            if len(sel) == 0:
                continue
            win = att[t:t + H]; tb = trend_bucket(slope)
            for th in sel:
                below = np.nonzero(win <= th)[0]
                T = below[0] * dt_s if len(below) else np.inf          # never recovered: +inf (censored)
                self.scores.setdefault((excess_bucket(last - th), tb), []).append(float(T))
        self._sorted.clear()

    def _arr(self, key):
        if key not in self._sorted:
            self._sorted[key] = np.sort(np.asarray(self.scores.get(key, []), float))
        return self._sorted[key]

    def t_upper(self, last_db: float, slope: Optional[float], theta_db: float, alpha: float) -> float:
        """Seconds within which attenuation is back at/below theta with prob >= 1-alpha (inf = no claim)."""
        if last_db <= theta_db:
            return 0.0
        th = snap_down(theta_db)
        if not np.isfinite(th) or th >= last_db:
            return 0.0
        eb, tb = excess_bucket(last_db - th), trend_bucket(slope)
        s = self._arr((eb, tb))
        if len(s) < self.min_stratum:
            s = np.sort(np.concatenate([self._arr((eb, t)) for t in range(3)]))
        n = len(s)
        if n < self.min_stratum:
            return float("inf")
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        return float("inf") if k > n else float(s[k - 1])


@dataclass
class SlackParams:
    lat_weight: float = 0.5               # cost of waiting, relative to the slack
    eta: float = 0.5
    guard_s: float = 1.0
    wait_threshold_margin_db: float = 0.5


def decide_with_slack(ctx, wc, ttf: TimeToFade, ttr: TimeToRecovery, params: EngineParams,
                      remaining_s: float, exec_window_s: float = 2.0,
                      sp: Optional[SlackParams] = None, sched: Optional[SchedulerParams] = None) -> Dict[str, Any]:
    from dataclasses import replace
    sp = sp or SlackParams(); sched = sched or SchedulerParams()
    msg, sc, table = ctx.msg, ctx.sc, ctx.table
    p = effective_params(params, msg.priority)
    slope = float(ctx.features["slope5"][0])
    dex = min(exec_window_s, remaining_s)
    cands: List[Dict[str, Any]] = []

    # --- SEND_NOW -----------------------------------------------------------------------------
    ctx_now = replace(ctx, msg=replace(msg, deadline_s=dex))
    r = search(ctx_now, wc, msg.eps, p)
    if r["feasible"]:
        cands.append(dict(option="SEND_NOW", action=r["action"], overhead=r["cost"], latency_s=0.0,
                          guarantee=f"window-conformal alpha_w={r['alpha_w']:.2g}", predicted_p=r["p_hat"]))
    # --- SEND_STRONGER (guaranteed pre-fade window) --------------------------------------------
    qm = QueuedMessage(0, msg.size_bits, msg.priority, msg.eps, 0.0, remaining_s)
    items, _ = plan(0.0, ctx.cur_att_db, slope, [qm], ttf, table, sc, sched)
    if items:
        it = items[0]
        cands.append(dict(option="SEND_STRONGER", action=it.action, overhead=it.cost, latency_s=0.0,
                          guarantee=f"time-to-fade lower bound {it.usable_lb_s:.1f}s >= tx {it.airtime_s:.2f}s (alpha_t={it.alpha_t:.2g})",
                          predicted_p=1.0 - it.alpha_t))
    # --- WAIT (guaranteed recovery) ------------------------------------------------------------
    best_clear = int(np.argmax(table.eff))
    theta_wait = protection_tolerance_db(table, sc, best_clear, 0.0, sp.wait_threshold_margin_db)
    alpha_t = max(1e-6, sp.eta * msg.eps)
    t_ub = ttr.t_upper(ctx.cur_att_db, slope, theta_wait, alpha_t)
    t_med = ttr.t_upper(ctx.cur_att_db, slope, theta_wait, 0.5)
    air = msg.size_bits / (sc.symbol_rate_baud * table.eff[best_clear])
    if ctx.cur_att_db > theta_wait and np.isfinite(t_ub) and t_ub + air + sp.guard_s + sc.one_way_delay_s <= remaining_s:
        cands.append(dict(option="WAIT", action=Action(best_clear, 0.0, 0.0, 0.0, interleave_s=0.0, dup=1),
                          overhead=1.0, latency_s=min(t_med, t_ub),
                          guarantee=f"time-to-recovery upper bound {t_ub:.1f}s (alpha_t={alpha_t:.2g}) fits the {remaining_s:.0f}s slack",
                          predicted_p=1.0 - alpha_t))
    for c in cands:
        c["total_cost"] = c["overhead"] + sp.lat_weight * c["latency_s"] / max(remaining_s, 1.0)
    if cands:
        best = min(cands, key=lambda c: c["total_cost"])
        return dict(decision=best["option"], chosen=best, candidates=cands, wait_upper_s=t_ub,
                    explanation=f"{best['option']}: {best['guarantee']}")
    return dict(decision="REPORT_INFEASIBLE", chosen=None, candidates=[], wait_upper_s=t_ub,
                explanation=("No option can be guaranteed within the %.0f s slack: send-now is infeasible at the window bound, "
                             "no protection fits before the predicted fade, and recovery is not guaranteed (upper bound %s s)."
                             % (remaining_s, "inf" if not np.isfinite(t_ub) else f"{t_ub:.0f}")))
