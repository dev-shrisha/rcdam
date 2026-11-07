"""Proactive message scheduling from conformal time-to-fade bounds (NEW).

Given the current (feedback-delayed) attenuation measurement, a queue of messages with
deadlines and reliability classes, the scheduler
  1. orders messages by urgency (earliest deadline first; stricter eps first on ties),
  2. for each message tries protection levels (MODCOD, power, parity), computing
        T_tx    = transmission time of the protected message,
        theta   = attenuation tolerance of that protection,
        T_lb    = conformal lower bound on time-to-fade for theta at level alpha_t = eta * eps,
  3. accepts the minimum-cost protection with   t_acc + T_tx <= T_lb   and  deadline OK,
     where t_acc is the time already committed to earlier messages in the plan,
  4. leaves unschedulable messages for the reactive engine (window-conformal V7).
Guarantee (per scheduled message, if the time-to-fade bound holds): attenuation stays below the
protection's tolerance for the whole transmission, so the frame loss probability stays at/below the
QEF level; the failure probability is then <= alpha_t (+ negligible BLER at the threshold).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import itertools
import numpy as np

from .optimizer import Action, FRAME_BITS
from .time_to_fade import TimeToFade, protection_tolerance_db


@dataclass
class SchedulerParams:
    eta: float = 0.5                                 # share of eps spent on the time-to-fade bound
    powers: Tuple[float, ...] = (0.0, 1.0, 2.0)
    parities: Tuple[float, ...] = (0.0, 0.05, 0.15)
    margin_db: float = 0.5
    w_energy: float = 0.25
    safety_s: float = 0.2                            # extra guard time before the bound


@dataclass
class QueuedMessage:
    id: int
    size_bits: float
    priority: str
    eps: float
    arrival_s: float
    deadline_abs_s: float
    attempts: int = 0


@dataclass
class PlanItem:
    msg_id: int
    start_offset_s: float
    action: Action
    airtime_s: float
    theta_db: float
    usable_lb_s: float
    alpha_t: float
    cost: float
    guaranteed: bool = True
    reason: str = "proactive"


def _tx_time(msg_bits, table, sc, modcod, parity):
    K = int(np.ceil(msg_bits / FRAME_BITS))
    fs = FRAME_BITS / (sc.symbol_rate_baud * table.eff[modcod])
    return int(np.ceil(K * (1.0 + parity))) * fs


def _cost(msg_bits, table, sc, a: Action, w_energy: float) -> float:
    ref = msg_bits / (sc.symbol_rate_baud * table.eff.max())
    air = _tx_time(msg_bits, table, sc, a.modcod, a.parity_frac) / ref
    return air * (1.0 + w_energy * (10 ** (a.power_db / 10.0) - 1.0))


def urgency_key(m: QueuedMessage, now_s: float, min_air_s: float):
    slack = m.deadline_abs_s - now_s - min_air_s
    return (slack, m.eps, m.arrival_s)


def plan(now_s: float, last_meas_db: float, slope_db_per_s: Optional[float],
         queue: List[QueuedMessage], ttf: TimeToFade, table, sc,
         params: Optional[SchedulerParams] = None) -> Tuple[List[PlanItem], List[QueuedMessage]]:
    """Return (guaranteed pre-fade plan in transmission order, messages left to the reactive engine)."""
    p = params or SchedulerParams()
    min_air = {m.id: _tx_time(m.size_bits, table, sc, int(np.argmax(table.eff)), 0.0) for m in queue}
    order = sorted(queue, key=lambda m: urgency_key(m, now_s, min_air[m.id]))
    items: List[PlanItem] = []
    left: List[QueuedMessage] = []
    t_acc = 0.0
    for m in order:
        alpha_t = max(1e-6, p.eta * m.eps)
        best = None
        for mc, pw, pr in itertools.product(range(len(table.eff)), p.powers, p.parities):
            tx = _tx_time(m.size_bits, table, sc, mc, pr)
            finish = t_acc + tx
            if now_s + finish + sc.one_way_delay_s > m.deadline_abs_s:
                continue
            theta = protection_tolerance_db(table, sc, mc, pw, p.margin_db)
            if theta <= last_meas_db:
                continue
            lb = ttf.t_lower(last_meas_db, slope_db_per_s, theta, alpha_t)
            if finish + p.safety_s > lb:
                continue
            a = Action(mc, pw, pr, 0.0, interleave_s=0.0, dup=1)
            c = _cost(m.size_bits, table, sc, a, p.w_energy)
            if best is None or c < best[0]:
                best = (c, a, tx, theta, lb)
        if best is None:
            left.append(m)
        else:
            c, a, tx, theta, lb = best
            items.append(PlanItem(m.id, t_acc, a, tx, theta, lb, alpha_t, c))
            t_acc += tx
    return items, left
