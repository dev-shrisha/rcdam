"""Defensible best-achievable alternatives for an infeasible message (Option 3).

For a message the engine cannot serve, answer: WHICH single relaxation would make it feasible
and by how much?  Each counterfactual re-runs the same joint search (same evaluator, same window
bound), so every alternative is itself a validated-style decision, not a guess.

  achievable_eps      : smallest reliability target in a ladder that IS feasible (explicit relaxation)
  min_deadline_s      : smallest deadline (<= calibration range 8 s) that is feasible
  min_power_db        : smallest power boost that is feasible
  max_payload_bits    : largest payload (fraction of the message) that is feasible
  limiting_constraints: relaxations that, applied alone, restore feasibility
"""
from dataclasses import replace
from typing import Any, Dict, Optional
import numpy as np

from .engine import EngineParams, search
from .infeasibility import action_to_dict

EPS_LADDER = (1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5)
POWER_LADDER = (3.0, 4.0, 6.0, 8.0, 10.0)
DEADLINE_FACTORS = (1.25, 1.5, 2.0, 3.0, 4.0)
SIZE_FRACTIONS = (0.75, 0.5, 0.35, 0.25, 0.15, 0.1)
MAX_DEADLINE_S = 8.0                     # window calibration range


def _ctx_with(ctx, **msg_kw):
    return replace(ctx, msg=replace(ctx.msg, **msg_kw))


def best_alternatives(ctx, wc, p: EngineParams) -> Dict[str, Any]:
    msg, table = ctx.msg, ctx.table
    out: Dict[str, Any] = {"original": dict(eps=msg.eps, deadline_s=msg.deadline_s, size_bits=msg.size_bits)}
    lim = []

    # (a) achievable reliability: smallest eps on the ladder that is feasible
    for e in EPS_LADDER:
        if e <= msg.eps:
            continue
        r = search(ctx, wc, e, p)
        if r["feasible"]:
            out["achievable_eps"] = dict(eps=e, predicted_p=r["p_hat"], lower_bound=r["lower_bound"],
                                         action=action_to_dict(r["action"], table),
                                         note="RELAXED requirement; original eps is NOT met")
            lim.append("reliability")
            break
    else:
        out["achievable_eps"] = None

    # (b) minimum deadline
    out["min_deadline_s"] = None
    for f in DEADLINE_FACTORS:
        d = msg.deadline_s * f
        if d > MAX_DEADLINE_S:
            break
        r = search(_ctx_with(ctx, deadline_s=d), wc, msg.eps, p)
        if r["feasible"]:
            out["min_deadline_s"] = dict(deadline_s=d, factor=f, action=action_to_dict(r["action"], table),
                                         overhead=r["cost"])
            lim.append("deadline")
            break

    # (c) minimum power boost
    out["min_power_db"] = None
    for cap in POWER_LADDER:
        pp = replace(p, powers=tuple(sorted(set(p.powers) | {cap, cap / 2.0})))
        r = search(ctx, wc, msg.eps, pp)
        if r["feasible"]:
            out["min_power_db"] = dict(power_db=r["action"].power_db, action=action_to_dict(r["action"], table),
                                       overhead=r["cost"])
            lim.append("power")
            break

    # (d) largest payload that fits
    out["max_payload_bits"] = None
    for fr in SIZE_FRACTIONS:
        r = search(_ctx_with(ctx, size_bits=msg.size_bits * fr), wc, msg.eps, p)
        if r["feasible"]:
            out["max_payload_bits"] = dict(bits=msg.size_bits * fr, fraction=fr,
                                           action=action_to_dict(r["action"], table))
            lim.append("message_size")
            break
    out["limiting_constraints"] = lim
    out["summary"] = ("No single tested relaxation restores feasibility: the channel/resources are the limit."
                      if not lim else "Feasibility is restored by relaxing any of: " + ", ".join(lim) + ".")
    return out
