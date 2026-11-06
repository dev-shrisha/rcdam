"""Explain WHY a message is infeasible, what the best achievable outcome is, and report any
relaxed-requirement outcome explicitly (never as satisfaction of the original requirement)."""
from typing import Any, Dict, List, Optional
import numpy as np
from .optimizer import FRAME_BITS

REASON_TEXT = {
    "DEADLINE_TOO_SHORT": "The deadline is shorter than the minimum possible delivery time.",
    "MESSAGE_TOO_LARGE": "The message cannot be transmitted within the deadline even at the best MODCOD.",
    "CHANNEL_ATTENUATION_TOO_HIGH": "The conformal window attenuation bound exceeds what any MODCOD "
                                    "can tolerate, even with the maximum available power.",
    "POWER_INSUFFICIENT": "The channel is recoverable, but the available transmit power is not enough.",
    "RELIABILITY_UNACHIEVABLE": "No candidate action combination reaches the required delivery probability.",
}


def action_to_dict(a, table=None) -> Optional[Dict[str, Any]]:
    if a is None:
        return None
    d = dict(modcod=int(a.modcod), power_db=float(a.power_db), parity_frac=float(a.parity_frac),
             defer_s=float(a.defer_s), interleave_s=float(a.interleave_s), dup=int(a.dup))
    if table is not None:
        d["modcod_name"] = table.names[a.modcod]
    return d


def diagnose(msg, sc, table, peak_db: float, p_max_db: float, eps: float, eps_delivery: float,
             best: Optional[Dict[str, Any]], hardware_power_cap_db: float = 6.0) -> Dict[str, Any]:
    """Return a structured infeasibility diagnosis.

    best = dict(p_hat, lower_bound, action) of the best candidate found (may be None).
    """
    K = int(np.ceil(msg.size_bits / FRAME_BITS))
    eff_max = float(table.eff.max())
    min_air = msg.size_bits / (sc.symbol_rate_baud * eff_max)
    min_completion = min_air + sc.one_way_delay_s
    reasons: List[Dict[str, Any]] = []

    if min_completion > msg.deadline_s:
        reasons.append(dict(code="DEADLINE_TOO_SHORT", evidence=dict(
            deadline_s=msg.deadline_s, min_delivery_time_s=round(min_completion, 3),
            required_deadline_s=round(min_completion, 3))))
        max_bits = max(0.0, (msg.deadline_s - sc.one_way_delay_s) * sc.symbol_rate_baud * eff_max)
        reasons.append(dict(code="MESSAGE_TOO_LARGE", evidence=dict(
            message_bits=msg.size_bits, max_bits_for_deadline=round(max_bits))))
    else:
        # most robust MODCOD that still fits one pass inside the deadline
        ok = [m for m in range(len(table.eff))
              if (K * FRAME_BITS / (sc.symbol_rate_baud * table.eff[m])) + sc.one_way_delay_s <= msg.deadline_s]
        m_rob = min(ok, key=lambda m: table.thr[m])
        tol_db = sc.clear_sky_esn0_db - float(table.thr[m_rob])          # attenuation tolerated at 0 dB boost
        needed_boost = max(0.0, peak_db - tol_db)
        if needed_boost > p_max_db:
            code = "POWER_INSUFFICIENT" if needed_boost <= hardware_power_cap_db \
                else "CHANNEL_ATTENUATION_TOO_HIGH"
            reasons.append(dict(code=code, evidence=dict(
                window_attenuation_bound_db=round(peak_db, 2), most_robust_modcod=table.names[m_rob],
                tolerated_attenuation_db=round(tol_db, 2), power_needed_db=round(needed_boost, 2),
                power_available_db=p_max_db)))

    p_best = best["p_hat"] if best else 0.0
    reasons.append(dict(code="RELIABILITY_UNACHIEVABLE", evidence=dict(
        required_delivery_prob=round(1.0 - eps, 6), required_prob_after_conformal_split=round(1.0 - eps_delivery, 6),
        best_achievable_prob=round(p_best, 4), achievable_eps=round(1.0 - p_best, 4))))
    for r in reasons:
        r["message"] = REASON_TEXT[r["code"]]
    return dict(primary_reason=reasons[0]["code"], reasons=reasons, best_achievable=best)
