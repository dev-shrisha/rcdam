"""Trajectory-aware delivery-probability evaluator (vectorised over Monte-Carlo trials).

Replaces the constant-frame-loss-probability evaluator for the levers that only matter when
the channel varies inside the transmission window (interleaving, deferral, power, parity,
duplication). It mirrors the frame timing of groundtruth.simulate_true_delivery but runs on a
*conservative attenuation profile* built from the conformal window bound, not on the true
trace, so evaluator and ground truth stay independent.

Profile (worst-case burst, adversarial placement):
    att(t) = peak_db  inside [t_on, t_on + burst_s),  base_db (last measured) elsewhere.
t_on is swept over the deadline window; the returned probability is the MINIMUM over burst
positions. Interleaving spreads frames across the window; deferral can step past a burst.
"""
from typing import Tuple
import numpy as np
from .optimizer import FRAME_BITS


def _p_loss(table, modcod, esn0):
    return float(table.bler([modcod], [esn0], use_curves=True)[0])


def p_deliver_profile_detail(action, msg_bits, deadline_s, table, clear_esn0_db,
                             symbol_rate, rtt_s, one_way_s,
                             base_db, peak_db, burst_s,
                             n_pos=4, n_mc=400, max_rounds=8, seed=0,
                             burst_lo_s=0.0) -> Tuple[float, int]:
    """Return (worst-case delivery probability over burst positions, MC trials per position)."""
    K = int(np.ceil(msg_bits / FRAME_BITS))
    parity = int(np.ceil(action.parity_frac * K))
    frame_s = FRAME_BITS / (symbol_rate * table.eff[action.modcod])
    dup = max(1, int(action.dup))
    il = max(0.0, float(action.interleave_s))
    rng = np.random.default_rng(seed)

    p_clear = _p_loss(table, action.modcod, clear_esn0_db + action.power_db - base_db)
    p_burst = _p_loss(table, action.modcod, clear_esn0_db + action.power_db - peak_db)

    lo = max(0.0, float(burst_lo_s))
    starts = np.linspace(lo, max(deadline_s - burst_s, lo), n_pos)
    worst = 1.0
    for t_on in starts:
        t_off = t_on + burst_s
        need = np.full(n_mc, K, dtype=np.int64)
        t_cur = np.full(n_mc, float(action.defer_s))
        alive = np.ones(n_mc, bool)
        success = np.zeros(n_mc, bool)
        for r in range(max_rounds):
            alive &= (t_cur + one_way_s) <= deadline_s
            if not alive.any():
                break
            send = np.where(alive, (K + parity) if r == 0 else need, 0)
            smax = int(send.max())
            if smax == 0:
                break
            tot = np.maximum(send * dup, 1)
            dur = np.maximum(tot * frame_s, il)
            step = dur / tot
            j = np.arange(smax * dup)[None, :]                       # (1, smax*dup)
            t_f = t_cur[:, None] + j * step[:, None]
            in_burst = (t_f >= t_on) & (t_f < t_off)
            p = np.where(in_burst, p_burst, p_clear)
            ok = rng.random(t_f.shape) >= p
            ok = ok.reshape(n_mc, smax, dup).any(axis=2)
            ok &= (np.arange(smax)[None, :] < send[:, None])
            got = ok.sum(axis=1)
            need = np.where(alive, np.maximum(0, need - got), need)
            fin = alive & (need == 0)
            success |= fin & ((t_cur + dur + one_way_s) <= deadline_s)
            alive &= ~fin
            t_cur = np.where(alive, t_cur + dur + rtt_s, t_cur)
        worst = min(worst, float(success.mean()))
    return worst, n_mc


def p_deliver_profile(action, msg_bits, deadline_s, table, clear_esn0_db,
                      symbol_rate, rtt_s, one_way_s, base_db, peak_db, burst_s,
                      n_pos=6, n_mc=400, max_rounds=8, seed=0) -> float:
    """Backward-compatible wrapper returning only the probability."""
    return p_deliver_profile_detail(action, msg_bits, deadline_s, table, clear_esn0_db,
                                    symbol_rate, rtt_s, one_way_s, base_db, peak_db,
                                    burst_s, n_pos, n_mc, max_rounds, seed)[0]
