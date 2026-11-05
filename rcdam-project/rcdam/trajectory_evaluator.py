"""Trajectory-aware delivery-probability evaluator (NEW).

Replaces the constant-frame-loss-probability evaluator for the levers that only
matter when the channel varies inside the window (interleaving, deferral, power,
parity). It mirrors the frame timing of groundtruth.simulate_true_delivery but
runs on a *conservative attenuation profile* built from the conformal window bound
instead of the true trace, so the evaluator and the ground truth stay independent.

Profile model (worst-case burst, one parameter pair):
    att(t) = peak_db for t in [t_on, t_on + burst_s), 0.0 dB-offset elsewhere
    where elsewhere = base_db (last measured attenuation).
The burst start t_on is swept over the window, and the delivery probability is the
MINIMUM over burst positions (adversarial placement) -> interleaving spreads loss,
deferral can step past an early burst.
"""
import numpy as np
from .optimizer import FRAME_BITS


def _p_loss(table, modcod, esn0):
    return float(table.bler([modcod], [esn0], use_curves=True)[0])


def p_deliver_profile(action, msg_bits, deadline_s, table, clear_esn0_db,
                      symbol_rate, rtt_s, one_way_s,
                      base_db, peak_db, burst_s,
                      n_pos=6, n_mc=400, max_rounds=8, seed=0):
    K = int(np.ceil(msg_bits / FRAME_BITS))
    parity = int(np.ceil(action.parity_frac * K))
    frame_s = FRAME_BITS / (symbol_rate * table.eff[action.modcod])
    dup = max(1, int(action.dup))
    il = max(0.0, float(action.interleave_s))
    rng = np.random.default_rng(seed)

    p_clear = _p_loss(table, action.modcod, clear_esn0_db + action.power_db - base_db)
    p_burst = _p_loss(table, action.modcod, clear_esn0_db + action.power_db - peak_db)

    # coarse burst start positions across the deadline window
    starts = np.linspace(0.0, max(deadline_s - burst_s, 0.0), n_pos)
    worst = 1.0
    for t_on in starts:
        t_off = t_on + burst_s
        ok = 0
        for _ in range(n_mc):
            need, send, t_cur = K, K + parity, action.defer_s
            done = False
            for _r in range(max_rounds):
                if t_cur + one_way_s > deadline_s:
                    break
                tot = send * dup
                dur = max(tot * frame_s, il)
                step = dur / max(1, tot)
                t_f = t_cur + (np.arange(send * dup)) * step
                p = np.where((t_f >= t_on) & (t_f < t_off), p_burst, p_clear)
                ok_f = (rng.random(send * dup) >= p).reshape(send, dup).any(axis=1)
                got = int(ok_f.sum())
                need = max(0, need - got)
                if need == 0:
                    done = (t_cur + dur + one_way_s) <= deadline_s
                    break
                t_cur += dur + rtt_s
                send = need
            ok += done
        worst = min(worst, ok / n_mc)
    return worst
