"""Monte Carlo Delivery-Probability Evaluator.

Implements RCDAM_Build_Guide.md Section 4.4:
Evaluates message delivery probability under a constant frame-loss probability p
(attenuation held at calibrated bound).
MDS erasure code: receiver needs K of K + parity frames.
Each ARQ round resends exactly the shortfall and starts one RTT after previous round ended.
Common random numbers across actions via fixed seed.
"""
from typing import Tuple
import numpy as np
from scipy.stats import binom


def p_deliver_by_deadline(K: int,
                          parity: int,
                          p: float,
                          frame_s: float,
                          rtt_s: float,
                          one_way_s: float,
                          defer_s: float,
                          deadline_s: float,
                          n_mc: int = 2000,
                          max_rounds: int = 8,
                          seed: int = 0) -> Tuple[float, float]:
    """Monte Carlo estimate under a CONSTANT frame-loss probability p (attenuation held at calibrated bound).

    Args:
        K: Number of information frames required for decoding.
        parity: Number of proactive erasure protection frames.
        p: Frame loss probability (BLER at the calibrated bound).
        frame_s: Frame transmission airtime duration in seconds.
        rtt_s: Round trip time in seconds.
        one_way_s: One-way propagation delay in seconds.
        defer_s: Delay before starting transmission in seconds.
        deadline_s: Message delivery deadline in seconds.
        n_mc: Number of Monte Carlo trials.
        max_rounds: Maximum ARQ rounds before giving up.
        seed: Fixed seed for common random numbers across actions.

    Returns:
        (p_ok, mean_sent_frames): Probability of delivery by deadline and average frames sent.
    """
    K = int(K)
    parity = int(parity)
    p = float(np.clip(p, 0.0, 1.0))

    if defer_s + K * frame_s + one_way_s > deadline_s:
        # Minimum physical transmission time alone exceeds deadline
        return 0.0, float(K + parity)

    if p <= 1e-6:
        # Quasi-error-free or clean channel: 1 round, no losses
        time_needed = defer_s + (K + parity) * frame_s + one_way_s
        return (1.0 if time_needed <= deadline_s else 0.0), float(K + parity)

    # Principled bound:
    # Under maximum possible continuous transmission without ARQ idle time,
    # the number of delivered frames X ~ Binomial(M, 1 - p).
    # If P(X >= K) < 1e-9, delivery by deadline is provably impossible.
    max_frames = int((deadline_s - defer_s - one_way_s) / frame_s)
    if max_frames >= K and (1.0 - p) > 0.0:
        if float(binom.sf(K - 1, max_frames, 1.0 - p)) < 1e-9:
            return 0.0, float(K + parity)

    rng = np.random.default_rng(seed)
    need = np.full(n_mc, K, dtype=np.int64)
    t = np.full(n_mc, float(defer_s))
    done = np.full(n_mc, np.inf)
    send = np.full(n_mc, K + parity, dtype=np.int64)
    sent_total = np.zeros(n_mc)

    for _ in range(max_rounds):
        active = need > 0
        if not active.any():
            break
        got = rng.binomial(send, 1.0 - p)
        end = t + send * frame_s
        sent_total += np.where(active, send, 0)
        need = np.where(active, np.maximum(need - got, 0), need)
        fin = active & (need == 0)
        done[fin] = end[fin] + one_way_s
        t = np.where(active, end + rtt_s, t)
        send = np.where(need > 0, need, 0)

    p_success = float(np.mean(done <= deadline_s))
    return p_success, float(sent_total.mean())
