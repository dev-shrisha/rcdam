"""Policy-level risk control for silent-miss certification.

Implements the Learn-then-Test (LTT) fixed-sequence procedure and
conformal risk control (CRC) from the RC-DAM Advancement Plan (Part 4).

Usage pattern:
    1. Calibrate: replay messages under each lambda setting on calibration traces.
       Record miss[i,j] and feasible[i,j] for message i, setting j.
    2. Certify: j_star, table = ltt_fixed_sequence(miss, feasible, eps, delta)
    3. Freeze lambda[j_star].
    4. Evaluate ONCE on separate test traces: never tune on test.
"""
import numpy as np
from scipy.stats import binom


def ltt_fixed_sequence(miss, feasible, eps, delta=0.05, min_feasible=None):
    """Learn-then-Test over an ordered safety parameter (fixed-sequence testing).

    miss, feasible: boolean arrays [n_messages, m]; column j is the policy with
    safety setting j, ordered from the LEAST conservative (j=0) to the MOST
    conservative (j=m-1). miss[i,j] is the replayed ground-truth miss of message i
    under policy j; feasible[i,j] is True if policy j did not flag it INFEASIBLE.

    Certifies the conditional silent-miss risk
        P(miss | flagged feasible) <= eps
    with probability >= 1 - delta over the calibration sample (no monotonicity
    assumption). The sequence starts at the most conservative setting that has at
    least `min_feasible` feasible messages. Testing moves toward less conservative
    settings and stops at the first failure.

    Returns (index of the least conservative certified setting or None, diagnostic table).
    Messages must be exchangeable with future ones; the safety setting must not be
    tuned on test messages.
    """
    miss = np.asarray(miss, dtype=bool)
    feasible = np.asarray(feasible, dtype=bool)
    n, m = miss.shape
    if min_feasible is None:
        # Rule of three: need log(delta)/log(1-eps) feasible messages to certify
        min_feasible = int(np.ceil(np.log(delta) / np.log(1 - eps)))

    table = []
    certified = None
    started = False

    for j in range(m - 1, -1, -1):  # most conservative first
        nf = int(feasible[:, j].sum())
        if not started and nf < min_feasible:
            continue
        started = True
        k = int((miss[:, j] & feasible[:, j]).sum())
        # P(X <= k) when true risk = eps exactly
        p_val = float(binom.cdf(k, nf, eps))
        ok = p_val <= delta
        table.append({
            "setting": j,
            "n_feasible": nf,
            "misses": k,
            "p_value": round(p_val, 6),
            "flagged_infeasible_rate": round(1.0 - nf / n, 4),
            "certified": ok,
        })
        if not ok:
            break
        certified = j

    return certified, table


def crc_lambda(losses, alpha, B=1.0):
    """Conformal risk control for a loss that is NON-INCREASING in the safety
    setting (columns ordered from least to most conservative).

    Returns the first column j with
        (n / (n+1)) * mean_loss[j] + B / (n+1) <= alpha,
    or None if no column qualifies.

    Guarantee: expected test loss <= alpha for exchangeable messages.
    Check monotonicity before relying on this guarantee.

    Args:
        losses: float array [n_messages, m], each entry is a loss value in [0, B]
        alpha: target expected loss
        B: loss upper bound (default 1.0 for 0/1 losses)
    """
    losses = np.asarray(losses, dtype=float)
    n = losses.shape[0]
    risk = losses.mean(axis=0)
    adjusted = n / (n + 1) * risk + B / (n + 1)
    ok = np.where(adjusted <= alpha)[0]
    return int(ok[0]) if len(ok) else None
