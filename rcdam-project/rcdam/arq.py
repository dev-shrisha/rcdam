"""Message-level ARQ with a real round-trip penalty (expected-value fluid model).

Model: the link transmits continuously at the rate chosen by the policy. In each slot a
fraction BLER of the transmitted bits is lost (expected value over ~tens of FEC frames).
A message of S bits is sent starting at its arrival. Bits lost in a round are retransmitted
in the next round, which starts one RTT after the last frame of the previous round was sent
(conservative: wait for all feedback). A message is complete when the remaining loss is
below `tol_bits` (about one FEC frame). Rounds stop when the deadline is already blown.

Simplifications to be aware of: slot granularity (partial slots counted whole), expected-value
losses instead of random frame losses, no HARQ soft combining, no queueing between messages.
"""
import numpy as np


def build_cumulative(rate_bps, bler, dt_s):
    cap = rate_bps * dt_s                                    # bits sent per slot
    c_tx = np.concatenate([[0.0], np.cumsum(cap)])           # cumulative bits sent
    c_ok = np.concatenate([[0.0], np.cumsum(cap * (1.0 - bler))])   # cumulative bits received
    return c_tx, c_ok


def message_delivery(c_tx, c_ok, dt_s, arrivals, msg_bits, rtt_steps, one_way_s,
                     deadline_s, max_rounds=8, tol_bits=65_000.0):
    n = len(c_tx) - 1
    m = len(arrivals)
    start = arrivals.astype(np.int64).copy()
    remaining = np.full(m, float(msg_bits))
    airtime = np.zeros(m)
    finish = np.full(m, -1, dtype=np.int64)
    active = np.ones(m, dtype=bool)

    for _ in range(max_rounds):
        if not active.any():
            break
        target = c_tx[start] + remaining
        end = np.searchsorted(c_tx, target, side="left")
        ended_ok = end <= n
        end_c = np.minimum(end, n)
        delivered = c_ok[end_c] - c_ok[start]
        lost = remaining - delivered
        airtime += np.where(active, (end_c - start) * dt_s, 0.0)

        done = active & ended_ok & (lost <= tol_bits)
        finish[done] = end_c[done]
        active = active & ended_ok & ~done

        remaining = np.where(active, lost, remaining)
        start = np.where(active, np.minimum(end_c + rtt_steps, n), start)
        active = active & ((start - arrivals) * dt_s <= deadline_s)

    delivered_mask = finish >= 0
    latency = np.where(delivered_mask, (finish - arrivals) * dt_s + one_way_s, np.inf)
    miss = ~delivered_mask | (latency > deadline_s)
    return {"finish_idx": finish, "airtime_s": airtime, "latency_s": latency, "miss": miss}
