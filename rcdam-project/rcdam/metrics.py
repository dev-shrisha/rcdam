"""Evaluation metrics: deadline-miss rate, goodput, outage, frame error rate."""
import numpy as np


def deadline_metrics(rate_bps, dt_s, msg_bits, deadline_s, one_way_s, interval_s):
    n = len(rate_bps)
    cum = np.concatenate([[0.0], np.cumsum(rate_bps) * dt_s])   # bits delivered before step k
    horizon = int(5 * deadline_s / dt_s)
    arrivals = np.arange(0, n - horizon, int(interval_s / dt_s))
    target = cum[arrivals] + msg_bits
    done = np.searchsorted(cum, target, side="left")             # first step where enough bits arrived
    delivered = done <= n
    latency = (done - arrivals) * dt_s + one_way_s
    miss = (~delivered) | (latency > deadline_s)
    return {
        "n_messages": len(arrivals),
        "miss_rate": float(miss.mean()),
        "median_latency_s": float(np.median(latency[delivered])) if delivered.any() else float("nan"),
    }


def link_metrics(res):
    tx, success, rate = res["tx"], res["success"], res["rate_bps"]
    return {
        "mean_goodput_mbps": float(rate.mean() / 1e6),
        "outage_frac": float(1.0 - success.mean()),
        "frame_error_rate": float((tx & ~success).sum() / max(tx.sum(), 1)),
    }
