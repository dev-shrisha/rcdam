"""Baseline 1: fixed-margin ACM with delayed channel feedback.

The transmitter picks the MODCOD from an Es/N0 measurement that is one RTT old,
minus a fixed safety margin. A frame is lost if the true Es/N0 at transmission is
below the chosen MODCOD's threshold. Goodput is counted as an idealized fluid
(lost frames contribute zero; no retransmission delay penalty yet).
"""
import numpy as np

from .link import ModcodTable


def run_fixed_margin_acm(esn0_db, table: ModcodTable, margin_db, delay_steps,
                         symbol_rate_baud):
    n = len(esn0_db)
    measured = np.empty(n)
    measured[delay_steps:] = esn0_db[: n - delay_steps]
    measured[:delay_steps] = esn0_db[0]

    idx = table.select(measured - margin_db)
    tx = idx >= 0
    safe = np.maximum(idx, 0)
    success = tx & (esn0_db >= table.thr[safe])
    rate_bps = np.where(success, symbol_rate_baud * table.eff[safe], 0.0)
    return {"tx": tx, "success": success, "rate_bps": rate_bps, "modcod_idx": idx}
