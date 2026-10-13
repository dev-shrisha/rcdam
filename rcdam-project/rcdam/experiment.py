"""One seed = one simulated channel; evaluate every method/parameter on it.

Besides the overall deadline-miss rate, every method is split into
  * unrecoverable-by-MODCOD messages: even the zero-delay ORACLE MODCOD policy misses them
    (capacity is simply too low during the fade), and
  * avoidable misses: messages the oracle delivers but the method does not.
Methods differ only on the second group; the first group is what an early INFEASIBLE
signal (and other levers: power, defer, priorities) must address.
"""
from dataclasses import replace

import numpy as np

from .config import Scenario
from .attenuation import generate_attenuation_db
from .link import ModcodTable
from .arq import build_cumulative, message_delivery
from . import policies

# (display name, policy key, parameter sweep in dB)
METHODS = [
    ("Oracle ACM (zero delay, bound)", "oracle", [0]),
    ("ARQ-only (static MODCOD)", "static", [4, 8, 10, 12, 14]),
    ("Fixed-margin ACM + ARQ", "margin", [0, 1, 2, 3, 4, 6]),
    ("Predictive ACM (linear) + ARQ", "predictive", [0, 1, 2, 3, 4, 6]),
]


def step2_scenario() -> Scenario:
    """Step 2 channel: same generator as Step 1 but with a less extreme rain tail
    (median event peak ~5 dB, capped at 35 dB). Still a placeholder: calibrate to ITU-R."""
    return replace(Scenario(), event_peak_ln_mean=1.6, event_peak_ln_std=0.7, event_peak_max_db=35.0)


def evaluate(idx, esn0, sc: Scenario, table: ModcodTable):
    n = len(esn0)
    rate = sc.symbol_rate_baud * table.eff[idx]
    p_loss = table.bler(idx, esn0, sc.bler_slope_decades_per_db, sc.bler_qef_log10)
    c_tx, c_ok = build_cumulative(rate, p_loss, sc.dt_s)

    horizon = int(5 * sc.deadline_s / sc.dt_s)
    arrivals = np.arange(0, n - horizon, int(sc.msg_interval_s / sc.dt_s))
    out = message_delivery(c_tx, c_ok, sc.dt_s, arrivals, sc.msg_bits,
                           sc.feedback_delay_steps, sc.one_way_delay_s, sc.deadline_s,
                           sc.arq_max_rounds, sc.arq_done_tol_bits)
    ref_airtime = sc.msg_bits / (sc.symbol_rate_baud * table.eff.max())   # best clear-sky MODCOD, no redundancy
    lat = out["latency_s"][np.isfinite(out["latency_s"])]
    summary = {
        "n_messages": int(len(arrivals)),
        "n_miss": int(out["miss"].sum()),
        "miss_rate": float(out["miss"].mean()),
        "overhead": float(out["airtime_s"].mean() / ref_airtime),
        "goodput_mbps": float(c_ok[-1] / (n * sc.dt_s) / 1e6),
        "mean_latency_s": float(lat.mean()) if len(lat) else float("nan"),
    }
    return summary, out["miss"]


def run_seed(args):
    seed, hours = args
    sc = replace(step2_scenario(), seed=seed, duration_h=hours)
    table = ModcodTable()
    att = generate_attenuation_db(sc)
    esn0 = sc.clear_sky_esn0_db - att
    n = len(esn0)
    win = max(1, int(round(sc.slope_window_s / sc.dt_s)))

    # reference: messages that even the zero-delay oracle MODCOD policy cannot deliver on time
    _, oracle_miss = evaluate(policies.oracle_policy(table, esn0, 0.0), esn0, sc, table)
    feasible = ~oracle_miss

    rows = []
    for name, key, params in METHODS:
        for prm in params:
            if key == "oracle":
                idx = policies.oracle_policy(table, esn0, prm)
            elif key == "static":
                idx = policies.static_policy(table, sc.clear_sky_esn0_db, prm, n)
            elif key == "margin":
                idx = policies.margin_policy(table, esn0, prm, sc.feedback_delay_steps)
            else:
                idx = policies.predictive_policy(table, esn0, prm, sc.feedback_delay_steps, win)
            r, miss = evaluate(idx, esn0, sc, table)
            r["infeasible_frac"] = float(oracle_miss.mean())
            r["n_avoidable_miss"] = int((miss & feasible).sum())
            r["n_feasible"] = int(feasible.sum())
            r["avoidable_miss_rate"] = float((miss & feasible).sum() / max(feasible.sum(), 1))
            r.update({"seed": seed, "method": name, "param_db": prm})
            rows.append(r)
    return rows
