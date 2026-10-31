"""Logging schema and miss taxonomy for per‑message diagnostics."""
import numpy as np
import pandas as pd

# Columns that appear in every per‑message log row
LOG_COLUMNS = [
    "seed", "msg_id", "arrival_idx", "priority", "eps", "size_bits", "deadline_s",
    "fade_state", "modcod", "power_db", "parity_frac", "defer_s", "alpha",
    "p_lb", "flagged_infeasible", "bound_db", "bound_horizon_steps",
    "true_window_max_db", "delivered", "completion_s", "airtime_s",
    "oracle_delivered", "oracle_airtime_s",
]


def classify_misses(df):
    """
    Adds a 'category' column. Priority order:
      ok                      delivered by the deadline
      flagged_unrecoverable   missed, flagged INFEASIBLE, oracle also misses
      flagged_avoidable       missed, flagged INFEASIBLE, oracle delivers
      silent_unrecoverable    missed, NOT flagged, oracle also misses
      bound_violation         missed, not flagged, oracle delivers, true window max exceeded the bound
      model_mismatch          missed, not flagged, oracle delivers, bound held
    """
    d = df.copy()
    miss = ~d["delivered"].astype(bool)
    flag = d["flagged_infeasible"].astype(bool)
    orc_ok = d["oracle_delivered"].astype(bool)
    viol = d["true_window_max_db"] > d["bound_db"]

    cat = np.full(len(d), "ok", dtype=object)
    cat[miss & flag & ~orc_ok] = "flagged_unrecoverable"
    cat[miss & flag & orc_ok] = "flagged_avoidable"
    cat[miss & ~flag & ~orc_ok] = "silent_unrecoverable"
    cat[miss & ~flag & orc_ok & viol] = "bound_violation"
    cat[miss & ~flag & orc_ok & ~viol] = "model_mismatch"

    d["category"] = cat
    return d


def infeasibility_report(df):
    """Confusion of the INFEASIBLE flag against the oracle plus the silent‑miss rate."""
    d = df
    truth_inf = ~d["oracle_delivered"].astype(bool)
    flag = d["flagged_infeasible"].astype(bool)
    miss = ~d["delivered"].astype(bool)

    tp = int((flag & truth_inf).sum())
    fp = int((flag & ~truth_inf).sum())
    fn = int((~flag & truth_inf).sum())

    return {
        "n": len(d),
        "oracle_infeasible": int(truth_inf.sum()),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "false_infeasible_rate": fp / max(int((~truth_inf).sum()), 1),
        "silent_miss_rate": float((miss & ~flag).mean()),
        "misses": int(miss.sum()),
        "misses_flagged": int((miss & flag).sum()),
    }
