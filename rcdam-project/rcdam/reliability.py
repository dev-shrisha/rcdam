"""Reliability accounting: keeps FOUR quantities separate and adds false-feasible metrics.

  1. predicted_p       - what the optimizer predicts for delivery probability (MC point estimate)
  2. calibrated_lb     - calibrated lower confidence bound the optimizer reports
  3. actual_rate_sim   - delivery success measured in the ground-truth simulator (same decisions)
  4. validated_lb      - one-sided Clopper-Pearson lower bound on success among messages the
                         system CLAIMED feasible, computed on INDEPENDENT test seeds only

False-feasible : optimizer says "feasible" (reliability + deadline OK) but the message fails.
False-infeasible: optimizer says "infeasible" although the oracle shows a deliverable action exists.
Input: a DataFrame with columns
  variant, priority, eps, delivered, flagged, oracle_feasible, pred_p, pred_lb
  (optional) delivered_frac - delivery fraction over repeated ground-truth simulations
"""
import numpy as np
import pandas as pd
from scipy.stats import beta


def cp_lower(k: int, n: int, conf: float = 0.95) -> float:
    if n <= 0:
        return float("nan")
    return 0.0 if k <= 0 else float(beta.ppf(1.0 - conf, k, n - k + 1))


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    if n <= 0:
        return float("nan")
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def reliability_report(df: pd.DataFrame, conf: float = 0.95) -> pd.DataFrame:
    rows = []
    for (v, pn), g in df.groupby(["variant", "priority"], sort=False):
        eps = float(g.eps.iloc[0])
        cl = g[~g.flagged]                                   # messages claimed feasible
        n, k = len(cl), int(cl.delivered.sum())
        ff = n - k
        of = g[g.oracle_feasible]
        actual_sim = float(cl.delivered_frac.mean()) if "delivered_frac" in cl and n else float("nan")
        vlb = cp_lower(k, n, conf)
        rows.append(dict(
            variant=v, priority=pn, eps=eps, required=1 - eps, n=len(g), n_claimed_feasible=n,
            predicted_p=float(cl.pred_p.mean()) if n else float("nan"),
            calibrated_lb=float(cl.pred_lb.mean()) if n else float("nan"),
            actual_rate_sim=actual_sim,
            actual_rate_binary=(k / n) if n else float("nan"),
            validated_lb=vlb, validated_meets_requirement=bool(vlb >= 1 - eps) if n else False,
            false_feasible=ff, false_feasible_rate=(ff / n) if n else float("nan"),
            false_feasible_ub=cp_upper(ff, n, conf),
            false_infeasible_rate=float(of.flagged.mean()) if len(of) else float("nan"),
            calibration_gap=(float(cl.pred_p.mean()) - k / n) if n else float("nan"),
        ))
    return pd.DataFrame(rows)


def infeasibility_report(df: pd.DataFrame) -> pd.DataFrame:
    """Confusion matrix of the 'infeasible' flag against the oracle's feasibility."""
    rows = []
    for v, g in df.groupby("variant", sort=False):
        tp = int((g.flagged & ~g.oracle_feasible).sum())     # correctly flagged infeasible
        fp = int((g.flagged & g.oracle_feasible).sum())      # false-infeasible
        fn = int((~g.flagged & ~g.oracle_feasible).sum())    # missed infeasible (silent)
        tn = int((~g.flagged & g.oracle_feasible).sum())
        n = tp + fp + fn + tn
        rows.append(dict(variant=v, n=n, TP_flagged_infeasible=tp, FP_false_infeasible=fp,
                         FN_missed_infeasible=fn, TN_correct_feasible=tn,
                         accuracy=(tp + tn) / n if n else np.nan,
                         precision=tp / (tp + fp) if tp + fp else np.nan,
                         recall=tp / (tp + fn) if tp + fn else np.nan))
    return pd.DataFrame(rows)


def reliability_curve(df: pd.DataFrame, variant: str, bins=(0, .5, .8, .9, .95, .99, .999, 1.0001)) -> pd.DataFrame:
    """Reliability-diagram data: predicted vs actual delivery in predicted-probability bins."""
    g = df[(df.variant == variant) & (~df.flagged)].copy()
    g["bin"] = pd.cut(g.pred_p, bins=list(bins), right=False)
    out = g.groupby("bin", observed=True).agg(n=("delivered", "size"), predicted=("pred_p", "mean"),
                                              actual=("delivered", "mean")).reset_index()
    return out
