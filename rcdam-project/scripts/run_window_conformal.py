"""Compare V1/V2/V4/V6 (existing) with V7 (window-conformal + trajectory evaluator).
Disjoint seeds: WC calibration 201.., test 301.. . Identical traces/messages for all variants."""
import argparse, os, sys, time
from dataclasses import replace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
from rcdam.config import preset_harsh
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.calibration import load_calibration_json
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.optimizer import Message
from rcdam.groundtruth import simulate_true_delivery
from rcdam.stats_utils import wilson_interval
from rcdam.window_conformal import WindowConformal
from rcdam.variants import (PolicyContext, policy_v1_fixed_margin, policy_v2_linear_predictive,
                            policy_v4_conformal_modcod, policy_v6_full_rcdam,
                            policy_v7_window_conformal, policy_oracle)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
ap = argparse.ArgumentParser()
ap.add_argument("--cal-seeds", type=int, default=6)
ap.add_argument("--test-seeds", type=int, default=3)
ap.add_argument("--hours", type=float, default=2.0)
ap.add_argument("--interval", type=float, default=30.0)
ap.add_argument("--skip-v6", action="store_true")
a = ap.parse_args()

table = ModcodTable()
sc0 = preset_harsh(duration_h=a.hours, msg_interval_s=a.interval)
_, cal_corr = load_calibration_json(os.path.join(OUT, "calibration.json"))
fc = EmpiricalQuantileForecaster.load(os.path.join(OUT, "forecaster_baseline.json"))

# ---- 1) calibrate window-conformal on disjoint traces
wc = WindowConformal()
for s in range(201, 201 + a.cal_seeds):
    sc = replace(sc0, seed=s, duration_h=max(a.hours, 4.0))
    wc.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps, [])
wc.save(os.path.join(OUT, "window_conformal_scores.json"))
print("calibration strata:", {k: len(v) for k, v in sorted(wc.scores.items())})

VARS = [("V1_Margin", policy_v1_fixed_margin), ("V2_Predictive", policy_v2_linear_predictive),
        ("V4_Conformal_MODCOD", policy_v4_conformal_modcod)]
if not a.skip_v6:
    VARS.append(("V6_RC_DAM_Full", policy_v6_full_rcdam))
VARS.append(("V7_WindowConformal", lambda c: policy_v7_window_conformal(c, wc)))

rows = []
prios = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]
for s in range(301, 301 + a.test_seeds):
    sc = replace(sc0, seed=s)
    rng = np.random.default_rng(s)
    att = generate_attenuation_db(sc); dt = sc.dt_s; d = sc.feedback_delay_steps
    t, i = 60.0, 0
    while t < sc.duration_h * 3600 - 5 * sc.deadline_s:
        t += a.interval + rng.uniform(-2, 2); idx = int(round(t / dt))
        if t >= sc.duration_h * 3600 - 5 * sc.deadline_s:
            break
        pn, eps = prios[i % 3]
        msg = Message(size_bits=sc.msg_bits, priority=pn, deadline_s=sc.deadline_s, eps=eps, arrival_idx=idx)
        last = max(600, idx - d); cur = float(att[last])
        feats = {"cur": np.array([cur]),
                 "slope5": np.array([(cur - att[max(0, last - 50)]) / 5.0]),
                 "slope30": np.array([(cur - att[max(0, last - 300)]) / 30.0]),
                 "max60": np.array([att[max(0, last - 600): last + 1].max()])}
        Hw = int(round((sc.deadline_s + sc.one_way_delay_s) / dt)) + 5
        ctx = PolicyContext(idx, last, cur, feats, att[idx: idx + Hw], msg, sc, table, fc, cal_corr)
        orc = policy_oracle(ctx).feasible
        for name, pol in VARS:
            dec = pol(ctx)
            res = simulate_true_delivery(dec.action, idx, att, msg, table, sc,
                                         np.random.default_rng(s * 10000 + i))
            rows.append(dict(seed=s, msg=i, variant=name, priority=pn, eps=eps,
                             delivered=res.delivered, flagged=not dec.feasible,
                             overhead=res.overhead, oracle_feasible=orc, window_max=float(att[idx: idx + Hw].max()),
                             cur=cur))
        i += 1
    print(f"seed {s}: {i} messages", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "window_conformal_eval.csv"), index=False)
out = []
for v, g in df.groupby("variant", sort=False):
    n = len(g); of = g[g.oracle_feasible]
    unflag = g[~g.flagged]
    silent = int((~unflag.delivered).sum())
    lo, hi = wilson_interval(silent, max(1, len(unflag)))
    out.append(dict(variant=v, n=n, oracle_infeasible=1 - g.oracle_feasible.mean(),
                    avoidable_miss=(~of.delivered).mean(),
                    false_infeasible=(of.flagged).mean(),
                    silent_miss_given_unflagged=silent / max(1, len(unflag)),
                    sm_lo=lo, sm_hi=hi, overhead_on_feasible=of.overhead.mean()))
res = pd.DataFrame(out); print(res.to_string(index=False))
res.to_csv(os.path.join(OUT, "window_conformal_summary.csv"), index=False)
for pn, g in df[df.variant == "V7_WindowConformal"].groupby("priority"):
    un = g[~g.flagged]; print(pn, "eps=", g.eps.iloc[0], "silent-miss|unflagged=",
          round(float((~un.delivered).mean()), 4), "n_unflagged=", len(un))
