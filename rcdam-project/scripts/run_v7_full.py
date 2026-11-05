"""V7 sweep + LTT certification + 20-seed test (with V6) + window-coverage check.

Stages (run all, or one at a time; each stage writes its own CSV/JSON in outputs/):
  sweep     : grid over (burst_s, alpha_split) on CALIBRATION seeds (401+), then
              Learn-then-Test per priority -> outputs/v7_sweep_raw.csv,
              v7_sweep_ltt.csv, v7_chosen_lambda.json
  test      : V1,V2,V4,V6,V7(chosen lambda) on TEST seeds (301..300+N)
              -> outputs/v7_test_eval.csv, v7_test_summary.csv, v7_test_stats.csv
  coverage  : empirical coverage of the window-max conformal bound on test seeds
              -> outputs/v7_window_coverage.csv

Seed hygiene: WC calibration 201+, sweep/LTT 401+, test 301+ (all disjoint, and
disjoint from the repo's own 1-18 / 101+ seeds).

Usage:
  python scripts/run_v7_full.py --stage all --workers 4
  python scripts/run_v7_full.py --stage test --test-seeds 20 --workers 8
"""
import argparse, itertools, json, os, sys
from dataclasses import replace
import multiprocessing as mp
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
from rcdam.config import preset_harsh
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.calibration import load_calibration_json
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.optimizer import Message
from rcdam.groundtruth import simulate_true_delivery
from rcdam.stats_utils import wilson_interval, exact_upper_bound, mcnemar_exact, cluster_bootstrap_diff
from rcdam.risk_control import ltt_fixed_sequence
from rcdam.window_conformal import WindowConformal, fade_state
from rcdam.variants import (PolicyContext, policy_v1_fixed_margin, policy_v2_linear_predictive,
                            policy_v4_conformal_modcod, policy_v6_full_rcdam,
                            policy_v7_window_conformal, policy_oracle)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
PRIOS = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]
# Ordered LEAST -> MOST conservative (shorter assumed burst and larger alpha_split = less conservative)
GRID = sorted(itertools.product((0.5, 1.0), (0.7, 0.5, 0.3)), key=lambda x: (x[0], -x[1]))

A = None  # parsed args (set in main, inherited by forked workers)
_G = {}


def _init(args):
    """Pool initializer: Windows/macOS start workers with 'spawn', so globals are NOT inherited."""
    global A
    A = args


def _globals():
    if not _G:
        _G["table"] = ModcodTable()
        _G["sc0"] = preset_harsh(duration_h=A.hours, msg_interval_s=A.interval)
        _G["cal_corr"] = load_calibration_json(os.path.join(OUT, "calibration.json"))[1]
        _G["fc"] = EmpiricalQuantileForecaster.load(os.path.join(OUT, "forecaster_baseline.json"))
        _G["wc"] = _get_wc()
    return _G


def _get_wc():
    path = os.path.join(OUT, "window_conformal_scores.json")
    if os.path.exists(path):
        return WindowConformal.load(path)
    wc = WindowConformal()
    sc0 = preset_harsh(duration_h=A.hours, msg_interval_s=A.interval)
    for s in range(201, 201 + A.wc_cal_seeds):
        sc = replace(sc0, seed=s, duration_h=max(A.hours, 4.0))
        wc.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps, [])
    wc.save(path)
    return wc


def _messages(seed):
    """Yield (i, idx, msg, ctx, att, sc) identically for every variant/config."""
    g = _globals(); sc = replace(g["sc0"], seed=seed); rng = np.random.default_rng(seed)
    att = generate_attenuation_db(sc); dt = sc.dt_s; d = sc.feedback_delay_steps
    t, i = 60.0, 0
    while True:
        t += A.interval + rng.uniform(-2, 2)
        if t >= sc.duration_h * 3600 - 5 * sc.deadline_s:
            break
        idx = int(round(t / dt)); pn, eps = PRIOS[i % 3]
        msg = Message(size_bits=sc.msg_bits, priority=pn, deadline_s=sc.deadline_s, eps=eps, arrival_idx=idx)
        last = max(600, idx - d); cur = float(att[last])
        feats = {"cur": np.array([cur]),
                 "slope5": np.array([(cur - att[max(0, last - 50)]) / 5.0]),
                 "slope30": np.array([(cur - att[max(0, last - 300)]) / 30.0]),
                 "max60": np.array([att[max(0, last - 600): last + 1].max()])}
        Hw = int(round((sc.deadline_s + sc.one_way_delay_s) / dt)) + 5
        ctx = PolicyContext(idx, last, cur, feats, att[idx: idx + Hw], msg, sc, g["table"], g["fc"], g["cal_corr"])
        yield i, idx, msg, ctx, att, sc
        i += 1


def _truth(dec, idx, att, msg, sc, seed, i):
    return simulate_true_delivery(dec.action, idx, att, msg, _globals()["table"], sc,
                                  np.random.default_rng(seed * 10000 + i))


# ------------------------------------------------------------------ sweep
def _sweep_seed(seed):
    g = _globals(); rows = []
    for i, idx, msg, ctx, att, sc in _messages(seed):
        for j, (b, a) in enumerate(GRID):
            dec = policy_v7_window_conformal(ctx, g["wc"], burst_s=b, alpha_split=a, n_mc=A.n_mc)
            res = _truth(dec, idx, att, msg, sc, seed, i)
            rows.append(dict(seed=seed, msg=i, priority=msg.priority, eps=msg.eps, setting=j,
                             burst_s=b, alpha_split=a, flagged=not dec.feasible,
                             miss=not res.delivered, overhead=res.overhead))
    return rows


def stage_sweep():
    seeds = range(401, 401 + A.sweep_seeds)
    with Pool(A.workers, initializer=_init, initargs=(A,)) as p:
        rows = [r for part in p.map(_sweep_seed, seeds) for r in part]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "v7_sweep_raw.csv"), index=False)
    chosen, tables = {}, []
    for pn, eps in PRIOS:
        d = df[df.priority == pn]
        ids = sorted(d[["seed", "msg"]].drop_duplicates().itertuples(index=False, name=None))
        key = {k: n for n, k in enumerate(ids)}
        n, m = len(ids), len(GRID)
        miss = np.zeros((n, m), bool); feas = np.zeros((n, m), bool)
        for r in d.itertuples():
            k = key[(r.seed, r.msg)]; miss[k, r.setting] = r.miss; feas[k, r.setting] = not r.flagged
        # certify at the priority's own epsilon; critical usually lacks the sample size -> falls back
        cert, tab = ltt_fixed_sequence(miss, feas, eps, delta=A.delta)
        t = pd.DataFrame(tab); t.insert(0, "priority", pn); t.insert(1, "eps", eps); tables.append(t)
        if cert is None:
            chosen[pn] = dict(setting=m - 1, burst_s=GRID[m - 1][0], alpha_split=GRID[m - 1][1], certified=False)
        else:
            chosen[pn] = dict(setting=int(cert), burst_s=GRID[cert][0], alpha_split=GRID[cert][1], certified=True)
    pd.concat(tables).to_csv(os.path.join(OUT, "v7_sweep_ltt.csv"), index=False)
    json.dump(chosen, open(os.path.join(OUT, "v7_chosen_lambda.json"), "w"), indent=2)
    print(json.dumps(chosen, indent=2))
    print("Rows with certified=False for 'critical' usually mean too few feasible messages: "
          "raise --sweep-seeds (need ~3000 feasible critical messages for eps=1e-3).")


# ------------------------------------------------------------------ test
def _test_seed(seed):
    g = _globals(); chosen = json.load(open(os.path.join(OUT, "v7_chosen_lambda.json"))); rows = []
    for i, idx, msg, ctx, att, sc in _messages(seed):
        orc = policy_oracle(ctx).feasible
        c = chosen[msg.priority]
        pols = [("V1_Margin", policy_v1_fixed_margin), ("V2_Predictive", policy_v2_linear_predictive),
                ("V4_Conformal_MODCOD", policy_v4_conformal_modcod), ("V6_RC_DAM_Full", policy_v6_full_rcdam),
                ("V7_WindowConformal", lambda cx: policy_v7_window_conformal(
                    cx, g["wc"], burst_s=c["burst_s"], alpha_split=c["alpha_split"], n_mc=A.n_mc))]
        for name, pol in pols:
            dec = pol(ctx); res = _truth(dec, idx, att, msg, sc, seed, i)
            rows.append(dict(seed=seed, msg=i, variant=name, priority=msg.priority, eps=msg.eps,
                             delivered=res.delivered, flagged=not dec.feasible, overhead=res.overhead,
                             oracle_feasible=orc))
    return rows


def stage_test():
    seeds = range(301, 301 + A.test_seeds)
    with Pool(A.workers, initializer=_init, initargs=(A,)) as p:
        rows = [r for part in p.map(_test_seed, seeds) for r in part]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "v7_test_eval.csv"), index=False)
    out = []
    for (v, pn), gp in df.groupby(["variant", "priority"], sort=False):
        un = gp[~gp.flagged]; k = int((~un.delivered).sum()); n = len(un)
        lo, hi = wilson_interval(k, max(1, n)); ex = exact_upper_bound(k, max(1, n))
        of = gp[gp.oracle_feasible]
        out.append(dict(variant=v, priority=pn, eps=gp.eps.iloc[0], n=len(gp), n_unflagged=n,
                        silent_misses=k, silent_miss_rate=k / max(1, n), wilson_hi=hi, exact_hi_95=ex,
                        meets_eps=bool(ex <= gp.eps.iloc[0]), false_infeasible=float(of.flagged.mean()),
                        avoidable_miss=float((~of.delivered).mean()), overhead_feasible=float(of.overhead.mean())))
    summ = pd.DataFrame(out); summ.to_csv(os.path.join(OUT, "v7_test_summary.csv"), index=False)
    print(summ.to_string(index=False))
    # paired stats: V7 vs each baseline on oracle-feasible messages
    st = []
    v7 = df[(df.variant == "V7_WindowConformal") & df.oracle_feasible].set_index(["seed", "msg"])
    for base in ("V1_Margin", "V2_Predictive", "V4_Conformal_MODCOD", "V6_RC_DAM_Full"):
        b = df[(df.variant == base) & df.oracle_feasible].set_index(["seed", "msg"]).loc[v7.index]
        b_cnt = int(((~v7.delivered) & b.delivered).sum()); c_cnt = int((v7.delivered & ~b.delivered).sum())
        d_oh = cluster_bootstrap_diff(v7.overhead.values, b.overhead.values, v7.index.get_level_values(0).values)
        st.append(dict(baseline=base, v7_only_misses=b_cnt, baseline_only_misses=c_cnt,
                       mcnemar_p=mcnemar_exact(b_cnt, c_cnt), overhead_diff_v7_minus_base=str(d_oh)))
    pd.DataFrame(st).to_csv(os.path.join(OUT, "v7_test_stats.csv"), index=False)
    print(pd.DataFrame(st).to_string(index=False))


# ------------------------------------------------------------------ coverage
def _cov_seed(seed):
    g = _globals(); rows = []
    for i, idx, msg, ctx, att, sc in _messages(seed):
        w = int(round(msg.deadline_s / sc.dt_s)); wmax = float(att[idx: idx + w].max())
        for frac in (0.1, 0.5, 1.0):  # alpha_split*eps class sweep
            alpha = round(frac * msg.eps, 6)
            rows.append(dict(seed=seed, state=fade_state(ctx.cur_att_db), alpha=alpha, nominal=1 - alpha,
                             covered=bool(wmax <= g["wc"].bound(ctx.cur_att_db, msg.deadline_s, alpha))))
    return rows


def stage_coverage():
    seeds = range(301, 301 + A.test_seeds)
    with Pool(A.workers, initializer=_init, initargs=(A,)) as p:
        df = pd.DataFrame([r for part in p.map(_cov_seed, seeds) for r in part])
    cov = df.groupby(["state", "alpha"]).agg(n=("covered", "size"), coverage=("covered", "mean")).reset_index()
    cov["nominal"] = 1 - cov.alpha; cov["ok"] = cov.coverage >= cov.nominal - 2 * np.sqrt(
        cov.nominal * (1 - cov.nominal) / cov.n)
    cov.to_csv(os.path.join(OUT, "v7_window_coverage.csv"), index=False); print(cov.to_string(index=False))


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sweep", "test", "coverage", "all"], default="all")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sweep-seeds", type=int, default=12)
    ap.add_argument("--test-seeds", type=int, default=20)
    ap.add_argument("--wc-cal-seeds", type=int, default=6)
    ap.add_argument("--n-mc", type=int, default=150)
    ap.add_argument("--delta", type=float, default=0.05)
    A = ap.parse_args()
    mp.set_start_method("spawn", force=True)  # same behaviour on Windows, Linux, macOS
    _globals()  # build/cache window-conformal scores once in the parent (avoids worker races)
    if A.stage in ("sweep", "all"): stage_sweep()
    if A.stage in ("test", "all"): stage_test()
    if A.stage in ("coverage", "all"): stage_coverage()


if __name__ == "__main__":
    main()
