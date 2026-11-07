"""Full benchmark: V0..V6 + V7 (window-conformal engine) + Oracle on IDENTICAL traces,
messages and ground-truth random streams. Reports success, deadline miss, overhead, energy,
four reliability quantities, false-feasible / false-infeasible and infeasibility accuracy.

  python scripts/run_full_comparison.py --seeds 20 --workers 4          # harsh channel
  python scripts/run_full_comparison.py --channel itu --hours 12 --min-att-db 1.0 --seeds 10

Per-seed results are cached (outputs/full_eval_<channel>_seed<N>.csv) so an interrupted run resumes.
Seeds: WC calibration 201+, test 301+ (disjoint from the repo's training/calibration seeds).
NOTE (channel=itu): V3-V6 use the repo's forecaster/calibration trained on the *events* channel;
retrain them on ITU traces for a like-for-like comparison. V7 re-calibrates automatically.
"""
import argparse, os, sys
import multiprocessing as mp
from dataclasses import replace
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
from rcdam.config import preset_harsh, preset_base, preset_itu_site
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.calibration import load_calibration_json
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.optimizer import Message
from rcdam.groundtruth import simulate_true_delivery
from rcdam.window_conformal import WindowConformal
from rcdam.engine import EngineParams, policy_v7_engine
from rcdam.reliability import reliability_report, infeasibility_report, reliability_curve
from rcdam.stats_utils import mcnemar_exact, cluster_bootstrap_diff
from rcdam.variants import (PolicyContext, policy_v0_static, policy_v1_fixed_margin,
                            policy_v2_linear_predictive, policy_v3_quantile_no_conformal,
                            policy_v4_conformal_modcod, policy_v5_stage_a_levers,
                            policy_v6_full_rcdam, policy_oracle)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
PRIOS = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]
A = None
_G = {}


def _init(args):
    global A
    A = args


def _scenario():
    kw = dict(duration_h=A.hours, msg_interval_s=A.interval)
    return {"harsh": preset_harsh, "base": preset_base, "itu": preset_itu_site}[A.channel](**kw)


def _wc():
    path = os.path.join(OUT, f"window_conformal_{A.channel}.json")
    if os.path.exists(path):
        wc = WindowConformal.load(path)
        if wc.supports_offsets:
            return wc
    wc = WindowConformal(); sc0 = _scenario()
    for s in range(201, 201 + A.wc_cal_seeds):
        sc = replace(sc0, seed=s, duration_h=max(A.hours, A.wc_cal_hours))
        wc.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps)
    wc.save(path)
    return wc


def _globals():
    if not _G:
        _G.update(table=ModcodTable(), sc0=_scenario(), wc=_wc(),
                  cal=load_calibration_json(os.path.join(OUT, "calibration.json"))[1],
                  fc=EmpiricalQuantileForecaster.load(os.path.join(OUT, "forecaster_baseline.json")),
                  params=EngineParams(burst_s=A.burst_s, alpha_split=A.alpha_split))
    return _G


def _run_seed(seed):
    tag = (f"{A.channel}_h{A.hours:g}_i{A.interval:g}_m{A.min_att_db:g}_n{A.n_rep}"
           f"_b{A.burst_s:g}_a{A.alpha_split:g}_r{A.rules.replace(',', '')}")
    path = os.path.join(OUT, f"full_eval_{tag}_seed{seed}.csv")      # cache key includes ALL settings
    if os.path.exists(path) and not A.force:
        return pd.read_csv(path).to_dict("records")
    g = _globals(); table, wc = g["table"], g["wc"]
    sc = replace(g["sc0"], seed=seed); rng = np.random.default_rng(seed)
    att = generate_attenuation_db(sc); dt = sc.dt_s; d = sc.feedback_delay_steps
    pols = [("V0_Static", policy_v0_static), ("V1_FixedMargin", policy_v1_fixed_margin),
            ("V2_Predictive", policy_v2_linear_predictive), ("V3_RawQuantile", policy_v3_quantile_no_conformal),
            ("V4_ConformalMODCOD", policy_v4_conformal_modcod), ("V5_JointNoRCDAM", policy_v5_stage_a_levers),
            ("V6_RCDAM_Legacy", policy_v6_full_rcdam),
            ("Oracle", policy_oracle)]
    for rule in A.rules.split(","):
        nm = "V7_RCDAM_WC" if rule == "R1" else f"V7_RCDAM_WC_{rule}"
        pols.append((nm, (lambda c, r=rule: policy_v7_engine(c, wc, g["params"], r))))
    rows, t, i = [], 60.0, 0
    while True:
        t += A.interval + rng.uniform(-2, 2)
        if t >= sc.duration_h * 3600 - 5 * sc.deadline_s:
            break
        idx = int(round(t / dt)); last = max(600, idx - d); cur = float(att[last])
        if cur < A.min_att_db:
            continue
        pn, eps = PRIOS[i % 3]
        msg = Message(size_bits=sc.msg_bits, priority=pn, deadline_s=sc.deadline_s, eps=eps, arrival_idx=idx)
        feats = {"cur": np.array([cur]), "slope5": np.array([(cur - att[max(0, last - 50)]) / 5.0]),
                 "slope30": np.array([(cur - att[max(0, last - 300)]) / 30.0]),
                 "max60": np.array([att[max(0, last - 600): last + 1].max()])}
        Hw = int(round((sc.deadline_s + sc.one_way_delay_s) / dt)) + 5
        ctx = PolicyContext(idx, last, cur, feats, att[idx: idx + Hw], msg, sc, table, g["fc"], g["cal"])
        ref = msg.size_bits / (sc.symbol_rate_baud * table.eff.max())
        orc = policy_oracle(ctx).feasible
        for name, pol in pols:
            dec = pol(ctx)
            outs = [simulate_true_delivery(dec.action, idx, att, msg, table, sc,
                                           np.random.default_rng(seed * 10000 + i + 7919 * r))
                    for r in range(A.n_rep)]
            res = outs[0]
            lg = dec.decision_log or {}
            pdl = float(lg.get("predicted_p", dec.p_lower_bound)) if np.isfinite(dec.p_lower_bound) else np.nan
            pw = dec.action.power_db if dec.action is not None else 0.0
            rows.append(dict(seed=seed, msg=i, variant=name, priority=pn, eps=eps,
                             delivered=bool(res.delivered), delivered_frac=float(np.mean([o.delivered for o in outs])),
                             flagged=not dec.feasible, overhead=res.overhead,
                             energy=res.airtime_s * 10 ** (pw / 10.0) / ref,
                             latency=res.latency_s, pred_p=pdl, pred_lb=float(dec.p_lower_bound),
                             oracle_feasible=bool(orc), att_cur=cur,
                             window_max=float(att[idx: idx + Hw].max()),
                             primary_reason=(lg.get("infeasibility") or {}).get("primary_reason", ""),
                             outcome=lg.get("outcome", "")))
        i += 1
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["harsh", "base", "itu"], default="harsh")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-rep", type=int, default=1, help="ground-truth repetitions per decision")
    ap.add_argument("--min-att-db", type=float, default=-1e9, help="only messages arriving during fades")
    ap.add_argument("--rules", default="R1", help="comma list of V7 infeasibility rules, e.g. R1,R2,R3,R4")
    ap.add_argument("--burst-s", type=float, default=1.0)
    ap.add_argument("--alpha-split", type=float, default=0.5)
    ap.add_argument("--wc-cal-seeds", type=int, default=6)
    ap.add_argument("--wc-cal-hours", type=float, default=4.0)
    ap.add_argument("--force", action="store_true")
    A = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    _globals()                                   # build/cache window-conformal scores once
    seeds = range(301, 301 + A.seeds)
    with Pool(A.workers, initializer=_init, initargs=(A,)) as p:
        rows = [r for part in p.map(_run_seed, seeds) for r in part]
    df = pd.DataFrame(rows); tag = A.channel
    df.to_csv(os.path.join(OUT, f"full_eval_{tag}.csv"), index=False)

    summ = []
    for v, g in df.groupby("variant", sort=False):
        of = g[g.oracle_feasible]
        summ.append(dict(variant=v, n=len(g), delivery_success=g.delivered.mean(),
                         deadline_miss_rate=1 - g.delivered.mean(),
                         avoidable_miss_rate=(~of.delivered).mean(),
                         overhead_mean=g.overhead.mean(), energy_mean=g.energy.mean(),
                         overhead_feasible=of.overhead.mean(), energy_feasible=of.energy.mean(),
                         flagged_rate=g.flagged.mean()))
    s = pd.DataFrame(summ); s.to_csv(os.path.join(OUT, f"full_summary_{tag}.csv"), index=False)
    r = reliability_report(df); r.to_csv(os.path.join(OUT, f"full_reliability_{tag}.csv"), index=False)
    ir = infeasibility_report(df); ir.to_csv(os.path.join(OUT, f"full_infeasibility_{tag}.csv"), index=False)
    for v in df.variant.unique():
        if v.startswith("V7"):
            reliability_curve(df, v).to_csv(os.path.join(OUT, f"full_reliability_curve_{v}_{tag}.csv"), index=False)
    # paired tests on oracle-feasible messages: V7 vs every other policy
    st = []
    v7 = df[(df.variant == "V7_RCDAM_WC") & df.oracle_feasible].set_index(["seed", "msg"])
    for base in df.variant.unique():
        if base.startswith("V7") or base == "Oracle":
            continue
        b = df[(df.variant == base) & df.oracle_feasible].set_index(["seed", "msg"]).reindex(v7.index)
        x = int(((~v7.delivered) & b.delivered).sum()); y = int((v7.delivered & ~b.delivered).sum())
        od = cluster_bootstrap_diff(v7.overhead.values, b.overhead.values, v7.index.get_level_values(0).values)
        st.append(dict(baseline=base, v7_only_misses=x, baseline_only_misses=y,
                       mcnemar_p=mcnemar_exact(x, y), overhead_diff_v7_minus_base=str(od)))
    pd.DataFrame(st).to_csv(os.path.join(OUT, f"full_paired_tests_{tag}.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n== SUMMARY =="); print(s.round(4).to_string(index=False))
    print("\n== RELIABILITY (4 quantities + false-feasible) =="); print(r.round(4).to_string(index=False))
    print("\n== INFEASIBILITY DETECTION vs ORACLE =="); print(ir.round(4).to_string(index=False))
    print("\n== PAIRED TESTS (oracle-feasible msgs) =="); print(pd.DataFrame(st).to_string(index=False))


if __name__ == "__main__":
    main()
