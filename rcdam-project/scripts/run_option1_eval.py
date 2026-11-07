"""Option 1: transmission-window-aware calibration - coverage, shift robustness, overhead.

  python scripts/run_option1_eval.py --part A                    # coverage across fade conditions
  python scripts/run_option1_eval.py --part B --seeds 10         # distribution shift: static vs rolling vs ACI
  python scripts/run_option1_eval.py --part C --seeds 20 --workers 4   # certified per-class lambda vs defaults
  python scripts/run_option1_eval.py --part all

Seeds: calibration 201-206 (harsh), shifted-recalibration reference 601-606, tests 301+.
"""
import argparse, json, os, sys
import multiprocessing as mp
from dataclasses import replace
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
from rcdam.config import preset_harsh, preset_itu_site
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.optimizer import Message
from rcdam.window_conformal import WindowConformal, fade_state, trend_bucket, LEN_UPPER_S
from rcdam.adaptive_window import RollingWindowConformal, AdaptiveWindowConformal
from rcdam.engine import EngineParams, decide, load_priority_params
from rcdam.variants import PolicyContext, policy_oracle
from rcdam.groundtruth import simulate_true_delivery
from rcdam.reliability import reliability_report, infeasibility_report
from rcdam.stats_utils import wilson_interval

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
ALPHAS = (5e-4, 5e-3, 5e-2)
A = None


def _fit(sc0, seeds, hours=4.0):
    wc = WindowConformal()
    for s in seeds:
        sc = replace(sc0, seed=s, duration_h=hours)
        wc.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps)
    return wc


def _base_wc():
    path = os.path.join(OUT, "window_conformal_harsh.json")
    if os.path.exists(path):
        w = WindowConformal.load(path)
        if w.supports_offsets:
            return w
    w = _fit(preset_harsh(), range(201, 207)); w.save(path); return w


def _samples(att, sc, step_s, rng):
    """(t_idx, last, slope) at regular times; windows are [t, t+L], L = deadline - one-way delay."""
    d, dt = sc.feedback_delay_steps, sc.dt_s
    L = sc.deadline_s - sc.one_way_delay_s
    for t in np.arange(70.0, sc.duration_h * 3600 - 10, step_s + rng.uniform(-1, 1)):
        idx = int(round(t / dt)); last = idx - d
        if last < 60 or idx + int(L / dt) + 2 >= len(att):
            continue
        yield idx, float(att[last]), (float(att[last]) - float(att[last - 50])) / 5.0, L


# ----------------------------------------------------------------------------- Part A
def part_a():
    wc = _base_wc() if A.cal_seeds == 6 else _fit(preset_harsh(), range(201, 201 + A.cal_seeds))
    print(f"[Part A] calibration seeds: {A.cal_seeds}"); sc0 = preset_harsh(duration_h=3.0); rows = []
    for seed in range(301, 301 + A.seeds):
        sc = replace(sc0, seed=seed); att = generate_attenuation_db(sc); dt = sc.dt_s
        rng = np.random.default_rng(seed)
        for idx, last, slope, L in _samples(att, sc, 20.0, rng):
            for off in (0.0, 0.5, 1.0):
                o = int(round(off / dt)); w = int(round((L - off) / dt)) if L - off > 0.2 else int(round(0.2 / dt))
                wmax = float(att[idx + o: idx + o + w].max())
                for a in ALPHAS:
                    rows.append(dict(alpha=a, offset=off, state=fade_state(last), trend=trend_bucket(slope),
                                     covered=wmax <= wc.bound(last, L - off, a, off, slope)))
    df = pd.DataFrame(rows); res = []
    for dim in ("state", "trend", "offset"):
        g = df.groupby(["alpha", dim]).covered.agg(["size", "sum"]).reset_index()
        for r in g.itertuples():
            nom = 1 - r.alpha; cov = r.sum / r.size; lo, hi = wilson_interval(int(r.sum), int(r.size))
            res.append(dict(dimension=dim, level=getattr(r, dim), alpha=r.alpha, n=int(r.size), coverage=cov, nominal=nom,
                            wilson_lo=lo, wilson_hi=hi, under_covered=bool(hi < nom)))
    pd.DataFrame(res).to_csv(os.path.join(OUT, f"opt1_coverage_conditions_cal{A.cal_seeds}.csv"), index=False)
    allc = df.groupby("alpha").covered.mean(); print("overall coverage:", allc.round(4).to_dict())
    r = pd.DataFrame(res); print(r[r.under_covered].to_string(index=False) if r.under_covered.any()
                                 else "No stratum is significantly under-covered (Wilson upper bound >= nominal everywhere).")


# ----------------------------------------------------------------------------- Part B
SCEN = {
    "calm":  lambda: preset_harsh(duration_h=A.hours, events_per_day=2.0, event_peak_ln_mean=1.8),
    "storm": lambda: preset_harsh(duration_h=A.hours, events_per_day=24.0, event_peak_ln_mean=3.0),
    "itu_ka": lambda: preset_itu_site(duration_h=A.hours),
}


def _shift_seed(task):
    scen, seed = task
    base = _base_wc(); sc = replace(SCEN[scen](), seed=seed); att = generate_attenuation_db(sc)
    meths = {"static": base,
             "rolling": RollingWindowConformal(base, cap=A.cap),
             "aci": AdaptiveWindowConformal(base, gamma=A.gamma)}
    rng = np.random.default_rng(seed); dt = sc.dt_s; rows = []
    for idx, last, slope, L in _samples(att, sc, 20.0, rng):
        w = int(round(L / dt)); wmax = float(att[idx: idx + w].max())
        for a in ALPHAS:
            for name, m in meths.items():
                b = m.bound(last, L, a, 0.0, slope)
                rows.append(dict(scenario=scen, seed=seed, method=name, alpha=a, covered=wmax <= b, slack_db=b - wmax))
        for a in ALPHAS:                                   # updates become available after the window ends
            meths["rolling"].update(last, wmax, L, 0.0, slope)
            meths["aci"].update(last, wmax, L, a, 0.0, slope)
    return rows


def part_b():
    sc0 = SCEN["calm"]()
    tasks = [(s, seed) for s in SCEN for seed in range(301, 301 + A.seeds)]
    with Pool(A.workers, initializer=_init, initargs=(A,)) as p:
        rows = [r for part in p.map(_shift_seed, tasks) for r in part]
    df = pd.DataFrame(rows)
    # reference: recalibrated on the shifted channel itself (disjoint seeds) = best possible static
    ref = []
    for scen in SCEN:
        w = _fit(SCEN[scen](), range(601, 601 + A.ref_seeds), hours=max(A.hours, 4.0))
        for seed in range(301, 301 + A.seeds):
            sc = replace(SCEN[scen](), seed=seed); att = generate_attenuation_db(sc); rng = np.random.default_rng(seed)
            for idx, last, slope, L in _samples(att, sc, 20.0, rng):
                wmax = float(att[idx: idx + int(round(L / sc.dt_s))].max())
                for a in ALPHAS:
                    b = w.bound(last, L, a, 0.0, slope)
                    ref.append(dict(scenario=scen, seed=seed, method="recalibrated_reference", alpha=a, covered=wmax <= b, slack_db=b - wmax))
    df = pd.concat([df, pd.DataFrame(ref)])
    df.to_csv(os.path.join(OUT, "opt1_shift_raw.csv"), index=False)
    s = df.groupby(["scenario", "method", "alpha"]).agg(n=("covered", "size"), coverage=("covered", "mean"),
                                                         mean_slack_db=("slack_db", "mean")).reset_index()
    s["nominal"] = 1 - s.alpha; s["coverage_gap"] = s.coverage - s.nominal
    s.to_csv(os.path.join(OUT, "opt1_shift_summary.csv"), index=False)
    pd.set_option("display.width", 220); print(s.round(4).to_string(index=False))


# ----------------------------------------------------------------------------- Part C
def _c_seed(seed):
    tag = f"opt1c_h{A.hours:g}_i{A.interval:g}"; path = os.path.join(OUT, f"{tag}_seed{seed}.csv")
    if os.path.exists(path) and not A.force:
        return pd.read_csv(path).to_dict("records")
    wc = _base_wc(); table = ModcodTable(); sc0 = preset_harsh(duration_h=A.hours)
    default = EngineParams()
    certified = load_priority_params(os.path.join(OUT, "v7_chosen_lambda.json"), default)
    sc = replace(sc0, seed=seed); att = generate_attenuation_db(sc); dt = sc.dt_s; d = sc.feedback_delay_steps
    rng = np.random.default_rng(seed); rows, t, i = [], 60.0, 0
    prios = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]
    while True:
        t += A.interval + rng.uniform(-2, 2)
        if t >= sc.duration_h * 3600 - 5 * sc.deadline_s:
            break
        idx = int(round(t / dt)); last = max(600, idx - d); cur = float(att[last]); pn, eps = prios[i % 3]
        msg = Message(size_bits=sc.msg_bits, priority=pn, deadline_s=sc.deadline_s, eps=eps, arrival_idx=idx)
        feats = {"cur": np.array([cur]), "slope5": np.array([(cur - att[max(0, last - 50)]) / 5.0]),
                 "slope30": np.array([0.0]), "max60": np.array([att[max(0, last - 600): last + 1].max()])}
        Hw = int(round((sc.deadline_s + sc.one_way_delay_s) / dt)) + 5
        ctx = PolicyContext(idx, last, cur, feats, att[idx: idx + Hw], msg, sc, table, None, None)
        ref = msg.size_bits / (sc.symbol_rate_baud * table.eff.max()); orc = policy_oracle(ctx).feasible
        for name, prm in (("V7_default", default), ("V7_certified_lambda", certified)):
            dec = decide(ctx, wc, prm, "R1")
            res = simulate_true_delivery(dec.action, idx, att, msg, table, sc, np.random.default_rng(seed * 10000 + i))
            lg = dec.decision_log or {}
            rows.append(dict(seed=seed, msg=i, variant=name, priority=pn, eps=eps, delivered=bool(res.delivered),
                             delivered_frac=float(res.delivered), flagged=not dec.feasible, overhead=res.overhead,
                             energy=res.airtime_s * 10 ** ((dec.action.power_db if dec.action is not None else 0.0) / 10) / ref,
                             pred_p=float(lg.get("predicted_p", np.nan)), pred_lb=float(dec.p_lower_bound),
                             oracle_feasible=bool(orc), inf_class=lg.get("infeasibility_class", "")))
        i += 1
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def part_c():
    seeds = range(301, 301 + A.seeds)
    with Pool(A.workers, initializer=_init, initargs=(A,)) as p:
        df = pd.DataFrame([r for part in p.map(_c_seed, seeds) for r in part])
    df.to_csv(os.path.join(OUT, "opt1c_all.csv"), index=False)
    out = []
    for v, g in df.groupby("variant", sort=False):
        of = g[g.oracle_feasible]
        out.append(dict(variant=v, n=len(g), avoidable_miss=(~of.delivered).mean(), false_infeasible=of.flagged.mean(),
                        overhead_feasible=of.overhead.mean(), energy_feasible=of.energy.mean(), flagged_rate=g.flagged.mean()))
    s = pd.DataFrame(out); s.to_csv(os.path.join(OUT, "opt1c_summary.csv"), index=False)
    r = reliability_report(df); r.to_csv(os.path.join(OUT, "opt1c_reliability.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(s.round(4).to_string(index=False)); print(r[["variant", "priority", "n_claimed_feasible", "false_feasible", "validated_lb", "false_infeasible_rate"]].round(4).to_string(index=False))
    print(infeasibility_report(df).round(4).to_string(index=False))


def _init(args):
    global A
    A = args


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--ref-seeds", type=int, default=4)
    ap.add_argument("--cal-seeds", type=int, default=6, help="Part A: number of 4-h calibration traces")
    ap.add_argument("--force", action="store_true")
    A = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    _base_wc()
    if A.part in ("A", "all"): part_a()
    if A.part in ("B", "all"): part_b()
    if A.part in ("C", "all"): part_c()


if __name__ == "__main__":
    main()
