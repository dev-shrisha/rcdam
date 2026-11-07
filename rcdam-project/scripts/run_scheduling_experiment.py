"""Queue experiment: does proactive time-to-fade scheduling deliver more messages?

Workload (identical for every policy): bursts of messages (mean gap --burst-gap-s, 1+Poisson(--burst-extra)
messages each) with SLACK deadlines (30/60/120/240 s after arrival) and reliability classes.
One link serves one transmission at a time.

Policies
  FIFO_reactive : oldest message first; window-conformal engine (V7) decides each attempt; waits if infeasible
  EDF_reactive  : earliest deadline first; same engine
  Proactive_TTF : conformal time-to-fade plan (guaranteed pre-fade windows); reactive EDF+engine for the rest

  python scripts/run_scheduling_experiment.py --seeds 12 --hours 3 --workers 4
Seeds: engine/TTF calibration 201+, test 301+ (disjoint). Per-seed results are cached with a settings-specific key.
"""
import argparse, os, sys
import multiprocessing as mp
from dataclasses import replace
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
from rcdam.config import preset_harsh
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.optimizer import Message, Action
from rcdam.groundtruth import simulate_true_delivery
from rcdam.window_conformal import WindowConformal
from rcdam.time_to_fade import TimeToFade
from rcdam.engine import EngineParams, decide
from rcdam.scheduler import plan, QueuedMessage, SchedulerParams
from rcdam.variants import PolicyContext
from rcdam.stats_utils import mcnemar_exact

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
PRIO = [("critical", 1e-3, .2), ("normal", 1e-2, .5), ("best_effort", 1e-1, .3)]
A = None
_G = {}


def _init(args):
    global A
    A = args


def _globals():
    if not _G:
        sc0 = preset_harsh(duration_h=A.hours)
        wcp = os.path.join(OUT, "window_conformal_harsh.json")
        wc = WindowConformal.load(wcp) if os.path.exists(wcp) else None
        if wc is None or not wc.supports_offsets:
            wc = WindowConformal()
            for s in range(201, 207):
                sc = replace(sc0, seed=s, duration_h=4.0)
                wc.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps)
            wc.save(wcp)
        ttp = os.path.join(OUT, "time_to_fade_harsh.json")
        if os.path.exists(ttp):
            ttf = TimeToFade.load(ttp)
        else:
            ttf = TimeToFade()
            for s in range(201, 207):
                sc = replace(sc0, seed=s, duration_h=4.0)
                ttf.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps)
            ttf.save(ttp)
        _G.update(sc0=sc0, table=ModcodTable(), wc=wc, ttf=ttf, ep=EngineParams(), sp=SchedulerParams())
    return _G


def _workload(seed, sc):
    rng = np.random.default_rng(seed + 9000)
    msgs, t, mid = [], 70.0, 0
    while t < sc.duration_h * 3600 - 400:
        t += rng.exponential(A.burst_gap_s)
        for _ in range(1 + rng.poisson(A.burst_extra)):
            pn, eps, _p = PRIO[rng.choice(3, p=[x[2] for x in PRIO])]
            slack = float(rng.choice([30, 60, 120, 240]))
            msgs.append(QueuedMessage(mid, sc.msg_bits, pn, eps, t, t + slack)); mid += 1
    return [m for m in msgs if m.deadline_abs_s < sc.duration_h * 3600 - 10]


def _simulate(policy, seed):
    g = _globals(); sc = replace(g["sc0"], seed=seed); table = g["table"]
    att = generate_attenuation_db(sc); dt = sc.dt_s; d = sc.feedback_delay_steps
    work = _workload(seed, sc); pending = sorted(work, key=lambda m: m.arrival_s)
    best_air = sc.msg_bits / (sc.symbol_rate_baud * table.eff.max())
    queue, rows, att_log = [], {}, []
    t, ai = 70.0, 0
    for m in work:
        rows[m.id] = dict(seed=seed, policy=policy, id=m.id, priority=m.priority, eps=m.eps, slack_s=m.deadline_abs_s - m.arrival_s,
                          delivered=False, latency=np.nan, attempts=0, airtime=0.0, energy=0.0, proactive=False,
                          flagged_send=False, ttf_violation=False)
    ref = sc.msg_bits / (sc.symbol_rate_baud * table.eff.max())
    end_t = sc.duration_h * 3600 - 5
    while t < end_t:
        while ai < len(pending) and pending[ai].arrival_s <= t:
            queue.append(pending[ai]); ai += 1
        queue = [m for m in queue if m.deadline_abs_s - t > best_air + sc.one_way_delay_s]   # expired -> stays missed
        if not queue:
            if ai >= len(pending): break
            t = max(t + dt, pending[ai].arrival_s); continue
        idx = int(round(t / dt)); last = max(d + 60, idx - d); cur = float(att[last])
        slope = (cur - att[last - 50]) / 5.0
        chosen = action = None; proactive = False; flagged = False; theta = None
        if policy == "Proactive_TTF":
            items, left = plan(t, cur, slope, queue, g["ttf"], table, sc, g["sp"])
            if items:
                it = items[0]; chosen = next(m for m in queue if m.id == it.msg_id)
                action, proactive, theta = it.action, True, it.theta_db
        if chosen is None:
            chosen = min(queue, key=(lambda m: m.arrival_s) if policy == "FIFO_reactive"
                         else (lambda m: (m.deadline_abs_s, m.eps)))
            rem = chosen.deadline_abs_s - t; dex = min(A.exec_window_s, rem)
            msg = Message(size_bits=chosen.size_bits, priority=chosen.priority, deadline_s=dex, eps=chosen.eps, arrival_idx=idx)
            feats = {"cur": np.array([cur]), "slope5": np.array([slope]), "slope30": np.array([0.0]), "max60": np.array([cur])}
            dec = decide(PolicyContext(idx, last, cur, feats, None, msg, sc, table, None, None), g["wc"], g["ep"], "R1")
            if dec.feasible:
                action = dec.action
            elif rem <= dex + A.retry_s and dec.action is not None:
                action, flagged = dec.action, True                              # last chance: send and flag
            else:
                t += A.retry_s; continue                                        # wait for a better channel
        rem = chosen.deadline_abs_s - t; dex = min(A.exec_window_s, rem)
        msg = Message(size_bits=chosen.size_bits, priority=chosen.priority, deadline_s=dex, eps=chosen.eps, arrival_idx=idx)
        res = simulate_true_delivery(action, idx, att, msg, table, sc,
                                     np.random.default_rng(seed * 100003 + chosen.id * 31 + chosen.attempts))
        r = rows[chosen.id]; r["attempts"] += 1; r["airtime"] += res.airtime_s
        r["energy"] += res.airtime_s * 10 ** (action.power_db / 10.0) / ref
        r["proactive"] |= proactive; r["flagged_send"] |= flagged
        if proactive:
            w = max(1, int(np.ceil(res.airtime_s / dt)))
            r["ttf_violation"] |= bool(att[idx: idx + w].max() > theta + 0.5)       # exceeded protection tolerance
        if res.delivered:
            r["delivered"] = True; r["latency"] = t + res.latency_s - chosen.arrival_s
            queue = [m for m in queue if m.id != chosen.id]
        else:
            chosen.attempts += 1
            if chosen.attempts >= A.max_attempts:
                queue = [m for m in queue if m.id != chosen.id]
        t += max(res.airtime_s, dt)
    return list(rows.values())


def _run(task):
    policy, seed = task
    tag = f"h{A.hours:g}_g{A.burst_gap_s:g}_x{A.burst_extra:g}_e{A.exec_window_s:g}_r{A.retry_s:g}_a{A.max_attempts}"
    path = os.path.join(OUT, f"sched_{tag}_{policy}_seed{seed}.csv")
    if os.path.exists(path) and not A.force:
        return pd.read_csv(path).to_dict("records")
    rows = _simulate(policy, seed)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--burst-gap-s", type=float, default=240.0)
    ap.add_argument("--burst-extra", type=float, default=3.0)
    ap.add_argument("--exec-window-s", type=float, default=2.0)
    ap.add_argument("--retry-s", type=float, default=5.0)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    A = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    _globals()
    pols = ["FIFO_reactive", "EDF_reactive", "Proactive_TTF"]
    tasks = [(p, s) for s in range(301, 301 + A.seeds) for p in pols]
    with Pool(A.workers, initializer=_init, initargs=(A,)) as pool:
        df = pd.DataFrame([r for part in pool.map(_run, tasks) for r in part])
    df.to_csv(os.path.join(OUT, "sched_all.csv"), index=False)
    s = []
    for p, g in df.groupby("policy", sort=False):
        dl = g[g.delivered]
        s.append(dict(policy=p, n=len(g), delivered=g.delivered.mean(), mean_latency_s=dl.latency.mean(),
                      attempts_per_msg=g.attempts.mean(),
                      wasted_airtime_s_per_msg=(g.airtime.sum() - 0.0) / len(g),
                      energy_per_delivered=g.energy.sum() / max(1, len(dl)),
                      proactive_share=g.proactive.mean(),
                      ttf_violation_rate_among_proactive=(g[g.proactive].ttf_violation.mean() if g.proactive.any() else np.nan),
                      **{f"deliv_{c}": g[g.priority == c].delivered.mean() for c, _, _ in PRIO}))
    s = pd.DataFrame(s); s.to_csv(os.path.join(OUT, "sched_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(s.round(4).to_string(index=False))
    pv = df.pivot_table(index=["seed", "id"], columns="policy", values="delivered", aggfunc="first")
    for base in ("FIFO_reactive", "EDF_reactive"):
        x = int((pv["Proactive_TTF"] & ~pv[base]).sum()); y = int((~pv["Proactive_TTF"] & pv[base]).sum())
        print(f"Proactive vs {base}: only-proactive-delivered={x}, only-{base}-delivered={y}, McNemar p={mcnemar_exact(y, x):.3g}")


if __name__ == "__main__":
    main()
