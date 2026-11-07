"""Option 2: time-to-fade-aware adaptive transmission for messages with slack.

Compares, on identical traces and messages (fade-conditioned arrivals, slack 30/60/120 s):
  SEND_NOW_ONLY : engine (V7) once; if infeasible send-and-flag immediately
  REACTIVE_WAIT : engine every --retry-s; wait while infeasible; last-chance send-and-flag
  TTF_AWARE     : decide_with_slack -> SEND_NOW / SEND_STRONGER / WAIT / REPORT_INFEASIBLE
                  (guarantees from conformal time-to-fade and time-to-recovery bounds)
Metrics: delivery, latency, airtime, false-feasible of guaranteed sends, and the quality of the
early REPORT_INFEASIBLE (precision/recall vs a slack-aware oracle that scans the true future).

  python scripts/run_option2_eval.py --seeds 6
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
from rcdam.optimizer import Message
from rcdam.groundtruth import simulate_true_delivery
from rcdam.window_conformal import WindowConformal
from rcdam.time_to_fade import TimeToFade
from rcdam.ttf_policy import TimeToRecovery, decide_with_slack
from rcdam.engine import EngineParams, decide
from rcdam.variants import PolicyContext, policy_oracle

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
PRIOS = [("critical", 1e-3), ("normal", 1e-2), ("best_effort", 1e-1)]
POLS = ["SEND_NOW_ONLY", "REACTIVE_WAIT", "TTF_AWARE"]
A = None
_G = {}


def _init(a):
    global A
    A = a


def _globals():
    if not _G:
        sc0 = preset_harsh(duration_h=A.hours)
        wcp = os.path.join(OUT, "window_conformal_harsh.json")
        wc = WindowConformal.load(wcp) if os.path.exists(wcp) else None
        if wc is None or not wc.supports_offsets:
            wc = WindowConformal()
            for s in range(201, 207):
                sc = replace(sc0, seed=s, duration_h=4.0); wc.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps)
            wc.save(wcp)
        ttf, ttr = TimeToFade(), TimeToRecovery()
        for s in range(201, 207):
            sc = replace(sc0, seed=s, duration_h=4.0); a = generate_attenuation_db(sc)
            ttf.fit_from_trace(a, sc.dt_s, sc.feedback_delay_steps); ttr.fit_from_trace(a, sc.dt_s, sc.feedback_delay_steps)
        _G.update(sc0=sc0, table=ModcodTable(), wc=wc, ttf=ttf, ttr=ttr, ep=EngineParams())
    return _G


def _ctx(att, sc, table, idx, msg, true_future=False):
    d = sc.feedback_delay_steps; last = max(d + 60, idx - d); cur = float(att[last])
    feats = {"cur": np.array([cur]), "slope5": np.array([(cur - att[last - 50]) / 5.0]),
             "slope30": np.array([0.0]), "max60": np.array([cur])}
    Hw = int(round((msg.deadline_s + sc.one_way_delay_s) / sc.dt_s)) + 5
    return PolicyContext(idx, last, cur, feats, att[idx: idx + Hw] if true_future else None, msg, sc, table, None, None)


def _deliverable(att, sc, table, idx0, msg_base, rem):
    """Slack-aware oracle: exists a start time within the slack at which a feasible action exists."""
    dex = min(A.exec_window_s, rem); step = int(round(5.0 / sc.dt_s))
    for k in range(0, int(round(max(rem - dex, 0) / sc.dt_s)) + 1, step):
        m = replace(msg_base, deadline_s=dex, arrival_idx=idx0 + k)
        if policy_oracle(_ctx(att, sc, table, idx0 + k, m, True)).feasible:
            return True
    return False


def _run_policy(pol, att, sc, table, idx0, base_msg, rem0, seed, mid):
    g = _globals(); dt = sc.dt_s
    t_off, attempts, air, reported, mode = 0.0, 0, 0.0, False, None
    rng = np.random.default_rng(seed * 7919 + mid)
    while True:
        rem = rem0 - t_off; idx = idx0 + int(round(t_off / dt)); dex = min(A.exec_window_s, rem)
        if rem <= 0.5:
            return dict(delivered=False, latency=np.nan, attempts=attempts, airtime=air, reported=reported, mode=mode or "NONE", guaranteed_send=False)
        msg = replace(base_msg, deadline_s=dex, arrival_idx=idx); ctx = _ctx(att, sc, table, idx, msg)
        action, guaranteed, tag = None, False, None
        if pol == "TTF_AWARE" and t_off == 0.0:
            out = decide_with_slack(ctx, g["wc"], g["ttf"], g["ttr"], g["ep"], rem, A.exec_window_s)
            mode = out["decision"]
            if out["decision"] in ("SEND_NOW", "SEND_STRONGER"):
                action, guaranteed = out["chosen"]["action"], True
            elif out["decision"] == "REPORT_INFEASIBLE":
                reported = True
        if action is None:
            dec = decide(ctx, g["wc"], g["ep"], "R1")
            if dec.feasible:
                action, guaranteed = dec.action, True
            elif pol == "SEND_NOW_ONLY" or rem <= dex + A.retry_s:
                action = dec.action                                   # send-and-flag
            else:
                t_off += A.retry_s; continue
        res = simulate_true_delivery(action, idx, att, replace(msg, deadline_s=dex), table, sc, rng)
        attempts += 1; air += res.airtime_s
        return dict(delivered=bool(res.delivered), latency=(t_off + res.latency_s) if res.delivered else np.nan,
                    attempts=attempts, airtime=air, reported=reported, mode=mode or ("SEND_NOW" if guaranteed else "SEND_FLAGGED"),
                    guaranteed_send=guaranteed)


def _seed(seed):
    path = os.path.join(OUT, f"opt2_h{A.hours:g}_i{A.interval:g}_m{A.min_att_db:g}_r{A.retry_s:g}_seed{seed}.csv")
    if os.path.exists(path) and not A.force:
        try:
            return pd.read_csv(path).to_dict("records")
        except pd.errors.EmptyDataError:             # a seed with no fade-conditioned messages
            return []
    g = _globals(); sc = replace(g["sc0"], seed=seed); table = g["table"]; att = generate_attenuation_db(sc)
    d, dt = sc.feedback_delay_steps, sc.dt_s; rng = np.random.default_rng(seed + 4242)
    rows, t, mid = [], 70.0, 0
    while True:
        t += A.interval + rng.uniform(-5, 5)
        if t >= sc.duration_h * 3600 - 300:
            break
        idx0 = int(round(t / dt)); last = idx0 - d
        if float(att[last]) < A.min_att_db:
            continue
        pn, eps = PRIOS[mid % 3]; rem0 = float(rng.choice([30, 60, 120]))
        base = Message(size_bits=sc.msg_bits, priority=pn, deadline_s=2.0, eps=eps, arrival_idx=idx0)
        deliverable = _deliverable(att, sc, table, idx0, base, rem0)
        for pol in POLS:
            r = _run_policy(pol, att, sc, table, idx0, base, rem0, seed, mid)
            rows.append(dict(seed=seed, msg=mid, policy=pol, priority=pn, eps=eps, slack_s=rem0, att_last=float(att[last]),
                             deliverable_oracle=deliverable, **r))
        mid += 1
    pd.DataFrame(rows).to_csv(path, index=False)    # empty -> header-less file, handled on read
    return rows


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=240.0)
    ap.add_argument("--min-att-db", type=float, default=1.0)
    ap.add_argument("--retry-s", type=float, default=10.0)
    ap.add_argument("--exec-window-s", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    A = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    _globals()
    with Pool(A.workers, initializer=_init, initargs=(A,)) as p:
        df = pd.DataFrame([r for part in p.map(_seed, range(301, 301 + A.seeds)) for r in part])
    df.to_csv(os.path.join(OUT, "opt2_all.csv"), index=False)
    out = []
    for pol, g in df.groupby("policy", sort=False):
        gs = g[g.guaranteed_send]; rp = g[g.reported]
        out.append(dict(policy=pol, n=len(g), delivered=g.delivered.mean(), mean_latency_s=g.latency.mean(),
                        airtime_s_per_msg=g.airtime.mean(), attempts=g.attempts.mean(),
                        guaranteed_sends=len(gs), false_feasible_of_guaranteed=(1 - gs.delivered.mean()) if len(gs) else np.nan,
                        reported_infeasible=len(rp),
                        report_precision=(1 - (rp.deliverable_oracle).mean()) if len(rp) else np.nan,
                        report_recall=(g[~g.deliverable_oracle].reported.mean() if (~g.deliverable_oracle).any() else np.nan),
                        undeliverable_share=(~g.deliverable_oracle).mean()))
    s = pd.DataFrame(out); s.to_csv(os.path.join(OUT, "opt2_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(s.round(4).to_string(index=False))
    print("TTF_AWARE decision mix:", df[df.policy == "TTF_AWARE"]["mode"].value_counts().to_dict())


if __name__ == "__main__":
    main()
