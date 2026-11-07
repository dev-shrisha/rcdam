"""Option 3: does the pruned engine find the MINIMUM-overhead feasible action?

Compares engine.search (screening + dominance pruning + probes) with an EXHAUSTIVE search that
evaluates every candidate at full Monte-Carlo size (same candidates, same random seeds, cost order).
A reduced action grid keeps the exhaustive run tractable.

  python scripts/check_optimality.py --messages 20
Outputs outputs/opt3_optimality.csv
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataclasses import replace
import numpy as np, pandas as pd
from rcdam.config import preset_harsh
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.optimizer import Message
from rcdam.window_conformal import WindowConformal
from rcdam.engine import EngineParams, search
from rcdam.variants import PolicyContext

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
ap = argparse.ArgumentParser(); ap.add_argument("--messages", type=int, default=20); ap.add_argument("--eps", type=float, default=0.01)
a = ap.parse_args()
wc = WindowConformal.load(os.path.join(OUT, "window_conformal_harsh.json")); table = ModcodTable()
small = dict(powers=(0.0, 2.0), parities=(0.0, 0.15), defers=(0.0,), interleaves=(0.0, 1.0), dups=(1,))
fast, full = EngineParams(**small), EngineParams(exhaustive=True, **small)
rows = []
sc = replace(preset_harsh(duration_h=3.0), seed=301); att = generate_attenuation_db(sc); d = sc.feedback_delay_steps
# pick messages spread over the interesting fade range (non-trivial choices)
cur_all = att[2000:len(att) - 2000]
targets = np.linspace(1.0, 11.0, a.messages)
for k, tgt in enumerate(targets):
    idx = int(2000 + np.argmin(np.abs(cur_all - tgt))); last = idx - d; cur = float(att[last])
    msg = Message(size_bits=sc.msg_bits, priority="normal", deadline_s=sc.deadline_s, eps=a.eps, arrival_idx=idx + 13 * k)
    feats = {"cur": np.array([cur]), "slope5": np.array([(cur - att[last - 50]) / 5.0]), "slope30": np.array([0.0]), "max60": np.array([cur])}
    ctx = PolicyContext(idx, last, cur, feats, None, msg, sc, table, None, None)
    t0 = time.time(); rf = search(ctx, wc, a.eps, fast); t1 = time.time(); rx = search(ctx, wc, a.eps, full); t2 = time.time()
    same_feas = rf["feasible"] == rx["feasible"]
    gap = (rf["cost"] - rx["cost"]) if (rf["feasible"] and rx["feasible"]) else np.nan
    rows.append(dict(msg=k, cur_db=round(cur, 2), engine_feasible=rf["feasible"], exhaustive_feasible=rx["feasible"],
                     engine_cost=rf["cost"] if rf["feasible"] else np.nan, exhaustive_cost=rx["cost"] if rx["feasible"] else np.nan,
                     cost_gap=gap, same_feasibility=same_feas, evals_engine=rf["n_eval"], evals_exhaustive=rx["n_eval"],
                     sec_engine=round(t1 - t0, 2), sec_exhaustive=round(t2 - t1, 2)))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "opt3_optimality.csv"), index=False)
both = df[df.engine_feasible & df.exhaustive_feasible]
print("\nfeasibility agreement: %d/%d" % (int(df.same_feasibility.sum()), len(df)))
print("both feasible: %d; engine cost == exhaustive cost in %d (max gap %.4f); engine speed-up x%.1f" %
      (len(both), int((both.cost_gap.abs() < 1e-9).sum()), float(both.cost_gap.max()) if len(both) else 0.0,
       df.sec_exhaustive.sum() / max(df.sec_engine.sum(), 1e-9)))
