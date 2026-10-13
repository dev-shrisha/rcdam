"""Step 1: simulate the channel, run the fixed-margin ACM baseline, sweep the margin.

Run from the project root:   python scripts/run_step1.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rcdam.config import Scenario
from rcdam.attenuation import generate_attenuation_db
from rcdam.link import ModcodTable
from rcdam.acm_baseline import run_fixed_margin_acm
from rcdam.metrics import deadline_metrics, link_metrics

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

sc = Scenario()
table = ModcodTable()

att = generate_attenuation_db(sc)
esn0 = sc.clear_sky_esn0_db - att

# ---- sanity checks on the synthetic channel ----
print("=== Channel sanity check ===")
print(f"samples: {len(att):,}  ({sc.duration_h:.0f} h at dt={sc.dt_s}s)")
for a in (1, 5, 10, 15, 20):
    print(f"  time with attenuation > {a:2d} dB: {100 * (att > a).mean():.3f} %")
k = int(round(1.0 / sc.dt_s))
slope = np.abs(att[k:] - att[:-k])                      # dB change over 1 s
raining = att[k:] > 1.0
if raining.any():
    s = slope[raining]
    print(f"  fade slope during rain (dB/s):  median {np.median(s):.3f}  "
          f"p99 {np.percentile(s, 99):.3f}  max {s.max():.2f}")
print(f"  lowest MODCOD needs {table.thr[0]:.2f} dB -> hard outage when attenuation > "
      f"{sc.clear_sky_esn0_db - table.thr[0]:.1f} dB\n")

# ---- baseline: fixed-margin ACM, sweep the margin ----
margins = [0, 0.5, 1, 2, 3, 4, 6, 8]
rows = []
for m in margins:
    res = run_fixed_margin_acm(esn0, table, m, sc.feedback_delay_steps, sc.symbol_rate_baud)
    d = deadline_metrics(res["rate_bps"], sc.dt_s, sc.msg_bits, sc.deadline_s,
                         sc.one_way_delay_s, sc.msg_interval_s)
    l = link_metrics(res)
    rows.append((m, d["miss_rate"], l["mean_goodput_mbps"], l["outage_frac"], l["frame_error_rate"]))

print("=== Baseline 1: fixed-margin ACM (feedback delayed by one RTT) ===")
print(f"messages: {d['n_messages']}, size {sc.msg_bits/1e6:.0f} Mbit, deadline {sc.deadline_s}s\n")
print(f"{'margin dB':>9} {'miss rate':>10} {'goodput Mbps':>13} {'outage':>8} {'FER':>8}")
for m, miss, gp, out, fer in rows:
    print(f"{m:9.1f} {miss:10.4f} {gp:13.2f} {out:8.4f} {fer:8.4f}")

# ---- plots ----
peak = int(np.argmax(att))
lo, hi = max(0, peak - 3000), min(len(att), peak + 3000)    # +-5 minutes around the deepest fade
t = (np.arange(lo, hi) * sc.dt_s) / 60.0
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(t, att[lo:hi]); ax[0].set_xlabel("time (min)"); ax[0].set_ylabel("attenuation (dB)")
ax[0].set_title("Deepest fade in the run")
ax[1].plot(t, esn0[lo:hi]); ax[1].axhline(table.thr[0], color="r", ls="--", label="lowest MODCOD threshold")
ax[1].set_xlabel("time (min)"); ax[1].set_ylabel("Es/N0 (dB)"); ax[1].legend(); ax[1].set_title("Es/N0 vs lowest threshold")
ax[2].plot([r[2] for r in rows], [r[1] for r in rows], "o-")
for m, miss, gp, *_ in rows:
    ax[2].annotate(f"{m:g} dB", (gp, miss), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax[2].set_xlabel("mean goodput (Mbps)"); ax[2].set_ylabel("deadline-miss rate")
ax[2].set_title("Baseline: reliability vs throughput")
plt.tight_layout()
path = os.path.join(OUT, "step1_baseline.png")
plt.savefig(path, dpi=130)
print(f"\nSaved plot: {os.path.normpath(path)}")
