"""Check the synthetic channel against ITU-R P.618 statistics (and optionally real beacon data).

  python scripts/validate_channel_itu.py --seeds 8 --hours 24
  python scripts/validate_channel_itu.py --beacon data/my_beacon.csv     # adds a real-data curve

Columns: itu_p618_pct (reference), fitted_lognormal_pct (ANALYTIC check of the calibration - the
reliable one for deep fades), simulated_pct (empirical; deep tails need years of simulated time
because rain events are slow, so it is noisy above ~5 dB).
Outputs: outputs/channel_ccdf_validation.csv and .png.  This is a CONSISTENCY check of the
simulator, not real-world validation of RC-DAM. Only a real beacon trace gives the latter.
"""
import argparse, os, sys
from dataclasses import replace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
from rcdam.config import preset_itu_site
from rcdam.attenuation import generate_attenuation_db
from rcdam.itur import itur_p618_rain_attenuation, EarthStationConfig, compute_empirical_ccdf

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--hours", type=float, default=24.0)
ap.add_argument("--beacon", default=None)
a = ap.parse_args()

sc0 = preset_itu_site(duration_h=a.hours)
cfg = EarthStationConfig(sc0.itu_freq_ghz, sc0.itu_elev_deg, sc0.itu_lat_deg, sc0.itu_alt_km,
                         sc0.itu_r001_mmh, sc0.itu_tilt_deg)
thr = np.array([1, 2, 3, 5, 8, 10, 15, 20.0])
p_grid = np.logspace(-3, np.log10(5.0), 200)
A_grid = itur_p618_rain_attenuation(p_grid, cfg)                      # decreasing in p
p_itu = np.interp(thr, A_grid[::-1], p_grid[::-1], left=np.nan, right=np.nan)
sim = np.mean([compute_empirical_ccdf(generate_attenuation_db(replace(sc0, seed=s)), thr)
               for s in range(501, 501 + a.seeds)], axis=0)
from scipy import stats
from rcdam.attenuation import _itu_params
_pr = _itu_params(sc0)
fit = 100.0 * _pr["p_rain"] * stats.norm.sf((np.log(thr) - _pr["rain_ln_mean"]) / _pr["rain_ln_std"])
df = pd.DataFrame(dict(threshold_db=thr, itu_p618_pct=p_itu, fitted_lognormal_pct=fit, simulated_pct=sim))
if a.beacon:
    from rcdam.beacon_loader import load_beacon_csv
    att, meta = load_beacon_csv(a.beacon, dt_s=sc0.dt_s)
    df["beacon_pct"] = compute_empirical_ccdf(att, thr)
df["ratio_sim_over_itu"] = df.simulated_pct / df.itu_p618_pct
df.to_csv(os.path.join(OUT, "channel_ccdf_validation.csv"), index=False)
print(df.round(4).to_string(index=False))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.loglog(thr, df.itu_p618_pct, "k-o", label="ITU-R P.618")
    ax.loglog(thr, df.fitted_lognormal_pct, "g-.d", label="Fitted lognormal (analytic)")
    ax.loglog(thr, df.simulated_pct, "r--s", label="Simulated (P.1853 dynamics)")
    if a.beacon: ax.loglog(thr, df.beacon_pct, "b-^", label="Beacon data")
    ax.set_xlabel("Attenuation threshold (dB)"); ax.set_ylabel("Time exceeded (%)"); ax.grid(True, which="both", alpha=.3)
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "channel_ccdf_validation.png"), dpi=200)
except Exception as e:
    print("plot skipped:", e)
