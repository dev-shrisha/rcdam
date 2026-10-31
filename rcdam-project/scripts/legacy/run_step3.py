"""Step 3 Runner: ITU-R calibration, ETSI MODCOD verification, and deadline sweep.

Run from the project root:
    python scripts/run_step3.py
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rcdam.itur import (
    EarthStationConfig,
    itur_p618_rain_attenuation,
    calibrate_maseng_bakken_to_itu_r,
    compute_empirical_ccdf,
)
from rcdam.config import Scenario
from rcdam.attenuation import generate_attenuation_db
from tests.test_step3 import test_modcod_dvbs2_verification

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def generate_itur_calibration_plot():
    os.makedirs(OUT, exist_ok=True)
    cfg = EarthStationConfig(freq_ghz=20.0, elevation_deg=35.0, rain_rate_001_mmh=35.0)

    # 1. Theoretical ITU-R P.618 curve
    p_pct = np.logspace(-3, 0.7, 100)  # 0.001% to 5%
    a_itu = itur_p618_rain_attenuation(p_pct, cfg)

    # 2. Calibrate Maseng-Bakken
    calib = calibrate_maseng_bakken_to_itu_r(cfg)
    print("Calibrated Maseng-Bakken parameters:")
    for k, v in calib.items():
        print(f"  {k}: {v}")

    # 3. Generate sample synthetic series using Gauss-Markov with calibrated params
    sc = Scenario(
        rain_model="gauss_markov",
        duration_h=720.0,  # 30 days of data for steady-state distribution
        dt_s=1.0,
        seed=42,
        p_rain=calib["p_rain"],
        rain_ln_mean=calib["rain_ln_mean"],
        rain_ln_std=calib["rain_ln_std"],
        rain_beta_per_s=calib["rain_beta_per_s"],
    )
    print("Simulating 30-day attenuation time series for empirical CCDF...")
    att = generate_attenuation_db(sc)
    thr = np.linspace(0.5, 25.0, 50)
    p_emp = compute_empirical_ccdf(att, thr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(a_itu, p_pct, "r-", linewidth=2.5, label="ITU-R P.618-13 (20 GHz, 35 deg el)")
    ax.plot(thr, p_emp, "b--o", markersize=4, label="Simulated Dynamic Maseng-Bakken (P.1853)")

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 10.0)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Rain Attenuation A (dB)")
    ax.set_ylabel("Percentage of time exceeded P(A > a) (%)")
    ax.set_title("Step 3: Rain Model Calibration to ITU-R P.618 Statistics")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_plot = os.path.join(OUT, "step3_itur_calibration.png")
    plt.savefig(out_plot, dpi=130)
    print(f"Saved ITU-R calibration plot: {out_plot}")


def main():
    print("=== Step 3: Verification, Calibration, and Deadline Sweep ===\n")
    print("1. Verifying MODCOD table against ETSI EN 302 307...")
    test_modcod_dvbs2_verification()
    print("MODCOD table is 100% verified against standard.\n")

    print("2. Calibrating rain model against ITU-R P.618 / P.1853...")
    generate_itur_calibration_plot()
    print("\n3. Running deadline sweep...")
    from scripts.run_deadline_sweep import main as run_sweep
    run_sweep()
    print("\n=== Step 3 Completed Successfully ===")


if __name__ == "__main__":
    main()
