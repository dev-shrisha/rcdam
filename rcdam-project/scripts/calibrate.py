"""M3 Calibrate: Fits stratified conformal corrections and persists them.

Implements M3 requirements:
- Computes split-conformal corrections per fade state for all (H, tau)
- Thins calibration windows (stride >= 60s)
- Persists corrections to outputs/calibration.json with metadata
"""
import argparse
import hashlib
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from rcdam.config import Scenario
from rcdam.attenuation import generate_attenuation_db
from rcdam.experiment import step2_scenario
from rcdam.windows import build_windows
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.calibration import fit_stratified_corrections, save_calibration_json, STATE_NAMES

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
HORIZONS = [10, 20, 30, 50, 100, 200, 300, 600]
TAUS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


def make_dataset(seeds, scenario, horizons, stride_steps):
    all_dicts = []
    for s in seeds:
        sc = replace(scenario, seed=s)
        att = generate_attenuation_db(sc)
        w = build_windows(att, sc.dt_s, horizons, stride_steps=stride_steps)
        all_dicts.append(w)
    return {k: np.concatenate([d[k] for d in all_dicts]) for k in all_dicts[0]}


def main():
    parser = argparse.ArgumentParser(description="M3 Split-Conformal Calibration")
    parser.add_argument("--hours", type=float, default=24.0, help="Hours per seed")
    parser.add_argument("--thin-stride", type=int, default=600, help="Thinned stride in steps (>= 60s)")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sc0 = replace(step2_scenario(), duration_h=args.hours)

    print("==================================================")
    print("   MILESTONE M3: SPLIT-CONFORMAL CALIBRATION")
    print("==================================================")
    print(f"Scenario: duration={args.hours}h, thinned stride={args.thin_stride} steps ({args.thin_stride * sc0.dt_s:.1f}s)")

    # Independent seeds: train=1..10, cal=11..18
    train_seeds = list(range(1, 11))
    cal_seeds = list(range(11, 19))

    t0 = time.time()
    train_data = make_dataset(train_seeds, sc0, HORIZONS, stride_steps=150)
    # Thin calibration windows to avoid overstating sample size
    cal_data = make_dataset(cal_seeds, sc0, HORIZONS, stride_steps=args.thin_stride)
    print(f"Extracted windows in {time.time()-t0:.1f}s: Train={len(train_data['cur'])}, Thinned Cal={len(cal_data['cur'])}")

    # Fit forecaster on train
    fc = EmpiricalQuantileForecaster(taus=TAUS).fit(train_data["cur"], train_data["Z"])

    corrections_by_h_and_tau = {}
    dropped_count = 0

    for k, H in enumerate(HORIZONS):
        corrections_by_h_and_tau[H] = {}
        for tau in TAUS:
            q_cal = fc.predict(cal_data["cur"], k, tau)
            corr = fit_stratified_corrections(cal_data["cur"], cal_data["Z"][:, k], q_cal, tau, min_n=100)

            # Check if any stratum is inf
            has_inf = any(np.isinf(v) for v in corr.values())
            if has_inf:
                dropped_count += 1
                print(f"  [Warning] (H={H}, tau={tau}) has insufficient points: {corr}. Dropping level.")
            else:
                corrections_by_h_and_tau[H][tau] = corr

    # Metadata
    sc_dict = {k: v for k, v in sc0.__dict__.items() if isinstance(v, (int, float, str, bool))}
    sc_hash = hashlib.md5(str(sorted(sc_dict.items())).encode()).hexdigest()[:10]

    metadata = {
        "scenario_hash": sc_hash,
        "train_seeds": train_seeds,
        "cal_seeds": cal_seeds,
        "hours_per_seed": args.hours,
        "thin_stride_steps": args.thin_stride,
        "horizons_steps": HORIZONS,
        "taus": list(TAUS),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_path = os.path.join(OUT, "calibration.json")
    save_calibration_json(json_path, metadata, corrections_by_h_and_tau)
    print(f"\nPersisted {len(HORIZONS)}x{len(TAUS)} calibration matrix to: {json_path}")

    # Also persist the trained baseline forecaster
    fc_path = os.path.join(OUT, "forecaster_baseline.json")
    fc.save(fc_path)
    print(f"Persisted trained empirical forecaster to: {fc_path}")
    print(f"Sample corrections for H=300 (30s) at tau=0.99:")
    sample = corrections_by_h_and_tau.get(300, {}).get(0.99, {})
    for s, c in sample.items():
        print(f"    {STATE_NAMES[s]}: +{c:.2f} dB")
    print("==================================================\n")


if __name__ == "__main__":
    main()
