"""End-to-end demo of the reference snippets on the Step 2 repo.

Implements RCDAM_Build_Guide.md Section 4.6.
"""
import sys
import os
import time
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from rcdam.experiment import step2_scenario
from rcdam.attenuation import generate_attenuation_db
from rcdam.link import ModcodTable
from rcdam.windows import build_windows
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.calibration import fade_state, fit_stratified_corrections, apply_corrections
from rcdam.optimizer import choose_action, ALPHAS

sc0 = step2_scenario()
HS = [10, 30, 100, 300, 600]                      # horizons in steps (1 s ... 60 s)


def make(seeds, scenario=sc0):
    ds = [build_windows(generate_attenuation_db(replace(scenario, seed=s)), sc0.dt_s, HS, 300) for s in seeds]
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def main():
    print("--- Running End-to-End Build Guide Section 4.6 Demo ---")
    train, cal, test = make(range(1, 9)), make(range(9, 13)), make(range(13, 21))
    fc = EmpiricalQuantileForecaster(taus=(0.9, 0.99)).fit(train["cur"], train["Z"])
    k, tau = 3, 0.99                                  # 30 s horizon, 99% bound
    q_cal, q_te = fc.predict(cal["cur"], k, tau), fc.predict(test["cur"], k, tau)
    corr = fit_stratified_corrections(cal["cur"], cal["Z"][:, k], q_cal, tau)
    bound = apply_corrections(test["cur"], q_te, corr)
    st = fade_state(test["cur"])
    print("stratified-CQR coverage at 0.99, 30 s horizon:",
          {s: round(float((test["Z"][st == s, k] <= bound[st == s]).mean()), 3) for s in range(4)},
          "corrections", {s: round(c, 2) for s, c in corr.items()})

    t = ModcodTable()
    rs, clear = sc0.symbol_rate_baud, sc0.clear_sky_esn0_db
    for name, zs in (("moderate", {a: 4.0 + 6.0 * (1 - a) for a in ALPHAS}),
                     ("deep", {a: 14.0 + 14.0 * (1 - a) for a in ALPHAS})):
        cands = [i for i in range(len(t.thr)) if t.thr[i] <= clear - min(zs.values()) + 3.0]
        t0 = time.time()
        a, alpha, ov, plb, feas = choose_action(zs, 20e6, 2.0, 1e-2, t, clear, rs, sc0.rtt_s, sc0.one_way_delay_s, cands)
        print(f"{name}: feasible={feas} {a} alpha={alpha} overhead={ov:.2f} P_lower_bound={plb:.3f} [{time.time()-t0:.1f}s, {len(cands)} MODCOD candidates]")


if __name__ == "__main__":
    main()
