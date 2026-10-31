"""Regression tests for Phase 1 defects (1.1 - 1.7).

Each test in this file targets a specific known defect described in the
RC-DAM Advancement Plan / Completion Prompt.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from rcdam.config import Scenario
from rcdam.link import ModcodTable
from rcdam.optimizer import (
    choose_decision,
    choose_action,
    Message,
    Action,
    ALPHAS,
    FRAME_BITS,
)
from rcdam.evaluator import p_deliver_by_deadline
from rcdam.validated_selection import horizon_steps_needed, pick_horizon
from rcdam.forecaster_baseline import EmpiricalQuantileForecaster
from rcdam.calibration import load_calibration_json


def test_1_1_pruning_reproduction():
    """Defect 1.1: choose_decision prunes candidates based on optimistic bound.
    
    Reproduction from prompt:
      Message: 20 Mbit, deadline 3 s, eps = 0.01
      Bounds: z = {a: 8 + 8*(1-a)**2 for a in ALPHAS}
      Clear-sky Es/N0 = 18 dB
    choose_action with all MODCODs finds feasible QPSK 3/4 with p_lb >= 0.99.
    choose_decision must return feasible=True and match the all-MODCOD search,
    instead of returning INFEASIBLE (16APSK 4/5) due to optimistic top-4 pruning.
    """
    table = ModcodTable()
    z = {a: 8.0 + 8.0 * (1.0 - a) ** 2 for a in ALPHAS}
    msg = Message(size_bits=20e6, deadline_s=3.0, eps=0.01)
    
    # All-MODCOD search benchmark
    all_mods = list(range(len(table.eff)))
    act_all, alpha_all, ov_all, plb_all, feas_all = choose_action(
        z, msg.size_bits, msg.deadline_s, msg.eps, table,
        clear_esn0_db=18.0, symbol_rate=10e6, rtt_s=0.54, one_way_s=0.27,
        mod_candidates=all_mods
    )
    assert feas_all is True, "All-MODCOD search must find a feasible action"
    assert plb_all >= 0.99, f"All-MODCOD p_lb must be >= 0.99, got {plb_all}"
    
    # choose_decision test
    dec = choose_decision(
        z, msg, table, clear_esn0_db=18.0,
        symbol_rate=10e6, rtt_s=0.54, one_way_s=0.27
    )
    assert dec.feasible is True, (
        f"choose_decision returned INFEASIBLE (reason={dec.reason}, modcod={dec.action.modcod} "
        f"{table.names[dec.action.modcod]}) despite all-MODCOD finding feasible {table.names[act_all.modcod]}"
    )
    assert dec.action.modcod == act_all.modcod, (
        f"choose_decision selected modcod {dec.action.modcod} ({table.names[dec.action.modcod]}), "
        f"expected {act_all.modcod} ({table.names[act_all.modcod]})"
    )


def test_1_2_validated_selection_independence():
    """Defect 1.2: Validated selection must use independent verification draws and Wilson lower bound."""
    K, p, frame_s, target = 309, 0.05, 0.0015, 0.99
    parities = [int(round(x * K)) for x in np.arange(0.0, 0.101, 0.005)]
    
    # True delivery probability computed with large Monte Carlo
    true_p = {
        par: p_deliver_by_deadline(K, par, p, frame_s, 0.54, 0.27, 0.0, 2.0, n_mc=30000, seed=999)[0]
        for par in parities
    }
    
    # 1. Selection only with small draws (n_mc=300) suffers from optimizer's curse
    def trial_selection_only(seed):
        for c in parities:
            p_sel = p_deliver_by_deadline(K, c, p, frame_s, 0.54, 0.27, 0.0, 2.0, n_mc=300, seed=10_000 + seed * 31 + c)[0]
            if p_sel >= target:
                return c
        return None
    
    # 2. Validated selection with independent verification draws (n_mc=3000, Wilson lower bound >= target)
    def trial_validated(seed):
        from rcdam.validated_selection import select_validated
        sel = lambda c: p_deliver_by_deadline(K, c, p, frame_s, 0.54, 0.27, 0.0, 2.0, n_mc=300, seed=10_000 + seed * 31 + c)[0]
        def ver(c):
            # Independent seed and larger sample size
            pr = p_deliver_by_deadline(K, c, p, frame_s, 0.54, 0.27, 0.0, 2.0, n_mc=3000, seed=500_000 + seed * 17 + c)[0]
            return int(round(pr * 3000)), 3000
        return select_validated(parities, sel, ver, target)
    
    n_trials = 100
    chosen_sel = [trial_selection_only(s) for s in range(n_trials)]
    chosen_val = [trial_validated(s) for s in range(n_trials)]
    
    ok_sel = [c for c in chosen_sel if c is not None]
    ok_val = [c for c in chosen_val if c is not None]
    
    fail_rate_sel = np.mean([true_p[c] < target for c in ok_sel])
    fail_rate_val = np.mean([true_p[c] < target for c in ok_val]) if ok_val else 0.0
    
    # Selection-only picks an action with true success < target a large fraction of the time (> 60%)
    assert fail_rate_sel > 0.60, f"Selection-only fail rate was unexpectedly low: {fail_rate_sel}"
    # Validated selection must reduce this to at most 5%
    assert fail_rate_val <= 0.05, f"Validated selection fail rate was {fail_rate_val}, expected <= 0.05"


def test_1_3_forecaster_dataset_validation():
    """Defect 1.3: Forecaster must be fit on independent training seeds, not toy linspace."""
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    cal_json = os.path.join(out_dir, "calibration.json")
    fc_json = os.path.join(out_dir, "forecaster_baseline.json")
    
    assert os.path.exists(cal_json), f"Missing {cal_json}"
    assert os.path.exists(fc_json), f"Missing {fc_json}"
    
    meta, corr = load_calibration_json(cal_json)
    fc = EmpiricalQuantileForecaster.load(fc_json)
    
    # Verify training and calibration seeds are disjoint
    train_seeds = set(meta.get("train_seeds", []))
    cal_seeds = set(meta.get("cal_seeds", []))
    assert len(train_seeds) > 0, "train_seeds must not be empty"
    assert len(cal_seeds) > 0, "cal_seeds must not be empty"
    assert not (train_seeds & cal_seeds), f"Train and cal seeds must be disjoint: {train_seeds & cal_seeds}"
    
    # Verify predictions on a non-trivial input are strictly positive
    pred = fc.predict(np.array([5.0]), k=2, tau=0.9)
    assert pred[0] > 5.0, f"Predicted envelope {pred[0]} must be > current attenuation 5.0"


def test_1_4_horizon_consistent_bounds():
    """Defect 1.4: An action that defers must require a larger horizon bound than no-deferral."""
    dt = 0.1
    d_steps = 5  # 0.5s feedback delay
    
    # Case A: no deferral, short 1.0s span -> horizon needed = 5 + 10 = 15 steps
    h_no_defer = horizon_steps_needed(d_steps=d_steps, defer_s=0.0, span_s=1.0, dt=dt)
    grid = [10, 20, 30, 50, 100, 200, 300, 600]
    picked_no_defer = pick_horizon(grid, h_no_defer)
    assert picked_no_defer == 20, f"Expected grid horizon 20, got {picked_no_defer}"
    
    # Case B: 2.0s deferral, 1.0s span -> horizon needed = 5 + 30 = 35 steps
    h_defer = horizon_steps_needed(d_steps=d_steps, defer_s=2.0, span_s=1.0, dt=dt)
    picked_defer = pick_horizon(grid, h_defer)
    assert picked_defer == 50, f"Expected grid horizon 50, got {picked_defer}"
    assert picked_defer > picked_no_defer, "Deferral action must use a strictly larger horizon bound"
    
    # Case C: horizon exceeds max grid -> returns None -> infeasible reason 'no_calibration_data'
    h_huge = horizon_steps_needed(d_steps=d_steps, defer_s=70.0, span_s=10.0, dt=dt)
    picked_huge = pick_horizon(grid, h_huge)
    assert picked_huge is None, "Exceeded grid should return None"


def test_1_5_evaluator_principled_bound():
    """Defect 1.5: Evaluator must use principled probability bound instead of magic 'p >= 0.95 and K > 50'."""
    K = 2
    frame_s = 0.001
    one_way_s = 0.1
    deadline_s = 1.0
    rtt_s = 0.2
    
    p_ok, sent = p_deliver_by_deadline(K=K, parity=0, p=0.96, frame_s=frame_s,
                                       rtt_s=rtt_s, one_way_s=one_way_s,
                                       defer_s=0.0, deadline_s=deadline_s, n_mc=2000, seed=42)
    # At p=0.96, true delivery probability is non-zero (~3.7%).
    # An unprincipled 'p >= 0.95' shortcut would have forced 0.0.
    # The principled bound correctly allows the Monte Carlo to compute the true outcome.
    assert p_ok > 0.01, f"Expected non-zero p_ok > 0.01 for K=2 at p=0.96, got {p_ok}"


def test_1_6_priority_differentiation_strict():
    """Defect 1.6: Priority test must assert a strict, justified difference."""
    from tests.test_step7 import setup_optimizer
    opt = setup_optimizer()
    msg_critical = Message(msg_id=3, bits=10e6, priority=1, deadline_s=3.0)  # eps = 0.001
    msg_standard = Message(msg_id=4, bits=10e6, priority=3, deadline_s=3.0)  # eps = 0.05
    res_crit = opt.optimize(msg_critical, calibrated_att_db=5.80, clear_sky_esn0_db=18.0)
    res_std = opt.optimize(msg_standard, calibrated_att_db=5.80, clear_sky_esn0_db=18.0)
    assert res_crit.overhead > res_std.overhead + 0.02, (
        f"Critical overhead ({res_crit.overhead}) must be strictly higher than standard ({res_std.overhead})"
    )


def test_1_7_hygiene():
    """Defect 1.7: Check line endings (LF only) and no hardcoded absolute user paths in source files."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checked_dirs = ["rcdam", "tests"]
    
    crlf_files = []
    hardcoded_path_files = []
    
    needle_win1 = b"c:" + b"\\users\\"
    needle_win2 = b"c:" + b"/users/"
    
    this_file = os.path.basename(__file__)
    
    for d in checked_dirs:
        dir_path = os.path.join(repo_root, d)
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py") and file != this_file:
                    full_p = os.path.join(root, file)
                    with open(full_p, "rb") as f:
                        content = f.read()
                    if b"\r\n" in content:
                        crlf_files.append(os.path.relpath(full_p, repo_root))
                    if needle_win1 in content.lower() or needle_win2 in content.lower():
                        hardcoded_path_files.append(os.path.relpath(full_p, repo_root))
                        
    assert len(crlf_files) == 0, f"Found CRLF line endings in: {crlf_files}"
    assert len(hardcoded_path_files) == 0, f"Found hardcoded Windows user paths in: {hardcoded_path_files}"


if __name__ == "__main__":
    print("Running Phase 1 regression tests...")
    test_1_1_pruning_reproduction()
    print("-> 1.1 Pruning reproduction passed.")
    test_1_2_validated_selection_independence()
    print("-> 1.2 Validated selection independence passed.")
    test_1_3_forecaster_dataset_validation()
    print("-> 1.3 Forecaster dataset validation passed.")
    test_1_4_horizon_consistent_bounds()
    print("-> 1.4 Horizon consistent bounds passed.")
    test_1_5_evaluator_principled_bound()
    print("-> 1.5 Evaluator principled bound passed.")
    test_1_6_priority_differentiation_strict()
    print("-> 1.6 Priority differentiation strict passed.")
    test_1_7_hygiene()
    print("-> 1.7 Hygiene passed.")
    print("\nAll Phase 1 regression tests passed successfully!")
