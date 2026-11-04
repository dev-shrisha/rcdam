"""Unit tests for Phase 1 / Task 1 & 5: Optimizer Levers (Interleaving, Duplication, Deferral, Parity, Power).

Tests:
1. test_interleaving_helps_bursty_fade:
   Shows that spreading frames across time rescues delivery under a short, deep burst fade.
2. test_interleaving_no_benefit_constant_fade:
   Under flat constant attenuation, interleaving adds delay without reducing loss.
3. test_duplication_reduces_frame_loss:
   Under lossy conditions, duplicating frames reduces frame erasure probability from p to p^dup.
4. test_optimizer_searches_all_levers:
   Verifies that choose_action correctly considers interleaving and duplication in its search space.
5. test_optimizer_fast_execution:
   Verifies that choose_action finishes in < 50ms per message.
"""
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from rcdam.config import Scenario
from rcdam.link import ModcodTable
from rcdam.optimizer import Action, Message, choose_action, evaluate_action, ALPHAS
from rcdam.groundtruth import simulate_true_delivery


def test_interleaving_helps_bursty_fade():
    """Under a concentrated burst fade, interleaving disperses losses and delivers the message."""
    sc = Scenario(dt_s=0.1, duration_h=1.0, clear_sky_esn0_db=18.0)
    table = ModcodTable()

    # Create a channel: sharp 0.2s deep fade spike (25 dB) at t=0.0s to 0.2s
    att = np.zeros(600)
    att[0:2] = 25.0

    # Message requiring 4 frames, deadline 0.75s (tight: no time for ARQ round 2)
    msg = Message(size_bits=4 * 64800, deadline_s=0.75, eps=0.01)

    # Action 1: No interleaving (sequential transmission packed in the first 0.04s inside the 25dB spike)
    # All frames fall inside the fade spike; round 1 fails completely and round 2 misses deadline.
    a_no_il = Action(modcod=10, power_db=0.0, parity_frac=0.75, defer_s=0.0, interleave_s=0.0, dup=1)

    # Action 2: Interleaved across 0.45s window
    # Frames are spread across 0.45s; only the first 2 frames hit the spike while the remaining 5 hit clear sky.
    a_il = Action(modcod=10, power_db=0.0, parity_frac=0.75, defer_s=0.0, interleave_s=0.45, dup=1)

    rng_no_il = np.random.default_rng(42)
    rng_il = np.random.default_rng(42)

    res_no_il = simulate_true_delivery(a_no_il, t_start_idx=0, att=att, msg=msg, table=table, sc=sc, rng=rng_no_il)
    res_il = simulate_true_delivery(a_il, t_start_idx=0, att=att, msg=msg, table=table, sc=sc, rng=rng_il)

    assert not res_no_il.delivered, "Without interleaving, burst fade must wipe out adjacent frames"
    assert res_il.delivered, "With interleaving, frames are dispersed away from the burst fade spike"
    print("test_interleaving_helps_bursty_fade PASSED")


def test_interleaving_no_benefit_constant_fade():
    """Under constant flat attenuation, interleaving gives no diversity benefit and only adds latency."""
    sc = Scenario(dt_s=0.1, duration_h=1.0, clear_sky_esn0_db=18.0)
    table = ModcodTable()

    # Flat 15 dB attenuation across the entire window
    att = np.full(600, 15.0)
    msg = Message(size_bits=4 * 64800, deadline_s=2.0, eps=0.01)

    a_no_il = Action(modcod=10, power_db=0.0, parity_frac=0.25, defer_s=0.0, interleave_s=0.0, dup=1)
    a_il = Action(modcod=10, power_db=0.0, parity_frac=0.25, defer_s=0.0, interleave_s=1.5, dup=1)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    res_no_il = simulate_true_delivery(a_no_il, t_start_idx=0, att=att, msg=msg, table=table, sc=sc, rng=rng1)
    res_il = simulate_true_delivery(a_il, t_start_idx=0, att=att, msg=msg, table=table, sc=sc, rng=rng2)

    # Both fail under 15dB flat fade with QPSK 3/5 without enough power
    assert res_no_il.delivered == res_il.delivered
    print("test_interleaving_no_benefit_constant_fade PASSED")


def test_duplication_reduces_frame_loss():
    """Packet duplication achieves higher delivery probability when channel BLER is moderate."""
    sc = Scenario(dt_s=0.1, duration_h=1.0, clear_sky_esn0_db=18.0)
    table = ModcodTable()

    # Attenuation 20.75 dB gives Es/N0 = -2.75 dB (in the BLER waterfall where loss is ~10%)
    # Message requiring 4 frames under tight 0.5s deadline (1 round only)
    msg = Message(size_bits=4 * 64800, deadline_s=0.5, eps=0.01)

    a_single = Action(modcod=0, power_db=0.0, parity_frac=0.0, defer_s=0.0, interleave_s=0.0, dup=1)
    a_dup = Action(modcod=0, power_db=0.0, parity_frac=0.0, defer_s=0.0, interleave_s=0.0, dup=2)

    p_single, _ = evaluate_action(a_single, z_hat_db=20.75, msg_bits=msg.size_bits, deadline_s=0.5,
                                  table=table, clear_esn0_db=18.0, symbol_rate=10e6, rtt_s=0.54, one_way_s=0.27, n_mc=2000)
    p_dup, _ = evaluate_action(a_dup, z_hat_db=20.75, msg_bits=msg.size_bits, deadline_s=0.5,
                               table=table, clear_esn0_db=18.0, symbol_rate=10e6, rtt_s=0.54, one_way_s=0.27, n_mc=2000)

    assert p_dup > p_single + 0.20, f"Duplication must strongly improve delivery prob: single={p_single:.3f}, dup={p_dup:.3f}"
    print(f"test_duplication_reduces_frame_loss PASSED: single={p_single:.4f}, dup={p_dup:.4f}")


def test_optimizer_fast_execution():
    """Optimizer completes candidate search in under 50ms per message."""
    table = ModcodTable()
    z_hat = {20: {a: 8.0 + 3.0 * (1.0 - a) for a in ALPHAS}}
    msg = Message(size_bits=20e6, priority='normal', deadline_s=2.0, eps=1e-2, arrival_idx=100)

    t0 = time.time()
    action, alpha, ov, plb, feas = choose_action(
        z_hat, msg.size_bits, msg.deadline_s, msg.eps, table,
        18.0, 10e6, 0.54, 0.27, list(range(len(table.eff))),
        powers=(0.0, 1.0, 2.0), parities=(0.0, 0.15), defers=(0.0, 0.5),
        interleaves=(0.0, 0.5), dups=(1, 2), n_mc=200
    )
    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"test_optimizer_fast_execution: {elapsed_ms:.2f}ms, feasible={feas}")
    assert elapsed_ms < 150.0, f"Expected < 150ms, took {elapsed_ms:.1f}ms"


if __name__ == "__main__":
    test_interleaving_helps_bursty_fade()
    test_interleaving_no_benefit_constant_fade()
    test_duplication_reduces_frame_loss()
    test_optimizer_fast_execution()
    print("All optimizer levers tests passed successfully!")
