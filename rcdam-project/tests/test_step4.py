"""Unit tests for Step 4: Cost Model and Erasure-Code / Interleaving Model."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcdam.link import ModcodTable
from rcdam.cost import Action, CostModel
from rcdam.erasure import ErasureModel


def test_cost_model_overhead():
    t = ModcodTable()
    cost_model = CostModel(table=t)
    msg_bits = 20e6

    # Action 1: Clear sky best MODCOD, rate 1.0, single copy
    best_idx = int(np.argmax(t.eff))
    a_clean = Action(modcod_idx=best_idx, power_step_db=0.0, code_rate=1.0, duplication=1)
    res_clean = cost_model.compute_overhead(msg_bits, a_clean)
    assert res_clean["airtime_overhead"] >= 1.0, "Airtime overhead must be >= 1.0 due to headers"
    # Overhead with headers should be slightly > 1.0 (e.g. 1.02 to 1.05)
    assert 1.0 <= res_clean["airtime_overhead"] <= 1.10

    # Action 2: Lower rate r = 0.5 (double the packets)
    a_fec = Action(modcod_idx=best_idx, power_step_db=0.0, code_rate=0.5, duplication=1)
    res_fec = cost_model.compute_overhead(msg_bits, a_fec)
    assert res_fec["airtime_overhead"] > 1.9 * res_clean["airtime_overhead"]

    # Action 3: Duplication = 2
    a_dup = Action(modcod_idx=best_idx, power_step_db=0.0, code_rate=1.0, duplication=2)
    res_dup = cost_model.compute_overhead(msg_bits, a_dup)
    assert res_dup["airtime_overhead"] > 1.95 * res_clean["airtime_overhead"]

    # Action 4: Power step = 3 dB adds power penalty
    a_pwr = Action(modcod_idx=best_idx, power_step_db=3.0, code_rate=1.0, duplication=1)
    res_pwr = cost_model.compute_overhead(msg_bits, a_pwr)
    assert res_pwr["power_penalty"] > 0.0
    assert res_pwr["overhead"] > res_clean["overhead"]

    print("CostModel overhead tests passed.")


def test_erasure_and_interleaving():
    t = ModcodTable()
    erasure = ErasureModel(table=t)
    msg_bits = 5e6  # 5 Mbit message

    mod_idx = 4  # QPSK 3/5, thr = 2.23 dB
    esn0_db = 2.0  # slightly marginal

    # 1. Deadline check: if deadline is too tight, P_delivery must be 0
    a_tight = Action(modcod_idx=mod_idx, code_rate=0.5, interleaver_depth=8)
    p_tight = erasure.delivery_probability(msg_bits, a_tight, esn0_db, deadline_s=0.1)
    assert p_tight == 0.0, "Should blow deadline when D is 0.1s"

    # 2. Comparison: coding rate 0.67 vs 1.0 under channel with loss
    a_nocode = Action(modcod_idx=mod_idx, code_rate=1.0, interleaver_depth=1)
    a_coded = Action(modcod_idx=mod_idx, code_rate=0.67, interleaver_depth=1)
    p_nocode = erasure.delivery_probability(msg_bits, a_nocode, esn0_db, deadline_s=5.0)
    p_coded = erasure.delivery_probability(msg_bits, a_coded, esn0_db, deadline_s=5.0)
    assert p_coded >= p_nocode, f"Coded P ({p_coded}) should be >= uncoded P ({p_nocode})"

    # 3. Interleaver depth tradeoff
    a_b1 = Action(modcod_idx=mod_idx, code_rate=0.8, interleaver_depth=1)
    a_b4 = Action(modcod_idx=mod_idx, code_rate=0.8, interleaver_depth=4)
    p_b1 = erasure.delivery_probability(msg_bits, a_b1, esn0_db, deadline_s=10.0, mean_burst_s=0.5)
    p_b4 = erasure.delivery_probability(msg_bits, a_b4, esn0_db, deadline_s=10.0, mean_burst_s=0.5)
    assert p_b4 >= p_b1, "Interleaving across burst should improve or maintain delivery prob"

    print("ErasureModel tests passed.")


if __name__ == "__main__":
    test_cost_model_overhead()
    test_erasure_and_interleaving()
    print("All Step 4 unit tests passed.")
