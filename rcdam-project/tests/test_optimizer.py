"""Unit tests for optimizer.py (Section 4.5 and M4 acceptance criteria)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rcdam.link import ModcodTable
from rcdam.optimizer import Action, Message, Decision, choose_action, choose_decision, ALPHAS


def test_clear_sky_highest_modcod():
    """Acceptance criterion: At a bound near 0 dB the optimizer picks the highest MODCOD
    with no parity and no extra power."""
    table = ModcodTable()
    clear_esn0 = 18.0
    symbol_rate = 25e6
    rtt_s = 0.54
    one_way_s = 0.27
    msg_bits = 20e6
    deadline_s = 2.0
    eps_p = 0.01

    # Bounds near 0 dB for all alphas
    z_clean = {a: 0.1 for a in ALPHAS}
    cands = list(range(len(table.eff)))

    a, alpha, ov, plb, feas = choose_action(
        z_clean, msg_bits, deadline_s, eps_p, table, clear_esn0, symbol_rate, rtt_s, one_way_s, cands
    )

    assert feas is True
    # Highest MODCOD index in table
    assert a.modcod == len(table.eff) - 1, f"Expected highest MODCOD, got {a.modcod}"
    assert a.power_db == 0.0, f"Expected 0 dB power, got {a.power_db}"
    assert a.parity_frac == 0.0, f"Expected 0 parity, got {a.parity_frac}"
    assert a.defer_s == 0.0
    assert abs(ov - 1.0) < 0.05
    print("optimizer.py: Clear sky highest MODCOD test verified.")


def test_more_severe_bound_never_lowers_overhead():
    """Acceptance criterion: A more severe bound never lowers the overhead of the chosen action."""
    table = ModcodTable()
    clear_esn0 = 18.0
    symbol_rate = 25e6
    rtt_s = 0.54
    one_way_s = 0.27
    msg_bits = 20e6
    deadline_s = 3.0
    eps_p = 0.05
    cands = list(range(len(table.eff)))

    # Mild bound: 2 dB
    z_mild = {a: 2.0 for a in ALPHAS}
    # Severe bound: 8 dB
    z_severe = {a: 8.0 for a in ALPHAS}

    a_mild, _, ov_mild, _, feas_mild = choose_action(
        z_mild, msg_bits, deadline_s, eps_p, table, clear_esn0, symbol_rate, rtt_s, one_way_s, cands
    )
    a_sev, _, ov_sev, _, feas_sev = choose_action(
        z_severe, msg_bits, deadline_s, eps_p, table, clear_esn0, symbol_rate, rtt_s, one_way_s, cands
    )

    if feas_sev:
        assert ov_sev >= ov_mild - 1e-4, f"Severe bound lowered overhead: {ov_sev} vs {ov_mild}"
    print(f"optimizer.py: Severe bound overhead check verified (Mild ov={ov_mild:.3f}, Severe ov={ov_sev:.3f}).")


def test_deadline_infeasible_reason():
    """Acceptance criterion: A deadline shorter than minimum transmission time plus one-way delay
    is infeasible with reason 'deadline'."""
    table = ModcodTable()
    clear_esn0 = 18.0
    symbol_rate = 25e6
    rtt_s = 0.54
    one_way_s = 0.27
    msg_bits = 20e6

    # Minimum airtime with 32APSK 9/10: 20e6 / (25e6 * 4.453) = 0.179s + 0.27s = 0.449s
    # With deadline_s = 0.1s, it is physically impossible
    msg = Message(size_bits=msg_bits, priority="critical", deadline_s=0.1, eps=0.01)
    z_clean = {a: 0.1 for a in ALPHAS}

    dec = choose_decision(z_clean, msg, table, clear_esn0, symbol_rate, rtt_s, one_way_s)
    assert dec.feasible is False
    assert dec.reason == "deadline", f"Expected reason 'deadline', got '{dec.reason}'"
    print("optimizer.py: Reason 'deadline' infeasibility verified.")


def test_capacity_infeasible_reason():
    """Extreme deep fade beyond lowest MODCOD returns infeasible with reason 'capacity'."""
    table = ModcodTable()
    clear_esn0 = 18.0
    symbol_rate = 25e6
    rtt_s = 0.54
    one_way_s = 0.27
    msg_bits = 20e6

    msg = Message(size_bits=msg_bits, priority="critical", deadline_s=2.0, eps=0.001)
    # 30 dB attenuation -> Es/N0 = -12 dB (lowest QPSK 1/4 needs -2.35 dB)
    z_deep = {a: 30.0 for a in ALPHAS}

    dec = choose_decision(z_deep, msg, table, clear_esn0, symbol_rate, rtt_s, one_way_s)
    assert dec.feasible is False
    assert dec.reason == "capacity", f"Expected reason 'capacity', got '{dec.reason}'"
    assert dec.p_lower_bound >= 0.0
    print("optimizer.py: Reason 'capacity' infeasibility verified.")


if __name__ == "__main__":
    test_clear_sky_highest_modcod()
    test_more_severe_bound_never_lowers_overhead()
    test_deadline_infeasible_reason()
    test_capacity_infeasible_reason()
    print("All optimizer tests passed.")
