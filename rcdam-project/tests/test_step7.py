"""Unit tests for Step 7: Redundancy Optimizer and Infeasibility Signaller."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcdam.link import ModcodTable
from rcdam.cost import CostModel
from rcdam.erasure import ErasureModel
from rcdam.optimizer import RedundancyOptimizer, Message


def setup_optimizer():
    table = ModcodTable()
    cost_model = CostModel(table=table)
    erasure = ErasureModel(table=table)
    return RedundancyOptimizer(table=table, cost_model=cost_model, erasure_model=erasure)


def test_clear_sky_optimization():
    opt = setup_optimizer()
    msg = Message(msg_id=1, bits=20e6, priority=3, deadline_s=2.0)
    # Clear sky: 0.5 dB attenuation (Es/N0 = 17.5 dB)
    res = opt.optimize(msg, calibrated_att_db=0.5, clear_sky_esn0_db=18.0)

    assert res.is_feasible, "Should be feasible under clear sky"
    assert res.delivery_prob >= msg.target_delivery_prob
    # Under clear sky, chosen MODCOD should be one of the top efficient ones
    chosen_eff = opt.table.eff[res.action.modcod_idx]
    assert chosen_eff >= 3.0, f"Expected efficient MODCOD, got eff {chosen_eff}"
    # Redundancy should be 1.0 (no extra parity)
    assert res.action.code_rate == 1.0
    assert res.overhead < 1.5
    print(f"Clear sky test passed: chosen MODCOD={opt.table.names[res.action.modcod_idx]}, overhead={res.overhead:.3f}")


def test_deep_fade_infeasibility_signalling():
    opt = setup_optimizer()
    msg = Message(msg_id=2, bits=20e6, priority=2, deadline_s=2.0)
    # Deep fade: 25 dB attenuation => Es/N0 = -7 dB (well below lowest QPSK 1/4 at -2.35 dB)
    res = opt.optimize(msg, calibrated_att_db=25.0, clear_sky_esn0_db=18.0)

    assert not res.is_feasible, "Must report INFEASIBLE under extreme fade"
    assert "INFEASIBLE" in res.status_message
    assert res.best_achievable_prob < msg.target_delivery_prob
    print(f"Deep fade infeasibility test passed: status={res.status_message}")


def test_priority_differentiation():
    opt = setup_optimizer()
    # At attenuation 5.80 dB (Es/N0 = 12.20 dB), critical priority (eps=0.001 -> target 0.999)
    # requires a strictly more robust MODCOD (QPSK 9/10, overhead ~1.298) than standard priority
    # (eps=0.05 -> target 0.95), which can safely use 8PSK 2/3 (overhead ~1.237).
    msg_critical = Message(msg_id=3, bits=10e6, priority=1, deadline_s=3.0)  # eps = 0.001 (target 0.999)
    msg_standard = Message(msg_id=4, bits=10e6, priority=3, deadline_s=3.0)  # eps = 0.05 (target 0.95)

    res_crit = opt.optimize(msg_critical, calibrated_att_db=5.80, clear_sky_esn0_db=18.0)
    res_std = opt.optimize(msg_standard, calibrated_att_db=5.80, clear_sky_esn0_db=18.0)

    # Standard priority must have strictly lower overhead and use a more efficient MODCOD than critical
    assert res_crit.overhead > res_std.overhead + 0.02, (
        f"Critical overhead ({res_crit.overhead:.3f}) must be strictly higher than Standard ({res_std.overhead:.3f})"
    )
    assert res_crit.action.modcod < res_std.action.modcod, (
        f"Critical MODCOD ({res_crit.action.modcod}) must be more robust than Standard ({res_std.action.modcod})"
    )
    print(f"Priority test passed: Standard overhead={res_std.overhead:.3f} (modcod {res_std.action.modcod}) "
          f"< Critical overhead={res_crit.overhead:.3f} (modcod {res_crit.action.modcod})")


if __name__ == "__main__":
    test_clear_sky_optimization()
    test_deep_fade_infeasibility_signalling()
    test_priority_differentiation()
    print("All Step 7 unit tests passed.")
