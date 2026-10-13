"""Step 4: Cost Model and Erasure/Interleaver Evaluation Script.

Demonstrates:
1. Consistent computation of Overhead O(a) across candidate actions.
2. The interleaver depth versus deadline tradeoff under burst fades.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcdam.link import ModcodTable
from rcdam.cost import Action, CostModel
from rcdam.erasure import ErasureModel

OUT = os.path.join(PROJECT_ROOT, "outputs")


def main():
    os.makedirs(OUT, exist_ok=True)
    table = ModcodTable()
    cost_model = CostModel(table=table)
    erasure = ErasureModel(table=table)

    msg_bits = 20e6
    mod_idx = 10  # QPSK 9/10

    # 1. Cost model comparison across code rates and power steps
    rates = [1.0, 0.8, 0.67, 0.5]
    powers = [0.0, 1.0, 2.0, 3.0]

    cost_matrix = np.zeros((len(powers), len(rates)))
    for i, pwr in enumerate(powers):
        for j, r in enumerate(rates):
            a = Action(modcod_idx=mod_idx, power_step_db=pwr, code_rate=r)
            res = cost_model.compute_overhead(msg_bits, a)
            cost_matrix[i, j] = res["overhead"]

    # 2. Interleaver depth vs deadline trade-off
    depths = [1, 2, 4, 8, 16]
    deadlines = [1.0, 2.0, 5.0, 10.0]
    # Challenging channel with burst
    burst_s = 0.4
    esn0_marginal = 6.0  # slightly near threshold

    p_deliv_matrix = np.zeros((len(deadlines), len(depths)))
    for i, d in enumerate(deadlines):
        for j, b in enumerate(depths):
            a = Action(modcod_idx=mod_idx, code_rate=0.8, interleaver_depth=b)
            p = erasure.delivery_probability(msg_bits, a, esn0_marginal, deadline_s=d, mean_burst_s=burst_s)
            p_deliv_matrix[i, j] = p

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap / grouped bars for cost model
    x = np.arange(len(rates))
    width = 0.18
    for i, pwr in enumerate(powers):
        ax1.bar(x + i * width, cost_matrix[i, :], width, label=f"Power +{pwr:g} dB")
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([f"r={r}" for r in rates])
    ax1.set_xlabel("Erasure Code Rate (r = K/N)")
    ax1.set_ylabel("Overhead O(a)")
    ax1.set_title("Cost Model: Overhead O(a) across Redundancy & Power Levers")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Lines for Interleaver Depth Tradeoff
    for i, d in enumerate(deadlines):
        ax2.plot(depths, p_deliv_matrix[i, :], marker="o", label=f"Deadline D = {d:g} s", linewidth=2)
    ax2.set_xlabel("Interleaver Depth B")
    ax2.set_ylabel("P(delivered on-time by D)")
    ax2.set_title("Interleaver Depth vs Deadline Trade-off under Burst Fade")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Step 4: Cost Model and Erasure-Code / Interleaving Trade-offs")
    plt.tight_layout()
    plot_path = os.path.join(OUT, "step4_cost_and_erasure.png")
    plt.savefig(plot_path, dpi=130)
    print(f"Saved: {plot_path}")
    print("=== Step 4 Completed Successfully ===")


if __name__ == "__main__":
    main()
