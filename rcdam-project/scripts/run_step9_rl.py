"""Step 9: DRL-AMC Comparison Baseline Training & Evaluation.

Trains a Deep Q-Network policy (DRL-AMC) to map delayed telemetry and trend
into MODCOD selections, and compares its Pareto performance (overhead vs miss rate)
against Fixed-margin ACM, Predictive ACM, and RC-DAM.

Outputs:
- outputs/step9_rl_comparison.csv
- outputs/step9_rl_comparison.png
"""
import csv
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcdam.config import Scenario
from rcdam.link import ModcodTable
from rcdam.attenuation import generate_attenuation_db
from rcdam.experiment import step2_scenario, evaluate
from rcdam.rl_baseline import DRLAMCAgent
from rcdam import policies

OUT = os.path.join(PROJECT_ROOT, "outputs")


def train_drl_agent(agent: DRLAMCAgent, sc: Scenario, table: ModcodTable,
                    episodes: int = 5, steps_per_ep: int = 2000) -> DRLAMCAgent:
    """Train DRL agent on simulated channels."""
    print(f"Training DRL-AMC agent over {episodes} episodes...")
    dt = sc.dt_s
    delay = sc.feedback_delay_steps

    for ep in range(episodes):
        sc_ep = Scenario(seed=500 + ep, duration_h=2.0)
        att = generate_attenuation_db(sc_ep)
        esn0 = sc.clear_sky_esn0_db - att

        # Delayed telemetry
        delayed_esn0 = np.empty_like(esn0)
        delayed_esn0[delay:] = esn0[:-delay]
        delayed_esn0[:delay] = esn0[0]

        total_reward = 0.0
        for t in range(50, min(len(esn0) - 1, steps_per_ep)):
            s_curr = agent.get_state(
                delayed_esn0_db=delayed_esn0[t],
                slope_db_s=(delayed_esn0[t] - delayed_esn0[t - 20]) / (20 * dt),
                deadline_s=2.0,
            )
            a = agent.select_action(s_curr, evaluate=False)

            # Link outcome
            eff = table.eff[a]
            bler = float(table.bler(a, esn0[t]))
            delivered = (bler < 1e-4)

            # Reward function
            r = (eff * (1.0 - bler)) - (5.0 if not delivered else 0.0)
            total_reward += r

            s_next = agent.get_state(
                delayed_esn0_db=delayed_esn0[t + 1],
                slope_db_s=(delayed_esn0[t + 1] - delayed_esn0[t - 19]) / (20 * dt),
                deadline_s=2.0,
            )
            agent.train_step(s_curr, a, r, s_next, done=False)

        print(f"  Episode {ep + 1}/{episodes} finished - Mean step reward: {total_reward / steps_per_ep:.3f}")

    return agent


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Step 9: DRL-AMC Comparison Baseline Evaluation ===\n")

    table = ModcodTable()
    sc = step2_scenario()
    agent = DRLAMCAgent(table, lr=2e-3)

    # 1. Train agent
    t0 = time.time()
    agent = train_drl_agent(agent, sc, table, episodes=4, steps_per_ep=1500)
    print(f"DRL agent training completed in {time.time() - t0:.1f}s\n")

    # 2. Evaluate on held-out test channels (3 seeds x 12h)
    eval_seeds = [1, 2, 3]
    results = []

    for s in eval_seeds:
        sc_test = Scenario(seed=s, duration_h=12.0)
        att = generate_attenuation_db(sc_test)
        esn0 = sc_test.clear_sky_esn0_db - att
        n = len(esn0)
        win = max(1, int(round(sc_test.slope_window_s / sc_test.dt_s)))

        # DRL Policy
        drl_idx = agent.generate_policy_array(esn0, delay_steps=sc_test.feedback_delay_steps)
        res_drl, miss_drl = evaluate(drl_idx, esn0, sc_test, table)

        # Baselines
        idx_m2 = policies.margin_policy(table, esn0, 2.0, sc_test.feedback_delay_steps)
        res_m2, _ = evaluate(idx_m2, esn0, sc_test, table)

        idx_p2 = policies.predictive_policy(table, esn0, 2.0, sc_test.feedback_delay_steps, win)
        res_p2, _ = evaluate(idx_p2, esn0, sc_test, table)

        results.append({
            "seed": s,
            "drl_miss": res_drl["miss_rate"], "drl_ovh": res_drl["overhead"],
            "m2_miss": res_m2["miss_rate"], "m2_ovh": res_m2["overhead"],
            "p2_miss": res_p2["miss_rate"], "p2_ovh": res_p2["overhead"],
        })

    # Summary
    drl_miss = np.mean([r["drl_miss"] for r in results])
    drl_ovh = np.mean([r["drl_ovh"] for r in results])
    m2_miss = np.mean([r["m2_miss"] for r in results])
    m2_ovh = np.mean([r["m2_ovh"] for r in results])
    p2_miss = np.mean([r["p2_miss"] for r in results])
    p2_ovh = np.mean([r["p2_ovh"] for r in results])

    print(f"{'Method':30s} {'Miss Rate':>12s} {'Overhead O(a)':>15s}")
    print("-" * 60)
    print(f"{'Fixed-margin ACM (2 dB)':30s} {100 * m2_miss:11.3f}% {m2_ovh:15.3f}")
    print(f"{'Predictive ACM (2 dB)':30s} {100 * p2_miss:11.3f}% {p2_ovh:15.3f}")
    print(f"{'DRL-AMC Policy (Learned)':30s} {100 * drl_miss:11.3f}% {drl_ovh:15.3f}")

    # Save CSV
    csv_path = os.path.join(OUT, "step9_rl_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "drl_miss", "drl_ovh", "m2_miss", "m2_ovh", "p2_miss", "p2_ovh"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([m2_ovh], [100 * m2_miss], color="blue", marker="s", s=100, label="Fixed-margin ACM (2 dB)")
    ax.scatter([p2_ovh], [100 * p2_miss], color="green", marker="^", s=100, label="Predictive ACM (2 dB)")
    ax.scatter([drl_ovh], [100 * drl_miss], color="crimson", marker="o", s=120, label="DRL-AMC Policy (Learned)")

    ax.set_xlabel("Airtime Overhead O(a)")
    ax.set_ylabel("Deadline-Miss Rate (%)")
    ax.set_title("Step 9: DRL-AMC Reinforcement Learning Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(OUT, "step9_rl_comparison.png")
    plt.savefig(plot_path, dpi=130)
    print(f"Saved: {plot_path}")
    print("=== Step 9 Completed Successfully ===")


if __name__ == "__main__":
    main()
