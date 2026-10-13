"""Unit tests for Step 9: DRL-AMC Reinforcement Learning Baseline."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcdam.link import ModcodTable
from rcdam.rl_baseline import DRLAMCAgent


def test_drl_agent_initialization_and_forward():
    table = ModcodTable()
    agent = DRLAMCAgent(table)

    s = agent.get_state(delayed_esn0_db=12.0, slope_db_s=-0.1, deadline_s=2.0)
    assert s.shape == (3,)

    action = agent.select_action(s, evaluate=True)
    assert 0 <= action < len(table.names), f"Action {action} out of range [0, {len(table.names)})"

    # Train step
    s_next = agent.get_state(delayed_esn0_db=11.8, slope_db_s=-0.2, deadline_s=1.9)
    agent.train_step(s, action, r=1.5, s_next=s_next, done=False)

    print("DRL-AMC agent initialization and train step verified.")


def test_drl_policy_generation():
    table = ModcodTable()
    agent = DRLAMCAgent(table)

    esn0_db = np.linspace(18.0, 5.0, 100)
    policy_arr = agent.generate_policy_array(esn0_db, delay_steps=5)

    assert len(policy_arr) == 100
    assert np.all((policy_arr >= 0) & (policy_arr < len(table.names)))
    print("DRL-AMC policy array generation verified.")


if __name__ == "__main__":
    test_drl_agent_initialization_and_forward()
    test_drl_policy_generation()
    print("All Step 9 unit tests passed.")
