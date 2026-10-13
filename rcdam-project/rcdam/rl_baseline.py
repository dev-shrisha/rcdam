"""Module 9: Deep Reinforcement Learning for Adaptive Modulation & Coding (DRL-AMC).

Implements the DRL-AMC comparison baseline per ARCHITECTURE.md Sections 5, 11, and 13.
State: [delayed_esn0, esn0_slope, normalized_deadline]
Action: candidate MODCOD index
Reward: delivered throughput minus deadline-miss penalty and overhead cost.
"""
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .link import ModcodTable


class QNetwork(nn.Module):
    """Deep Q-Network for DRL-AMC policy."""
    def __init__(self, state_dim: int = 3, action_dim: int = 28, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRLAMCAgent:
    """Q-learning agent for satellite ACM decision-making under delayed feedback."""
    def __init__(self,
                 table: ModcodTable,
                 state_dim: int = 3,
                 hidden_dim: int = 64,
                 lr: float = 1e-3,
                 gamma: float = 0.95,
                 epsilon: float = 0.1):
        self.table = table
        self.action_dim = len(table.names)
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_net = QNetwork(state_dim, self.action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def get_state(self, delayed_esn0_db: float, slope_db_s: float, deadline_s: float) -> np.ndarray:
        """Form normalized state vector."""
        return np.array([
            (delayed_esn0_db - 10.0) / 10.0,
            slope_db_s / 2.0,
            deadline_s / 5.0,
        ], dtype=np.float32)

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and np.random.uniform(0.0, 1.0) < self.epsilon:
            return int(np.random.randint(0, self.action_dim))
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = self.q_net(s_t)
            return int(torch.argmax(q_vals, dim=1).item())

    def train_step(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool):
        """Single Q-learning update step."""
        self.q_net.train()
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        s_next_t = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0)

        q_curr = self.q_net(s_t)[0, a]
        with torch.no_grad():
            q_next = torch.max(self.q_net(s_next_t)) if not done else 0.0
            q_target = r + self.gamma * q_next

        loss = self.criterion(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def generate_policy_array(self, esn0_db: np.ndarray, delay_steps: int,
                              slope_window: int = 50, dt_s: float = 0.1) -> np.ndarray:
        """Apply learned DRL policy across continuous time series to generate MODCOD indices."""
        self.q_net.eval()
        n = len(esn0_db)
        out = np.zeros(n, dtype=np.int64)

        # Delayed Es/N0
        delayed = np.empty_like(esn0_db)
        delayed[delay_steps:] = esn0_db[:n - delay_steps]
        delayed[:delay_steps] = esn0_db[0]

        # Slope estimate
        slope = np.zeros_like(delayed)
        slope[slope_window:] = (delayed[slope_window:] - delayed[:-slope_window]) / (slope_window * dt_s)

        # Vectorized inference
        norm_esn0 = (delayed - 10.0) / 10.0
        norm_slope = slope / 2.0
        norm_deadline = np.full(n, 2.0 / 5.0)

        states = np.stack([norm_esn0, norm_slope, norm_deadline], axis=1).astype(np.float32)
        with torch.no_grad():
            states_t = torch.tensor(states, dtype=torch.float32)
            q_vals = self.q_net(states_t)
            out = torch.argmax(q_vals, dim=1).cpu().numpy()

        return out
