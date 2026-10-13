"""Module 7: RC-DAM Deadline-Constrained Redundancy Optimizer & Infeasibility Signaller.

Implements ARCHITECTURE.md Section 6, 7, and 8:
For message m = (S, p, D) at time t0:
- Delivery probability P(a) = probability decoded message is complete by D,
  evaluated via calibrated attenuation forecast through the link / erasure model.
- Constraint: P(a) >= 1 - epsilon_p (stricter epsilon for higher priority).
- Objective: minimize overhead O(a).
- Fallback: if no action is feasible, flag INFEASIBLE and return a* = argmax P(a) with P(a*).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

from .link import ModcodTable
from .cost import Action, CostModel
from .erasure import ErasureModel


@dataclass(frozen=True)
class Message:
    msg_id: int
    bits: float = 20e6
    priority: int = 2                 # 1: Critical (eps=0.001), 2: High (eps=0.01), 3: Standard (eps=0.05)
    deadline_s: float = 2.0
    arrival_s: float = 0.0

    @property
    def target_epsilon(self) -> float:
        if self.priority == 1:
            return 0.001
        elif self.priority == 2:
            return 0.01
        else:
            return 0.05

    @property
    def target_delivery_prob(self) -> float:
        return 1.0 - self.target_epsilon


@dataclass
class OptimizationResult:
    is_feasible: bool
    action: Action
    delivery_prob: float
    overhead: float
    best_achievable_prob: float
    status_message: str


class RedundancyOptimizer:
    """Joint deadline-constrained redundancy optimizer for RC-DAM."""
    def __init__(self,
                 table: ModcodTable,
                 cost_model: CostModel,
                 erasure_model: ErasureModel,
                 allow_power_control: bool = True,
                 allow_erasure_coding: bool = True,
                 allow_interleaving: bool = True,
                 allow_deferral: bool = True,
                 allow_duplication: bool = True):
        self.table = table
        self.cost_model = cost_model
        self.erasure = erasure_model

        self.allow_power = allow_power_control
        self.allow_erasure = allow_erasure_coding
        self.allow_interleaving = allow_interleaving
        self.allow_deferral = allow_deferral
        self.allow_duplication = allow_duplication

        self._candidate_actions = self._build_action_space()

    def _build_action_space(self) -> List[Action]:
        """Build candidate action set spanning physical, packet, and timing levers."""
        # 1. MODCODs: select a representative subset spanning the DVB-S2 SNR range
        # Indices: 0 (QPSK 1/4: -2.35dB), 3 (QPSK 1/2: 1.0dB), 6 (QPSK 3/4: 4.03dB),
        #          10 (QPSK 9/10: 6.42dB), 13 (8PSK 3/4: 7.91dB), 18 (16APSK 3/4: 10.21dB),
        #          23 (32APSK 3/4: 12.73dB), 27 (32APSK 9/10: 16.05dB)
        mod_indices = [0, 3, 6, 10, 13, 18, 23, 27]

        # 2. Power steps
        power_steps = [0.0, 1.0, 2.0, 3.0] if self.allow_power else [0.0]

        # 3. Erasure code rates
        code_rates = [1.0, 0.8, 0.67, 0.5] if self.allow_erasure else [1.0]

        # 4. Interleaver depths
        depths = [1, 2, 4, 8] if self.allow_interleaving else [1]

        # 5. Duplication
        duplications = [1, 2] if self.allow_duplication else [1]

        # 6. Deferral options
        defers = [0.0, 0.5, 1.0] if self.allow_deferral else [0.0]

        # Build curated candidates: avoid combinatorial explosion by pairing levers logically
        actions = []
        for m in mod_indices:
            # Baseline: no extra redundancy
            actions.append(Action(modcod_idx=m, power_step_db=0.0, code_rate=1.0, interleaver_depth=1))

            # Power stepped actions
            for p in power_steps:
                if p > 0:
                    actions.append(Action(modcod_idx=m, power_step_db=p, code_rate=1.0, interleaver_depth=1))

            # Packet-level erasure coding + interleaving
            for r in code_rates:
                if r < 1.0:
                    for b in depths:
                        actions.append(Action(modcod_idx=m, power_step_db=0.0, code_rate=r, interleaver_depth=b))
                        # Joint power + erasure
                        if self.allow_power:
                            actions.append(Action(modcod_idx=m, power_step_db=2.0, code_rate=r, interleaver_depth=b))

            # Duplication for critical protection
            if self.allow_duplication:
                actions.append(Action(modcod_idx=m, duplication=2))

            # Deferral options
            if self.allow_deferral:
                for d_sec in defers:
                    if d_sec > 0:
                        actions.append(Action(modcod_idx=m, defer_s=d_sec))

        return actions

    def optimize(self, msg: Message,
                 calibrated_att_db: float,
                 clear_sky_esn0_db: float = 18.0) -> OptimizationResult:
        """Find the minimum overhead action satisfying P(delivered by D) >= 1 - epsilon_p.

        If infeasible, return INFEASIBLE flag and a* = argmax P(a) with best achievable prob.
        """
        target_p = msg.target_delivery_prob
        predicted_esn0 = clear_sky_esn0_db - calibrated_att_db

        feasible_actions = []
        best_overall_action = None
        best_overall_p = -1.0
        best_overall_cost = float("inf")

        for a in self._candidate_actions:
            # 1. Physical deadline constraint check
            latency = self.erasure.block_latency_s(msg.bits, a)
            if latency > msg.deadline_s:
                continue

            # 2. Compute delivery probability through erasure/delivery model
            p_deliv = self.erasure.delivery_probability(
                msg_bits=msg.bits,
                action=a,
                esn0_db=predicted_esn0,
                deadline_s=msg.deadline_s,
            )

            # 3. Compute action overhead O(a)
            cost_info = self.cost_model.compute_overhead(
                msg_bits=msg.bits,
                action=a,
                residual_bler=1.0 - p_deliv,
                deadline_s=msg.deadline_s,
            )
            overhead = cost_info["overhead"]

            # Track best achievable delivery probability across all actions
            if (p_deliv > best_overall_p) or (abs(p_deliv - best_overall_p) < 1e-6 and overhead < best_overall_cost):
                best_overall_p = p_deliv
                best_overall_cost = overhead
                best_overall_action = a

            # 4. Check reliability constraint
            if p_deliv >= target_p:
                feasible_actions.append((overhead, p_deliv, a))

        # Decision rule evaluation
        if feasible_actions:
            # Sort by minimum overhead O(a)
            feasible_actions.sort(key=lambda item: item[0])
            best_cost, best_p, best_a = feasible_actions[0]
            return OptimizationResult(
                is_feasible=True,
                action=best_a,
                delivery_prob=best_p,
                overhead=best_cost,
                best_achievable_prob=best_p,
                status_message=f"FEASIBLE: P={best_p:.4f} >= {target_p:.4f}, Overhead={best_cost:.3f}",
            )
        else:
            # Infeasible fallback: report INFEASIBLE + best achievable P
            fallback_action = best_overall_action or self._candidate_actions[0]
            fallback_cost = best_overall_cost if best_overall_action else 1.0
            return OptimizationResult(
                is_feasible=False,
                action=fallback_action,
                delivery_prob=max(best_overall_p, 0.0),
                overhead=fallback_cost,
                best_achievable_prob=max(best_overall_p, 0.0),
                status_message=f"INFEASIBLE: Best P={max(best_overall_p, 0.0):.4f} < {target_p:.4f}",
            )
