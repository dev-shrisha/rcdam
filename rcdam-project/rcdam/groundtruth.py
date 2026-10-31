"""Closed-Loop Frame-Level Ground Truth Simulator.

Implements RCDAM_Build_Guide.md Section 5 M5:
Simulates actual transmission of chosen action over the true time-varying attenuation
trajectory with Bernoulli frame losses and RTT-delayed ARQ rounds.
Prevents circular scoring (evaluator's model != ground truth).
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from .config import Scenario
from .link import ModcodTable
from .optimizer import Action, Message, FRAME_BITS


@dataclass
class DeliveryResult:
    delivered: bool
    latency_s: float
    airtime_s: float
    overhead: float
    frames_sent: int
    rounds_used: int
    reason: str


def simulate_true_delivery(action: Optional[Action],
                           t_start_idx: int,
                           att: np.ndarray,
                           msg: Message,
                           table: ModcodTable,
                           sc: Scenario,
                           rng: np.random.Generator) -> DeliveryResult:
    """Simulates frame-by-frame transmission of a message under true channel attenuation.

    Args:
        action: Chosen Action (or None if sender took no action)
        t_start_idx: Time step index when message arrives at sender
        att: True attenuation array [dB] across the scenario duration
        msg: Message to deliver
        table: ModcodTable
        sc: Scenario configuration
        rng: NumPy random generator for Bernoulli loss trials

    Returns:
        DeliveryResult with ground-truth outcome
    """
    ref_airtime = msg.size_bits / (sc.symbol_rate_baud * table.eff.max())

    if action is None:
        return DeliveryResult(
            delivered=False,
            latency_s=float("inf"),
            airtime_s=0.0,
            overhead=0.0,
            frames_sent=0,
            rounds_used=0,
            reason="no_action",
        )

    K = int(np.ceil(msg.size_bits / FRAME_BITS))
    parity = int(np.ceil(action.parity_frac * K))
    modcod = action.modcod
    eff = table.eff[modcod]
    frame_s = FRAME_BITS / (sc.symbol_rate_baud * eff)

    t0_s = t_start_idx * sc.dt_s
    deadline_timestamp = t0_s + msg.deadline_s
    n_steps = len(att)

    t_cur = t0_s + action.defer_s
    need = K
    send_count = K + parity
    total_frames_sent = 0
    rounds = 0
    max_rounds = sc.arq_max_rounds

    while need > 0 and rounds < max_rounds:
        rounds += 1
        # Check if transmission starts after deadline
        if t_cur + sc.one_way_delay_s > deadline_timestamp:
            break

        round_successful = 0
        for j in range(send_count):
            t_frame = t_cur + j * frame_s
            idx = int(np.clip(round(t_frame / sc.dt_s), 0, n_steps - 1))
            channel_att = att[idx]
            esn0 = sc.clear_sky_esn0_db + action.power_db - channel_att
            p_loss = float(table.bler([modcod], [esn0], use_curves=True)[0])

            # Bernoulli trial
            lost = rng.random() < p_loss
            if not lost:
                round_successful += 1

        total_frames_sent += send_count
        round_duration = send_count * frame_s
        round_arrival_at_rx = t_cur + round_duration + sc.one_way_delay_s

        need = max(0, need - round_successful)
        if need == 0:
            # Complete message received
            if round_arrival_at_rx <= deadline_timestamp:
                latency = round_arrival_at_rx - t0_s
                airtime = total_frames_sent * frame_s
                power_penalty = 0.03 * action.power_db
                overhead = (airtime / ref_airtime) + power_penalty
                return DeliveryResult(
                    delivered=True,
                    latency_s=latency,
                    airtime_s=airtime,
                    overhead=overhead,
                    frames_sent=total_frames_sent,
                    rounds_used=rounds,
                    reason="delivered",
                )
            else:
                # Decoded after deadline
                airtime = total_frames_sent * frame_s
                overhead = (airtime / ref_airtime) + 0.03 * action.power_db
                return DeliveryResult(
                    delivered=False,
                    latency_s=round_arrival_at_rx - t0_s,
                    airtime_s=airtime,
                    overhead=overhead,
                    frames_sent=total_frames_sent,
                    rounds_used=rounds,
                    reason="deadline_miss",
                )

        # Prepare next round (starts 1 RTT after round transmission completed)
        t_cur = t_cur + round_duration + sc.rtt_s
        send_count = need  # shortfall

    # Never completed within rounds or deadline
    airtime = total_frames_sent * frame_s
    overhead = (airtime / ref_airtime) + 0.03 * action.power_db
    return DeliveryResult(
        delivered=False,
        latency_s=float("inf"),
        airtime_s=airtime,
        overhead=overhead,
        frames_sent=total_frames_sent,
        rounds_used=rounds,
        reason="exhausted",
    )
