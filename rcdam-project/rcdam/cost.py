"""Module 3: Cost and Overhead Model for RC-DAM.

Defines the measurable cost quantity per ARCHITECTURE.md Section 8:
O(a) = expected channel airtime used to deliver the message under action a,
       divided by the airtime needed for the same payload at the best clear-sky
       MODCOD with no redundancy.

Airtime includes:
- Base payload bits
- Protocol headers and control fields
- Packet-level erasure parity packets (code rate r = K/N)
- Duplicated packets
- Expected retransmissions under residual loss
- Transmit power step penalty term
- Delay penalty tie-breaker
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .link import ModcodTable


@dataclass(frozen=True)
class Action:
    """A transmission action candidate a."""
    modcod_idx: int                       # Index into ModcodTable
    power_step_db: float = 0.0            # Extra transmit power step (0, 1, 2, 3 dB)
    code_rate: float = 1.0                # Packet erasure code rate K/N <= 1.0 (e.g. 1.0, 0.8, 0.67, 0.5)
    interleaver_depth: int = 1            # Interleaving depth factor B (1, 2, 4, 8)
    duplication: int = 1                  # Transmission copies (1 = single, 2 = duplicated)
    defer_s: float = 0.0                  # Transmission delay offset (seconds)
    header_bytes_per_pkt: int = 40        # Protocol headers (e.g. GSE / BBFrame overhead)
    packet_payload_bytes: int = 1460      # Typical MTU payload bytes per packet


@dataclass
class CostModel:
    """Computes airtime overhead and total action cost O(a)."""
    table: ModcodTable
    symbol_rate_baud: float = 10e6
    power_penalty_weight: float = 0.1     # Tradeoff weight for extra RF power
    delay_penalty_weight: float = 0.01    # Tie-breaker weight for added transmission delay

    @property
    def max_clear_sky_eff(self) -> float:
        return float(self.table.eff.max())

    def clear_sky_reference_airtime(self, msg_bits: float) -> float:
        """Ideal airtime at best clear-sky MODCOD with zero redundancy and zero headers."""
        return float(msg_bits / (self.symbol_rate_baud * self.max_clear_sky_eff))

    def packet_count(self, msg_bits: float, action: Action) -> tuple[int, int]:
        """Compute (K source packets, N coded packets)."""
        payload_bits_per_pkt = action.packet_payload_bytes * 8
        k_src = int(np.ceil(msg_bits / payload_bits_per_pkt))
        r = max(min(action.code_rate, 1.0), 0.1)
        n_coded = int(np.ceil(k_src / r))
        return k_src, n_coded

    def total_tx_bits(self, msg_bits: float, action: Action) -> float:
        """Total bits over the air including payload, headers, parity, and duplication."""
        k_src, n_coded = self.packet_count(msg_bits, action)
        total_pkt_bytes = action.packet_payload_bytes + action.header_bytes_per_pkt
        total_bits_single = n_coded * total_pkt_bytes * 8.0
        return float(total_bits_single * max(1, action.duplication))

    def transmission_duration(self, msg_bits: float, action: Action) -> float:
        """Physical transmission airtime in seconds."""
        tx_bits = self.total_tx_bits(msg_bits, action)
        eff = self.table.eff[action.modcod_idx]
        phy_rate_bps = self.symbol_rate_baud * eff
        return float(tx_bits / phy_rate_bps)

    def total_span_duration(self, msg_bits: float, action: Action) -> float:
        """Total time from deferral start until complete reception of interleaved span."""
        t_tx = self.transmission_duration(msg_bits, action)
        # Interleaver spreads packets across time by factor action.interleaver_depth
        # when interleaving across transmission bursts
        t_span = action.defer_s + t_tx * action.interleaver_depth
        return float(t_span)

    def compute_overhead(self, msg_bits: float, action: Action,
                         residual_bler: float = 0.0,
                         deadline_s: float = 2.0) -> dict:
        """Compute single measurable quantity O(a) and its breakdown components.

        O(a) = expected channel airtime / clear-sky ideal airtime + power penalty + delay penalty.
        """
        ref_airtime = self.clear_sky_reference_airtime(msg_bits)
        base_airtime = self.transmission_duration(msg_bits, action)

        # Expected retransmissions factor under residual loss:
        p_loss = np.clip(residual_bler, 0.0, 0.95)
        expected_tx_multiplier = 1.0 / (1.0 - p_loss) if p_loss < 0.95 else 20.0
        expected_airtime = base_airtime * expected_tx_multiplier

        airtime_overhead = expected_airtime / ref_airtime

        # Power penalty: (10^(delta_P/10) - 1.0)
        power_linear = 10.0 ** (action.power_step_db / 10.0)
        power_penalty = self.power_penalty_weight * (power_linear - 1.0)

        # Delay penalty tie breaker
        total_delay = action.defer_s + base_airtime
        delay_penalty = self.delay_penalty_weight * (total_delay / max(deadline_s, 0.1))

        total_cost = airtime_overhead + power_penalty + delay_penalty

        return {
            "overhead": float(total_cost),
            "airtime_overhead": float(airtime_overhead),
            "expected_airtime_s": float(expected_airtime),
            "base_airtime_s": float(base_airtime),
            "ref_airtime_s": float(ref_airtime),
            "power_penalty": float(power_penalty),
            "delay_penalty": float(delay_penalty),
            "total_span_s": float(self.total_span_duration(msg_bits, action)),
        }
