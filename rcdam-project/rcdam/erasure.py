"""Erasure coding and interleaving model for fast fades and burst loss.

Implements ARCHITECTURE.md Sections 7 and 10:
- Message S bits partitioned into K source packets.
- Coded into N packets (rate r = K/N <= 1.0).
- Interleaver depth B disperses burst errors across time:
  a burst of duration T_burst hits approx ceil(L_burst / B) packets of a given codeword.
- Tradeoff: interleaver depth B increases block span latency:
  total transmission span T_span = N * T_pkt * B.
  If T_span + RTT/2 + defer_s > deadline_s, the message misses its deadline!
- Delivery probability P_delivery(a) = P(received packets >= K and latency <= D).
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from scipy import stats

from .link import ModcodTable
from .cost import Action


@dataclass
class ErasureModel:
    table: ModcodTable
    symbol_rate_baud: float = 10e6
    pkt_payload_bytes: int = 1460
    header_bytes: int = 40
    one_way_delay_s: float = 0.27

    @property
    def total_pkt_bytes(self) -> int:
        return self.pkt_payload_bytes + self.header_bytes

    @property
    def total_pkt_bits(self) -> float:
        return self.total_pkt_bytes * 8.0

    def packet_transmission_time_s(self, modcod_idx: int) -> float:
        """Airtime for one single packet at the chosen MODCOD."""
        eff = self.table.eff[modcod_idx]
        phy_rate_bps = self.symbol_rate_baud * eff
        return self.total_pkt_bits / phy_rate_bps

    def packet_counts(self, msg_bits: float, code_rate: float) -> Tuple[int, int]:
        """Return (K source packets, N coded packets)."""
        src_bits_per_pkt = self.pkt_payload_bytes * 8.0
        k = max(1, int(np.ceil(msg_bits / src_bits_per_pkt)))
        r = np.clip(code_rate, 0.1, 1.0)
        n = max(k, int(np.ceil(k / r)))
        return k, n

    def block_latency_s(self, msg_bits: float, action: Action) -> float:
        """Total latency from message release to complete reception of all packets."""
        _, n = self.packet_counts(msg_bits, action.code_rate)
        t_pkt = self.packet_transmission_time_s(action.modcod_idx)
        # Interleaving spreads packets across B parallel streams / slot spacing
        t_span = n * t_pkt * action.interleaver_depth * action.duplication
        return action.defer_s + t_span + self.one_way_delay_s

    def delivery_probability(self, msg_bits: float,
                             action: Action,
                             esn0_db: float,
                             deadline_s: float,
                             mean_burst_s: float = 0.2) -> float:
        """Calculate P(delivered by deadline) under physical BLER and burst fade.

        1. Checks deadline feasibility: if latency > deadline_s, delivery probability is 0.
        2. Computes physical layer packet error rate (BLER) p_e.
        3. Computes burst loss impact mitigated by interleaver depth B.
        4. Computes probability of collecting at least K packets out of N.
        """
        latency = self.block_latency_s(msg_bits, action)
        if latency > deadline_s:
            return 0.0

        # Physical layer BLER with applied power boost
        eff_esn0 = esn0_db + action.power_step_db
        p_bler = float(self.table.bler(action.modcod_idx, eff_esn0))
        p_bler = float(np.clip(p_bler, 1e-7, 1.0))

        k, n = self.packet_counts(msg_bits, action.code_rate)
        t_pkt = self.packet_transmission_time_s(action.modcod_idx)

        # Burst loss occurs during rain / scintillation fades (when attenuation A > 1.0 dB)
        att_db = max(0.0, 18.0 - eff_esn0)
        fade_active = float(np.clip((att_db - 1.0) / 2.0, 0.0, 1.0))

        # Burst loss model:
        # A burst of duration T_burst packets = mean_burst_s / t_pkt
        # Interleaver depth B divides the burst length hitting this block:
        burst_pkts = max(1.0, mean_burst_s / max(t_pkt, 1e-6)) * fade_active
        effective_burst_on_block = burst_pkts / max(1, action.interleaver_depth)

        # Effective independent packet erasure rate combining random error and dispersed burst
        # When B is large, burst erasures are dispersed into independent losses
        burst_loss_factor = np.clip(effective_burst_on_block / max(n, 1), 0.0, 1.0)
        p_erasure = 1.0 - (1.0 - p_bler) * (1.0 - 0.5 * burst_loss_factor)
        p_erasure = float(np.clip(p_erasure, 1e-7, 1.0 - 1e-7))

        # Probability that at least K packets are received = P(Binomial(N, 1 - p_erasure) >= K)
        # For duplication > 1, any of the duplicates arriving successfully delivers the message:
        p_single_copy_success = float(stats.binom.sf(k - 1, n, 1.0 - p_erasure))

        if action.duplication > 1:
            p_success = 1.0 - ((1.0 - p_single_copy_success) ** action.duplication)
        else:
            p_success = p_single_copy_success

        return float(np.clip(p_success, 0.0, 1.0))

    def simulate_packet_transmission(self, msg_bits: float,
                                     action: Action,
                                     esn0_profile_db: np.ndarray,
                                     dt_s: float,
                                     rng: np.random.Generator) -> bool:
        """Monte Carlo simulation of transmission of the message over an Es/N0 profile."""
        k, n = self.packet_counts(msg_bits, action.code_rate)
        t_pkt = self.packet_transmission_time_s(action.modcod_idx)
        step_per_pkt = max(1, int(round(t_pkt / dt_s)))

        # Send packets spaced by interleaver_depth
        pkts_received = 0
        curr_step = int(round(action.defer_s / dt_s))
        profile_len = len(esn0_profile_db)

        for _ in range(action.duplication):
            for i in range(n):
                slot_idx = curr_step + i * action.interleaver_depth * step_per_pkt
                if slot_idx >= profile_len:
                    break
                slot_esn0 = esn0_profile_db[slot_idx] + action.power_step_db
                bler = float(self.table.bler(action.modcod_idx, slot_esn0))
                if rng.uniform(0.0, 1.0) > bler:
                    pkts_received += 1

            if pkts_received >= k:
                return True
        return pkts_received >= k
