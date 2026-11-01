"""Monte Carlo delivery-probability evaluator.

Models a constant frame-loss probability over the action window. A receiver
needs K successfully received logical frames. The initial transmission includes
K + parity frames; each ARQ round retransmits the remaining shortfall.

Duplication assumes independent copies, so a logical frame is lost only when
all copies are lost. Interleaving is modelled as a minimum per-round transmission
span: max(serial airtime, interleave_s).
"""
from typing import Tuple
import numpy as np
from scipy.stats import binom


def p_deliver_by_deadline(
    K: int,
    parity: int,
    p: float,
    frame_s: float,
    rtt_s: float,
    one_way_s: float,
    defer_s: float,
    deadline_s: float,
    n_mc: int = 2000,
    max_rounds: int = 8,
    seed: int = 0,
    dup: int = 1,
    interleave_s: float = 0.0,
) -> Tuple[float, float]:
    """Return (delivery probability by deadline, mean physical frames sent)."""
    if int(K) != K or K <= 0:
        raise ValueError("K must be a positive integer")
    if int(parity) != parity or parity < 0:
        raise ValueError("parity must be a non-negative integer")
    if int(n_mc) != n_mc or n_mc <= 0:
        raise ValueError("n_mc must be a positive integer")
    if int(max_rounds) != max_rounds or max_rounds <= 0:
        raise ValueError("max_rounds must be a positive integer")
    if int(dup) != dup or dup < 1:
        raise ValueError("dup must be a positive integer")
    for name, value in (
        ("frame_s", frame_s), ("rtt_s", rtt_s), ("one_way_s", one_way_s),
        ("defer_s", defer_s), ("deadline_s", deadline_s),
        ("interleave_s", interleave_s),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if frame_s <= 0:
        raise ValueError("frame_s must be greater than zero")
    if not np.isfinite(p):
        raise ValueError("p must be finite")

    K, parity, n_mc, max_rounds, dup = map(
        int, (K, parity, n_mc, max_rounds, dup)
    )
    p_eff = float(np.clip(p, 0.0, 1.0)) ** dup
    first_batch = K + parity
    first_round_span = max(first_batch * dup * frame_s, interleave_s)

    # Even the minimum K-frame transmission cannot meet the deadline.
    min_span = max(K * dup * frame_s, interleave_s)
    if defer_s + min_span + one_way_s > deadline_s:
        return 0.0, 0.0

    # Exact clean-channel shortcut.
    if p_eff <= 1e-12:
        if defer_s + first_round_span + one_way_s <= deadline_s:
            return 1.0, float(first_batch * dup)
        return 0.0, 0.0

    # Optimistic continuous-transmission upper bound. If even this cannot
    # deliver K frames, ARQ/RTT cannot rescue the message.
    available = deadline_s - defer_s - one_way_s
    # Ignore interleaving in this upper bound to keep it optimistic and safe.
    max_logical_frames = max(0, int(available / (dup * frame_s)))
    if max_logical_frames >= K:
        p_upper = float(binom.sf(K - 1, max_logical_frames, 1.0 - p_eff))
        if p_upper < 1e-12:
            return 0.0, 0.0

    rng = np.random.default_rng(seed)
    need = np.full(n_mc, K, dtype=np.int64)
    current_time = np.full(n_mc, float(defer_s))
    done_time = np.full(n_mc, np.inf)
    logical_to_send = np.full(n_mc, first_batch, dtype=np.int64)
    physical_sent = np.zeros(n_mc, dtype=np.int64)

    for _ in range(max_rounds):
        required_span = np.maximum(
            logical_to_send * dup * frame_s, interleave_s
        )
        active = (
            (need > 0)
            & (current_time + required_span + one_way_s <= deadline_s)
        )
        if not np.any(active):
            break

        received = rng.binomial(logical_to_send, 1.0 - p_eff)
        physical_round = logical_to_send * dup
        round_span = np.maximum(physical_round * frame_s, interleave_s)
        round_end = current_time + round_span

        physical_sent += np.where(active, physical_round, 0)
        need = np.where(active, np.maximum(need - received, 0), need)

        completed = active & (need == 0)
        done_time[completed] = round_end[completed] + one_way_s

        # Feedback arrives one RTT after the previous round ends.
        current_time = np.where(active & (need > 0), round_end + rtt_s, current_time)
        logical_to_send = np.where(need > 0, need, 0)

    success_probability = float(np.mean(done_time <= deadline_s))
    return success_probability, float(physical_sent.mean())
