"""Redundancy Optimizer and Infeasibility Signaller.

Implements RCDAM_Build_Guide.md Section 3, Section 4.5 & Milestone M4:
- Data contracts: Message, Action, Decision
- evaluate_action(a, z_hat_db, msg_bits, deadline_s, table, ...)
- choose_action(z_hat_by_alpha, msg_bits, deadline_s, eps_p, ...)
- Infeasibility reason diagnostics: 'ok', 'capacity', 'deadline', 'eps_too_strict', 'no_calibration_data'
- Full backward compatibility for RedundancyOptimizer and OptimizationResult
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import hashlib

from .evaluator import p_deliver_by_deadline
from .link import ModcodTable
from .validated_selection import select_validated, horizon_steps_needed, pick_horizon

FRAME_BITS = 64_800.0          # information bits per FEC frame
ALPHAS = (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)


@dataclass(frozen=True)
class Action:
    """Action tuple across physical, packet, and scheduling levers."""
    modcod: int = 0
    power_db: float = 0.0
    parity_frac: float = 0.0
    defer_s: float = 0.0
    # Stage B levers
    interleave_s: float = 0.0
    dup: int = 1

    # Aliases for backwards compatibility with legacy tests
    @property
    def modcod_idx(self) -> int:
        return self.modcod

    @property
    def power_step_db(self) -> float:
        return self.power_db

    @property
    def code_rate(self) -> float:
        # code_rate = K / (K + parity) = 1 / (1 + parity_frac)
        return 1.0 / (1.0 + self.parity_frac)

    @property
    def interleaver_depth(self) -> int:
        return max(1, int(round(self.interleave_s / 0.1))) if self.interleave_s > 0 else 1

    @property
    def duplication(self) -> int:
        return self.dup


@dataclass(frozen=True)
class Message:
    """Message arrival descriptor."""
    size_bits: float = 20e6
    priority: Union[str, int] = "normal"          # "critical" | "normal" | "best_effort"
    deadline_s: float = 2.0
    eps: float = 0.01                             # permitted deadline-failure probability
    arrival_idx: int = 0

    # Backwards compatibility fields
    msg_id: int = 0
    bits: float = 20e6
    arrival_s: float = 0.0

    def __post_init__(self):
        # Synchronize legacy bits / size_bits
        if self.bits != 20e6 and self.size_bits == 20e6:
            object.__setattr__(self, "size_bits", float(self.bits))
        elif self.size_bits != 20e6 and self.bits == 20e6:
            object.__setattr__(self, "bits", float(self.size_bits))

        # Handle numeric priority for legacy tests (1: critical, 2: high/normal, 3: standard)
        if isinstance(self.priority, int) and self.eps == 0.01:
            if self.priority == 1:
                object.__setattr__(self, "eps", 0.001)
            elif self.priority == 2:
                object.__setattr__(self, "eps", 0.01)
            else:
                object.__setattr__(self, "eps", 0.05)

    @property
    def target_epsilon(self) -> float:
        return self.eps

    @property
    def target_delivery_prob(self) -> float:
        return 1.0 - self.eps


@dataclass
class Decision:
    """Decision output for message delivery."""
    action: Optional[Action]
    feasible: bool
    alpha: float           # calibration level used
    p_lower_bound: float   # p_link - alpha when feasible; P* (clamped at 0) when infeasible
    overhead: float
    reason: str            # "ok" | "capacity" | "deadline" | "eps_too_strict" | "no_calibration_data"
    decision_log: Optional[Dict[str, Any]] = None


def evaluate_action(a: Action,
                    z_hat_db: float,
                    msg_bits: float,
                    deadline_s: float,
                    table: ModcodTable,
                    clear_esn0_db: float,
                    symbol_rate: float,
                    rtt_s: float,
                    one_way_s: float,
                    n_mc: int = 2000,
                    power_cost_per_db: float = 0.03,
                    seed: int = 0,
                    header_bytes: int = 40) -> Tuple[float, float]:
    """Evaluates delivery probability and airtime overhead for a candidate action."""
    K = int(np.ceil(msg_bits / FRAME_BITS))
    parity = int(np.ceil(a.parity_frac * K))
    esn0 = clear_esn0_db + a.power_db - z_hat_db
    p = float(table.bler(np.array([a.modcod]), np.array([esn0]))[0])
    frame_s = FRAME_BITS / (symbol_rate * table.eff[a.modcod])
    p_ok, sent = p_deliver_by_deadline(
        K, parity, p, frame_s, rtt_s, one_way_s, a.defer_s, deadline_s,
        n_mc=n_mc, seed=seed, interleave_s=a.interleave_s, dup=a.dup
    )
    ref_airtime = msg_bits / (symbol_rate * table.eff.max())
    header_factor = (1460.0 + header_bytes) / 1500.0
    overhead = (sent * frame_s / ref_airtime) * header_factor + power_cost_per_db * a.power_db
    return p_ok, overhead


def choose_action(z_hat_by_alpha: Dict[float, float],
                  msg_bits: float,
                  deadline_s: float,
                  eps_p: float,
                  table: ModcodTable,
                  clear_esn0_db: float,
                  symbol_rate: float,
                  rtt_s: float,
                  one_way_s: float,
                  mod_candidates: List[int],
                  powers: Tuple[float, ...] = (0.0, 1.0, 2.0),
                  parities: Tuple[float, ...] = (0.0, 0.05, 0.15, 0.3),
                  defers: Tuple[float, ...] = (0.0, 0.5, 1.0),
                  interleaves: Tuple[float, ...] = (0.0,),
                  dups: Tuple[int, ...] = (1,),
                  n_mc: int = 2000,
                  use_validated_selection: bool = False,
                  header_bytes: int = 40, seed: int = 0, extra_defer_s: float = 0.0) -> Tuple[Optional[Action], float, float, float, bool]:
    """Joint deadline-constrained optimization over action candidates and alpha confidence levels.

    Parameters
    ----------
    use_validated_selection : bool
        When True, guard against optimizer's curse by verifying the best feasible
        action with independent MC draws via select_validated / wilson_interval.
        Defaults to False for backward compatibility.

    Returns:
        (action, alpha, overhead, p_lower_bound, feasible)
    """
    if len(z_hat_by_alpha) == 0:
        return None, 1.0, float("inf"), 0.0, False

    # Check physical deadline feasibility: minimum airtime with fastest MODCOD
    min_frame_s = FRAME_BITS / (symbol_rate * table.eff.max())
    min_tx_s = int(np.ceil(msg_bits / FRAME_BITS)) * min_frame_s + one_way_s
    if deadline_s < min_tx_s:
        # Physical deadline violation
        a_dummy = Action(modcod=len(table.eff) - 1, power_db=0.0, parity_frac=0.0, defer_s=0.0)
        return a_dummy, 1.0, 1.0, 0.0, False

    best_feasible = None
    best_any = (None, None, float("inf"), -1.0)

    # Check if z_hat_by_alpha is multi-horizon (keyed by H_steps) or single-horizon (keyed by alpha)
    is_multi_horizon = len(z_hat_by_alpha) > 0 and isinstance(next(iter(z_hat_by_alpha.keys())), (int, np.integer))
    grid_horizons = sorted(z_hat_by_alpha.keys()) if is_multi_horizon else None
    K_msg = int(np.ceil(msg_bits / FRAME_BITS))
    ref_airtime = msg_bits / (symbol_rate * table.eff.max())
    d_steps = int(round(rtt_s / 0.1))

    # Minimum attenuation across all horizons/alphas (most optimistic channel)
    if is_multi_horizon:
        z_min = min(min(zd.values()) for zd in z_hat_by_alpha.values())
    else:
        z_min = min(z_hat_by_alpha.values())
    max_pw = max(powers) if powers else 0.0

    # Fast BLER cache
    bler_cache = {}
    def get_bler(m_idx: int, esn0_val: float) -> float:
        k_b = (m_idx, round(esn0_val, 2))
        if k_b not in bler_cache:
            bler_cache[k_b] = float(table.bler(np.array([m_idx]), np.array([esn0_val]))[0])
        return bler_cache[k_b]

    # Evaluation cache by (modcod, power_db, parity_frac, defer_s, interleave_s, dup, round(z, 2))
    cache = {}

    # Collect all feasible candidates for validated selection
    feasible_candidates = []  # list of (overhead, action, alpha, p_lb, z)

    # Sort MODCODs by efficiency descending (cheapest airtime first)
    sorted_mod_cands = sorted(mod_candidates, key=lambda idx: table.eff[idx], reverse=True)

    for m in sorted_mod_cands:
        frame_s = FRAME_BITS / (symbol_rate * table.eff[m])
        min_mod_ov = (K_msg * frame_s) / ref_airtime
        if best_feasible is not None and min_mod_ov >= best_feasible[2]:
            # Any lower MODCOD will have even higher overhead; can stop searching lower MODCODs
            break

        # Early prune: if with max power and lowest attenuation, BLER is >= 0.999, this MODCOD cannot work
        if get_bler(m, clear_esn0_db + max_pw - z_min) >= 0.999:
            continue

        for pw in sorted(powers):
            # If base overhead already exceeds best feasible, break out of higher powers
            if best_feasible is not None and min_mod_ov + 0.03 * pw >= best_feasible[2]:
                break

            # If for this specific power and lowest attenuation, BLER is >= 0.999, skip
            if get_bler(m, clear_esn0_db + pw - z_min) >= 0.999:
                continue

            for pr in sorted(parities):
                for dp in sorted(dups):
                    # Duplication >= 2 doubles frames; cannot beat a feasible dup=1 action
                    if dp > 1 and best_feasible is not None and best_feasible[2] < 1.8:
                        continue

                    for intlv in sorted(interleaves):
                        # Under constant-p bound, interleaving > 0 only increases span
                        if intlv > 0.0 and best_feasible is not None:
                            continue

                        total_first_pkts = int(np.ceil(K_msg * (1.0 + pr))) * dp
                        span_s = max(total_first_pkts * frame_s, intlv) + one_way_s

                        for df in defers:
                            if df + min_tx_s > deadline_s:
                                continue

                            if is_multi_horizon:
                                needed_steps = horizon_steps_needed(
                                    d_steps, df + extra_defer_s, span_s, dt=0.1
                                )
                                h_pick = pick_horizon(grid_horizons, needed_steps)
                                if h_pick is None:
                                    continue
                                z_dict = z_hat_by_alpha[h_pick]
                            else:
                                z_dict = z_hat_by_alpha

                            sorted_alphas = sorted(z_dict.keys())
                            a = Action(m, pw, pr, df, interleave_s=intlv, dup=dp)

                            for alpha in sorted_alphas:
                                z = z_dict[alpha]
                                # Quick check: if BLER at this specific z is >= 0.999 and no duplication can rescue it
                                if get_bler(m, clear_esn0_db + pw - z) >= 0.999 and dp == 1:
                                    continue

                                cache_key = (m, pw, pr, df, intlv, dp, round(z, 2))
                                if cache_key in cache:
                                    p_ok, ov = cache[cache_key]
                                else:
                                    p_ok, ov = evaluate_action(
                                        a, z, msg_bits, deadline_s, table, clear_esn0_db,
                                        symbol_rate, rtt_s, one_way_s, n_mc=n_mc,
                                        seed=seed, header_bytes=header_bytes
                                    )
                                    cache[cache_key] = (p_ok, ov)

                                p_lb = p_ok - alpha
                                if p_lb > best_any[3]:
                                    best_any = (a, alpha, ov, p_lb)
                                if p_lb >= 1.0 - eps_p and (best_feasible is None or ov < best_feasible[2]):
                                    best_feasible = (a, alpha, ov, p_lb)
                                    if use_validated_selection:
                                        feasible_candidates.append((ov, a, alpha, p_lb, z))

    if best_feasible is not None:
        if use_validated_selection and len(feasible_candidates) > 0:
            feasible_candidates.sort(key=lambda x: x[0])
            target_p = 1.0 - eps_p
            # A Wilson lower bound can never equal 1.0, so candidates that
            # spend the entire reliability budget on calibration uncertainty
            # cannot be independently certified.
            certifiable_candidates = [
                cand for cand in feasible_candidates
                if target_p + cand[2] < 1.0
            ]

            def p_select_fn(cand):
                return cand[3]

            def p_verify_fn(cand):
                a_cand, alpha_cand, z = cand[1], cand[2], cand[4]
                # Critical targets need thousands of independent trials. The
                # cap is explicit: if the bound cannot be certified, return
                # INFEASIBLE rather than silently weakening the confidence rule.
                n_verify = min(10000, max(1000, int(np.ceil(6.0 / max(eps_p, 1e-6)))))
                seed_material = (
                    f"{seed}|{a_cand.modcod}|{a_cand.power_db}|{a_cand.parity_frac}|"
                    f"{a_cand.defer_s}|{a_cand.interleave_s}|{a_cand.dup}|"
                    f"{round(z, 5)}|{msg_bits}|{deadline_s}|{eps_p}"
                ).encode("utf-8")
                verify_seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:4], "big"
                )
                p_v, _ = evaluate_action(
                    a_cand, z, msg_bits, deadline_s, table, clear_esn0_db,
                    symbol_rate, rtt_s, one_way_s, n_mc=n_verify,
                    seed=verify_seed, header_bytes=header_bytes
                )
                k_success = int(round(p_v * n_verify))
                return k_success, n_verify

            # Verification is performed on the single lowest-cost candidate
            # selected using the separate selection sample. This avoids
            # uncorrected repeated 95% tests across many candidates.
            validated = select_validated(
                certifiable_candidates[:1], p_select_fn, p_verify_fn, target_p,
                target_for_candidate=lambda cand: target_p + cand[2]
            )
            if validated is not None:
                ov_v, a_v, alpha_v, plb_v, _ = validated
                return a_v, alpha_v, ov_v, plb_v, True
            else:
                a_b, alpha_b, ov_b, plb_b = best_any
                if a_b is None:
                    a_b = Action(modcod=0, power_db=max(powers), parity_frac=max(parities), defer_s=0.0)
                    alpha_b = sorted_alphas[0]
                    ov_b = 1.0
                    plb_b = 0.0
                return a_b, alpha_b, ov_b, max(plb_b, 0.0), False

        return (*best_feasible, True)

    a, alpha, ov, p_lb = best_any
    if a is None:
        a = Action(modcod=0, power_db=max(powers), parity_frac=max(parities), defer_s=0.0)
        alpha = sorted_alphas[0] if 'sorted_alphas' in locals() else 1.0
        ov = 1.0
        p_lb = 0.0
    return a, alpha, ov, max(p_lb, 0.0), False


def choose_decision(z_hat_by_alpha: Dict[float, float],
                    msg: Message,
                    table: ModcodTable,
                    clear_esn0_db: float,
                    symbol_rate: float,
                    rtt_s: float,
                    one_way_s: float,
                    mod_candidates: Optional[List[int]] = None,
                    interleaves: Tuple[float, ...] = (0.0,),
                    dups: Tuple[int, ...] = (1,),
                    n_mc: int = 500,
                    use_validated_selection: bool = False,
                    header_bytes: int = 40, seed: Optional[int] = None, extra_defer_s: float = 0.0) -> Decision:
    """Wrapper returning structured Decision dataclass with exact reason."""
    if len(z_hat_by_alpha) == 0:
        return Decision(None, False, 1.0, 0.0, float("inf"), reason="no_calibration_data")

    min_frame_s = FRAME_BITS / (symbol_rate * table.eff.max())
    min_tx_s = int(np.ceil(msg.size_bits / FRAME_BITS)) * min_frame_s + one_way_s
    if msg.deadline_s < min_tx_s:
        a_dummy = Action(modcod=len(table.eff) - 1, power_db=0.0, parity_frac=0.0, defer_s=0.0)
        return Decision(a_dummy, False, 1.0, 0.0, 1.0, reason="deadline")

    if msg.eps <= 0.0:
        return Decision(None, False, 1.0, 0.0, float("inf"), reason="eps_too_strict")

    if mod_candidates is None:
        # Evaluate all MODCODs whose minimum payload transmission time physically fits
        # within the deadline, rather than pruning to an optimistic top-4 subset that
        # discards robust lower MODCODs under severe attenuation (Defect 1.1).
        mod_candidates = [
            i for i in range(len(table.eff))
            if int(np.ceil(msg.size_bits / FRAME_BITS)) * (FRAME_BITS / (symbol_rate * table.eff[i])) + one_way_s <= msg.deadline_s
        ]
        if len(mod_candidates) == 0:
            mod_candidates = [0]

    if seed is None:
        seed = int(msg.arrival_idx) % 2147483647

    a, alpha, ov, plb, feas = choose_action(
        z_hat_by_alpha, msg.size_bits, msg.deadline_s, msg.eps, table,
        clear_esn0_db, symbol_rate, rtt_s, one_way_s, mod_candidates,
        interleaves=interleaves, dups=dups,
        n_mc=n_mc, use_validated_selection=use_validated_selection,
        header_bytes=header_bytes, seed=seed,
        extra_defer_s=extra_defer_s
    )

    if feas:
        reason = "ok"
    elif a is None:
        reason = "no_calibration_data"
    else:
        reason = "capacity"
    return Decision(action=a, feasible=feas, alpha=alpha, p_lower_bound=plb, overhead=ov, reason=reason)


# ==============================================================================
# Legacy Adapter Classes for backward compatibility
# ==============================================================================
@dataclass
class OptimizationResult:
    is_feasible: bool
    action: Action
    delivery_prob: float
    overhead: float
    best_achievable_prob: float
    status_message: str


class RedundancyOptimizer:
    """Backward compatibility adapter for Step 7 / Step 8 / Step 9 legacy scripts and tests."""
    def __init__(self,
                 table: ModcodTable,
                 cost_model=None,
                 erasure_model=None,
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

    def optimize(self, msg: Message, calibrated_att_db: float, clear_sky_esn0_db: float = 18.0) -> OptimizationResult:
        # Build synthetic z_hat_by_alpha with calibrated_att_db
        z_dict = {a: calibrated_att_db + 2.0 * (1.0 - a) for a in ALPHAS}
        symbol_rate = 25e6
        rtt_s = 0.54
        one_way_s = 0.27
        min_bound = min(z_dict.values())
        cands = [i for i in range(len(self.table.thr)) if self.table.thr[i] <= clear_sky_esn0_db - min_bound + 3.0]
        if not cands:
            cands = [0]

        powers = (0.0, 1.0, 2.0, 3.0) if self.allow_power else (0.0,)
        parities = (0.0, 0.05, 0.15, 0.3) if self.allow_erasure else (0.0,)
        defers = (0.0, 0.5, 1.0) if self.allow_deferral else (0.0,)
        interleaves = (0.0, 0.5, 1.0) if self.allow_interleaving else (0.0,)
        dups = (1, 2) if self.allow_duplication else (1,)
        header_bytes = getattr(self.erasure, 'header_bytes', 40) if self.erasure else 40

        a, alpha, ov, plb, feas = choose_action(
            z_dict, msg.size_bits, msg.deadline_s, msg.eps, self.table,
            clear_sky_esn0_db, symbol_rate, rtt_s, one_way_s, cands,
            powers=powers, parities=parities, defers=defers,
            interleaves=interleaves, dups=dups, n_mc=200,
            header_bytes=header_bytes
        )

        status = (
            f"FEASIBLE: P={plb:.4f} >= {msg.target_delivery_prob:.4f}, Overhead={ov:.3f}"
            if feas else f"INFEASIBLE: Best P={plb:.4f} < {msg.target_delivery_prob:.4f}"
        )

        return OptimizationResult(
            is_feasible=feas,
            action=a,
            delivery_prob=plb,
            overhead=ov,
            best_achievable_prob=plb,
            status_message=status,
        )
