"""Redundancy Optimizer and Infeasibility Signaller.

Implements RCDAM_Build_Guide.md Section 3, Section 4.5 & Milestone M4:
- Data contracts: Message, Action, Decision
- evaluate_action(a, z_hat_db, msg_bits, deadline_s, table, ...)
- choose_action(z_hat_by_alpha, msg_bits, deadline_s, eps_p, ...)
- Infeasibility reason diagnostics: 'ok', 'capacity', 'deadline', 'eps_too_strict', 'no_calibration_data'
- Full backward compatibility for RedundancyOptimizer and OptimizationResult
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

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
        if isinstance(self.priority, int):
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
                    seed: int = 0) -> Tuple[float, float]:
    """Evaluates delivery probability and airtime overhead for a candidate action."""
    K = int(np.ceil(msg_bits / FRAME_BITS))
    parity = int(np.ceil(a.parity_frac * K))
    esn0 = clear_esn0_db + a.power_db - z_hat_db
    p = float(table.bler(np.array([a.modcod]), np.array([esn0]))[0])
    frame_s = FRAME_BITS / (symbol_rate * table.eff[a.modcod])
    p_ok, sent = p_deliver_by_deadline(K, parity, p, frame_s, rtt_s, one_way_s, a.defer_s, deadline_s, n_mc=n_mc, seed=seed)
    ref_airtime = msg_bits / (symbol_rate * table.eff.max())
    overhead = sent * frame_s / ref_airtime + power_cost_per_db * a.power_db
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
                  n_mc: int = 2000,
                  use_validated_selection: bool = False) -> Tuple[Optional[Action], float, float, float, bool]:
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
    d_steps = int(round(rtt_s / 0.1))

    # Evaluation cache by (modcod, power_db, parity_frac, defer_s, round(z, 2))
    cache = {}

    # Collect all feasible candidates for validated selection
    feasible_candidates = []  # list of (overhead, action, alpha, p_lb, z)

    for m in mod_candidates:
        frame_s = FRAME_BITS / (symbol_rate * table.eff[m])
        for pw in powers:
            for pr in parities:
                span_s = int(np.ceil(K_msg * (1.0 + pr))) * frame_s + one_way_s
                for df in defers:
                    if df + min_tx_s > deadline_s:
                        continue
                    
                    if is_multi_horizon:
                        needed_steps = horizon_steps_needed(d_steps, df, span_s, dt=0.1)
                        h_pick = pick_horizon(grid_horizons, needed_steps)
                        if h_pick is None:
                            # Required horizon exceeds calibrated grid -> action not covered
                            continue
                        z_dict = z_hat_by_alpha[h_pick]
                    else:
                        z_dict = z_hat_by_alpha

                    sorted_alphas = sorted(z_dict.keys())
                    a = Action(m, pw, pr, df)
                    for alpha in sorted_alphas:
                        z = z_dict[alpha]
                        cache_key = (m, pw, pr, df, round(z, 2))
                        if cache_key in cache:
                            p_ok, ov = cache[cache_key]
                        else:
                            p_ok, ov = evaluate_action(
                                a, z, msg_bits, deadline_s, table, clear_esn0_db,
                                symbol_rate, rtt_s, one_way_s, n_mc=n_mc
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
            # Sort by overhead (cheapest first) and apply optimizer's-curse guard
            feasible_candidates.sort(key=lambda x: x[0])
            target_p = 1.0 - eps_p

            def p_select_fn(cand):
                return cand[3]  # p_lb from selection phase

            def p_verify_fn(cand):
                # Independent verification draws for the candidate action
                a_cand, alpha_cand, z = cand[1], cand[2], cand[4]
                # Defect 1.2 fix: Use independent seed and n_verify >= 3000
                n_verify = max(n_mc, 3000)
                verify_seed = int((hash((a_cand.modcod, a_cand.power_db, a_cand.parity_frac, a_cand.defer_s, round(z, 2))) + 777777) % 2147483647)
                p_v, _ = evaluate_action(
                    a_cand, z, msg_bits, deadline_s, table, clear_esn0_db,
                    symbol_rate, rtt_s, one_way_s, n_mc=n_verify, seed=verify_seed
                )
                k_success = int(round(p_v * n_verify))
                return k_success, n_verify

            validated = select_validated(feasible_candidates, p_select_fn, p_verify_fn, target_p)
            if validated is not None:
                ov_v, a_v, alpha_v, plb_v, _ = validated
                return a_v, alpha_v, ov_v, plb_v, True
            else:
                # Defect 1.2 fix: Never fall back silently to the unvalidated best!
                # If no candidate validates under independent draws, return INFEASIBLE.
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
        alpha = sorted_alphas[0]
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
                    n_mc: int = 500,
                    use_validated_selection: bool = False) -> Decision:
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

    a, alpha, ov, plb, feas = choose_action(
        z_hat_by_alpha, msg.size_bits, msg.deadline_s, msg.eps, table,
        clear_esn0_db, symbol_rate, rtt_s, one_way_s, mod_candidates,
        n_mc=n_mc, use_validated_selection=use_validated_selection
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

        a, alpha, ov, plb, feas = choose_action(
            z_dict, msg.size_bits, msg.deadline_s, msg.eps, self.table,
            clear_sky_esn0_db, symbol_rate, rtt_s, one_way_s, cands,
            powers=powers, parities=parities, defers=defers, n_mc=1000
        )

        status = f"FEASIBLE: P={plb:.4f} >= {msg.target_delivery_prob:.4f}, Overhead={ov:.3f}" if feas else \
                 f"INFEASIBLE: Best P={plb:.4f} < {msg.target_delivery_prob:.4f}"

        return OptimizationResult(
            is_feasible=feas,
            action=a,
            delivery_prob=plb,
            overhead=ov,
            best_achievable_prob=plb,
            status_message=status,
        )
