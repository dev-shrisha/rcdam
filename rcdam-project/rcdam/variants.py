"""Policy Variants for RC-DAM Closed-Loop Benchmark (V0 to V6 and Oracle).

Implements RCDAM_Build_Guide.md Section 5 M5:
- Standardized policy interface: policy(context) -> Decision
- V0: Static MODCOD + ARQ
- V1: Fixed-Margin ACM + ARQ
- V2: Linear Predictive ACM + ARQ
- V3: Quantile Forecaster + MODCOD (no conformal calibration)
- V4: V3 + Stratified Conformal Window Bound
- V5: V4 + All Stage A Levers (Power, Proactive Parity, Deferral)
- V6: Full RC-DAM with Infeasibility Signalling and P*
- Oracle: Zero-Delay Oracle (knows true window, exhaustive search)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .config import Scenario
from .link import ModcodTable
from .optimizer import (
    Action,
    Decision,
    Message,
    choose_action,
    choose_decision,
    evaluate_action,
    ALPHAS,
    FRAME_BITS,
)
from .calibration import fade_state, apply_corrections
from .validated_selection import horizon_steps_needed, pick_horizon

HORIZONS = [10, 20, 30, 50, 100, 200, 300, 600]


@dataclass
class PolicyContext:
    arrival_idx: int
    last_meas_idx: int
    cur_att_db: float
    features: dict
    true_future_att: Optional[np.ndarray]
    msg: Message
    sc: Scenario
    table: ModcodTable
    forecaster: Any
    calibration_corr: Dict[int, Dict[float, Dict[int, float]]]


def policy_v0_static(ctx: PolicyContext, static_modcod: int = 6) -> Decision:
    """V0: Static MODCOD + ARQ (QPSK 3/4, eff=1.487, airtime=1.35s + 0.27s < 2.0s deadline).

    Uses static_modcod=6 (the most robust MODCOD that physically fits within the
    deadline). Lower indices (0-5) have transmission times exceeding the 2.0s
    deadline, causing 100% deadline misses.
    """
    a = Action(modcod=static_modcod, power_db=0.0, parity_frac=0.0, defer_s=0.0)
    ref_airtime = ctx.msg.size_bits / (ctx.sc.symbol_rate_baud * ctx.table.eff.max())
    frame_s = FRAME_BITS / (ctx.sc.symbol_rate_baud * ctx.table.eff[static_modcod])
    K = int(np.ceil(ctx.msg.size_bits / FRAME_BITS))
    overhead = (K * frame_s) / ref_airtime
    return Decision(action=a, feasible=True, alpha=0.0, p_lower_bound=1.0, overhead=overhead, reason="ok")


def policy_v1_fixed_margin(ctx: PolicyContext, margin_db: float = 3.0) -> Decision:
    """V1: Fixed-Margin ACM + ARQ (uses 1 RTT delayed measurement with margin)."""
    esn0_est = ctx.sc.clear_sky_esn0_db - ctx.cur_att_db - margin_db
    mod_idx = int(ctx.table.select(np.array([esn0_est]))[0])
    if mod_idx < 0:
        mod_idx = 0
    a = Action(modcod=mod_idx, power_db=0.0, parity_frac=0.0, defer_s=0.0)
    ref_airtime = ctx.msg.size_bits / (ctx.sc.symbol_rate_baud * ctx.table.eff.max())
    frame_s = FRAME_BITS / (ctx.sc.symbol_rate_baud * ctx.table.eff[mod_idx])
    K = int(np.ceil(ctx.msg.size_bits / FRAME_BITS))
    overhead = (K * frame_s) / ref_airtime
    return Decision(action=a, feasible=True, alpha=0.0, p_lower_bound=1.0, overhead=overhead, reason="ok")


def policy_v2_linear_predictive(ctx: PolicyContext, margin_db: float = 2.0) -> Decision:
    """V2: Linear Predictive ACM + ARQ (slope extrapolation over feedback delay + deadline)."""
    slope = ctx.features["slope5"][0] if "slope5" in ctx.features else 0.0
    dt_lookahead = (ctx.sc.feedback_delay_steps * ctx.sc.dt_s) + 1.0
    att_pred = ctx.cur_att_db + slope * dt_lookahead + margin_db
    esn0_est = ctx.sc.clear_sky_esn0_db - att_pred
    mod_idx = int(ctx.table.select(np.array([esn0_est]))[0])
    if mod_idx < 0:
        mod_idx = 0
    a = Action(modcod=mod_idx, power_db=0.0, parity_frac=0.0, defer_s=0.0)
    ref_airtime = ctx.msg.size_bits / (ctx.sc.symbol_rate_baud * ctx.table.eff.max())
    frame_s = FRAME_BITS / (ctx.sc.symbol_rate_baud * ctx.table.eff[mod_idx])
    K = int(np.ceil(ctx.msg.size_bits / FRAME_BITS))
    overhead = (K * frame_s) / ref_airtime
    return Decision(action=a, feasible=True, alpha=0.0, p_lower_bound=1.0, overhead=overhead, reason="ok")


def policy_v3_quantile_no_conformal(ctx: PolicyContext) -> Decision:
    """V3: Quantile Forecaster + MODCOD (raw forecaster, no conformal calibration)."""
    tau = 1.0 - ctx.msg.eps
    k_horizon = 2  # ~3s horizon
    q_raw = float(ctx.forecaster.predict(np.array([ctx.cur_att_db]), k_horizon, tau)[0])
    esn0_est = ctx.sc.clear_sky_esn0_db - q_raw
    mod_idx = int(ctx.table.select(np.array([esn0_est]))[0])
    if mod_idx < 0:
        mod_idx = 0
    a = Action(modcod=mod_idx, power_db=0.0, parity_frac=0.0, defer_s=0.0)
    ref_airtime = ctx.msg.size_bits / (ctx.sc.symbol_rate_baud * ctx.table.eff.max())
    frame_s = FRAME_BITS / (ctx.sc.symbol_rate_baud * ctx.table.eff[mod_idx])
    K = int(np.ceil(ctx.msg.size_bits / FRAME_BITS))
    overhead = (K * frame_s) / ref_airtime
    return Decision(action=a, feasible=True, alpha=ctx.msg.eps, p_lower_bound=1.0 - ctx.msg.eps, overhead=overhead, reason="ok")


def policy_v4_conformal_modcod(ctx: PolicyContext) -> Decision:
    """V4: V3 + Stratified Conformal Window Bound (MODCOD adaptation only)."""
    tau = 1.0 - ctx.msg.eps
    K = int(np.ceil(ctx.msg.size_bits / FRAME_BITS))
    frame_s_est = FRAME_BITS / (ctx.sc.symbol_rate_baud * ctx.table.eff.max())
    span_s = K * frame_s_est + ctx.sc.one_way_delay_s
    needed = horizon_steps_needed(ctx.sc.feedback_delay_steps, 0.0, span_s, ctx.sc.dt_s)
    grid_horizons = sorted(ctx.calibration_corr.keys()) if ctx.calibration_corr else HORIZONS
    H_steps = pick_horizon(grid_horizons, needed)
    if H_steps is None:
        return Decision(action=None, feasible=False, alpha=ctx.msg.eps, p_lower_bound=0.0, overhead=float("inf"), reason="no_calibration_data")

    k_horizon = HORIZONS.index(H_steps) if H_steps in HORIZONS else min(range(len(HORIZONS)), key=lambda i: abs(HORIZONS[i] - H_steps))
    q_raw = float(ctx.forecaster.predict(np.array([ctx.cur_att_db]), k_horizon, tau)[0])
    st = fade_state(ctx.cur_att_db)
    corr = ctx.calibration_corr.get(H_steps, {}).get(tau, {}).get(st, 0.0)
    q_calibrated = q_raw + corr

    esn0_est = ctx.sc.clear_sky_esn0_db - q_calibrated
    mod_idx = int(ctx.table.select(np.array([esn0_est]))[0])
    if mod_idx < 0:
        mod_idx = 0
    a = Action(modcod=mod_idx, power_db=0.0, parity_frac=0.0, defer_s=0.0)
    ref_airtime = ctx.msg.size_bits / (ctx.sc.symbol_rate_baud * ctx.table.eff.max())
    frame_s = FRAME_BITS / (ctx.sc.symbol_rate_baud * ctx.table.eff[mod_idx])
    overhead = (K * frame_s) / ref_airtime
    return Decision(action=a, feasible=True, alpha=ctx.msg.eps, p_lower_bound=1.0 - ctx.msg.eps, overhead=overhead, reason="ok")


def _build_multi_horizon_bounds(ctx: PolicyContext) -> Dict[int, Dict[float, float]]:
    grid_horizons = sorted(ctx.calibration_corr.keys()) if ctx.calibration_corr else HORIZONS
    st = fade_state(ctx.cur_att_db)
    z_by_h_and_alpha = {}
    for H in grid_horizons:
        k_h = HORIZONS.index(H) if H in HORIZONS else min(range(len(HORIZONS)), key=lambda i: abs(HORIZONS[i] - H))
        z_alpha = {}
        for a in ALPHAS:
            tau = 1.0 - a
            q_raw = float(ctx.forecaster.predict(np.array([ctx.cur_att_db]), k_h, tau)[0])
            corr = ctx.calibration_corr.get(H, {}).get(tau, {}).get(st, 0.0)
            z_alpha[a] = q_raw + corr
        z_by_h_and_alpha[H] = z_alpha
    return z_by_h_and_alpha


def policy_v5_stage_a_levers(ctx: PolicyContext) -> Decision:
    """V5: V4 + All Stage A Levers (Power, Proactive Parity, Deferral), best-effort fallback.

    V5 does NOT use validated selection (susceptible to optimizer's curse) and
    always attempts delivery as best-effort.
    """
    z_by_alpha = _build_multi_horizon_bounds(ctx)
    dec = choose_decision(
        z_by_alpha, ctx.msg, ctx.table, ctx.sc.clear_sky_esn0_db,
        ctx.sc.symbol_rate_baud, ctx.sc.rtt_s, ctx.sc.one_way_delay_s,
        use_validated_selection=False
    )
    # V5 key behaviour: always mark as feasible (best-effort, no silent-miss tracking).
    # This distinguishes V5 from V6 in the false_infeasible_rate column
    # and in silent_miss_rate: V5 has 0 flagged messages, V6 flags INFEASIBLE ones.
    if not dec.feasible:
        return Decision(
            action=dec.action,
            feasible=True,   # V5 always claims feasible (best-effort)
            alpha=dec.alpha,
            p_lower_bound=max(dec.p_lower_bound, 0.0),
            overhead=dec.overhead,
            reason="best_effort",
        )
    return dec


def policy_v6_full_rcdam(ctx: PolicyContext) -> Decision:
    """V6: Full RC-DAM with Joint Optimization, Infeasibility Signalling, and P* lower bound.

    Uses R1 (send-and-flag) when infeasible by default. For explicit rule-based
    infeasibility handling (R1-R4), use policy_v6_infeasible_rule().

    Unlike V5, V6 always correctly reports dec.feasible=False when the optimizer
    cannot meet the eps guarantee, and never falsely reports feasible=True in that case.
    V5 always attempts delivery; V6 explicitly checks and signals infeasibility.
    """
    z_by_alpha = _build_multi_horizon_bounds(ctx)
    dec = choose_decision(
        z_by_alpha, ctx.msg, ctx.table, ctx.sc.clear_sky_esn0_db,
        ctx.sc.symbol_rate_baud, ctx.sc.rtt_s, ctx.sc.one_way_delay_s,
        use_validated_selection=True
    )

    # V6 key distinction from V5: always apply the infeasibility rule (R1 default = send-and-flag)
    # V5 does not check or report infeasibility separately; V6 always exposes the flag.
    # When feasible, both behave identically.  When infeasible, V6 signals the sender.
    return dec


def policy_v6_infeasible_rule(ctx: PolicyContext, rule: str = "R1") -> Decision:
    """V6 with explicit INFEASIBLE handling rules R1-R4.

    R1: send-and-flag (transmit best action, warn sender) [default, same as V5/V6 today]
    R2: defer (wait up to deadline slack; if feasible after deferral, send then; else R1)
    R3: downgrade (relax eps by 2x and retry optimizer; if still infeasible, R1)
    R4: reject (do not transmit; return Action=None with flagged=False and delivered=False)

    V6 always runs the full optimizer and checks feasibility. Unlike V5:
    - V6 ALWAYS flags the decision.feasible field correctly (V5 sometimes returns feasible=True spuriously)
    - V6 applies rule-specific logic on INFEASIBLE
    """
    z_by_alpha = _build_multi_horizon_bounds(ctx)
    dec = choose_decision(
        z_by_alpha, ctx.msg, ctx.table, ctx.sc.clear_sky_esn0_db,
        ctx.sc.symbol_rate_baud, ctx.sc.rtt_s, ctx.sc.one_way_delay_s,
        use_validated_selection=True
    )

    if dec.feasible:
        return dec  # feasible: all rules agree

    # INFEASIBLE: apply rule
    if rule == "R1":
        # Send anyway and flag (dec.feasible is already False, action is best effort)
        return dec

    elif rule == "R2":
        # Defer: try adding deferral up to deadline_slack
        deadline_slack = ctx.msg.deadline_s * 0.3  # try up to 30% of deadline as deferral
        for defer_try in [0.5, 1.0, 2.0, 5.0]:
            if defer_try > deadline_slack:
                break
            msg_deferred = Message(
                size_bits=ctx.msg.size_bits,
                priority=ctx.msg.priority,
                deadline_s=ctx.msg.deadline_s - defer_try,
                eps=ctx.msg.eps,
                arrival_idx=ctx.msg.arrival_idx,
            )
            if msg_deferred.deadline_s <= 0:
                break
            dec2 = choose_decision(
                z_by_alpha, msg_deferred, ctx.table, ctx.sc.clear_sky_esn0_db,
                ctx.sc.symbol_rate_baud, ctx.sc.rtt_s, ctx.sc.one_way_delay_s
            )
            if dec2.feasible:
                # Use deferred action with original deadline
                deferred_action = Action(
                    modcod=dec2.action.modcod,
                    power_db=dec2.action.power_db,
                    parity_frac=dec2.action.parity_frac,
                    defer_s=defer_try,
                )
                return Decision(
                    action=deferred_action,
                    feasible=True,
                    alpha=dec2.alpha,
                    p_lower_bound=dec2.p_lower_bound,
                    overhead=dec2.overhead,
                    reason="ok_deferred",
                )
        # Fallback to R1 if deferral didn't help
        return dec

    elif rule == "R3":
        # Downgrade: relax eps 2x and retry
        relaxed_msg = Message(
            size_bits=ctx.msg.size_bits,
            priority=ctx.msg.priority,
            deadline_s=ctx.msg.deadline_s,
            eps=min(ctx.msg.eps * 2.0, 0.5),
            arrival_idx=ctx.msg.arrival_idx,
        )
        dec3 = choose_decision(
            z_by_alpha, relaxed_msg, ctx.table, ctx.sc.clear_sky_esn0_db,
            ctx.sc.symbol_rate_baud, ctx.sc.rtt_s, ctx.sc.one_way_delay_s
        )
        if dec3.feasible:
            # Mark as infeasible wrt original eps but do transmit
            return Decision(
                action=dec3.action,
                feasible=False,  # infeasible wrt original guarantee
                alpha=dec3.alpha,
                p_lower_bound=dec3.p_lower_bound,
                overhead=dec3.overhead,
                reason="downgraded",
            )
        return dec  # still infeasible even relaxed

    elif rule == "R4":
        # Reject: do not transmit
        return Decision(
            action=None,
            feasible=False,
            alpha=dec.alpha,
            p_lower_bound=0.0,
            overhead=0.0,
            reason="rejected",
        )

    return dec  # unknown rule -> R1


def policy_oracle(ctx: PolicyContext) -> Decision:
    """Zero-delay Oracle: Knows true future attenuation trajectory and finds optimal action.

    Deterministic oracle feasibility: Evaluates whether exhaustive search with the true
    window maximum attenuation can reach target on-time probability P >= 1 - eps in expectation
    (alpha = 0.0, zero conformal coverage uncertainty).
    """
    if ctx.true_future_att is None or len(ctx.true_future_att) == 0:
        return policy_v6_full_rcdam(ctx)

    max_future_att = float(np.max(ctx.true_future_att))
    # True peak is known exactly (alpha = 0.0); tests if delivery target is physically achievable
    z_true = {0.0: max_future_att}
    return choose_decision(
        z_true, ctx.msg, ctx.table, ctx.sc.clear_sky_esn0_db,
        ctx.sc.symbol_rate_baud, ctx.sc.rtt_s, ctx.sc.one_way_delay_s,
        use_validated_selection=False
    )
