"""RC-DAM engine v7: window-conformal risk + trajectory evaluation + exhaustive joint search.

    minimise   Cost(a) = airtime_ratio(a) + w_E * energy_extra(a) + w_D * prio_mult * defer/deadline
    subject to P(deliver by deadline | a) >= 1 - eps      (risk split: eps = alpha_w + eps_d)
               T_delivery <= D

Risk accounting (union bound):
    alpha_w = alpha_split * eps   -> conformal bound on the WINDOW-MAX attenuation (window starts
                                     after feedback delay + deferral, length = deadline-aware)
    eps_d   = (1-alpha_split)*eps -> miss budget of the action on the worst-case burst profile
All levers (MODCOD, power, parity, interleave, duplication, deferral) are searched JOINTLY; the
cost-ordered search returns the minimum-cost feasible combination in the grid.
Reported per decision: predicted_p (MC point estimate), lower_bound (Clopper-Pearson MC lower
limit minus alpha_w), plus infeasibility diagnosis and explicit rule outcome.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import itertools
import numpy as np
from scipy.stats import beta as _beta

from .optimizer import Action, Decision, FRAME_BITS
from .trajectory_evaluator import p_deliver_profile_detail
from .window_conformal import WindowConformal, OFFSETS_S
from .infeasibility import diagnose, action_to_dict


@dataclass
class EngineParams:
    burst_s: float = 1.0
    alpha_split: float = 0.5
    powers: Tuple[float, ...] = (0.0, 1.0, 2.0)
    parities: Tuple[float, ...] = (0.0, 0.05, 0.15, 0.3)
    defers: Tuple[float, ...] = (0.0, 0.5)               # must be a subset of OFFSETS_S
    interleaves: Tuple[float, ...] = (0.0, 1.0)
    dups: Tuple[int, ...] = (1, 2)
    w_energy: float = 0.25
    w_delay: float = 0.05
    prio_delay_mult: Dict[str, float] = field(default_factory=lambda: {
        "critical": 2.0, "normal": 1.0, "best_effort": 0.5})
    n_screen: int = 60
    n_confirm_min: int = 300
    n_confirm_max: int = 3000
    n_pos: int = 4
    relax_factor: float = 2.0
    max_eps_relaxed: float = 0.25
    hardware_power_cap_db: float = 6.0
    cp_conf: float = 0.95
    exhaustive: bool = False                  # verification mode: no screening / pruning (slow)
    classify_infeasible: bool = True          # separate uncertainty-limited from physically infeasible
    by_priority: Dict[str, Dict[str, float]] = field(default_factory=dict)   # per-class burst_s / alpha_split


def effective_params(p: EngineParams, priority: str) -> EngineParams:
    """Apply per-priority (LTT-certified) burst_s / alpha_split overrides, if configured."""
    from dataclasses import replace as _rep
    ov = p.by_priority.get(str(priority)) if p.by_priority else None
    if not ov:
        return p
    return _rep(p, burst_s=float(ov.get("burst_s", p.burst_s)), alpha_split=float(ov.get("alpha_split", p.alpha_split)))


def load_priority_params(path: str, base: Optional[EngineParams] = None) -> EngineParams:
    """Load outputs/v7_chosen_lambda.json (written by run_v7_full.py --stage sweep)."""
    import json
    from dataclasses import replace as _rep
    with open(path) as f:
        raw = json.load(f)
    return _rep(base or EngineParams(), by_priority={k: dict(burst_s=v["burst_s"], alpha_split=v["alpha_split"])
                                                    for k, v in raw.items()})


def cost_breakdown(a: Action, msg, sc, table, p: EngineParams) -> Dict[str, float]:
    K = int(np.ceil(msg.size_bits / FRAME_BITS))
    fs = FRAME_BITS / (sc.symbol_rate_baud * table.eff[a.modcod])
    frames = int(np.ceil(K * (1.0 + a.parity_frac))) * max(1, int(a.dup))
    ref = msg.size_bits / (sc.symbol_rate_baud * table.eff.max())
    airtime_ratio = frames * fs / ref
    energy_extra = (10 ** (a.power_db / 10.0) - 1.0) * airtime_ratio
    mult = p.prio_delay_mult.get(str(msg.priority), 1.0)
    delay_pen = mult * a.defer_s / max(msg.deadline_s, 1e-9)
    total = airtime_ratio + p.w_energy * energy_extra + p.w_delay * delay_pen
    return dict(airtime_ratio=airtime_ratio, energy_extra=energy_extra, delay_penalty=delay_pen, total=total)


def _cp_lower(p_hat: float, n: int, conf: float) -> float:
    k = int(round(p_hat * n))
    return 0.0 if k <= 0 else float(_beta.ppf(1.0 - conf, k, n - k + 1))


def search(ctx, wc: WindowConformal, eps: float, p: EngineParams,
           defers: Optional[Tuple[float, ...]] = None,
           alpha_w_override: Optional[float] = None) -> Dict[str, Any]:
    msg, sc, table = ctx.msg, ctx.sc, ctx.table
    slope = float(ctx.features["slope5"][0]) if ctx.features and "slope5" in ctx.features else None
    alpha_w = alpha_w_override if alpha_w_override is not None else p.alpha_split * eps
    eps_d = max(1e-6, (1.0 - p.alpha_split) * eps)
    n_conf = int(np.clip(np.ceil(3.0 / eps_d), p.n_confirm_min, p.n_confirm_max))
    K = int(np.ceil(msg.size_bits / FRAME_BITS))
    defers = tuple(d for d in (defers if defers is not None else p.defers) if
                   (wc.supports_offsets or d == 0.0) and any(abs(d - o) < 1e-9 for o in OFFSETS_S))
    if not defers:
        defers = (0.0,)

    cands = []
    for m in range(len(table.eff)):
        fs = FRAME_BITS / (sc.symbol_rate_baud * table.eff[m])
        for pw, pr, df, il, dp in itertools.product(p.powers, p.parities, defers, p.interleaves, p.dups):
            frames = int(np.ceil(K * (1 + pr))) * dp
            if frames * fs + df + sc.one_way_delay_s > msg.deadline_s:
                continue
            a = Action(m, pw, pr, df, interleave_s=il, dup=dp)
            cands.append((cost_breakdown(a, msg, sc, table, p)["total"], a))
    cands.sort(key=lambda x: x[0])

    peak_cache: Dict[float, float] = {}
    def peak_for(df):
        if df not in peak_cache:
            L = max(0.1, msg.deadline_s - sc.one_way_delay_s - df)
            peak_cache[df] = wc.bound(ctx.cur_att_db, L, alpha_w, df, slope)
        return peak_cache[df]

    best = None            # (p_hat, -cost, action, n_mc, peak)
    failed: List[Tuple] = []   # groups that failed badly -> dominance pruning
    dead = set()               # (modcod, defer) groups that fail even with maximum resources
    n_eval = 0
    probed = False
    base_seed = (int(msg.arrival_idx) * 1315423911) % 2**31

    def _kw(a, pk):
        return dict(msg_bits=msg.size_bits, deadline_s=msg.deadline_s, table=table,
                    clear_esn0_db=sc.clear_sky_esn0_db, symbol_rate=sc.symbol_rate_baud, rtt_s=sc.rtt_s,
                    one_way_s=sc.one_way_delay_s, base_db=ctx.cur_att_db, peak_db=pk, burst_s=p.burst_s,
                    n_pos=p.n_pos, burst_lo_s=a.defer_s)

    def _probe_groups():
        """Cheap upper-bound probe per (MODCOD, deferral): maximum power/parity/interleave."""
        nonlocal best, n_eval
        for m in range(len(table.eff)):
            fs = FRAME_BITS / (sc.symbol_rate_baud * table.eff[m])
            for df in defers:
                pr = max([x for x in p.parities
                          if int(np.ceil(K * (1 + x))) * fs + df + sc.one_way_delay_s <= msg.deadline_s],
                         default=None)
                if pr is None:
                    continue
                a = Action(m, max(p.powers), pr, df, interleave_s=max(p.interleaves), dup=1)
                pk = peak_for(df)
                n_eval += 1
                pp, _ = p_deliver_profile_detail(a, n_mc=30, seed=(base_seed + 99991 + m) % 2**31, **_kw(a, pk))
                c = cost_breakdown(a, msg, sc, table, p)["total"]
                if best is None or (pp, -c) > (best[0], best[1]):
                    best = (pp, -c, a, 30, pk)
                if pp < 0.05:
                    dead.add((m, df))

    for idx, (cost, a) in enumerate(cands):
        if not p.exhaustive:
            if not probed and n_eval >= 60:
                probed = True
                _probe_groups()
            if (a.modcod, a.defer_s) in dead:
                continue
            if any(f[0] == a.modcod and f[1] == a.defer_s and f[2] == a.interleave_s and f[3] == a.dup
                   and a.power_db <= f[4] and a.parity_frac <= f[5] for f in failed):
                continue
        pk = peak_for(a.defer_s)
        kw = _kw(a, pk)
        n_eval += 1
        if not p.exhaustive:
            ps, _ = p_deliver_profile_detail(a, n_mc=p.n_screen, seed=(base_seed + idx) % 2**31, **kw)
            if best is None or (ps, -cost) > (best[0], best[1]):
                best = (ps, -cost, a, p.n_screen, pk)
            if ps < 0.5:
                failed.append((a.modcod, a.defer_s, a.interleave_s, a.dup, a.power_db, a.parity_frac))
                continue
            if ps < 1.0 - eps_d - 0.25:
                continue
        ph, n = p_deliver_profile_detail(a, n_mc=n_conf, seed=(base_seed + idx + 7) % 2**31, **kw)
        if best is None or best[2] is a or (ph, -cost) > (best[0], best[1]):
            best = (ph, -cost, a, n, pk)
        if ph >= 1.0 - eps_d:
            return _pack(True, a, ph, n, pk, alpha_w, eps_d, eps, cost, len(cands), n_eval, p, msg, sc, table)
    if not probed:
        _probe_groups()
    # Best-effort fallback: the most robust MODCOD that fits, with maximum resources.
    fit = [m for m in range(len(table.eff))
           if K * FRAME_BITS / (sc.symbol_rate_baud * table.eff[m]) + sc.one_way_delay_s <= msg.deadline_s]
    if fit:
        m_r = min(fit, key=lambda m: table.thr[m])
        fs_r = FRAME_BITS / (sc.symbol_rate_baud * table.eff[m_r])
        pr_r = max([x for x in p.parities if int(np.ceil(K * (1 + x))) * fs_r + sc.one_way_delay_s <= msg.deadline_s],
                   default=0.0)
        a_r = Action(m_r, max(p.powers), pr_r, 0.0, interleave_s=max(p.interleaves), dup=1)
        pk_r = peak_for(0.0)
        p_r, n_r = p_deliver_profile_detail(a_r, n_mc=n_conf, seed=(base_seed + 424242) % 2**31, **_kw(a_r, pk_r))
        c_r = cost_breakdown(a_r, msg, sc, table, p)["total"]
        if best is None or p_r >= best[0]:
            best = (p_r, -c_r, a_r, n_r, pk_r)
    if best is None:
        return dict(feasible=False, action=None, p_hat=0.0, lower_bound=0.0, n_cand=len(cands),
                    n_eval=n_eval, alpha_w=alpha_w, eps_d=eps_d, eps=eps, peak_db=peak_for(0.0),
                    cost=float("inf"))
    ph, negc, a, n, pk = best
    return _pack(False, a, ph, n, pk, alpha_w, eps_d, eps, -negc, len(cands), n_eval, p, msg, sc, table)


def _pack(feasible, a, p_hat, n, peak, alpha_w, eps_d, eps, cost, n_cand, n_eval, p, msg, sc, table):
    p_lb_mc = _cp_lower(p_hat, n, p.cp_conf)
    lb = max(0.0, p_lb_mc - alpha_w)
    return dict(feasible=feasible, action=a, p_hat=float(p_hat), p_lb_mc=p_lb_mc, lower_bound=lb,
                n_mc=n, peak_db=float(peak), alpha_w=alpha_w, eps_d=eps_d, eps=eps, cost=float(cost),
                cost_breakdown=cost_breakdown(a, msg, sc, table, p), n_cand=n_cand, n_eval=n_eval,
                lb_meets_requirement=bool(lb >= 1.0 - eps))


def decide(ctx, wc: WindowConformal, params: Optional[EngineParams] = None, rule: str = "R1",
           explain: bool = False) -> Decision:
    p = effective_params(params or EngineParams(), ctx.msg.priority)
    msg, sc, table = ctx.msg, ctx.sc, ctx.table
    r = search(ctx, wc, msg.eps, p)
    log: Dict[str, Any] = dict(engine="window_conformal_v7", rule=rule, eps=msg.eps,
                               alpha_window=r["alpha_w"], eps_delivery=r["eps_d"],
                               window_bound_db=r["peak_db"], predicted_p=r["p_hat"],
                               p_lb_mc=r.get("p_lb_mc"), lower_bound=r["lower_bound"],
                               n_mc=r.get("n_mc"), cost=r["cost"], cost_breakdown=r.get("cost_breakdown"),
                               n_candidates=r["n_cand"], n_evaluated=r["n_eval"],
                               meets_original_requirement=r["feasible"], relaxed_eps=None,
                               lb_meets_requirement=r.get("lb_meets_requirement"))
    if r["feasible"]:
        log.update(outcome="sent_normal", action=action_to_dict(r["action"], table), infeasibility=None)
        return Decision(r["action"], True, r["alpha_w"], r["lower_bound"], r["cost"], "ok", log)

    best = dict(p_hat=r["p_hat"], lower_bound=r["lower_bound"], action=action_to_dict(r["action"], table))
    diag = diagnose(msg, sc, table, r["peak_db"], max(p.powers), msg.eps, r["eps_d"], best,
                    p.hardware_power_cap_db)
    log["infeasibility"] = diag
    if p.classify_infeasible:
        # UNCERTAINTY_LIMITED: feasible if the window maximum is the MEDIAN one (alpha=0.5), infeasible
        # only because the safety margin demanded by eps; PHYSICALLY_INFEASIBLE: infeasible even then.
        r_nom = search(ctx, wc, msg.eps, p, alpha_w_override=0.5)
        cls = "UNCERTAINTY_LIMITED" if r_nom["feasible"] else "PHYSICALLY_INFEASIBLE"
        log.update(infeasibility_class=cls, nominal_feasible=bool(r_nom["feasible"]),
                   nominal_predicted_p=r_nom["p_hat"])
        diag["infeasibility_class"] = cls
    if explain:
        from .alternatives import best_alternatives
        log["alternatives"] = best_alternatives(ctx, wc, p)
    reason = "deadline" if diag["primary_reason"] in ("DEADLINE_TOO_SHORT", "MESSAGE_TOO_LARGE") else "capacity"

    def _flag(res, outcome):
        log.update(outcome=outcome, action=action_to_dict(res["action"], table))
        return Decision(res["action"], False, res["alpha_w"], res["lower_bound"],
                        res["cost"] if np.isfinite(res["cost"]) else 999.0, reason, log)

    if rule == "R2":
        r2 = search(ctx, wc, msg.eps, p, defers=OFFSETS_S)
        if r2["feasible"]:
            log.update(outcome="deferred", meets_original_requirement=True, predicted_p=r2["p_hat"],
                       lower_bound=r2["lower_bound"], window_bound_db=r2["peak_db"],
                       action=action_to_dict(r2["action"], table))
            return Decision(r2["action"], True, r2["alpha_w"], r2["lower_bound"], r2["cost"], "ok_deferred", log)
        log["r2_note"] = "no deferral up to %.1f s satisfies the original requirement" % max(OFFSETS_S)
        return _flag(r, "sent_and_flagged")
    if rule == "R3":
        eps2 = min(p.relax_factor * msg.eps, p.max_eps_relaxed)
        r3 = search(ctx, wc, eps2, p)
        log.update(relaxed_eps=eps2, relaxed_feasible=bool(r3["feasible"]),
                   relaxed_predicted_p=r3["p_hat"], relaxed_lower_bound=r3["lower_bound"])
        if r3["feasible"]:
            log.update(outcome="sent_relaxed_original_NOT_met", meets_original_requirement=False,
                       action=action_to_dict(r3["action"], table),
                       warning="Requirement RELAXED from eps=%g to eps=%g; the original requirement is NOT satisfied."
                               % (msg.eps, eps2))
            return Decision(r3["action"], False, r3["alpha_w"], r3["lower_bound"], r3["cost"],
                            "relaxed_feasible", log)
        return _flag(r, "sent_and_flagged")
    if rule == "R4":
        log.update(outcome="rejected", action=None)
        return Decision(None, False, r["alpha_w"], r["lower_bound"], 0.0, "rejected", log)
    return _flag(r, "sent_and_flagged")


def policy_v7_engine(ctx, wc: WindowConformal, params: Optional[EngineParams] = None, rule: str = "R1") -> Decision:
    """Standard policy interface used by the benchmark harness (V7)."""
    return decide(ctx, wc, params, rule)
