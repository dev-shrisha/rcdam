# RC-DAM Build Guide

How to turn the RC-DAM project document (*RC_DAM_Complete_Project_Document_v3*) into working code. It extends the existing Step 1/Step 2 repository (`rcdam-step2-project.zip`) milestone by milestone.

**Tested vs untested.** Code blocks in Section 4 were run against the Step 2 repository (Python 3, NumPy, SciPy) and the outputs quoted here come from those runs. The PyTorch blocks were **not** run (PyTorch was not available where I tested). Blocks marked *outline* are structure to implement, not finished code. All numbers quoted come from the **placeholder synthetic channel**; they are checks that the pipeline works, not research results.


## 1. Ground rules

- Build in the order M0 to M6. Each milestone has acceptance criteria; do not start the next one until they pass.
- Every stochastic function takes a seed. All parameters live in `Scenario` (`rcdam/config.py`) or a new config dataclass. Every script writes a CSV and a PNG to `outputs/` and prints the config it used.
- Use **independent seeds (or real time periods) for train, calibration and test**. Never tune on test.
- One link model, one retransmission rule and one overhead definition for **every** method, baselines included.
- Units: attenuation and Es/N0 in dB, time in seconds, simulation grid `dt = 0.1 s`, indices are steps.
- Calibrate on **attenuation** (always observed), never on delivery outcomes (observed only for the chosen action, hence biased).

## 2. What already exists (Step 1 and Step 2)

| File | Provides |
|---|---|
| `rcdam/config.py` | `Scenario` dataclass: time base, link, rain models, BLER model, ARQ model, message model; properties `n_steps`, `feedback_delay_steps`, `one_way_delay_s` |
| `rcdam/attenuation.py` | `generate_attenuation_db(sc)`; `events` and `gauss_markov` rain models plus fast jitter and scintillation |
| `rcdam/link.py` | `MODCODS` table (verify against ETSI EN 302 307), `ModcodTable.select`, `ModcodTable.bler` |
| `rcdam/arq.py` | `build_cumulative`, `message_delivery` (expected-value fluid ARQ with RTT rounds) |
| `rcdam/policies.py` | `static_policy`, `margin_policy`, `predictive_policy` (linear), `oracle_policy` |
| `rcdam/experiment.py` | `step2_scenario()`, `evaluate`, `run_seed` (splits misses into oracle-infeasible and avoidable) |
| `rcdam/stats.py` | `aggregate` (mean and 95% t-interval across seeds), `save_csv` |
| `rcdam/metrics.py`, `rcdam/acm_baseline.py` | Step 1 metrics and baseline |
| `scripts/run_step1.py`, `scripts/run_step2.py`, `tests/test_step2.py` | Runners and correctness tests |

Run (PowerShell or terminal, from the project root):

```
pip install -r requirements.txt
python tests/test_step2.py
python scripts/run_step2.py --seeds 20 --workers 4
```

**Placeholders to replace before any claim:** the rain generators (calibrate to ITU-R P.618 / P.1853 or real data), the BLER waterfall (use real LDPC curves), the MODCOD table (verify), and the expected-value ARQ model (M5 in Section 5 adds a frame-level ground truth).

**Step 2 baseline results on the placeholder channel (20 seeds x 48 h, 2 s deadline):** about 0.34% of messages are unrecoverable by MODCOD adaptation (even the zero-delay oracle misses them); the best fixed-margin ACM misses about 0.47%; linear predictive ACM performs the same as fixed-margin ACM.

## 3. Data contracts

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Message:
    size_bits: float
    priority: str          # "critical" | "normal" | "best_effort"
    deadline_s: float
    eps: float             # permitted deadline-failure probability for this priority
    arrival_idx: int

@dataclass(frozen=True)
class Action:
    modcod: int
    power_db: float
    parity_frac: float     # parity frames as a fraction of K
    defer_s: float
    # Stage B adds: interleave_s, dup

@dataclass
class Decision:
    action: "Action | None"
    feasible: bool
    alpha: float           # calibration level used
    p_lower_bound: float   # p_link - alpha when feasible; P* (clamped at 0) when infeasible
    overhead: float
    reason: str            # "ok" | "capacity" | "deadline" | "eps_too_strict" | "no_calibration_data"
```

**Time conventions (get these exactly right).**
- A message arrives at index `i0`. The last measurement the sender has is index `m = i0 - d`, where `d = sc.feedback_delay_steps` (the feedback is one RTT old).
- The forecast target is `Z_H(m) = max A[m+1 .. m+H]`, the maximum attenuation over the window **after** the last measurement. Nested windows: `Z_H` is non-decreasing in `H`.
- For an action with deferral `defer_steps` and time span `span_steps` (transmission plus interleaving plus any ARQ rounds you must cover), use the smallest grid horizon `H >= d + defer_steps + span_steps`. If none exists, the decision is infeasible with reason `deadline` or `no_calibration_data`.
- Fade state is computed from the current measured attenuation: edges at 1, 5 and 10 dB give clear, light, moderate, deep.

---

## 4. Reference code (tested)

Put these in `rcdam/` as `windows.py`, `forecaster_baseline.py`, `calibration.py`, `evaluator.py`, `optimizer.py`. In `optimizer.py` the import `from evaluator import ...` becomes `from .evaluator import ...` (already adjusted below).

### 4.1 Window statistic and feature windows (`windows.py`)

```python
import numpy as np
from scipy.ndimage import maximum_filter1d


def forward_window_max(a, H):
    """Z[m] = max(a[m+1 : m+H+1]) for every index m (window of H steps AFTER m).
    The last H entries use the shorter available tail."""
    z = maximum_filter1d(a, size=H, origin=-(H // 2), mode="nearest")
    z = np.roll(z, -1)
    z[-1] = a[-1]
    return z


def build_windows(att, dt, horizons_steps, stride_steps, hist_s=60.0):
    """One row per decision point m = index of the LAST MEASUREMENT.
    Features use only att[:m+1] (no leakage); targets Z[:, k] = max attenuation over (m, m + H_k]."""
    n = len(att)
    L = int(hist_s / dt)
    idx = np.arange(L, n - max(horizons_steps) - 2, stride_steps)
    s5, s30 = int(5 / dt), int(30 / dt)
    cur = att[idx]
    feats = {
        "idx": idx,
        "cur": cur,
        "slope5": (cur - att[idx - s5]) / 5.0,
        "slope30": (cur - att[idx - s30]) / 30.0,
        "max60": np.array([att[i - L:i + 1].max() for i in idx]),
    }
    feats["Z"] = np.stack([forward_window_max(att, H)[idx] for H in horizons_steps], axis=1)
    return feats
```

Checked against brute force for several window sizes.

### 4.2 Non-neural baseline forecaster (`forecaster_baseline.py`)

```python
import numpy as np


class EmpiricalQuantileForecaster:
    """Non-neural baseline: quantiles of (Z_H - current attenuation) within bins of the current attenuation.
    Fit on TRAIN seeds only. Build this first; only add a neural model if it beats it."""

    def __init__(self, edges=(0.5, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30), taus=(0.5, 0.9, 0.95, 0.99), min_n=20):
        self.edges = np.asarray(edges, dtype=float)
        self.taus = tuple(taus)
        self.min_n = min_n
        self.q = {}

    def fit(self, cur, Z):
        b = np.digitize(cur, self.edges)
        for bi in range(len(self.edges) + 1):
            sel = b == bi
            for k in range(Z.shape[1]):
                r = Z[sel, k] - cur[sel]
                for t in self.taus:
                    self.q[(bi, k, t)] = float(np.quantile(r, t)) if sel.sum() >= self.min_n else None
        return self

    def predict(self, cur, k, tau):
        """Upper-quantile forecast of Z_H(k) for each row; sparse bins fall back to the nearest lower bin."""
        b = np.digitize(cur, self.edges)
        out = np.empty(len(cur))
        for i, bi in enumerate(b):
            j = bi
            while self.q[(j, k, tau)] is None and j > 0:
                j -= 1
            out[i] = cur[i] + (self.q[(j, k, tau)] if self.q[(j, k, tau)] is not None else 0.0)
        return out
```

### 4.3 Conformal calibration, stratified, plus online update (`calibration.py`)

```python
import numpy as np

STATE_EDGES_DB = (1.0, 5.0, 10.0)          # clear < 1 <= light < 5 <= moderate < 10 <= deep


def fade_state(cur_db):
    return np.digitize(cur_db, STATE_EDGES_DB)


def cqr_correction(scores, tau):
    """Finite-sample split-conformal quantile of the scores (Z - q_hat). Returns +inf if too few points,
    which must be treated as 'no valid bound' (never silently as zero)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * tau))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def fit_stratified_corrections(cur_cal, z_cal, qhat_cal, tau, min_n=300):
    """One correction per fade state. A state with fewer than min_n calibration windows borrows the next
    lower state's pool (log this in real code). Thin the calibration windows beforehand (stride >= horizon)."""
    st = fade_state(cur_cal)
    scores = z_cal - qhat_cal
    corr = {}
    for s in range(len(STATE_EDGES_DB) + 1):
        pool = scores[st == s]
        j = s
        while len(pool) < min_n and j > 0:
            j -= 1
            pool = scores[st <= s][st[st <= s] >= j]
        corr[s] = cqr_correction(pool, tau)
    return corr


def apply_corrections(cur, qhat, corr):
    st = fade_state(cur)
    return qhat + np.array([corr[s] for s in st])


def aci_step(alpha_t, err, alpha_target, gamma=0.005):
    """Adaptive conformal inference update. err = 1 if the window maximum exceeded the bound issued at level
    alpha_t, else 0. Call it when the outcome becomes available, i.e. about H + RTT after the bound was issued."""
    return alpha_t + gamma * (alpha_target - err)
```

### 4.4 Delivery-probability evaluator (`evaluator.py`)

```python
import numpy as np


def p_deliver_by_deadline(K, parity, p, frame_s, rtt_s, one_way_s, defer_s, deadline_s,
                          n_mc=2000, max_rounds=8, seed=0):
    """Monte Carlo estimate under a CONSTANT frame-loss probability p (i.e. attenuation held at the calibrated bound).
    MDS erasure code: the receiver needs K of the K + parity frames. Each ARQ round resends exactly the shortfall;
    the next round starts one RTT after the previous round finished sending.
    Returns (P(delivered by deadline), mean frames sent). A fixed seed gives common random numbers across actions."""
    rng = np.random.default_rng(seed)
    need = np.full(n_mc, K, dtype=np.int64)
    t = np.full(n_mc, float(defer_s))
    done = np.full(n_mc, np.inf)
    send = np.full(n_mc, K + parity, dtype=np.int64)
    sent_total = np.zeros(n_mc)
    for _ in range(max_rounds):
        active = need > 0
        if not active.any():
            break
        got = rng.binomial(send, 1.0 - p)
        end = t + send * frame_s
        sent_total += np.where(active, send, 0)
        need = np.where(active, np.maximum(need - got, 0), need)
        fin = active & (need == 0)
        done[fin] = end[fin] + one_way_s
        t = np.where(active, end + rtt_s, t)
        send = np.where(need > 0, need, 0)
    return float(np.mean(done <= deadline_s)), float(sent_total.mean())
```

Sanity results (K = 309 frames, 1.5 ms per frame, RTT 0.54 s, deadline 2 s): at frame-loss probability 0.05, success is 0.963 with no parity and 0.995 with 5% parity (parity avoids one RTT retransmission round); deferring by 1 s at 2% loss drops success to about 0.002, because deferral consumes the deadline slack.

### 4.5 Optimizer with infeasibility report (`optimizer.py`)

```python
from dataclasses import dataclass

import numpy as np

from .evaluator import p_deliver_by_deadline

FRAME_BITS = 64_800.0          # information bits per FEC frame (simplification shared with the Step 2 ARQ model)
ALPHAS = (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)   # wide: P* needs low-confidence levels too


@dataclass(frozen=True)
class Action:
    modcod: int
    power_db: float
    parity_frac: float
    defer_s: float
    # Stage B adds: interleave_s, dup


def evaluate_action(a, z_hat_db, msg_bits, deadline_s, table, clear_esn0_db, symbol_rate, rtt_s, one_way_s,
                    n_mc=2000, power_cost_per_db=0.03):
    K = int(np.ceil(msg_bits / FRAME_BITS))
    parity = int(np.ceil(a.parity_frac * K))
    esn0 = clear_esn0_db + a.power_db - z_hat_db
    p = float(table.bler(np.array([a.modcod]), np.array([esn0]))[0])
    frame_s = FRAME_BITS / (symbol_rate * table.eff[a.modcod])
    p_ok, sent = p_deliver_by_deadline(K, parity, p, frame_s, rtt_s, one_way_s, a.defer_s, deadline_s, n_mc=n_mc)
    ref_airtime = msg_bits / (symbol_rate * table.eff.max())
    overhead = sent * frame_s / ref_airtime + power_cost_per_db * a.power_db
    return p_ok, overhead


def choose_action(z_hat_by_alpha, msg_bits, deadline_s, eps_p, table, clear_esn0_db, symbol_rate, rtt_s, one_way_s,
                  mod_candidates, powers=(0.0, 1.0, 2.0), parities=(0.0, 0.05, 0.15, 0.3), defers=(0.0, 0.5, 1.0)):
    """z_hat_by_alpha: {alpha: calibrated window bound (dB)}.
    Feasible if some (action, alpha) has p_link - alpha >= 1 - eps_p (union bound); pick the minimum overhead.
    Returns (action, alpha, overhead, p_lower_bound, feasible). When infeasible, p_lower_bound is P* clamped at 0."""
    best_feasible, best_any = None, (None, None, None, -1.0)
    for m in mod_candidates:
        for pw in powers:
            for pr in parities:
                for df in defers:
                    a = Action(m, pw, pr, df)
                    for alpha, z in z_hat_by_alpha.items():
                        p_ok, ov = evaluate_action(a, z, msg_bits, deadline_s, table, clear_esn0_db,
                                                   symbol_rate, rtt_s, one_way_s)
                        p_lb = p_ok - alpha
                        if p_lb > best_any[3]:
                            best_any = (a, alpha, ov, p_lb)
                        if p_lb >= 1.0 - eps_p and (best_feasible is None or ov < best_feasible[2]):
                            best_feasible = (a, alpha, ov, p_lb)
    if best_feasible is not None:
        return (*best_feasible, True)
    a, alpha, ov, p_lb = best_any
    return a, alpha, ov, max(p_lb, 0.0), False
```

### 4.6 End-to-end demo (`demo.py`)

```python
"""End-to-end demo of the reference snippets on the Step 2 repo (run with PYTHONPATH pointing at rcdam-project)."""
import sys, time
import numpy as np
from dataclasses import replace
from rcdam.experiment import step2_scenario
from rcdam.attenuation import generate_attenuation_db
from rcdam.link import ModcodTable
from windows import build_windows
from forecaster_baseline import EmpiricalQuantileForecaster
from calibration import fade_state, fit_stratified_corrections, apply_corrections
from optimizer import choose_action, ALPHAS

sc0 = step2_scenario()
HS = [10, 30, 100, 300, 600]                      # horizons in steps (1 s ... 60 s)
def make(seeds, scenario=sc0):
    ds = [build_windows(generate_attenuation_db(replace(scenario, seed=s)), sc0.dt_s, HS, 300) for s in seeds]
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}
train, cal, test = make(range(1, 9)), make(range(9, 13)), make(range(13, 21))
fc = EmpiricalQuantileForecaster(taus=(0.9, 0.99)).fit(train["cur"], train["Z"])
k, tau = 3, 0.99                                  # 30 s horizon, 99% bound
q_cal, q_te = fc.predict(cal["cur"], k, tau), fc.predict(test["cur"], k, tau)
corr = fit_stratified_corrections(cal["cur"], cal["Z"][:, k], q_cal, tau)
bound = apply_corrections(test["cur"], q_te, corr)
st = fade_state(test["cur"])
print("stratified-CQR coverage at 0.99, 30 s horizon:",
      {s: round(float((test["Z"][st == s, k] <= bound[st == s]).mean()), 3) for s in range(4)}, "corrections", {s: round(c, 2) for s, c in corr.items()})

t = ModcodTable()
rs, clear = sc0.symbol_rate_baud, sc0.clear_sky_esn0_db
for name, zs in (("moderate", {a: 4.0 + 6.0 * (1 - a) for a in ALPHAS}), ("deep", {a: 14.0 + 14.0 * (1 - a) for a in ALPHAS})):
    cands = [i for i in range(len(t.thr)) if t.thr[i] <= clear - min(zs.values()) + 3.0]
    t0 = time.time()
    a, alpha, ov, plb, feas = choose_action(zs, 20e6, 2.0, 1e-2, t, clear, rs, sc0.rtt_s, sc0.one_way_delay_s, cands)
    print(f"{name}: feasible={feas} {a} alpha={alpha} overhead={ov:.2f} P_lower_bound={plb:.3f} [{time.time()-t0:.1f}s, {len(cands)} MODCOD candidates]")
```

What this demo printed here: stratified conformal coverage at 0.99 for the 30 s horizon was 0.989 (clear), 0.994 (light), 0.986 (moderate) and 0.978 (deep). Deep-state coverage is below target because only a few hundred deep windows were available; use more calibration seeds before judging it. The optimizer chose a feasible action for the moderate case and returned INFEASIBLE with P* = 0.1 (at alpha = 0.9) for the deep case. The bounds in the demo are toy values, not forecaster outputs.

### 4.7 Quantile training pieces (PyTorch, **not run**)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def pinball_loss(pred, target, taus):
    """pred [B, H, Q], target [B, H, 1], taus [Q] (float tensor). Mean pinball (quantile) loss."""
    diff = target - pred
    return torch.maximum(taus * diff, (taus - 1.0) * diff).mean()

class QuantileMLP(nn.Module):
    """Predicts non-crossing quantiles of (Z_H - current attenuation) for every horizon."""
    def __init__(self, d_in, n_h, n_q, hidden=64):
        super().__init__()
        self.n_h, self.n_q = n_h, n_q
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_h * n_q)
    def forward(self, x):
        z = self.head(self.body(x)).view(-1, self.n_h, self.n_q)
        base, inc = z[..., :1], F.softplus(z[..., 1:])
        return torch.cat([base, base + inc.cumsum(-1)], dim=-1)    # non-crossing in the quantile axis
```

---

## 5. Milestones

### M0. Gate: literature table and headroom analysis

**Goal:** know whether there is room for improvement before building the forecaster.

1. *Reading (no code).* Read the closest 6 to 8 papers in full; fix the coverage matrix in the document.
2. *Code.* Extend `run_seed` so a deadline can be passed, and write `scripts/run_headroom.py`:

```python
# rcdam/experiment.py (outline of the change)
def run_seed(args):
    seed, hours = args[0], args[1]
    deadline_s = args[2] if len(args) > 2 else None
    sc = replace(step2_scenario(), seed=seed, duration_h=hours)
    if deadline_s is not None:
        # make sure the ARQ round limit never binds for long deadlines
        sc = replace(sc, deadline_s=deadline_s, arq_max_rounds=int(deadline_s / sc.rtt_s) + 4)
    ...  # rest unchanged
```

```python
# scripts/run_headroom.py (outline)
# for deadline in (2, 10, 60, 300): run N seeds, aggregate, and record for each method
#   miss_rate, avoidable_miss_rate, infeasible_frac, overhead (with 95% CI)
# headroom(deadline) = min over baselines of avoidable_miss_rate   (the oracle has 0 by construction)
# save outputs/headroom.csv and outputs/headroom.png (x: deadline, y: miss rates, one line per method)
```

**Decision rule.** If the 95% interval of the best baseline's avoidable miss rate includes 0 at every deadline, MODCOD adaptation alone has no demonstrable headroom: reframe the contribution around the extra levers and infeasibility reporting, and reword H1.

**Acceptance:** `headroom.csv` and plot exist; the coverage matrix is verified; the decision above is written down and shown to your supervisor.

### M1. Data and channel realism

1. `rcdam/data.py`: `load_attenuation(path) -> (t_s, att_db)` (resample to `dt`, mark gaps, convert to differential attenuation by removing the clear-sky offset), and `split_by_time(att, fractions=(0.5, 0.25, 0.25))`. Split by time or by independent seeds, never by random windows.
2. Add a rain model option using ITU-R P.1853-style synthesis; keep `generate_attenuation_db` as the single entry point.
3. Verify the MODCOD table: transcribe the standard's table to a CSV and add a test that compares every row of `MODCODS` to it.
4. Replace the BLER waterfall with per-MODCOD curves (Es/N0 grid versus log10 BLER, linear interpolation), keeping the slope model as a fallback.
5. Real data: find public attenuation or beacon measurements (check access terms and sampling rate). Without them every result is simulation-only and must be labelled so.

**Acceptance:** a statistics table (time above 1, 5, 10, 15, 20 dB; fade-slope percentiles) compared against the reference; the MODCOD test passes.

### M2. Forecaster

1. Build windows with `build_windows` (4.1). Horizon grid in steps at `dt = 0.1`: `[10, 20, 30, 50, 100, 200, 300, 600, 1200, 3000]` (1 s to 5 min). Features: current attenuation, slopes over 5 s and 30 s, maximum over the last 60 s; optionally rain-rate or weather inputs and time of day.
2. Fit the **empirical baseline** (4.2) first. It is numpy-only and may turn out hard to beat.
3. Neural model (4.7): MLP or small LSTM on the last 60 s (1 s resolution) plus the summary features; outputs quantiles of `Z_H - current` at the levels you will calibrate, `tau = 1 - alpha` for the alpha grid. Adam, learning rate 1e-3, early stopping on validation pinball loss, small ensemble if useful.
4. Evaluate on held-out seeds: pinball loss per (tau, H), raw coverage per tau and per fade state, and a comparison against persistence and the empirical baseline.

**Acceptance:** the forecaster beats persistence on held-out pinball loss; raw coverage tables exist. If the neural model does not beat the empirical baseline, report that and use the baseline.

### M3. Conformal calibration

1. For every (tau, H) level, compute split-conformal corrections per fade state (4.3). **Thin** calibration windows (stride at least the horizon, or at least 60 s) because overlapping windows overstate the sample size. A correction of `+inf` means not enough data: drop that level and log it.
2. Persist corrections to JSON (`outputs/calibration.json`) together with the config hash and seed ranges.
3. Online update (optional first, then required for the shift experiment): keep `alpha_t` per fade state; when a window's outcome becomes available (about `H + RTT` after the bound was issued) compute `err = 1[Z > bound]` and call `aci_step`. Process events in chronological order of **availability**, not of issue time.
4. Shift experiment: train and calibrate on one parameter set, test on another, for example `replace(step2_scenario(), event_peak_ln_mean=2.0, events_per_day=6.0)`. Report coverage per fade state with sample counts.

What the prototype showed (synthetic channel): with matched train and test channels the raw empirical quantiles were already calibrated, so conformal added nothing. Under the shifted channel, deep-fade coverage of a 0.99 target fell to about 0.96; a pooled correction stayed near 0 dB because clear-sky windows dominate it; stratified calibration added about 6 dB in the deep state and restored about 0.99. Expect calibration to matter only under shift or on real data.

**Acceptance:** per-state coverage within sampling error of the target (use a binomial interval and report n) on held-out seeds, with and without shift.

### M4. Action evaluator and optimizer

1. Use `evaluator.py` and `optimizer.py` (4.4, 4.5). The evaluator assumes attenuation is held at the calibrated bound, so frame losses are independent with a constant probability.
2. **Known limitation (important).** Under a constant worst-case fade, interleaving and partial erasure protection show no benefit, and the window maximum is pessimistic. **Stage A** (build this first): evaluate MODCOD, power, parity against residual random loss, deferral. **Stage B** (research extension): forecast and calibrate a window *profile* (for example the peak and the time spent above two or three attenuation levels), split the miscoverage budget across those statistics by a union bound, and evaluate actions by simulating frames across the worst-case profile, so interleaving, duplication and burst-aware parity can matter.
3. Overhead: airtime ratio from the evaluator plus a small power term; add header and control bytes per frame when you want them counted (`header_bytes` in the config).
4. Alpha grid: keep large values (0.9, 0.7, ...) in `ALPHAS`, otherwise P* is meaningless for deep fades. Levels without calibrated bounds must be removed from the grid.
5. Speed: the demo grid takes a few seconds per message. Prune MODCOD candidates (as in the demo), cache evaluations by `(modcod, power, parity, defer, bound)`, and use common random numbers (already built in via the fixed seed). For `eps_p = 1e-3` Monte Carlo with 2000 trials is too coarse: use more trials or an analytic bound for the first-round success probability.
6. Return a `reason` for every infeasible decision.

**Acceptance (unit tests):**
- At a bound near 0 dB the optimizer picks the highest MODCOD with no parity and no extra power.
- A more severe bound never lowers the overhead of the chosen action.
- A deadline shorter than the minimum transmission time plus one-way delay is infeasible with reason `deadline`.
- With common random numbers, success probability is non-decreasing in parity at fixed other settings.

### M5. Closed-loop evaluation harness

1. **Frame-level ground truth.** Do not score RC-DAM with the evaluator's own constant-loss model. Simulate the chosen action against the **true** attenuation trajectory with Bernoulli frame losses (outline):

```python
def simulate_true_delivery(action, t_start_idx, att, msg, table, sc, rng):
    """Outline. Frames go out back to back from t_start. Frame j is sent at t_j = t_start + j * frame_s and is lost with
    probability bler(modcod, clear + power - att[int(t_j / dt)]). The receiver needs K of K + parity frames. Each ARQ round
    resends the shortfall and starts one RTT after the previous round ended. Returns (delivered_by_deadline, airtime_s)."""
    ...
```

2. Message stream: arrivals every `msg_interval_s` with jitter; a mixture of priorities with their `eps`.
3. Variants as functions `policy(context) -> Decision`: V0 static + ARQ, V1 fixed-margin ACM, V2 linear predictive ACM, V3 quantile forecaster + MODCOD without conformal, V4 V3 + conformal window bound, V5 V4 + all Stage A levers, V6 full RC-DAM with infeasibility, and the zero-delay oracle (knows the true window, exhaustive search).
4. Metrics: miss rate versus `eps` per priority; overhead; mean and tail completion time; coverage of `Z` per fade state; infeasibility confusion matrix versus the oracle (false-feasible, false-infeasible); reliability diagram of reported P* against empirical success.
5. Statistics: independent seeds as replicates, 95% intervals over seeds, **paired** differences (same seed across variants). Report "overhead needed to reach the target miss rate" as well as miss at matched overhead.

**Acceptance:** tables and plots for H1 to H4; the fairness checklist in Section 6 is satisfied.

### M6. Reporting and reproducibility

- Figures: miss-versus-overhead curves, coverage per fade state, infeasibility confusion matrix, deadline sweep, ablation bars.
- Save for every run: config dump, git commit, seed list, library versions, raw per-seed CSV.
- Update the document's status table and the README; rerun the novelty search just before submission.

---

## 6. Test and fairness checklist

- Features use only indices `<= m` (no future leakage); a test shifts the series and checks the features change.
- Windows are `(m, m + H]`; off-by-one test against brute force (done for `forward_window_max`).
- Feedback delay applied once: the sender sees attenuation up to `i0 - d`, never later.
- Train, calibration and test windows come from different seeds or periods.
- No method gets information another lacks, except the explicitly labelled oracle.
- Same BLER curves, ARQ rule, frame size and overhead definition for all methods.
- Calibration never uses delivery outcomes.
- Report sample counts next to every coverage number.

## 7. Pitfalls

- **Overlapping windows** overstate the calibration sample size; thin them.
- **Selection bias:** outcomes exist only for the action taken; calibrate on attenuation.
- **Circular scoring:** the evaluator's Monte Carlo must not be the ground truth.
- **Stationary synthetic channel** makes calibration look unnecessary; test under shift.
- **Rare events:** 1e-3 miss rates need many independent windows; count independent weather events, not messages.
- **Model mixing:** the repository's ARQ is an expected-value fluid model; M5's ground truth is frame-level Bernoulli. Keep them separate and labelled.
- **ARQ round limit:** raise `arq_max_rounds` for long deadlines so it never causes artificial misses.
- **dB versus linear:** attenuation adds in dB; BLER curves are per Es/N0 in dB.
- **Runtime:** 48 h at 0.1 s is 1.7M samples per seed; use window strides and workers.

## 8. Open decisions to fix before coding

| Decision | Suggestion |
|---|---|
| eps_p per priority | critical 1e-3, normal 1e-2, best-effort 1e-1 (illustrative); 1e-3 needs far more data |
| Message size and deadlines | several Mbit to tens of Mbit; deadline sweep 2, 10, 60, 300 s |
| Regime | GEO Ka-band, RTT about 0.54 s |
| Horizon grid | 1 s to 5 min as in M2 |
| Fade-state edges | 1, 5, 10 dB (revisit with real data) |
| Alpha grid | the `ALPHAS` tuple; drop levels lacking calibration data |
| Action grid | Stage A: MODCOD, power {0, 1, 2} dB, parity {0, 5, 15, 30}%, defer {0, 0.5, 1} s |
| Overhead weights | airtime ratio plus 0.03 per dB of extra power; add header bytes later |
| Split protocol | by seed first; by time and season on real data |

## 9. Suggested layout after M6

```
rcdam-project/
  rcdam/
    config.py  attenuation.py  link.py  arq.py  policies.py  experiment.py  stats.py  metrics.py   (existing)
    data.py  windows.py  forecaster_baseline.py  forecaster_nn.py  calibration.py                  (M1-M3)
    evaluator.py  optimizer.py  groundtruth.py  variants.py                                        (M4-M5)
  scripts/
    run_step1.py  run_step2.py  run_headroom.py  train_forecaster.py  calibrate.py
    run_shift_test.py  run_h1_h4.py  make_figures.py
  tests/
    test_step2.py  test_windows.py  test_calibration.py  test_evaluator.py  test_optimizer.py
  outputs/
  RCDAM_Build_Guide.md  README.md
```
