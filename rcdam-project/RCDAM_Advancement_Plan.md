# RC-DAM Advancement Plan

What to change in the present project, and the advancements that make RC-DAM's claim testable. It follows from the first end-to-end benchmark (`h1_h4_results.csv`), which was inconclusive.

**Status of the evidence.** The benchmark has about 118 messages and one replicate (every confidence interval is 0.0). Oracle, V1 and V2 missed 0; V3/V4 missed 2; V5/V6 missed 5 (95% Wilson interval 1.8% to 9.5%); V5 and V6 are identical in every column; V0 fails every message; the oracle misses nothing, so there are no genuinely infeasible messages. It supports neither "RC-DAM beats the baselines" nor "it does not". Fix the evaluation first (Part 1), then the mechanics (Parts 2 and 3), then add the risk-control advancement (Part 4).

**Tested vs untested.** All code in Part 9 was run (Python 3, NumPy, SciPy, pandas). The numbers quoted from simulations are checks of the utilities on synthetic or placeholder-channel data; they are not results about RC-DAM on real channels.

---

## 1. Required changes, in priority order

### P0: before any further claim

| # | Change | Why | Acceptance |
|---|---|---|---|
| 1 | Run at least 20 independent seeds (or real time periods) with thousands of messages per variant. Treat seeds or weather events as the replicate, not messages. | One replicate, n of about 118, zero-width intervals. Zero misses in 118 messages is compatible with a true rate near 3%. | Intervals computed over seeds; counts printed next to every rate |
| 2 | Log every message (schema in 9.3) and classify every miss | The 5 V5/V6 misses cannot be explained from aggregates | Category table for every variant |
| 3 | Define V6 separately from V5: V6 = V5 plus an explicit INFEASIBLE handling rule and its own metrics | Identical rows mean V6 is only a relabelled V5 | V6 differs from V5 in at least one outcome column, or the report says why it cannot |
| 4 | Repair or drop V0 | Miss rate 1.0 and NaN latency carry no information | V0 delivers messages, or is removed |
| 5 | Compare at matched reliability: sweep a safety setting for every variant and report overhead at a target miss rate | V5/V6 are cheaper only because they accept more misses | Overhead-versus-miss curves per variant, `overhead_at_target` table |
| 6 | Create genuinely infeasible messages: deadline sweep and a harsher channel until the oracle misses at least 100 test messages | With zero oracle misses H3 and H4 cannot be tested | Oracle-infeasible count reported; at least 100 |
| 7 | Horizon-consistent bounds and validated selection (Part 3) | Optimizer's curse and uncovered deferral windows are the leading suspects for V5 > V4 misses | V5 misses no longer exceed V4 at the same target, or the cause is shown |

### P1: the main advancement

| # | Change | Acceptance |
|---|---|---|
| 8 | Policy-level risk control of the silent-miss risk (Part 4) | Certified setting found; held-out silent-miss risk at or below target with counts |
| 9 | Distribution-shift experiment for calibration | Stratified coverage under shift reported |
| 10 | Real attenuation data | At least one real dataset evaluated, or the paper says simulation-only |

### P2: extensions

| # | Change |
|---|---|
| 11 | Stage B window-profile statistics so interleaving and erasure protection can matter (Part 6) |
| 12 | Richer INFEASIBLE handling: defer, downgrade, tiered delivery (Part 5) |
| 13 | Several messages competing for airtime (queueing) |

---

## 2. Evaluation protocol

**Hypotheses, restated so they can be falsified.**

- **H1′.** At the same certified silent-miss target (for example epsilon = 1e-2), RC-DAM needs less overhead than predictive ACM and fixed-margin ACM. *Test:* `overhead_at_target` on sweeps, paired by seed, cluster bootstrap over seeds.
- **H2′.** Stratified calibration keeps coverage of the window maximum near target in every fade state, including under channel shift. *Test:* coverage per state with n and binomial intervals.
- **H3′.** The INFEASIBLE flag is informative: high recall against the oracle, low false-infeasible rate, and miss rate much higher among flagged than unflagged messages. *Test:* `infeasibility_report`, needs at least 100 oracle-infeasible messages.
- **H4′.** Levers beyond MODCOD rescue a measurable share of messages that MODCOD adaptation alone cannot deliver, and the share depends on the deadline relative to fade duration. *Test:* ablation across a deadline sweep.

**Sample sizes (computed with the utilities in 9.1).**

| Question | Messages needed |
|---|---|
| Zero misses certify miss rate below 1e-2 (95%) | 299 |
| Zero misses certify below 1e-3 | 2995 |
| Detect 1% versus 0.5% miss rate, unpaired, per variant | about 4700 |
| Certify a conditional risk with the Part 4 method at epsilon = 1e-2 | at least 299 flagged-feasible calibration messages per tested setting |

Messages inside one fade are correlated, so these are lower bounds on independent weather events. Prefer more seeds over more messages per seed.

**Rules.** Paired comparisons (same seed, same messages) with an exact McNemar test or a cluster bootstrap over seeds. Report counts, not only rates. Do not average latency over delivered messages only: report completion-time percentiles with misses counted as censored. Treat 8 variants and 4 hypotheses as multiple comparisons.

---

## 3. Fix the mechanics of V5 and V6

### 3.1 Per-message logging and miss taxonomy

Log the columns in 9.3 for every message. `classify_misses` assigns every miss to one category:

| Category | Meaning | What to do |
|---|---|---|
| `bound_violation` | Not flagged, true window maximum exceeded the calibrated bound | Calibration or horizon problem: check stratification, horizon and shift |
| `model_mismatch` | Not flagged, bound held, yet missed | Evaluator and ground truth disagree, optimizer's curse, or a bug (power, parity, deferral modelled differently) |
| `silent_unrecoverable` | Not flagged, oracle also misses | The system should have warned: the infeasibility rule is too optimistic |
| `flagged_unrecoverable` | Flagged, oracle also misses | Correct warning |
| `flagged_avoidable` | Flagged, oracle could deliver | False infeasible with a loss: rule too pessimistic |

Aggregate counts of the first three are the diagnosis of V5/V6.

### 3.2 Horizon-consistent bounds

An action that defers or interleaves occupies a longer window. The bound must be taken at the horizon the action needs (`horizon_steps_needed`, then `pick_horizon`). Using the bound for the shorter default window under-covers exactly the actions that use these levers. If no grid horizon is long enough, the action is infeasible with reason `no_calibration_data`.

### 3.3 Optimizer's-curse guard

Selecting the cheapest action whose Monte Carlo estimate clears the target picks actions whose estimates are favourably wrong, especially with a large action set. In a controlled test on the evaluator (constant frame loss 0.05, 309 frames, 2 s deadline, target 0.99, 21 candidate parity levels, 150 repeated selections):

| Selection rule | Mean true success of chosen action | Choices whose true success is below 0.99 |
|---|---|---|
| Selection only (300 draws) | 0.9835 | 88% |
| Selection, then verification on independent draws (3000) with Wilson lower bound at least 0.99 | 0.9943 | 0% |

The guard costs about 5 extra parity frames on average here (15.0 versus 9.6): that is the price of not violating the target. Use `select_validated`.

---

## 4. Advancement: policy-level risk control of silent misses

**Problem.** Calibrated attenuation bounds do not by themselves guarantee delivery: the evaluator can be wrong, the union bound is loose, and selection is biased. So certify the thing you actually claim, end to end.

**Define the claim.** A *silent miss* is a message that the system did **not** flag INFEASIBLE and that missed its deadline. The honest guarantee is: *when RC-DAM says feasible, it fails with probability at most epsilon.* This is the "does not pretend success" requirement, expressed as a risk.

**Safety setting.** Give the policy one scalar lambda per fade state, for example extra dB added to the calibrated bound before evaluation (lambda = 0, 0.5, 1, ... dB). Larger lambda is more conservative: more messages flagged INFEASIBLE, fewer silent misses, more overhead.

**Procedure.**
1. On held-out **calibration** traces, replay every message under every lambda with the frame-level ground-truth simulator (M5). Record `feasible[i, j]` and `miss[i, j]`. Evaluator draws can be cached per (action, bound), so extra lambdas are cheap.
2. Run `ltt_fixed_sequence(miss, feasible, eps, delta)` (Learn-then-Test, fixed-sequence). It returns the least conservative lambda whose conditional silent-miss risk P(miss | flagged feasible) is certified at most epsilon with probability at least 1 - delta over the calibration sample. It needs no monotonicity assumption and starts at the most conservative setting that has enough flagged-feasible messages.
3. Freeze lambda. Evaluate once on separate **test** traces and report silent-miss rate with counts and a Wilson interval, plus the price: flagged-infeasible rate and overhead.
4. Alternative for the marginal silent-miss loss: `crc_lambda` (conformal risk control). It needs the loss to be non-increasing in lambda; check that before using it.

**Validation of the utilities (simulation of repeated calibrations, toy policy family, 2000 calibration messages, epsilon = 1%, delta = 5%).**
- Learn-then-Test certified a setting in 974 of 1000 calibrations; in 956 of those (98.2%) the true conditional risk was at most epsilon (the guarantee asks for at least 95%).
- When the true risk was 3%, above epsilon, a setting was falsely certified in 0.1% of 1000 calibrations (allowed: 5%).
- The price: the certified setting flagged about 28% of messages as infeasible, against 7% for the unattainable best setting. More calibration messages narrow this gap.
- Conformal risk control on the marginal silent-miss loss (target 0.005) had mean true risk 0.00456 over 600 calibrations.

**Caveats.** The guarantee needs calibration and test messages to be exchangeable (different seeds or time periods, never the same trace), and lambda must never be tuned on test traces. Under distribution shift it can fail; use per-state settings and recalibrate. Certifying epsilon = 1e-3 needs about 3000 flagged-feasible calibration messages. Replay on real traces needs the link model to be believable, since counterfactual outcomes are simulated.

**Why this matters for the contribution.** If it holds on held-out data, the claim becomes concrete: *a deadline-constrained protection policy with an end-to-end certified bound on silent failures, at a measured cost in flagged messages and overhead.* Treat it as a hypothesis until it passes on held-out traces and, ideally, real data.

---

## 5. Advancement: make infeasibility a decision, not a label

Today V6 only attaches a flag. Define what the system does when no action meets the target, and evaluate each rule:

| Rule | Behaviour | Cost |
|---|---|---|
| R1 send anyway and flag | Transmit the best action; sender is warned | Uses airtime that may be wasted |
| R2 defer | Wait until the forecast says the target becomes feasible, if the deadline slack allows | Consumes deadline |
| R3 downgrade | Split the message into priority layers and guarantee the critical layer | Needs layered messages |
| R4 reject and notify | Do not transmit; return P* | Message lost by design |

**Metrics beyond miss rate:** silent-miss rate; flag precision, recall and false-infeasible rate against the oracle; **flag lift** = P(miss | flagged) divided by P(miss | not flagged) (a useful flag has high lift); reliability diagram of reported P* against empirical success; airtime wasted on flagged messages.

## 6. Advancement: window-profile statistics (Stage B)

Bounding only the window maximum treats the whole window as at its worst, so interleaving and partial erasure protection show no benefit. Calibrate a small profile instead: the maximum Z and the time spent above two or three attenuation levels. Split the miscoverage budget across these statistics with a union bound, build a worst-case profile from them (for example a step outage of the calibrated duration at the calibrated depth), and evaluate actions by simulating frames across it. Interleaving then matters because it spreads frames across the outage; deferral matters because it can skip a short one. Implement after Parts 1 to 4 work; test that interleaving gives a benefit on bursty fades and none on constant fades.

---

## 7. Revised milestones

| Milestone | Work | Done when |
|---|---|---|
| M0 | Literature table and headroom analysis (build guide) | Matrix verified; headroom with intervals |
| M4b | Horizon-consistent bounds, validated selection (Part 3) | Optimizer's-curse test passes; V5 vs V4 explained |
| M5a | Per-message logs, classifier, matched-reliability sweeps, stress scenario with at least 100 oracle-infeasible messages | Category tables and curves for all variants |
| M5b | Policy-level risk control (Part 4) on calibration traces, evaluation on test traces | Certified lambda; held-out silent-miss risk with counts |
| M5c | Infeasibility rules R1 to R4 (Part 5) | Flag lift, precision and recall tables |
| M5d | Shift experiment and real data | Coverage per state under shift |
| M6 | Stage B, reporting, final novelty recheck | Draft paper |

## 8. Edits to the project document (v3 to v4)

- Replace H1 to H4 with H1′ to H4′ (Part 2) and add the sample-size table.
- Add a section "Policy-level risk control" (Part 4) and, in the abstract, mention it only **after** it passes held-out tests; until then call it a proposed extension.
- Add an "Empirical status" paragraph: the first end-to-end benchmark (about 118 messages, one replicate) was inconclusive, with the diagnosis plan.
- Add to the risks: optimizer's curse, horizon mismatch for deferral, evaluator versus ground-truth mismatch, too few infeasible messages.
- Add the miss taxonomy to the evaluation section.

---

## 9. Reference code (tested)

Place in `rcdam/` (`evaluator.py` is the one from the build guide). `validated_selection.py` and the tests import `stats_utils.py`; adjust imports to `from .stats_utils import ...` inside the package.

### 9.1 Statistics (`stats_utils.py`)

```python
import numpy as np
from scipy.stats import binom, binomtest, norm


def wilson_interval(k, n, conf=0.95):
    """Wilson score interval for a proportion. Always report it next to k and n."""
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=conf, method="wilson")
    return float(ci.low), float(ci.high)


def exact_upper_bound(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound; with k = 0 this is the 'rule of three' bound."""
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=1 - 2 * (1 - conf), method="exact")
    return float(ci.high)


def mcnemar_exact(b, c):
    """Exact two-sided paired test. b = messages variant A missed and B delivered; c = the reverse."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))


def n_zero_miss(eps, conf=0.95):
    """Messages needed with ZERO misses to show miss rate < eps at the given confidence."""
    return int(np.ceil(np.log(1 - conf) / np.log(1 - eps)))


def n_for_difference(p1, p2, alpha=0.05, power=0.8):
    """Messages per variant to detect p1 versus p2 (normal approximation; unpaired, independent messages)."""
    z_a, z_b = norm.isf(alpha / 2), norm.isf(1 - power)
    pbar = 0.5 * (p1 + p2)
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


def cluster_bootstrap_diff(a, b, clusters, n_boot=4000, seed=0, conf=0.95):
    """Paired difference mean(a - b) with a cluster bootstrap. Use independent seeds (or weather events) as clusters,
    never single messages: messages inside one fade are strongly correlated."""
    a, b, clusters = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    ids = np.unique(clusters)
    per = np.array([(a[clusters == i] - b[clusters == i]).mean() for i in ids])
    rng = np.random.default_rng(seed)
    boots = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(per.mean()), float(lo), float(hi)


def overhead_at_target(miss_k, miss_n, overhead, target, conf=0.95):
    """Matched-reliability comparison: the cheapest configuration of a sweep whose miss-rate UPPER bound is below target.
    Returns the index, or None if no configuration is certified."""
    best = None
    for i, (k, n, o) in enumerate(zip(miss_k, miss_n, overhead)):
        if wilson_interval(k, n, conf)[1] <= target and (best is None or o < overhead[best]):
            best = i
    return best
```

Checks: Wilson(5 of 118) = (0.0182, 0.0954); upper bound for 0 of 118 = 0.0251; exact paired test for 5 versus 0 discordant gives p = 0.0625; 299 and 2995 zero-miss messages for 1e-2 and 1e-3.

### 9.2 Risk control (`risk_control.py`)

```python
import numpy as np
from scipy.stats import binom


def ltt_fixed_sequence(miss, feasible, eps, delta=0.05, min_feasible=None):
    """Learn-then-Test over an ordered safety parameter (fixed-sequence testing).

    miss, feasible: boolean arrays [n_messages, m]; column j is the policy with safety setting j, ordered from the
    LEAST conservative (j = 0) to the MOST conservative (j = m - 1). miss[i, j] is the replayed ground-truth miss of
    message i under policy j; feasible[i, j] is True if policy j did not flag it INFEASIBLE.

    Certifies the conditional silent-miss risk P(miss | flagged feasible) <= eps with probability >= 1 - delta over the
    calibration sample (no monotonicity assumption). The sequence starts at the most conservative setting that has at
    least `min_feasible` feasible messages (default: the number of zero-miss messages needed to certify eps at level
    1 - delta; fewer feasible messages can never certify anything). The start depends only on the feasibility flags,
    not on outcomes, so validity is kept. Testing then moves toward less conservative settings and stops at the first
    failure. Returns (index of the least conservative certified setting or None, table of diagnostics).
    Messages must be exchangeable with future ones, and the safety setting must not be tuned on test messages."""
    n, m = miss.shape
    if min_feasible is None:
        min_feasible = int(np.ceil(np.log(delta) / np.log(1 - eps)))
    table, certified, started = [], None, False
    for j in range(m - 1, -1, -1):
        nf = int(feasible[:, j].sum())
        if not started and nf < min_feasible:
            continue
        started = True
        k = int((miss[:, j] & feasible[:, j]).sum())
        p = float(binom.cdf(k, nf, eps))                    # P(X <= k) if the true risk were exactly eps
        ok = p <= delta
        table.append({"setting": j, "n_feasible": nf, "misses": k, "p_value": p,
                      "flagged_infeasible_rate": 1.0 - nf / n, "certified": ok})
        if not ok:
            break
        certified = j
    return certified, table


def crc_lambda(losses, alpha, B=1.0):
    """Conformal risk control for a loss that is NON-INCREASING in the safety setting (columns ordered from least to most
    conservative). Returns the first column with (n/(n+1)) * mean_loss + B/(n+1) <= alpha, or None.
    Guarantee: expected test loss <= alpha for exchangeable messages. Check monotonicity before relying on it."""
    n = losses.shape[0]
    risk = losses.mean(axis=0)
    ok = np.where(n / (n + 1) * risk + B / (n + 1) <= alpha)[0]
    return int(ok[0]) if len(ok) else None
```

### 9.3 Logging schema and miss taxonomy (`diagnostics.py`)

```python
import numpy as np
import pandas as pd

LOG_COLUMNS = [
    "seed", "msg_id", "arrival_idx", "priority", "eps", "size_bits", "deadline_s", "fade_state",
    "modcod", "power_db", "parity_frac", "defer_s", "alpha", "p_lb", "flagged_infeasible",
    "bound_db", "bound_horizon_steps", "true_window_max_db", "delivered", "completion_s", "airtime_s",
    "oracle_delivered", "oracle_airtime_s",
]


def classify_misses(df):
    """Adds a 'category' column. Priority order:
      ok                      delivered by the deadline
      flagged_unrecoverable   missed, flagged INFEASIBLE, oracle also misses       (the system warned, correctly)
      flagged_avoidable       missed, flagged INFEASIBLE, oracle delivers          (warning + loss: false infeasible)
      silent_unrecoverable    missed, NOT flagged, oracle also misses              (should have warned)
      bound_violation         missed, not flagged, oracle delivers, true window max exceeded the bound
      model_mismatch          missed, not flagged, oracle delivers, bound held     (evaluator/ground-truth gap,
                                                                                    optimizer's curse, or a bug)"""
    d = df.copy()
    miss = ~d["delivered"].astype(bool)
    flag = d["flagged_infeasible"].astype(bool)
    orc_ok = d["oracle_delivered"].astype(bool)
    viol = d["true_window_max_db"] > d["bound_db"]
    cat = np.full(len(d), "ok", dtype=object)
    cat[miss & flag & ~orc_ok] = "flagged_unrecoverable"
    cat[miss & flag & orc_ok] = "flagged_avoidable"
    cat[miss & ~flag & ~orc_ok] = "silent_unrecoverable"
    cat[miss & ~flag & orc_ok & viol] = "bound_violation"
    cat[miss & ~flag & orc_ok & ~viol] = "model_mismatch"
    d["category"] = cat
    return d


def infeasibility_report(df):
    """Confusion of the INFEASIBLE flag against the oracle plus the silent-miss rate."""
    d = df
    truth_inf = ~d["oracle_delivered"].astype(bool)
    flag = d["flagged_infeasible"].astype(bool)
    miss = ~d["delivered"].astype(bool)
    tp, fp, fn = int((flag & truth_inf).sum()), int((flag & ~truth_inf).sum()), int((~flag & truth_inf).sum())
    return {
        "n": len(d), "oracle_infeasible": int(truth_inf.sum()),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "false_infeasible_rate": fp / max(int((~truth_inf).sum()), 1),
        "silent_miss_rate": float((miss & ~flag).mean()),
        "misses": int(miss.sum()), "misses_flagged": int((miss & flag).sum()),
    }
```

Checked on a 12-message synthetic log: every category assigned correctly; precision and recall computed against the oracle by hand and matched.

### 9.4 Horizon helper and validated selection (`validated_selection.py`)

```python
import numpy as np
from stats_utils import wilson_interval


def horizon_steps_needed(d_steps, defer_s, span_s, dt):
    """Forecast horizon an action needs: feedback age + deferral + the time span the action occupies (transmission,
    interleaving, ARQ rounds you must cover). The calibrated bound must be taken at THIS horizon, not at a shorter one."""
    return int(d_steps + np.ceil((defer_s + span_s) / dt))


def pick_horizon(grid_steps, needed):
    """Smallest grid horizon >= needed, or None (then the action is infeasible for lack of a calibrated bound)."""
    ok = [h for h in grid_steps if h >= needed]
    return min(ok) if ok else None


def select_validated(candidates, p_select, p_verify, target, conf=0.95):
    """Optimizer's-curse guard. `candidates` are sorted by increasing overhead. p_select(c) -> estimated success
    probability from the selection draws. p_verify(c) -> (k_success, n) from INDEPENDENT draws.
    Walk up the overhead order; accept the first candidate that passes the selection test AND whose verification
    success proportion has a Wilson LOWER bound >= target."""
    for c in candidates:
        if p_select(c) < target:
            continue
        k, n = p_verify(c)
        if wilson_interval(k, n, conf)[0] >= target:
            return c
    return None
```

### 9.5 Test script (`test_adv.py`)

```python
import numpy as np
import pandas as pd
from scipy.stats import binom

from stats_utils import (wilson_interval, exact_upper_bound, mcnemar_exact, n_zero_miss, n_for_difference,
                         cluster_bootstrap_diff, overhead_at_target)
from risk_control import ltt_fixed_sequence, crc_lambda
from diagnostics import classify_misses, infeasibility_report, LOG_COLUMNS
from validated_selection import horizon_steps_needed, pick_horizon, select_validated
from evaluator import p_deliver_by_deadline

print("== stats_utils")
print(" wilson(5,118) =", tuple(round(x, 4) for x in wilson_interval(5, 118)), "(expected about 0.0182, 0.0954)")
print(" exact upper bound (0 of 118) =", round(exact_upper_bound(0, 118), 4), "(rule of three ~ 0.025)")
print(" mcnemar_exact(5, 0) =", mcnemar_exact(5, 0), "(expected 0.0625)")
print(" n_zero_miss(1e-2) =", n_zero_miss(1e-2), " n_zero_miss(1e-3) =", n_zero_miss(1e-3))
print(" n_for_difference(1%, 0.5%) =", n_for_difference(0.01, 0.005), "messages per variant (independent messages)")
rng = np.random.default_rng(0)
cl = np.repeat(np.arange(20), 200); a = rng.random(4000) < 0.012; b = rng.random(4000) < 0.006
print(" cluster bootstrap diff (a-b) =", tuple(round(x, 4) for x in cluster_bootstrap_diff(a, b, cl)))
print(" overhead_at_target:", overhead_at_target([0, 1, 3], [3000, 3000, 3000], [1.30, 1.10, 1.05], target=0.01))

print("\n== risk control: toy policy family (coverage check over repeated calibrations)")
eps, delta, n_cal, n_trials = 0.01, 0.05, 2000, 1000
taus = 1.0 - 0.05 * np.arange(19)                          # setting j: flag infeasible when difficulty s > tau_j; j=0 least conservative
true_cond = lambda tau: 0.05 * tau ** 3 / 4                 # P(miss | feasible) when q(s) = 0.05 s^3, s ~ U(0,1)
rng = np.random.default_rng(1)
good, chosen_tau, none_cnt = 0, [], 0
for t in range(n_trials):
    s = rng.random(n_cal); miss_i = rng.random(n_cal) < 0.05 * s ** 3
    feasible = s[:, None] <= taus[None, :]; miss = np.repeat(miss_i[:, None], len(taus), axis=1)
    j, _ = ltt_fixed_sequence(miss, feasible, eps, delta)
    if j is None: none_cnt += 1; continue
    chosen_tau.append(taus[j]); good += true_cond(taus[j]) <= eps
cert = n_trials - none_cnt
print(f" LTT certified a setting in {cert}/{n_trials} calibrations; true conditional risk <= eps in {good}/{cert} "
      f"({100*good/cert:.1f}%, target >= {100*(1-delta):.0f}%); mean chosen tau {np.mean(chosen_tau):.3f} "
      f"(oracle-optimal tau = {0.8 ** (1/3):.3f}); flagged-infeasible rate {1-np.mean(chosen_tau):.3f} vs {1-0.8**(1/3):.3f}")
bad = 0
for t in range(n_trials):                                   # true risk 3% > eps everywhere: must (almost) never be certified
    s = rng.random(n_cal); miss_i = rng.random(n_cal) < 0.03
    feasible = s[:, None] <= taus[None, :]; miss = np.repeat(miss_i[:, None], len(taus), axis=1)
    j, _ = ltt_fixed_sequence(miss, feasible, eps, delta); bad += j is not None
print(f" false certification rate when the true risk is 3% (> eps): {bad / n_trials:.3f} (allowed <= {delta})")
# CRC on the marginal silent-miss loss (monotone in the setting)
alpha_m, losses_test = 0.005, []
for t in range(600):
    s = rng.random(n_cal); miss_i = rng.random(n_cal) < 0.05 * s ** 3
    L = (miss_i[:, None] & (s[:, None] <= taus[None, :])).astype(float)
    j = crc_lambda(L, alpha_m)
    losses_test.append(0.05 * taus[j] ** 4 / 4 if j is not None else 0.0)
print(f" CRC marginal silent-miss risk: mean true risk over 600 calibrations = {np.mean(losses_test):.5f} (target <= {alpha_m})")

print("\n== diagnostics")
n = 12
df = pd.DataFrame({
    "delivered": [1,1,1,0,0,0,0,0,1,0,1,1],
    "flagged_infeasible": [0,0,0,1,1,0,0,0,0,0,1,0],
    "oracle_delivered": [1,1,1,0,1,0,1,1,1,1,0,1],
    "true_window_max_db": [3,2,4,30,9,28,12,6,5,5,25,1],
    "bound_db": [5,5,5,20,10,20,10,9,9,9,20,3],
})
d = classify_misses(df); print(d["category"].value_counts().to_dict())
print(infeasibility_report(df))
print("log columns:", len(LOG_COLUMNS))

print("\n== horizon helper")
print(" needed steps (d=5, defer 1 s, span 2 s, dt 0.1):", horizon_steps_needed(5, 1.0, 2.0, 0.1),
      "-> grid pick:", pick_horizon([10, 30, 100, 300, 600], horizon_steps_needed(5, 1.0, 2.0, 0.1)))

print("\n== optimizer's curse demo (evaluator.py, constant frame loss p = 0.05, K = 309, deadline 2 s, target 0.99)")
K, p, frame_s, target = 309, 0.05, 0.0015, 0.99
parities = [int(round(x * K)) for x in np.arange(0.0, 0.101, 0.005)]            # 21 candidates, overhead increases with parity
true_p = {par: p_deliver_by_deadline(K, par, p, frame_s, 0.54, 0.27, 0.0, 2.0, n_mc=60000, seed=999)[0] for par in parities}
feas_true = [par for par in parities if true_p[par] >= target]
print(f" cheapest truly feasible parity: {feas_true[0]} frames; true success there = {true_p[feas_true[0]]:.4f}")
def trial(seed, validate):
    sel = lambda c: p_deliver_by_deadline(K, c, p, frame_s, 0.54, 0.27, 0.0, 2.0, n_mc=300, seed=10_000 + seed * 31 + c)[0]
    def ver(c):
        pr = p_deliver_by_deadline(K, c, p, frame_s, 0.54, 0.27, 0.0, 2.0, n_mc=3000, seed=500_000 + seed * 17 + c)[0]
        return int(round(pr * 3000)), 3000
    if validate:
        from validated_selection import select_validated as sv
        return sv(parities, sel, ver, target)
    for c in parities:
        if sel(c) >= target: return c
    return None
for validate in (False, True):
    chosen = [trial(s, validate) for s in range(150)]
    ok = [c for c in chosen if c is not None]
    below = np.mean([true_p[c] < target for c in ok]); mean_par = np.mean(ok)
    print(f" {'validated' if validate else 'selection only'}: chosen parity mean {mean_par:.1f} frames; "
          f"share of choices whose TRUE success is below {target}: {100*below:.0f}%; "
          f"mean true success of choices {np.mean([true_p[c] for c in ok]):.4f}")
```

Every number quoted in Parts 3 and 4 and in 9.1 is printed by this script.
