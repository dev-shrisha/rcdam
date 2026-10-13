# RC-DAM: Complete Solution and System Architecture (v2)

**RC-DAM = Risk-Calibrated, Deadline-Aware Mitigation** for a satellite link that is the only path from A to B.

> **What changed from v1**
> 1. Decision principle fixed: one constraint (on-time delivery probability) plus one objective (overhead), instead of "reliability, then deadline, then cost".
> 2. Redundancy is now organised by **fade timescale** (physical layer, packet level, timing), including the interleaver-depth versus deadline tradeoff.
> 3. "Overhead" is defined as one measurable cost; header and control bytes are a **cost term**, not a headline contribution.
> 4. Calibration is done on the **channel** (always observed), not on delivery outcomes (biased by the chosen action); the forecaster outputs **quantiles**; coverage is reported **per fade state**.
> 5. Simulator requirements added (smooth block-error curves, erasure-decoding model); a concrete ablation plan added.

---

## 1. Problem and goal

A sends messages to B over a single satellite link with no backup path. Fades (rain, scintillation, geometry) cut capacity, and the long round-trip time (about 0.5 s for GEO) means the sender cannot react to what the channel is doing *now*.

Delivery cannot be guaranteed under every channel condition: if a fade pushes capacity below what a message needs, nothing can deliver it on time. The goal is therefore:

> For each message (size S, priority p, deadline D), choose transmission actions on the single link so that **P(delivered by D) >= 1 - epsilon_p**, at **minimum overhead**. If no action satisfies this, report INFEASIBLE together with the best achievable delivery probability.

Reliability and the deadline are **one** requirement, because a message delivered late has failed. They are not ranked against each other.

## 2. Research question and scope

> Can RC-DAM size and structure redundancy (physical-layer coding and modulation, packet-level erasure coding and interleaving, and transmission timing) from calibrated forecasts of predicted channel conditions, so that overhead is minimised while the probability of successful, on-time delivery stays above a specified target?

**In scope:** forecast, calibration, joint redundancy sizing under a deadline, infeasibility signalling.
**Cost term only:** header and control overhead (counted in the cost; compression is well studied and not claimed as novel).
**Out of scope:** rerouting or any second path.

## 3. Solution in one paragraph

Forecast the fade distribution over a horizon covering the RTT and the transmission time. Calibrate the forecast on observed attenuation so that it yields an upper-bound fade envelope with a controlled miss rate. For each message, evaluate candidate actions (MODCOD, power step, erasure redundancy, interleaver depth, selective duplication, send time) through a channel-to-delivery model, and pick the lowest-overhead action whose delivery probability meets the target. If none does, return INFEASIBLE with the best achievable probability. Delayed feedback recalibrates the system online.

## 4. System architecture

```
 INPUTS              FORECAST               CALIBRATION              DECISION                    LINK
┌────────────┐   ┌─────────────────┐   ┌───────────────────┐   ┌─────────────────────┐   ┌───────────────┐
│ beacon/SNR │   │ Quantile fade   │   │ Conformal risk    │   │ Deadline-constrained│   │ Transmitter   │
│ weather    │──>│ forecaster      │──>│ control, stratified│──>│ redundancy optimizer│──>│ MODCOD, power │
│ geometry   │   │ (multi-quantile │   │ by fade state     │   │ + delivery-prob.    │   │ erasure code, │
└────────────┘   │  over horizon)  │   │ -> calibrated     │   │ model + cost model  │   │ interleaver,  │
                 └─────────────────┘   │ envelope + CDF    │   └───────┬─────────────┘   │ duplication,  │
                                       └────────▲──────────┘           │ infeasible      │ send time     │
                                                │                      v                 └──────┬────────┘
                              (1) ATTENUATION   │               ALERT to sender                 │
                              measurements      │               + best achievable P             v
                              (RTT-delayed)     │               (defer / downgrade /       GEO satellite link
                                                │                escalate)                       │
                                       ┌────────┴───────────────────────────────────────────────v────────┐
                                       │ Receiver: measured Es/N0 (always observed),                      │
                                       │ ACK/NACK + delivered-by-deadline? (observed only for chosen      │
                                       │ action) -> (2) used for END-TO-END VALIDATION only               │
                                       └──────────────────────────────────────────────────────────────────┘
```

Two feedback paths, kept separate on purpose:
1. **Channel measurements** recalibrate the forecast. They are observed whatever action was chosen, so they are unbiased.
2. **Delivery outcomes** validate the end-to-end guarantee. They are observed only for the chosen action, so they are not used to recalibrate the forecast directly.

Decisions are open-loop for about one RTT, which is why there is no pre-transmission probe.

## 5. Modules and interfaces

| # | Module | Input | Output | Notes |
|---|---|---|---|---|
| 1 | Channel simulator | Scenario config | attenuation(t), Es/N0(t) | Events model for stress tests; Gauss-Markov for calibration; later real beacon data |
| 2 | Link / delivery model | Es/N0, action | block-error probability, erasure-decoding probability | **Smooth** error curves, not a hard threshold (see Section 10) |
| 3 | Cost model | action, message | overhead O(a), power cost, delay | Single cost definition (Section 8) |
| 4 | Baselines | Es/N0(t), config | miss rate, overhead | HARQ/ARQ, fixed-margin ACM, predictive ACM (point forecast), DRL-AMC |
| 5 | Forecaster | recent telemetry window | quantiles of attenuation over horizon H >= RTT + T_tx | PyTorch, pinball loss |
| 6 | Conformal calibrator | quantiles, past attenuation | calibrated envelope and delivery-probability distribution | Stratified by fade state; online update with delayed feedback |
| 7 | Redundancy optimizer | message (S, p, D), calibrated forecast, action set | chosen action or INFEASIBLE + best P | Enumerates a small discrete set, so it stays interpretable |
| 8 | Evaluation harness | all of the above | miss rate vs. epsilon, overhead, coverage by state, latency | Multi-seed, confidence intervals |
| 9 | (Optional) RL policy | state, reward | action | Comparison baseline only; the paper must not depend on it |

## 6. Decision rule

For message m = (S, p, D) at time t0, and action a = (MODCOD, power step, erasure redundancy, interleaver depth, duplication set, send time):

- **Delivery probability** P(a) = probability that the decoded message is complete by D, computed by passing the calibrated attenuation distribution through the link / delivery model (module 2).
- **Constraint:** P(a) >= 1 - epsilon_p (stricter epsilon for higher-priority messages).
- **Objective:** minimise overhead O(a) (Section 8) over feasible actions.
- **Fallback:** if no action is feasible, return `a* = argmax P(a)` (ties broken by lower cost) and flag the message INFEASIBLE with P(a*).

## 7. Redundancy layers matched to fade timescales

| Fade timescale | Typical cause | Mitigation layer | Cost / limit |
|---|---|---|---|
| Slow (seconds to minutes) | Rain | MODCOD and power step | Lower spectral efficiency; power budget |
| Fast (sub-second to seconds), bursty | Scintillation, short rain-cell edges | Packet-level erasure coding plus interleaving across the burst | Parity airtime; **interleaver depth adds delay, bounded by D** |
| Predicted short outage | Known geometry, forecast dip | Defer the send | Delay, bounded by D |
| Critical message, residual risk | Any | Selective duplication | Extra airtime for that message only |

**Key design tradeoff.** Deeper interleaving survives longer bursts but consumes deadline. The optimizer therefore chooses (code rate, interleaver depth, MODCOD) *jointly* under the deadline, which is stronger than picking from a fixed action list.

**Where this sits in the field.** Predictive ACM, adaptive FEC and deadline-constrained erasure coding each exist separately. The proposed contribution is the *calibrated, joint* sizing of these layers under a deadline, with an infeasibility signal and a per-fade-state coverage check. Re-verify this before claiming novelty.

## 8. Overhead (cost) definition

Define one measurable quantity:

> **O(a) = expected channel airtime used to deliver the message under action a, divided by the airtime needed for the same payload at the best clear-sky MODCOD with no redundancy.**

Airtime includes: protocol headers and control fields, padding, parity, duplicated packets, and expected retransmissions. Add separate terms for **extra transmit power** and, for the tie-breaker, **added delay**. Report the curve **overhead versus on-time delivery probability**, not a single point.

Headers and padding enter here as part of O(a). Compression experiments (for example ROHC-style) are an optional separate study.

## 9. Calibration design

- **Calibrate on attenuation, not on delivery outcomes.** Attenuation is observed at the receiver regardless of the action chosen, so there is no selection bias.
- **Delayed feedback.** Measurements arrive one RTT late; the online conformal update must handle this.
- **Distribution, not one level.** Multi-quantile forecasts allow an estimate of P(a) when the target is infeasible.
- **Stratify by fade state** (clear, light, heavy, deep, or rate-of-change classes) so coverage is checked per state. A guarantee averaged over all messages can hide poor coverage in deep fades.
- **Validate end to end separately:** measure the actual on-time-delivery miss rate against epsilon on held-out data. The conformal guarantee is marginal and assumes roughly exchangeable data, so report where it degrades (new season, new climate).

## 10. Simulator requirements (gaps in the Step 1 code)

The Step 1 simulator decodes with a hard threshold and counts goodput as an idealised fluid. That is fine for a first baseline but will make adaptive-redundancy results an artifact. Before testing RC-DAM:

1. Replace hard thresholds with **smooth block-error curves** per MODCOD, from published results or simulated LDPC.
2. Add an **erasure-code decoding model**: probability of recovery versus parity overhead and loss pattern.
3. Add **interleaving and burst effects** (burst length distribution from the fade timescales).
4. Add **retransmission delay** (HARQ/ARQ) so the long RTT actually costs time.
5. Add **queueing and priorities** once single messages work.
6. Calibrate the rain model to ITU-R statistics, or use real beacon data.

## 11. Evaluation and ablation plan

Baselines: HARQ/ARQ, fixed-margin ACM, predictive ACM (point forecast), DRL-AMC, deadline-aware adaptive erasure coding without forecasting.

Ablations (add one thing at a time, so each contribution is measured):

| ID | System | Question answered |
|---|---|---|
| A0 | Baselines above | What must be beaten? |
| A1 | Calibrated forecast + MODCOD only | Does calibration alone beat point-forecast ACM? |
| A2 | A1 + power step | Value of power control |
| A3 | A2 + packet-level erasure coding | Value of fast-fade protection |
| A4 | A3 + interleaver depth chosen under the deadline | Value of joint depth/deadline optimisation |
| A5 | A4 + defer | Value of timing |
| A6 | Full RC-DAM + infeasibility signal | Precision/recall of infeasibility detection |
| B1 | A6 with header/overhead reduction (separate study) | Does overhead trimming add anything? |

Metrics: deadline-miss rate versus epsilon (does the guarantee hold?), overhead O(a), latency, **empirical coverage per fade state**, infeasibility precision/recall, robustness to distribution shift. Present **Pareto curves** (overhead versus on-time probability) with confidence intervals over many seeds.

## 12. Repo layout

```
rcdam-project/
  rcdam/
    config.py          Scenario parameters
    attenuation.py     module 1
    link.py            module 2   [add: smooth BLER curves, erasure model]
    acm_baseline.py    module 4   [add: harq.py, predictive_acm.py, drl_amc.py]
    metrics.py         module 8
    [cost.py]          module 3
    [forecaster.py]    module 5
    [conformal.py]     module 6
    [optimizer.py]     module 7
  scripts/             one runnable script per experiment
  outputs/             plots and tables
  ARCHITECTURE.md
```

## 13. Build plan

| Step | Build | Done when |
|---|---|---|
| 1 | Simulator + fixed-margin ACM baseline (**in the zip**) | You reproduce the table/plot and understand every number |
| 2 | Smooth BLER curves, retransmission delay, ARQ / ACM / predictive-ACM / oracle baselines, multi-seed runner with confidence intervals **(built, see README)** | Baseline table with error bars |
| 3 | Calibrate the rain model to ITU-R statistics; look for real beacon data | Statistics match, or a real dataset is in hand |
| 4 | Cost model (Section 8) and erasure-code / interleaving model | Overhead computed consistently for every action |
| 5 | Quantile forecaster | Pinball loss and calibration beat the persistence baseline |
| 6 | Conformal calibrator, stratified, with delayed feedback | Coverage close to target in each fade state on held-out data |
| 7 | Redundancy optimizer + infeasibility signal | Full RC-DAM end to end |
| 8 | Ablations A0-A6 (and B1) | Results tables and Pareto plots |
| 9 | RL comparison (optional) | One ablation row |
| 10 | Paper draft | Full draft |

## 14. Status and next action

Steps 1 and 2 are built (see README). Step 2 findings on the **placeholder** channel (20 seeds x 48 h, 115,200 messages per method, 2 s deadline):

- About **0.34% of messages (+/- 0.17)** cannot be delivered even by a zero-delay oracle MODCOD policy, because capacity during deep fades is too low. These are the messages an early INFEASIBLE signal and levers beyond MODCOD must address.
- The best fixed-margin ACM (1-2 dB margin) misses about 0.47% of messages overall, of which only about **0.13 percentage points are avoidable** by a better MODCOD choice.
- Predictive ACM with a **linear point forecast performs the same** as fixed-margin ACM on this channel (no gain).
- Static MODCOD with a large design margin reaches lower avoidable misses (about 0.07%) but at roughly **3x the airtime overhead** of ACM.

**Implications for RC-DAM.** On this channel the headroom from better MODCOD prediction alone is small. RC-DAM's value has to come from (a) calibrated, state-dependent margins instead of one fixed margin, (b) levers beyond MODCOD (erasure coding, interleaving, power, deferral) for fast fades and for fades short relative to the deadline, and (c) detecting infeasible messages early. All of this depends on the deadline relative to fade duration.

**Next (Step 3):**
1. Run a **deadline sweep** (for example 2 s, 10 s, 60 s, 300 s) to see when deferral and extra redundancy can rescue the "unrecoverable" messages.
2. Calibrate the rain model to ITU-R statistics and look for real beacon data.
3. Verify the MODCOD table against ETSI EN 302 307.
4. Then Step 4 (cost model, erasure-code and interleaving model) per the build plan.

## 15. Risks and limitations

- The conformal guarantee is marginal and assumes roughly stable statistics; report per-state coverage and where it breaks.
- A conservative envelope wastes capacity; the overhead-versus-reliability curve is the real result.
- More directions dilute the paper: keep the core as calibrated joint redundancy sizing, with headers as a cost term.
- Without real data the contribution is weaker; plan for data access early.
- Novelty must be rechecked just before submission.
- The Step 1 simulator is idealised (hard thresholds, fluid goodput); results from it are not evidence for RC-DAM.
