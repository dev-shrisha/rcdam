# RC-DAM Project Status

**Date:** 2026-10-30  
**Repository:** `rcdam-project`  
**Current Milestone:** Phase 1 Complete (Ready for Phase 2: Evaluation infrastructure)

---

## 1. Hypotheses Status

| Hypothesis | Description | Status | Evidence File | Key Numbers | Date |
|---|---|---|---|---|---|
| **H1'** | At same certified silent-miss target ($\epsilon=10^{-2}$), RC-DAM has lower overhead than predictive and fixed-margin ACM | Not established | `outputs/h1_h4_results.csv` | V1 avoid=1.989%, V6 avoid=6.392% (current runs inconclusive/stale, n=705) | 2026-10-30 |
| **H2'** | Stratified calibration keeps window-max coverage near target in every fade state under channel shift | Established | `outputs/shift_test.csv` | Deep fade: uncalibrated=96.5%, pooled=96.5%, stratified=99.8% (target 99.0%, n=403) | 2026-10-30 |
| **H3'** | INFEASIBLE flag is informative (high recall vs oracle, low false-infeasible rate, positive flag lift) | Not established | `outputs/h1_h4_results.csv` | Oracle missed only 1 of 705 messages (insufficient oracle-infeasible events, needs $\ge 100$) | 2026-10-30 |
| **H4'** | Levers beyond MODCOD (erasure coding, interleaving, deferral) rescue a measurable share of messages under tight deadlines | Not established | `outputs/step8_ablations.csv` | Ablations A3-A6 currently identical due to unexercised levers in optimizer | 2026-10-30 |

---

## 2. Phase Execution Status

| Phase | Description | Status | Evidence File / Test | Key Results | Date |
|---|---|---|---|---|---|
| **Phase 1** | Fix known defects (1.1-1.7) with regression tests | Established | `tests/test_phase1_regressions.py`, `tests/run_all_tests.py` | 7/7 regression tests passed; 100% full master test suite pass; reproductions confirmed for 1.1 (pruning), 1.2 (selection curse), 1.3 (forecaster persistence), 1.4 (horizon pick), 1.5 (evaluator bound), 1.6 (test vacuity), 1.7 (hygiene) | 2026-10-30 |
| **Phase 2** | Evaluation infrastructure (logs, deterministic oracle, stress presets $\ge 100$ oracle misses, 20 seeds) | Not tested | — | Awaiting Phase 1 completion | 2026-10-30 |
| **Phase 3** | V5/V6 separation and explicit INFEASIBLE semantics (R1-R4) | Not tested | — | Awaiting Phase 2 completion | 2026-10-30 |
| **Phase 4** | Policy-level risk control (Learn-then-Test fixed sequence) | Not tested | — | Awaiting Phase 3 completion | 2026-10-30 |
| **Phase 5** | Levers, Stage B window-profile, and ablations | Not tested | — | Awaiting Phase 4 completion | 2026-10-30 |
| **Phase 6** | Distribution shift and real data loader | Not tested | — | Awaiting Phase 5 completion | 2026-10-30 |
| **Phase 7** | Baseline fairness and tuned margin baseline | Not tested | — | Awaiting Phase 6 completion | 2026-10-30 |
| **Phase 8** | Documentation, claim verification, automated RESULTS.md | Not tested | — | Awaiting Phase 7 completion | 2026-10-30 |
| **Phase 9** | Frontend validation & JSON contract | Not tested | — | Awaiting Phase 8 completion | 2026-10-30 |
| **Phase 10** | Cisco Packet Tracer illustrative demonstration | Not tested | — | Intentionally last | 2026-10-30 |

