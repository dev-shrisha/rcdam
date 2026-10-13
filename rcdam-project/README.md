# RC-DAM: Risk-Calibrated, Deadline-Aware Mitigation for Satellite Links

Complete implementation of the **RC-DAM** system architecture for high-reliability, latency-constrained communications over single GEO satellite links.

## Project Structure & Architecture Modules

| Module | File | Purpose | Step Completed |
|---|---|---|---|
| **Module 1: Channel Simulator** | `rcdam/attenuation.py`, `rcdam/itur.py` | Rain attenuation + scintillation time series, calibrated to ITU-R P.618 and P.1853 | Step 1, 2, 3 |
| **Module 2: Link & Delivery Model** | `rcdam/link.py`, `rcdam/erasure.py` | ETSI EN 302 307 DVB-S2 MODCODs, smooth BLER waterfall, packet erasure coding & block interleaving | Step 2, 3, 4 |
| **Module 3: Cost Model** | `rcdam/cost.py` | Single measurable quantity $O(a)$ (airtime overhead, headers, power step, delay tie-breaker) | Step 4 |
| **Module 4: Baselines** | `rcdam/policies.py`, `rcdam/arq.py` | Static MODCOD ARQ, Fixed-margin ACM, Predictive ACM (point forecast), Oracle upper bound | Step 2 |
| **Module 5: Quantile Forecaster** | `rcdam/forecaster.py` | Deep neural network with non-crossing heads predicting multi-quantiles via pinball loss | Step 5 |
| **Module 6: Conformal Calibrator** | `rcdam/conformal.py` | Stratified conformal risk control with delayed feedback FIFO and adaptive ACI margin | Step 6 |
| **Module 7: Redundancy Optimizer** | `rcdam/optimizer.py` | Joint deadline-constrained optimizer across all fade timescales + early INFEASIBLE alert | Step 7 |
| **Module 8: Evaluation Harness** | `rcdam/ablations.py`, `rcdam/stats.py` | Multi-seed ablation harness (A0–A6, B1) with 95% confidence intervals and Pareto plots | Step 8 |
| **Module 9: RL Policy** | `rcdam/rl_baseline.py` | Deep Reinforcement Learning (DRL-AMC) comparison baseline | Step 9 |
| **Module 10: Research Paper** | `PAPER_DRAFT.md` | Complete publication-ready paper draft synthesizing all methods and empirical findings | Step 10 |

---

## Running the Project Steps

All experiment scripts are in `scripts/` and output tables and figures to `outputs/`:

```bash
# Step 1: Simulator + initial single-seed baseline
python scripts/run_step1.py

# Step 2: Realistic baselines with smooth BLER curves, ARQ, and multi-seed confidence intervals
python scripts/run_step2.py --seeds 20 --workers 4

# Step 3: ITU-R P.618 calibration, ETSI EN 302 307 MODCOD verification, and deadline sweep
python scripts/run_step3.py

# Step 4: Cost model O(a) and packet erasure code / interleaver depth tradeoff evaluation
python scripts/run_step4.py

# Step 5: Multi-quantile deep fade forecaster training and pinball loss evaluation
python scripts/run_step5.py

# Step 6: Stratified conformal calibrator evaluation under delayed feedback (per fade state)
python scripts/run_step6.py

# Step 7: End-to-end full RC-DAM redundancy optimizer and early infeasibility signaller
python scripts/run_step7.py

# Step 8: Comprehensive multi-seed ablation study (A0 to A6 and B1) and Pareto curve generation
python scripts/run_step8_ablations.py --seeds 5 --hours 12.0

# Step 9: DRL-AMC Reinforcement Learning baseline training and comparison
python scripts/run_step9_rl.py
```

---

## Running Unit Tests

Run the complete test suite across all modules:
```bash
python tests/run_all_tests.py
```
Or run individual step test suites:
- `python tests/test_step2.py`
- `python tests/test_step3.py`
- `python tests/test_step4.py`
- `python tests/test_step5.py`
- `python tests/test_step6.py`
- `python tests/test_step7.py`
- `python tests/test_step8.py`
- `python tests/test_step9.py`

---

## Output Artifacts in `outputs/`

1. `step1_baseline.png`: Initial Step 1 single-seed baseline plot.
2. `step2_baselines.png`, `step2_results.csv`: Step 2 multi-seed baseline table and error bars.
3. `step3_itur_calibration.png`, `step3_deadline_sweep.csv`, `step3_deadline_sweep.png`: ITU-R calibration and horizon sweep.
4. `step4_cost_and_erasure.png`: Cost model breakdown and interleaver depth vs deadline tradeoff.
5. `step5_forecaster_weights.pt`, `step5_quantile_forecast.png`: Trained multi-quantile forecaster and fan chart.
6. `step6_conformal_coverage.png`: Empirical coverage per fade state under delayed feedback.
7. `step7_rcdam_decisions.png`: Full RC-DAM decision dynamics (MODCOD, power, code rate, infeasibility alerts).
8. `step8_ablations.csv`, `step8_pareto_curves.png`: Comprehensive ablation results and Pareto frontiers.
9. `step9_rl_comparison.csv`, `step9_rl_comparison.png`: DRL-AMC comparison baseline results.
10. `PAPER_DRAFT.md`: Full research paper draft.
