# Legacy Step Scripts

This directory contains step-by-step runner scripts and modules from the original RC-DAM implementation track (Steps 1–9).

These scripts are **superseded** by the M0–M6 milestone track:

| Legacy Script            | Superseded By                     |
|--------------------------|-----------------------------------|
| `run_step1.py`           | M1: `scripts/run_deadline_sweep.py` |
| `run_step2.py`           | M1/M2: experiment + forecaster    |
| `run_step3.py`           | M1: `scripts/run_deadline_sweep.py` |
| `run_step4.py`           | M3: calibration pipeline          |
| `run_step5.py`           | M2: `scripts/train_forecaster.py` |
| `run_step6.py`           | M3: calibration pipeline          |
| `run_step7.py`           | M4: optimizer / evaluator         |
| `run_step8_ablations.py` | M5: `scripts/run_h1_h4.py`       |
| `run_step9_rl.py`        | M5/M6: closed-loop evaluation     |
| `acm_baseline.py`        | `rcdam/variants.py` (V0–V1)       |
| `metrics.py`             | `rcdam/stats_utils.py`            |

> **Do not use these scripts for new experiments.** They are retained for historical reference and reproducibility of earlier paper drafts only.
