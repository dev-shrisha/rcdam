# RC-DAM

Risk-Calibrated, Deadline-Aware Mitigation (RC-DAM) is an experimental framework for evaluating deadline-constrained message transmission over a single GEO satellite link subject to time-varying atmospheric attenuation. It combines delayed-feedback forecasting, conformal risk calibration, adaptive coding and modulation (ACM), power adaptation, packet-level erasure protection, interleaving, and deadline-bounded transmission deferral.

This is the repository's only Markdown document. It is the canonical reference for the project layout, current validation status, and reproducible commands.

## Scope

The framework evaluates a joint action over DVB-S2-like MODCOD selection, transmit-power steps, packet redundancy, interleaver depth, duplication, and deferral. For each message, the optimizer seeks the lowest-overhead feasible action subject to a delivery-by-deadline reliability target. When no candidate satisfies that target, it returns an explicit infeasibility diagnosis.

The simulator supports synthetic attenuation traces and ITU-R-inspired channel models. Results from synthetic channels are implementation evidence only; they are not a substitute for validation against field measurements or a basis for deployment claims.

## Architecture

```
Delayed channel telemetry
          |
          v
Quantile forecasting --> stratified conformal calibration
          |                         |
          +------------+------------+
                       v
      deadline-aware joint optimizer
       /       |        |        \\
   MODCOD    power   protection  deferral
                       |
                       v
       frame-level ground-truth evaluation
```

Key implementation areas:

- `rcdam/attenuation.py`, `rcdam/itur.py`, `rcdam/data.py`, and `rcdam/beacon_loader.py`: channel generation, ITU-R-inspired propagation support, chronological data handling, and beacon ingestion.
- `rcdam/windows.py`, `rcdam/forecaster_baseline.py`, `rcdam/forecaster_nn.py`, and `rcdam/forecaster.py`: non-leaking horizon windows and quantile forecasters.
- `rcdam/calibration.py`, `rcdam/conformal.py`, and `rcdam/risk_control.py`: split/stratified conformal calibration and risk-control utilities.
- `rcdam/link.py`, `rcdam/arq.py`, `rcdam/erasure.py`, `rcdam/cost.py`, and `rcdam/window_profile.py`: link, retransmission, packet-protection, cost, and window-profile models.
- `rcdam/optimizer.py`, `rcdam/evaluator.py`, `rcdam/groundtruth.py`, `rcdam/variants.py`, and `rcdam/ablations.py`: decision optimization, evaluation, frame-level simulation, standardized variants, and ablations.
- `rcdam/api.py`, `rcdam/api_models.py`, and `run_api.py`: optional FastAPI interface.

## Project layout

```
rcdam-project/
├── rcdam/                 # Core simulator, policies, optimizer, and API
├── scripts/               # Reproducible experiments and figure generation
│   └── legacy/            # Historical step scripts; not for new experiments
├── tests/                 # Unit, regression, API, and infrastructure tests
├── outputs/               # Generated CSV files, models, and figures (when run)
├── demo.py                # End-to-end demonstration
├── run_api.py             # API server entry point
└── requirements.txt       # Python dependencies
```

## Setup

Use Python 3.10 or later in an isolated environment.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Validation and experiments

Run commands from this directory (`rcdam-project/`). Outputs are written by the scripts to the configured output location.

```powershell
# Complete test suite
python tests/run_all_tests.py

# End-to-end demonstration
python demo.py

# Core experiments
python scripts/run_headroom.py --seeds 5 --hours 12.0 --workers 4
python scripts/train_forecaster.py --hours 12.0 --epochs 12
python scripts/calibrate.py --hours 12.0 --thin-stride 300
python scripts/run_shift_test.py --hours 12.0
python scripts/run_h1_h4.py --seeds 3 --hours 6.0 --interval 45.0

# Reliability, infeasibility, and robustness evaluations
python scripts/run_matched_reliability.py
python scripts/run_infeasible_eval.py
python scripts/run_risk_validation.py
python scripts/run_stress_tests.py

# Generate figures from experiment outputs
python scripts/make_figures.py
```

Some scripts provide additional command-line options; inspect them with `python <script> --help`. Run competing variants with identical traces, workloads, deadlines, and seed schedules. Keep training, calibration, and evaluation periods separate.

## Current evidence and limitations

The repository includes regression and infrastructure tests, including tests for data/link handling, forecast windows, calibration, evaluator and optimizer behavior, ground-truth simulation, ablations, risk validation, API contracts, and Phase 1 regression fixes.

The recorded project status establishes Phase 1 regression fixes. Later evaluation phases and claims about comparative performance, calibrated coverage across every fade state, infeasibility-detection power, or the contribution of individual mitigation levers require regenerated results under the corresponding evaluation scripts. In particular, historical ablation outputs with identical configurations must not be interpreted as evidence that those levers improve performance.

For an IEEE submission, archive the exact configuration, random seeds, raw CSV outputs, generated figures, dependency versions, and test logs alongside every reported result. Do not present synthetic-channel results as real-world validation.

## API (optional)

Start the local API server after installing the requirements:

```powershell
python run_api.py
```

Refer to the OpenAPI interface exposed by the running FastAPI service for endpoint schemas.

## Citation and research use

This codebase is a research prototype. Before publishing quantitative claims, reproduce the experiment, retain the raw outputs, and verify that reported metrics use a shared workload and channel trace across compared methods.
