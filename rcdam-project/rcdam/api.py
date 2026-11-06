"""FastAPI Application for RC-DAM (Risk-Calibrated, Deadline-Aware Mitigation).

Connects frontend applications, satellite terminals, and simulation harnesses
to the RC-DAM decision engine via standard JSON REST endpoints.
"""
import os
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api_models import (
    DecisionRequest,
    DecisionResponse,
    ActionModel,
    ReportOutcomeRequest,
    ReportOutcomeResponse,
    CalibrationStatusResponse,
    QuantilePredictionsResponse,
    StatsResponse,
    HealthResponse,
)
from .api_state import ApiState
from .optimizer import Message, Action
from .calibration import fade_state
from .variants import PolicyContext, policy_v6_infeasible_rule, HORIZONS, ALPHAS
from .engine import decide as engine_decide

app = FastAPI(
    title="RC-DAM Decision Engine API",
    description="REST API for Risk-Calibrated, Deadline-Aware Mitigation on GEO Satellite Downlinks",
    version="1.0.0",
)

# Enable CORS for web frontend integration
# Set RCDAM_CORS_ORIGINS="https://your-frontend.com,http://localhost:5173" in production.
_origins = [o.strip() for o in os.environ.get("RCDAM_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,  # wildcard origin + credentials is unsafe and unnecessary (no cookies used)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize application state
state = ApiState()


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health and model status probe."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=state.forecaster_loaded,
    )


@app.post("/api/v1/decide", response_model=DecisionResponse, tags=["Optimization"])
def decide_action(req: DecisionRequest):
    """Computes the optimal MODCOD, power, parity, deferral, and interleaving for a message arrival."""
    # Build Message object
    msg = Message(
        size_bits=req.message_size_bits,
        priority=req.priority,
        deadline_s=req.deadline_s,
        eps=req.eps,
        arrival_idx=0,
    )

    # Features
    feats = req.features or {}
    f_dict = {
        "cur": np.array([req.current_attenuation_db]),
        "slope5": np.array([feats.get("slope5", 0.0)]),
        "slope30": np.array([feats.get("slope30", 0.0)]),
        "max60": np.array([feats.get("max60", req.current_attenuation_db)]),
    }

    ctx = PolicyContext(
        arrival_idx=0,
        last_meas_idx=0,
        cur_att_db=req.current_attenuation_db,
        features=f_dict,
        true_future_att=None,
        msg=msg,
        sc=state.sc,
        table=state.table,
        forecaster=state.forecaster,
        calibration_corr=state.cal_corr,
    )

    if req.engine == "window_conformal":
        dec = engine_decide(ctx, state.wc, state.engine_params, rule=req.infeasibility_rule)
    else:
        dec = policy_v6_infeasible_rule(ctx, rule=req.infeasibility_rule)
    log = dec.decision_log or {}

    action_model = None
    if dec.action is not None:
        action_model = ActionModel(
            modcod=dec.action.modcod,
            modcod_name=state.table.names[dec.action.modcod],
            spectral_eff=float(state.table.eff[dec.action.modcod]),
            power_db=dec.action.power_db,
            parity_frac=dec.action.parity_frac,
            defer_s=dec.action.defer_s,
            interleave_s=dec.action.interleave_s,
            dup=dec.action.dup,
        )

    # Record decision in thread-safe state store
    d_id = state.record_decision({
        "feasible": dec.feasible,
        "action": dec.action,
        "predicted_p_lb": dec.p_lower_bound,
        "predicted_lb": dec.p_lower_bound,
        "predicted_p": log.get("predicted_p"),
        "overhead": dec.overhead,
        "reason": dec.reason,
        "rule": req.infeasibility_rule,
        "req": req.model_dump(),
    })

    return DecisionResponse(
        decision_id=d_id,
        feasible=dec.feasible,
        action=action_model,
        predicted_delivery_prob=float(dec.p_lower_bound),
        overhead=float(dec.overhead) if np.isfinite(dec.overhead) else 999.0,
        reason=dec.reason,
        infeasibility_rule=req.infeasibility_rule,
        decision_log=dec.decision_log,
        engine=req.engine,
        requested_eps=req.eps,
        predicted_p=log.get("predicted_p"),
        calibrated_lower_bound=float(dec.p_lower_bound),
        window_bound_db=log.get("window_bound_db"),
        meets_original_requirement=log.get("meets_original_requirement", dec.feasible),
        relaxed_eps=log.get("relaxed_eps"),
        warning=log.get("warning"),
        infeasibility=log.get("infeasibility"),
    )


@app.post("/api/v1/report_outcome", response_model=ReportOutcomeResponse, tags=["Telemetry"])
def report_outcome(req: ReportOutcomeRequest):
    """Terminal telemetry feedback reporting whether a decision was successfully delivered."""
    if req.decision_id not in state.decisions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Decision ID '{req.decision_id}' not found in active decision store"
        )

    state.record_outcome(req.model_dump())
    return ReportOutcomeResponse(status="recorded", decision_id=req.decision_id)


@app.get("/api/v1/calibration/status", response_model=CalibrationStatusResponse, tags=["Calibration"])
def calibration_status():
    """Returns current conformal calibration matrix status and horizons."""
    return CalibrationStatusResponse(
        calibrated=state.calibrated,
        fade_states=[0, 1, 2, 3],
        horizons=list(state.cal_corr.keys()) if state.cal_corr else HORIZONS,
        n_horizons=len(state.cal_corr) if state.cal_corr else len(HORIZONS),
        calibration_file=state.calibration_file,
    )


@app.get("/api/v1/predictions/{attenuation_db}", response_model=QuantilePredictionsResponse, tags=["Forecasting"])
def get_predictions(attenuation_db: float):
    """Retrieves multi-horizon raw quantile forecasts and conformal corrections for a given attenuation."""
    st = fade_state(attenuation_db)
    q_preds = {}
    c_corrs = {}

    for i, H in enumerate(HORIZONS[:4]):  # first 4 key horizons
        h_key = f"horizon_{H}_steps"
        q_preds[h_key] = {}
        c_corrs[h_key] = {}
        for a in [0.1, 0.05, 0.01]:
            tau = 1.0 - a
            q_raw = float(state.forecaster.predict(np.array([attenuation_db]), i, tau)[0])
            corr = state.cal_corr.get(H, {}).get(tau, {}).get(st, 0.0)
            q_preds[h_key][f"q_{int(tau*100)}"] = round(q_raw, 2)
            c_corrs[h_key][f"q_{int(tau*100)}"] = round(corr, 2)

    return QuantilePredictionsResponse(
        current_attenuation_db=attenuation_db,
        fade_state=st,
        quantile_predictions=q_preds,
        calibration_corrections=c_corrs,
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["System"])
def system_stats():
    """Retrieves operational telemetry, total decisions, delivery rates, and mean overhead."""
    return StatsResponse(**state.get_stats())


# Serve the operator console at http://localhost:8000/ui/
_ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(_ui_dir):
    app.mount("/ui", StaticFiles(directory=_ui_dir, html=True), name="ui")
