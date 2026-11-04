"""Pydantic data models for the RC-DAM FastAPI backend service."""
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    """Payload to request an optimized deadline-aware transmission decision."""
    message_size_bits: float = Field(20e6, gt=0, description="Payload size in bits")
    priority: Literal["critical", "normal", "best_effort"] = Field("normal", description="'critical', 'normal', or 'best_effort'")
    deadline_s: float = Field(2.0, gt=0, description="End-to-end delivery deadline in seconds")
    eps: float = Field(0.01, gt=0.0, le=0.25, description="Permitted deadline-failure probability epsilon")
    current_attenuation_db: float = Field(..., ge=0.0, description="Estimated/measured channel attenuation in dB")
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional channel features (slope5, slope30, max60)"
    )
    infeasibility_rule: Literal["R1", "R2", "R3", "R4"] = Field(
        "R1",
        description="Rule on INFEASIBLE: R1 (send-and-flag), R2 (defer), R3 (downgrade), R4 (reject)"
    )


class ActionModel(BaseModel):
    """Selected physical, coding, and scheduling levers."""
    modcod: int = Field(..., description="MODCOD index in DVB-S2X table")
    modcod_name: str = Field(..., description="MODCOD standard name (e.g. '16APSK 2/3')")
    spectral_eff: float = Field(..., description="Information bits per symbol")
    power_db: float = Field(0.0, description="Uplink power boost in dB")
    parity_frac: float = Field(0.0, description="Proactive erasure code parity fraction")
    defer_s: float = Field(0.0, description="Intentional transmission delay in seconds")
    interleave_s: float = Field(0.0, description="Interleaver span in seconds")
    dup: int = Field(1, description="Packet duplication factor")


class DecisionResponse(BaseModel):
    """Response containing optimizer decision, action, and risk certificates."""
    decision_id: str = Field(..., description="Unique UUID for this decision")
    feasible: bool = Field(..., description="True if delivery guarantee is certified feasible")
    action: Optional[ActionModel] = Field(None, description="Recommended transmission action")
    predicted_delivery_prob: float = Field(..., description="Calibrated lower bound on delivery probability P*")
    overhead: float = Field(..., description="Relative airtime and power overhead")
    reason: str = Field(..., description="'ok', 'ok_deferred', 'capacity', 'deadline', 'rejected', etc.")
    infeasibility_rule: str = Field(..., description="Active infeasibility rule applied")
    decision_log: Optional[Dict[str, Any]] = Field(None, description="Detailed decision audit trace")


class ReportOutcomeRequest(BaseModel):
    """Ground-truth delivery outcome report from satellite terminal."""
    decision_id: str = Field(..., description="Decision UUID being reported")
    delivered: bool = Field(..., description="True if message was decoded on-time at receiver")
    latency_s: Optional[float] = Field(None, description="End-to-end delivery latency in seconds")
    frames_sent: Optional[int] = Field(None, description="Total physical frames transmitted")


class ReportOutcomeResponse(BaseModel):
    status: str = "recorded"
    decision_id: str


class CalibrationStatusResponse(BaseModel):
    calibrated: bool
    fade_states: List[int]
    horizons: List[int]
    n_horizons: int
    calibration_file: str


class QuantilePredictionsResponse(BaseModel):
    current_attenuation_db: float
    fade_state: int
    quantile_predictions: Dict[str, Dict[str, float]]
    calibration_corrections: Dict[str, Dict[str, float]]


class StatsResponse(BaseModel):
    total_decisions: int
    total_feasible: int
    total_infeasible: int
    total_reported_outcomes: int
    total_delivered: int
    delivery_rate: Optional[float]
    mean_overhead: Optional[float]


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    model_loaded: bool

