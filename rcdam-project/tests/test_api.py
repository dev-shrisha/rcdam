"""Unit and integration tests for the RC-DAM FastAPI backend service."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from rcdam.api import app

client = TestClient(app)


def test_health_endpoint():
    """GET /api/v1/health returns 200 with healthy status."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data


def test_decide_feasible_clearsky():
    """POST /api/v1/decide under clear sky returns feasible decision with highest MODCOD."""
    payload = {
        "message_size_bits": 20e6,
        "priority": "normal",
        "deadline_s": 2.0,
        "eps": 0.01,
        "current_attenuation_db": 0.5,
        "infeasibility_rule": "R1",
    }
    resp = client.post("/api/v1/decide", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["feasible"] is True
    assert data["action"] is not None
    assert data["action"]["modcod"] >= 20  # high MODCOD in clear sky
    assert data["predicted_delivery_prob"] >= 0.99
    assert data["overhead"] >= 1.0
    assert "decision_id" in data


def test_decide_infeasible_fade_and_r4():
    """POST /api/v1/decide under severe fade triggers infeasibility rule."""
    payload = {
        "message_size_bits": 50e6,  # huge message
        "priority": "critical",
        "deadline_s": 1.0,          # tight deadline
        "eps": 0.001,
        "current_attenuation_db": 25.0,  # extreme fade
        "infeasibility_rule": "R4",      # reject rule
    }
    resp = client.post("/api/v1/decide", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["feasible"] is False
    assert data["reason"] == "rejected"
    assert data["action"] is None
    assert data["overhead"] == 0.0


def test_report_outcome_flow():
    """POST /api/v1/report_outcome records feedback for a decision ID and updates stats."""
    # First make a decision
    dec_resp = client.post("/api/v1/decide", json={
        "message_size_bits": 10e6,
        "priority": "normal",
        "deadline_s": 2.0,
        "eps": 0.01,
        "current_attenuation_db": 1.0,
    })
    assert dec_resp.status_code == 200
    d_id = dec_resp.json()["decision_id"]

    # Report delivery outcome
    outcome_resp = client.post("/api/v1/report_outcome", json={
        "decision_id": d_id,
        "delivered": True,
        "latency_s": 0.65,
        "frames_sent": 154,
    })
    assert outcome_resp.status_code == 200
    assert outcome_resp.json()["status"] == "recorded"

    # Verify stats update
    stats_resp = client.get("/api/v1/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_reported_outcomes"] >= 1
    assert stats["total_delivered"] >= 1


def test_calibration_status_endpoint():
    """GET /api/v1/calibration/status returns calibration metadata."""
    resp = client.get("/api/v1/calibration/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "calibrated" in data
    assert "horizons" in data
    assert len(data["horizons"]) > 0


def test_predictions_endpoint():
    """GET /api/v1/predictions/{att} returns quantile forecasts and corrections."""
    resp = client.get("/api/v1/predictions/5.5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_attenuation_db"] == 5.5
    assert "quantile_predictions" in data
    assert "calibration_corrections" in data


def test_input_validation_error():
    """POST /api/v1/decide with invalid input returns 422 Unprocessable Entity."""
    # Negative deadline is invalid
    bad_payload = {
        "message_size_bits": 20e6,
        "priority": "normal",
        "deadline_s": -2.0,
        "eps": 0.01,
        "current_attenuation_db": 1.0,
    }
    resp = client.post("/api/v1/decide", json=bad_payload)
    assert resp.status_code == 422


if __name__ == "__main__":
    test_health_endpoint()
    test_decide_feasible_clearsky()
    test_decide_infeasible_fade_and_r4()
    test_report_outcome_flow()
    test_calibration_status_endpoint()
    test_predictions_endpoint()
    test_input_validation_error()
    print("All API integration tests passed successfully!")

