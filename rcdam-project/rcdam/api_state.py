"""Thread-safe application state and decision store for the RC-DAM API service."""
import os
import threading
import uuid
from collections import OrderedDict
from typing import Dict, Any, Optional, List
import numpy as np

from .config import Scenario, preset_base
from .link import ModcodTable
from .forecaster_baseline import EmpiricalQuantileForecaster
from .calibration import load_calibration_json, fade_state
from .variants import HORIZONS, ALPHAS
from .window_conformal import WindowConformal
from .engine import EngineParams
from .reliability import cp_lower


class ApiState:
    MAX_STORED = 10000  # cap in-memory decision/outcome history

    def __init__(self, out_dir: Optional[str] = None):
        if out_dir is None:
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
        self.out_dir = out_dir
        self.table = ModcodTable()
        self.sc = preset_base()

        self._lock = threading.Lock()
        self.decisions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.outcomes: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        # Load trained forecaster if present
        fc_path = os.path.join(out_dir, "forecaster_baseline.json")
        if os.path.exists(fc_path):
            self.forecaster = EmpiricalQuantileForecaster.load(fc_path)
            self.forecaster_loaded = True
        else:
            # Fallback fit
            train_cur = np.linspace(0.0, 30.0, 500)
            train_Z = np.column_stack([np.linspace(0.1, 35.0, 500) for _ in range(8)])
            self.forecaster = EmpiricalQuantileForecaster().fit(train_cur, train_Z)
            self.forecaster_loaded = False

        # Load calibration if present
        cal_path = os.path.join(out_dir, "calibration.json")
        if os.path.exists(cal_path):
            _, self.cal_corr = load_calibration_json(cal_path)
            self.calibrated = True
            self.calibration_file = cal_path
        else:
            self.cal_corr = {}
            self.calibrated = False
            self.calibration_file = "none"

        # Window-level conformal calibration (drives the default decision engine)
        self.engine_params = EngineParams()
        self.wc = self._load_or_fit_wc(out_dir)

    def _load_or_fit_wc(self, out_dir: str) -> WindowConformal:
        """Load outputs/window_conformal_api.json, or fit it once from simulated calibration traces."""
        from dataclasses import replace
        from .attenuation import generate_attenuation_db
        path = os.path.join(out_dir, "window_conformal_api.json")
        if os.path.exists(path):
            wc = WindowConformal.load(path)
            if wc.supports_offsets:
                return wc
        wc = WindowConformal()
        for s in range(201, 204):                      # seeds disjoint from training/test seeds
            sc = replace(self.sc, seed=s, duration_h=4.0)
            wc.fit_from_trace(generate_attenuation_db(sc), sc.dt_s, sc.feedback_delay_steps)
        try:
            os.makedirs(out_dir, exist_ok=True)
            wc.save(path)
        except OSError:
            pass
        return wc

    def record_decision(self, dec_data: Dict[str, Any]) -> str:
        d_id = str(uuid.uuid4())
        with self._lock:
            self.decisions[d_id] = {**dec_data, "decision_id": d_id}
            while len(self.decisions) > self.MAX_STORED:
                old, _ = self.decisions.popitem(last=False)
                self.outcomes.pop(old, None)
        return d_id

    def record_outcome(self, outcome_data: Dict[str, Any]):
        d_id = outcome_data["decision_id"]
        with self._lock:
            self.outcomes[d_id] = outcome_data
            while len(self.outcomes) > self.MAX_STORED:
                self.outcomes.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            n_tot = len(self.decisions)
            n_feas = sum(1 for d in self.decisions.values() if d.get("feasible"))
            n_infeas = n_tot - n_feas
            n_outcomes = len(self.outcomes)
            n_delivered = sum(1 for o in self.outcomes.values() if o.get("delivered"))

            deliv_rate = (n_delivered / n_outcomes) if n_outcomes > 0 else None
            overheads = [d["overhead"] for d in self.decisions.values() if "overhead" in d and np.isfinite(d["overhead"])]
            mean_ov = float(np.mean(overheads)) if overheads else None

            # live reliability tracking on decisions that CLAIMED feasibility and got an outcome
            cl = [(d, self.outcomes[i]) for i, d in self.decisions.items()
                  if d.get("feasible") and i in self.outcomes]
            n_cl = len(cl)
            n_ok = sum(1 for _, o in cl if o.get("delivered"))
            pp = [d.get("predicted_p") for d, _ in cl if d.get("predicted_p") is not None]
            lb = [d.get("predicted_lb") for d, _ in cl if d.get("predicted_lb") is not None]
            return {
                "claimed_feasible_with_outcome": n_cl,
                "false_feasible": n_cl - n_ok,
                "false_feasible_rate": ((n_cl - n_ok) / n_cl) if n_cl else None,
                "mean_predicted_p": float(np.mean(pp)) if pp else None,
                "mean_calibrated_lb": float(np.mean(lb)) if lb else None,
                "actual_rate_on_claimed_feasible": (n_ok / n_cl) if n_cl else None,
                "validated_lower_bound": cp_lower(n_ok, n_cl) if n_cl else None,
                "total_decisions": n_tot,
                "total_feasible": n_feas,
                "total_infeasible": n_infeas,
                "total_reported_outcomes": n_outcomes,
                "total_delivered": n_delivered,
                "delivery_rate": deliv_rate,
                "mean_overhead": mean_ov,
            }

