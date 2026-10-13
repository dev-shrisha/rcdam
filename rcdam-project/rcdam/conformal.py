"""Module 6: Stratified Conformal Calibrator with Delayed Feedback.

Implements ARCHITECTURE.md Sections 4, 6, and 9:
- Calibrate on channel attenuation (unbiased, observed regardless of action).
- Delayed feedback: observations arrive RTT steps late; predictions are buffered in a FIFO queue.
- Stratified by fade state (Clear, Light, Moderate, Deep) so coverage is guaranteed in deep fades.
- Generates calibrated attenuation envelope and CDF adjustments with controlled miss rate.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Fade strata thresholds (dB)
STATE_CLEAR = 0     # < 1.0 dB
STATE_LIGHT = 1     # 1.0 - 3.0 dB
STATE_MODERATE = 2  # 3.0 - 8.0 dB
STATE_DEEP = 3      # >= 8.0 dB

STATE_NAMES = ["Clear (<1dB)", "Light (1-3dB)", "Moderate (3-8dB)", "Deep (>=8dB)"]


def classify_fade_state(att_db: float) -> int:
    """Classify attenuation into fade stratum."""
    if att_db < 1.0:
        return STATE_CLEAR
    elif att_db < 3.0:
        return STATE_LIGHT
    elif att_db < 8.0:
        return STATE_MODERATE
    else:
        return STATE_DEEP


@dataclass
class PendingPrediction:
    step_tx: int
    stratum: int
    raw_upper: float
    calibrated_upper: float


class StratifiedConformalCalibrator:
    """Online stratified conformal calibrator with delayed feedback."""
    def __init__(self,
                 target_miss_rate: float = 0.05,
                 feedback_delay_steps: int = 5,
                 buffer_size_per_stratum: int = 1000,
                 aci_gamma: float = 0.005,
                 initial_margin_db: float = 0.5):
        self.epsilon = target_miss_rate
        self.delay_steps = feedback_delay_steps
        self.buffer_size = buffer_size_per_stratum
        self.gamma = aci_gamma

        # Buffers of non-conformity scores (residuals: y_true - raw_pred) per stratum
        self.residuals = {s: deque(maxlen=self.buffer_size) for s in range(4)}
        for s in range(4):
            # Seed with modest positive residual to start conservatively
            self.residuals[s].extend([initial_margin_db] * 20)

        # Adaptive alpha per stratum (Adaptive Conformal Inference)
        self.alpha_adaptive = {s: self.epsilon for s in range(4)}

        # FIFO queue for delayed feedback: holds (step_tx, PendingPrediction)
        self.pending_queue = deque()

        # Tracking metrics
        self.history_true = {s: [] for s in range(4)}
        self.history_covered = {s: [] for s in range(4)}

    def get_stratum_correction(self, stratum: int) -> float:
        """Compute conformal quantile correction q_s for the given stratum."""
        res_list = list(self.residuals[stratum])
        n = len(res_list)
        if n == 0:
            return 1.0
        alpha_eff = np.clip(self.alpha_adaptive[stratum], 0.001, 0.40)
        # Standard conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, (n + 1) * (1.0 - alpha_eff) / n)
        q = float(np.quantile(res_list, level))
        return max(q, 0.0)

    def predict_envelope(self, step_t: int, raw_upper_quantile: float,
                         recent_att_db: float) -> float:
        """Predict calibrated upper bound for step_t and enqueue for delayed feedback update."""
        stratum = classify_fade_state(recent_att_db)
        correction = self.get_stratum_correction(stratum)
        calibrated_upper = raw_upper_quantile + correction

        # Enqueue pending prediction
        self.pending_queue.append(PendingPrediction(
            step_tx=step_t,
            stratum=stratum,
            raw_upper=raw_upper_quantile,
            calibrated_upper=calibrated_upper,
        ))
        return float(calibrated_upper)

    def calibrate_quantiles(self, raw_quantiles: np.ndarray,
                            recent_att_db: float) -> np.ndarray:
        """Apply stratified conformal shift to the multi-quantile forecast array."""
        stratum = classify_fade_state(recent_att_db)
        correction = self.get_stratum_correction(stratum)
        return raw_quantiles + correction

    def receive_feedback(self, current_step: int, observed_att_db: float):
        """Process delayed channel attenuation measurement that arrived at current_step."""
        while self.pending_queue:
            front = self.pending_queue[0]
            if current_step - front.step_tx >= self.delay_steps:
                pred = self.pending_queue.popleft()
                stratum = pred.stratum

                # Residual: y_true - raw_upper
                residual = observed_att_db - pred.raw_upper
                self.residuals[stratum].append(residual)

                # Did the calibrated envelope cover the true observation?
                covered = bool(observed_att_db <= pred.calibrated_upper)
                self.history_true[stratum].append(observed_att_db)
                self.history_covered[stratum].append(covered)

                # Adaptive Conformal Inference (ACI) step per stratum:
                # If missed (covered = False, err = 1): increase alpha => increase margin
                err = 0.0 if covered else 1.0
                curr_alpha = self.alpha_adaptive[stratum]
                new_alpha = curr_alpha + self.gamma * (self.epsilon - err)
                self.alpha_adaptive[stratum] = float(np.clip(new_alpha, 0.001, 0.40))
            else:
                break

    def get_coverage_report(self) -> Dict[str, Dict[str, float]]:
        """Return empirical coverage and sample count per fade stratum."""
        report = {}
        for s in range(4):
            cov_list = self.history_covered[s]
            n_samples = len(cov_list)
            coverage = float(np.mean(cov_list)) if n_samples > 0 else float("nan")
            miss_rate = 1.0 - coverage if n_samples > 0 else float("nan")
            report[STATE_NAMES[s]] = {
                "stratum_id": s,
                "n_samples": n_samples,
                "empirical_coverage": coverage,
                "empirical_miss_rate": miss_rate,
                "target_coverage": 1.0 - self.epsilon,
                "adaptive_alpha": self.alpha_adaptive[s],
            }
        return report
