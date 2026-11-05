"""Window-maximum conformal bound with delay- and span-aware stratification (NEW).

Why this exists
---------------
The original calibrator conformalizes attenuation at ONE future horizon, but a
message occupies a transmission *window*; a frame is lost when attenuation
anywhere inside that window exceeds what its MODCOD tolerates. The guarantee that
matters is therefore on the window maximum, not on a single-horizon sample.

Score:   s = max_{t in window} A(t) - A_last_measured          (dB, one-sided)
Strata:  (fade state of last measurement) x (window-length bucket)
Bound:   A_last_measured + Q_{ceil((n+1)(1-alpha))/n}(scores in stratum)

Finite-sample marginal coverage P(window_max <= bound) >= 1-alpha holds within
each stratum for exchangeable calibration/test windows (split conformal).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json
import numpy as np

STATE_EDGES = (1.0, 3.0, 8.0)          # dB -> 4 fade states
SPAN_EDGES_S = (0.5, 1.0, 2.0, 4.0)    # window-length buckets (seconds)


def fade_state(att_db: float) -> int:
    return int(np.searchsorted(STATE_EDGES, att_db, side="right"))


def span_bucket(span_s: float) -> int:
    return int(np.searchsorted(SPAN_EDGES_S, span_s, side="left"))


@dataclass
class WindowConformal:
    min_stratum: int = 30
    scores: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)

    # ---- calibration -------------------------------------------------------
    def add(self, last_meas_db: float, window_max_db: float, span_s: float) -> None:
        key = (fade_state(last_meas_db), span_bucket(span_s))
        self.scores.setdefault(key, []).append(float(window_max_db - last_meas_db))

    def fit_from_trace(self, att: np.ndarray, dt_s: float, delay_steps: int,
                       start_offset_steps: List[int], span_s_grid=(1.0, 2.0, 3.0),
                       stride: int = 20) -> None:
        """Harvest scores from a calibration trace (never from test traces)."""
        n = len(att)
        for span_s in span_s_grid:
            w = int(round(span_s / dt_s))
            for t in range(max(delay_steps, 600), n - w - 1, stride):
                last = att[t - delay_steps]
                self.add(last, float(np.max(att[t: t + w])), span_s)

    # ---- inference ---------------------------------------------------------
    def quantile(self, state: int, bucket: int, alpha: float) -> float:
        s = self.scores.get((state, bucket), [])
        if len(s) < self.min_stratum:
            # back off to the pooled-over-buckets stratum, then to the global pool
            pooled = [x for (st, _), v in self.scores.items() if st == state for x in v]
            s = pooled if len(pooled) >= self.min_stratum else \
                [x for v in self.scores.values() for x in v]
        if not s:
            return 3.0
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        return float(np.quantile(s, level, method="higher"))

    def bound(self, last_meas_db: float, span_s: float, alpha: float) -> float:
        return last_meas_db + max(0.0, self.quantile(fade_state(last_meas_db),
                                                     span_bucket(span_s), alpha))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({f"{k[0]},{k[1]}": v for k, v in self.scores.items()}, f)

    @classmethod
    def load(cls, path: str) -> "WindowConformal":
        wc = cls()
        with open(path) as f:
            for k, v in json.load(f).items():
                a, b = k.split(",")
                wc.scores[(int(a), int(b))] = v
        return wc
