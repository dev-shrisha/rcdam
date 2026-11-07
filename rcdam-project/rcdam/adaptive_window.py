"""Online adaptation of the window-maximum conformal bound under channel shift (Option 1).

Two complementary mechanisms (both use ONLY realised window maxima, which become known after
the window ends + feedback delay):

RollingWindowConformal  - sliding-window recalibration: new (last_meas, window_max) scores are added
                          to their stratum and each stratum keeps at most `cap` newest scores, so the
                          quantile tracks the recent channel (adapts in BOTH directions: tighter in
                          calm weather, wider in a harsher climate).
AdaptiveWindowConformal - adaptive conformal inference (Gibbs & Candes): the working miscoverage level
                          alpha_t is nudged by the observed coverage error, alpha_{t+1} = alpha_t +
                          gamma*(alpha - err_t), with a floor so that tiny targets do not collapse.
Both expose the same bound()/update() interface as WindowConformal so they can be dropped into the
engine (engine.decide(ctx, wc=adaptive, ...)).
"""
from collections import deque
from typing import Dict, Optional
import numpy as np

from .window_conformal import (WindowConformal, fade_state, span_bucket, offset_bucket, trend_bucket)


class RollingWindowConformal(WindowConformal):
    """WindowConformal whose strata keep only the `cap` newest scores."""

    def __init__(self, base: WindowConformal, cap: int = 2000, min_stratum: int = 30):
        super().__init__(min_stratum=min_stratum)
        self.supports_offsets = base.supports_offsets
        self.cap = cap
        for k, v in base.scores.items():
            self.scores[k] = list(v[-cap:])

    def update(self, last_meas_db: float, window_max_db: float, span_s: float,
               offset_s: float = 0.0, slope_db_per_s: Optional[float] = None) -> None:
        key = (fade_state(last_meas_db), span_bucket(span_s), offset_bucket(offset_s),
               trend_bucket(slope_db_per_s))
        v = self.scores.setdefault(key, [])
        v.append(float(window_max_db - last_meas_db))
        if len(v) > self.cap:
            del v[: len(v) - self.cap]
        self._cache.clear()


class AdaptiveWindowConformal:
    """Adaptive conformal inference around a (frozen) WindowConformal."""

    def __init__(self, base: WindowConformal, gamma: float = 0.005, floor_ratio: float = 0.05,
                 ceil_alpha: float = 0.5):
        self.base, self.gamma, self.floor_ratio, self.ceil = base, gamma, floor_ratio, ceil_alpha
        self.state: Dict[float, float] = {}

    @property
    def supports_offsets(self) -> bool:
        return self.base.supports_offsets

    def _a(self, alpha: float) -> float:
        return self.state.setdefault(round(alpha, 9), alpha)

    def bound(self, last_meas_db: float, span_s: float, alpha: float, offset_s: float = 0.0,
              slope_db_per_s: Optional[float] = None) -> float:
        return self.base.bound(last_meas_db, span_s, self._a(alpha), offset_s, slope_db_per_s)

    def update(self, last_meas_db: float, window_max_db: float, span_s: float, alpha: float,
               offset_s: float = 0.0, slope_db_per_s: Optional[float] = None) -> None:
        a_t = self._a(alpha)
        err = float(window_max_db > self.base.bound(last_meas_db, span_s, a_t, offset_s, slope_db_per_s))
        new = a_t + self.gamma * (alpha - err)
        self.state[round(alpha, 9)] = float(np.clip(new, alpha * self.floor_ratio, self.ceil))
