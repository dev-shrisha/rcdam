"""Window-maximum conformal bound: delay-, span-, deferral- and trend-aware (v2).

A message occupies a transmission WINDOW, and a frame is lost when attenuation anywhere in the
window exceeds what its MODCOD tolerates. The guarantee that matters is therefore on the window
maximum, not on one single-horizon sample.

Nonconformity score   s = max_{t in [t0+offset, t0+offset+L]} A(t) - A_last_measured
  * A_last_measured is the FEEDBACK-DELAYED measurement (age = RTT), so feedback delay is part
    of every score.
  * offset = deferral time (0, 0.5, 1.0, 1.5 s): the window starts that much later.
  * L      = window length, calibrated at the UPPER edge of its length bucket (conservative).
Strata: (fade state) x (length bucket) x (deferral offset) x (fade trend: falling/flat/rising).
Bound : A_last + Q_{ceil((n+1)(1-alpha))/n}(scores in stratum)  (split conformal; marginal
        coverage >= 1-alpha within each stratum under exchangeability of calibration/test windows).
Back-off when a stratum has < min_stratum scores:  drop trend -> pool lengths -> pool all.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

STATE_EDGES = (1.0, 3.0, 8.0)                  # dB -> 4 fade states
LEN_EDGES_S = (0.5, 1.0, 2.0, 4.0)             # window-length buckets (upper edges; last = 8 s)
LEN_UPPER_S = (0.5, 1.0, 2.0, 4.0, 8.0)
OFFSETS_S = (0.0, 0.5, 1.0, 1.5)               # supported deferral offsets (s)
TREND_EDGES = (-0.02, 0.02)                    # dB/s: falling / flat / rising
SPAN_EDGES_S = LEN_EDGES_S                     # backward-compat alias

Key = Tuple[int, int, int, int]


def fade_state(att_db: float) -> int:
    return int(np.searchsorted(STATE_EDGES, att_db, side="right"))


def span_bucket(span_s: float) -> int:
    return int(np.searchsorted(LEN_EDGES_S, span_s, side="left"))


def offset_bucket(offset_s: float) -> int:
    return int(np.argmin([abs(offset_s - o) for o in OFFSETS_S]))


def trend_bucket(slope_db_per_s: Optional[float]) -> int:
    if slope_db_per_s is None:
        return 1
    return int(np.searchsorted(TREND_EDGES, slope_db_per_s, side="right"))


@dataclass
class WindowConformal:
    min_stratum: int = 30
    scores: Dict[Key, List[float]] = field(default_factory=dict)
    supports_offsets: bool = True
    _cache: Dict = field(default_factory=dict, repr=False)

    # ---- calibration -------------------------------------------------------
    def add(self, last_meas_db: float, window_max_db: float, span_s: float,
            offset_s: float = 0.0, slope_db_per_s: Optional[float] = None) -> None:
        key = (fade_state(last_meas_db), span_bucket(span_s), offset_bucket(offset_s),
               trend_bucket(slope_db_per_s))
        self.scores.setdefault(key, []).append(float(window_max_db - last_meas_db))
        self._cache.clear()

    def fit_from_trace(self, att: np.ndarray, dt_s: float, delay_steps: int,
                       start_offset_steps=None, span_s_grid=LEN_UPPER_S,
                       stride: int = 20, offsets_s=OFFSETS_S) -> None:
        """Harvest scores from a CALIBRATION trace (never from test traces)."""
        n = len(att)
        for span_s in span_s_grid:
            w = int(round(span_s / dt_s))
            for off in offsets_s:
                o = int(round(off / dt_s))
                for t in range(max(delay_steps, 600), n - w - o - 1, stride):
                    last = att[t - delay_steps]
                    slope = (last - att[max(0, t - delay_steps - 50)]) / 5.0
                    self.add(last, float(np.max(att[t + o: t + o + w])), span_s, off, slope)

    # ---- inference ---------------------------------------------------------
    def _pool(self, state, sb, ob, tb, level: int) -> List[float]:
        ck = (state, sb, ob, tb, level)
        if ck in self._cache:
            return self._cache[ck]
        out: List[float] = []
        for (s, b, o, t), v in self.scores.items():
            if s != state or o != ob:
                continue
            if level == 0 and (b == sb and t == tb):
                out += v
            elif level == 1 and b == sb:
                out += v
            elif level == 2:
                out += v
        self._cache[ck] = out
        return out

    def quantile(self, state: int, bucket: int, alpha: float, offset_b: int = 0,
                 trend_b: int = 1) -> float:
        s: List[float] = []
        for lvl in (0, 1, 2):
            s = self._pool(state, bucket, offset_b, trend_b, lvl)
            if len(s) >= self.min_stratum:
                break
        if len(s) < self.min_stratum:                       # last resort: same offset, any state
            s = [x for (st, b, o, t), v in self.scores.items() if o == offset_b for x in v]
        if not s:
            return 3.0
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        return float(np.quantile(s, level, method="higher"))

    def bound(self, last_meas_db: float, span_s: float, alpha: float,
              offset_s: float = 0.0, slope_db_per_s: Optional[float] = None) -> float:
        ob = offset_bucket(offset_s) if self.supports_offsets else 0
        q = self.quantile(fade_state(last_meas_db), span_bucket(span_s), alpha, ob,
                          trend_bucket(slope_db_per_s))
        return max(0.0, last_meas_db + q)

    # ---- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"format": 2, "scores": {",".join(map(str, k)): v
                                               for k, v in self.scores.items()}}, f)

    @classmethod
    def load(cls, path: str) -> "WindowConformal":
        wc = cls()
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and raw.get("format") == 2:
            for k, v in raw["scores"].items():
                wc.scores[tuple(int(x) for x in k.split(","))] = v
        else:                                              # legacy 2-tuple file (no offsets/trend)
            wc.supports_offsets = False
            for k, v in raw.items():
                a, b = k.split(",")
                wc.scores[(int(a), int(b), 0, 1)] = v
        return wc
