"""Conformal time-to-fade (first-passage) prediction and usable-window estimation (NEW).

Question answered:  "From now, for how many seconds can I count on attenuation staying
below a threshold theta (the tolerance of the chosen protection)?"

Score (per calibration instant t0 and threshold theta):
    T = first time after t0 at which A(t) > theta        (censored at H_MAX_S if never)
anchored on the FEEDBACK-DELAYED measurement A_hat = A(t0 - tau).
Lower confidence bound (split conformal, one-sided):
    T_lb(alpha) = k-th smallest score in the stratum, k = floor(alpha (n+1))
    =>  P(T_new >= T_lb) >= 1 - alpha   (exchangeability within a stratum)
If k < 1 (too few scores for this alpha) the bound is 0 s: the system refuses to claim a window.
Strata: (gap bucket of theta - A_hat) x (fade trend bucket). gap <= 0 -> usable time 0 s.
Thresholds are snapped DOWN to the calibration grid (an earlier crossing => conservative).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

H_MAX_S = 600.0
THR_GRID = tuple(np.arange(1.0, 21.5, 0.5))          # dB
GAP_EDGES = (0.5, 1.0, 2.0, 4.0, 8.0)                # (0,.5],(.5,1],(1,2],(2,4],(4,8],>8
TREND_EDGES = (-0.02, 0.02)                          # dB/s: falling / flat / rising


def gap_bucket(gap_db: float) -> int:
    return int(np.searchsorted(GAP_EDGES, gap_db, side="left"))


def trend_bucket(slope: Optional[float]) -> int:
    return 1 if slope is None else int(np.searchsorted(TREND_EDGES, slope, side="right"))


def snap_down(theta_db: float) -> float:
    g = np.asarray(THR_GRID)
    ok = g[g <= theta_db + 1e-9]
    return float(ok.max()) if len(ok) else float("nan")


@dataclass
class TimeToFade:
    min_stratum: int = 50
    scores: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    _sorted: Dict = field(default_factory=dict, repr=False)

    def fit_from_trace(self, att: np.ndarray, dt_s: float, delay_steps: int, stride: int = 20,
                       thr_grid=THR_GRID, horizon_s: float = H_MAX_S) -> None:
        H = int(round(horizon_s / dt_s)); n = len(att)
        thr = np.asarray(thr_grid)
        for t in range(max(delay_steps + 50, 600), n - H - 1, stride):
            last = float(att[t - delay_steps])
            slope = (last - att[t - delay_steps - 50]) / 5.0
            cm = np.maximum.accumulate(att[t:t + H])
            sel = thr[thr > last]
            if len(sel) == 0:
                continue
            idx = np.searchsorted(cm, sel, side="right")           # first index with cm > theta
            T = np.where(idx < H, idx * dt_s, horizon_s)
            tb = trend_bucket(slope)
            for th, tt in zip(sel, T):
                self.scores.setdefault((gap_bucket(th - last), tb), []).append(float(tt))
        self._sorted.clear()

    def _arr(self, key):
        if key not in self._sorted:
            self._sorted[key] = np.sort(np.asarray(self.scores.get(key, []), float))
        return self._sorted[key]

    def t_lower(self, last_meas_db: float, slope_db_per_s: Optional[float], theta_db: float,
                alpha: float) -> float:
        """Seconds for which attenuation stays <= theta with probability >= 1-alpha."""
        th = snap_down(theta_db)
        if not np.isfinite(th) or th <= last_meas_db:
            return 0.0
        gb, tb = gap_bucket(th - last_meas_db), trend_bucket(slope_db_per_s)
        s = self._arr((gb, tb))
        if len(s) < self.min_stratum:                       # back off: pool trends
            s = np.sort(np.concatenate([self._arr((gb, t)) for t in range(3)]))
        n = len(s)
        k = int(np.floor(alpha * (n + 1)))
        if n < self.min_stratum or k < 1:
            return 0.0
        return float(s[k - 1])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({f"{a},{b}": v for (a, b), v in self.scores.items()}, f)

    @classmethod
    def load(cls, path: str) -> "TimeToFade":
        o = cls()
        with open(path) as f:
            for k, v in json.load(f).items():
                a, b = k.split(","); o.scores[(int(a), int(b))] = v
        return o


def protection_tolerance_db(table, sc, modcod: int, power_db: float, margin_db: float = 0.5) -> float:
    """Attenuation (dB) up to which MODCOD m at boost P stays at/below its QEF threshold minus margin."""
    return float(sc.clear_sky_esn0_db + power_db - table.thr[modcod] - margin_db)
