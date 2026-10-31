"""DVB-S2 MODCOD table and selection.

(name, spectral efficiency [bit/symbol], required Es/N0 [dB] for quasi-error-free
decoding, normal FECFRAME, AWGN).

IMPORTANT: values were entered from memory of ETSI EN 302 307 (DVB-S2). Verify every
row against the standard before you publish anything. For DVB-S2X, use EN 302 307-2.
"""
import numpy as np

MODCODS = [
    ("QPSK 1/4", 0.490243, -2.35),
    ("QPSK 1/3", 0.656448, -1.24),
    ("QPSK 2/5", 0.789412, -0.30),
    ("QPSK 1/2", 0.988858, 1.00),
    ("QPSK 3/5", 1.188304, 2.23),
    ("QPSK 2/3", 1.322253, 3.10),
    ("QPSK 3/4", 1.487473, 4.03),
    ("QPSK 4/5", 1.587196, 4.68),
    ("QPSK 5/6", 1.654663, 5.18),
    ("QPSK 8/9", 1.766451, 6.20),
    ("QPSK 9/10", 1.788612, 6.42),
    ("8PSK 3/5", 1.779991, 5.50),
    ("8PSK 2/3", 1.980636, 6.62),
    ("8PSK 3/4", 2.228124, 7.91),
    ("8PSK 5/6", 2.478562, 9.35),
    ("8PSK 8/9", 2.646012, 10.69),
    ("8PSK 9/10", 2.679207, 10.98),
    ("16APSK 2/3", 2.637201, 8.97),
    ("16APSK 3/4", 2.966728, 10.21),
    ("16APSK 4/5", 3.165623, 11.03),
    ("16APSK 5/6", 3.300184, 11.61),
    ("16APSK 8/9", 3.523143, 12.89),
    ("16APSK 9/10", 3.567342, 13.13),
    ("32APSK 3/4", 3.703295, 12.73),
    ("32APSK 4/5", 3.951571, 13.64),
    ("32APSK 5/6", 4.119540, 14.28),
    ("32APSK 8/9", 4.397854, 15.69),
    ("32APSK 9/10", 4.453027, 16.05),
]


class ModcodTable:
    def __init__(self, entries=MODCODS):
        entries = sorted(entries, key=lambda e: e[2])      # ascending required Es/N0
        self.names = [e[0] for e in entries]
        self.eff = np.array([e[1] for e in entries])
        self.thr = np.array([e[2] for e in entries])
        # best_upto[k] = index of the most efficient MODCOD among entries 0..k
        best = np.zeros(len(entries), dtype=int)
        cur = 0
        for k in range(len(entries)):
            if self.eff[k] > self.eff[cur]:
                cur = k
            best[k] = cur
        self.best_upto = best

        # Precompute per-MODCOD LDPC waterfall interpolation tables (Es/N0 grid vs log10 BLER)
        # Offset from QEF threshold: delta = esn0 - thr
        self._delta_grid = np.array([-1.5, -0.8, -0.5, -0.3, -0.15, -0.05, 0.0, 0.2, 0.5])
        self._log10_bler_profile = np.array([0.0, 0.0, -0.3, -1.0, -2.5, -5.0, -7.0, -9.0, -12.0])

    def select(self, esn0_db: np.ndarray) -> np.ndarray:
        """Most efficient MODCOD decodable at the given Es/N0; -1 where none is."""
        k = np.searchsorted(self.thr, esn0_db, side="right") - 1
        return np.where(k >= 0, self.best_upto[np.maximum(k, 0)], -1)

    def bler(self, idx, esn0_db, slope_decades_per_db=15.0, qef_log10=-7.0, use_curves=False):
        """Block-error probability.

        When use_curves=True: Evaluates per-MODCOD LDPC waterfall curves via linear
        interpolation in the log10(BLER) domain over an Es/N0 grid.
        When use_curves=False (default): Evaluates standard smooth exponential slope model.
        """
        idx = np.asarray(idx, dtype=int)
        esn0_db = np.asarray(esn0_db, dtype=float)
        idx_clamped = np.maximum(idx, 0)

        if not use_curves:
            exponent = qef_log10 - slope_decades_per_db * (esn0_db - self.thr[idx_clamped])
            return np.power(10.0, np.minimum(exponent, 0.0))

        # Per-MODCOD curves with linear interpolation in log10 BLER
        delta = esn0_db - self.thr[idx_clamped]
        log10_bler = np.interp(delta, self._delta_grid, self._log10_bler_profile,
                               left=0.0, right=-12.0)
        return np.power(10.0, np.minimum(log10_bler, 0.0))

    def verify_against_standard(self, csv_path: str, tol_eff: float = 1e-4, tol_esn0: float = 0.01) -> bool:
        """Verifies every row of MODCODS against the transcribed standard CSV."""
        import csv
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {r["name"]: (float(r["spectral_efficiency_bps_per_hz"]), float(r["required_esn0_qef_db"]))
                    for r in reader}

        for name, eff, thr in zip(self.names, self.eff, self.thr):
            if name not in rows:
                raise ValueError(f"MODCOD {name} not found in standard CSV.")
            std_eff, std_thr = rows[name]
            if abs(eff - std_eff) > tol_eff:
                raise ValueError(f"MODCOD {name} efficiency mismatch: table={eff}, standard={std_eff}")
            if abs(thr - std_thr) > tol_esn0:
                raise ValueError(f"MODCOD {name} threshold mismatch: table={thr}, standard={std_thr}")
        return True
