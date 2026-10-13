"""Tests for Step 3: ITU-R calibration and ETSI EN 302 307 MODCOD verification."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcdam.link import ModcodTable, MODCODS
from rcdam.itur import (
    EarthStationConfig,
    itur_p618_rain_attenuation,
    calibrate_maseng_bakken_to_itu_r,
    specific_attenuation_itu_r_p838,
)


def test_modcod_dvbs2_verification():
    """Verify MODCOD table matches ETSI EN 302 307 standard parameters."""
    t = ModcodTable()
    assert len(t.names) == 28, f"Expected 28 standard MODCODs, got {len(t.names)}"

    # 1. Assert threshold monotonically non-decreasing
    assert np.all(np.diff(t.thr) >= 0), "Required Es/N0 must be sorted in ascending order"

    # 2. Assert efficiency ranges
    assert t.eff.min() >= 0.49
    assert t.eff.max() <= 4.46

    # 3. Check representative standard rows from ETSI EN 302 307 Table 13 (QEF at AWGN)
    expected_samples = {
        "QPSK 1/4": (-2.35, 0.490243),
        "QPSK 1/2": (1.00, 0.988858),
        "8PSK 3/4": (7.91, 2.228124),
        "16APSK 5/6": (11.61, 3.300184),
        "32APSK 9/10": (16.05, 4.453027),
    }

    for name, (req_esn0, eff) in expected_samples.items():
        assert name in t.names, f"{name} missing from ModcodTable"
        idx = t.names.index(name)
        assert abs(t.thr[idx] - req_esn0) < 1e-4, f"{name} req Es/N0 mismatch: {t.thr[idx]} vs {req_esn0}"
        assert abs(t.eff[idx] - eff) < 1e-4, f"{name} eff mismatch: {t.eff[idx]} vs {eff}"

    print("ETSI EN 302 307 MODCOD verification passed.")


def test_itur_p618_attenuation():
    """Verify ITU-R P.618 rain attenuation calculation produces realistic Ka-band values."""
    cfg = EarthStationConfig(
        freq_ghz=20.0,
        elevation_deg=35.0,
        station_lat_deg=45.0,
        station_alt_km=0.2,
        rain_rate_001_mmh=35.0,
    )
    p_percentages = np.array([0.001, 0.01, 0.1, 1.0, 5.0])
    a_db = itur_p618_rain_attenuation(p_percentages, cfg)

    # Sanity checks:
    # 1. Attenuation increases as exceedance probability decreases
    assert np.all(np.diff(a_db) <= 0), "A(p) must decrease as p increases"

    # 2. For 20 GHz Ka-band at 35 deg elevation and R0.01=35mm/h:
    # A_0.01 is typically between 12 and 22 dB
    a_001 = a_db[1]
    assert 10.0 <= a_001 <= 25.0, f"Unrealistic A_0.01: {a_001:.2f} dB"

    # A_0.1 is typically between 3 and 10 dB
    a_01 = a_db[2]
    assert 3.0 <= a_01 <= 12.0, f"Unrealistic A_0.1: {a_01:.2f} dB"

    # A_1.0 is typically between 0.5 and 4 dB
    a_1 = a_db[3]
    assert 0.5 <= a_1 <= 5.0, f"Unrealistic A_1: {a_1:.2f} dB"

    print(f"ITU-R P.618 checks passed (A_0.01 = {a_001:.2f} dB, A_0.1 = {a_01:.2f} dB, A_1 = {a_1:.2f} dB).")


def test_maseng_bakken_calibration():
    """Verify calibration of dynamic Maseng-Bakken parameters."""
    cfg = EarthStationConfig(freq_ghz=20.0, elevation_deg=35.0, rain_rate_001_mmh=35.0)
    calib = calibrate_maseng_bakken_to_itu_r(cfg)

    assert "p_rain" in calib
    assert "rain_ln_mean" in calib
    assert "rain_ln_std" in calib
    assert calib["rain_ln_std"] > 0.5, "Standard deviation should be reasonable"
    assert calib["rain_beta_per_s"] == 2e-4

    print(f"Maseng-Bakken calibration passed: {calib}")


if __name__ == "__main__":
    test_modcod_dvbs2_verification()
    test_itur_p618_attenuation()
    test_maseng_bakken_calibration()
    print("All Step 3 unit tests passed.")
