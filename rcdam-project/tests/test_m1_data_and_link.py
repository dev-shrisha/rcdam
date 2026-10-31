"""Tests for Milestone M1: Data and channel realism."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rcdam.config import Scenario
from rcdam.attenuation import generate_attenuation_db
from rcdam.link import ModcodTable
from rcdam.data import load_attenuation, split_by_time, compute_fade_statistics, save_synthetic_dataset


def test_modcod_table_standard_verification():
    """Verify that every row of ModcodTable matches ETSI EN 302 307."""
    table = ModcodTable()
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rcdam", "data", "etsi_en_302_307.csv"))
    assert os.path.exists(csv_path), f"Standard CSV not found at {csv_path}"
    verified = table.verify_against_standard(csv_path)
    assert verified is True
    print("M1: All MODCODs verified against ETSI EN 302 307 standard CSV.")


def test_bler_per_modcod_curves():
    """Verify per-MODCOD LDPC curves and slope fallback."""
    table = ModcodTable()
    # At QEF threshold, both curve and slope should be near 1e-7
    idx = 3  # QPSK 1/2, thr = 1.0 dB
    thr = table.thr[idx]

    bler_slope = float(table.bler([idx], [thr], use_curves=False)[0])
    bler_curve = float(table.bler([idx], [thr], use_curves=True)[0])
    assert abs(np.log10(bler_slope) - (-7.0)) < 1e-3
    assert abs(np.log10(bler_curve) - (-7.0)) < 1e-3

    # At 2 dB above threshold, BLER should be <= 1e-9
    bler_high = float(table.bler([idx], [thr + 2.0], use_curves=True)[0])
    assert bler_high <= 1e-8

    # At 2 dB below threshold, BLER should be 1.0
    bler_low = float(table.bler([idx], [thr - 2.0], use_curves=True)[0])
    assert abs(bler_low - 1.0) < 1e-3
    print("M1: BLER waterfall curves verified.")


def test_data_loading_and_splitting():
    """Verify time series loading, time-based split, and propagation statistics."""
    sc = Scenario(seed=123, duration_h=1.0)
    att = np.maximum(generate_attenuation_db(sc), 0.0)

    # Save temporary dataset
    tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "tmp_m1_att.csv"))
    save_synthetic_dataset(tmp_path, att, dt_s=sc.dt_s)

    # Load back
    t, loaded_att = load_attenuation(tmp_path, dt_s=sc.dt_s, clear_sky_offset_db=0.0)
    assert len(t) == len(att)
    assert np.allclose(loaded_att, att, atol=1e-4)

    # Chronological split
    tr, cal, te = split_by_time(loaded_att, (0.5, 0.25, 0.25))
    assert len(tr) + len(cal) + len(te) == len(loaded_att)
    assert np.array_equal(loaded_att[:len(tr)], tr)
    assert np.array_equal(loaded_att[len(tr):len(tr)+len(cal)], cal)
    assert np.array_equal(loaded_att[len(tr)+len(cal):], te)

    # Propagation statistics
    stats = compute_fade_statistics(loaded_att, dt_s=sc.dt_s)
    assert "pct_above_1db" in stats
    assert "pct_above_5db" in stats
    assert "slope_p50_db_per_s" in stats
    assert stats["duration_h"] > 0.99
    print(f"M1: Data statistics: Time above 1dB = {stats['pct_above_1db']:.2f}%, slope_p50 = {stats['slope_p50_db_per_s']:.3f} dB/s")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_itur_p1853_attenuation_generator():
    """Verify ITU-R P.1853 rain synthesis mode in generate_attenuation_db."""
    sc = Scenario(seed=999, duration_h=1.0, rain_model="itur_p1853")
    att = generate_attenuation_db(sc)
    assert len(att) == sc.n_steps
    assert np.all(np.isfinite(att))
    print("M1: ITU-R P.1853 attenuation generator verified.")


if __name__ == "__main__":
    test_modcod_table_standard_verification()
    test_bler_per_modcod_curves()
    test_data_loading_and_splitting()
    test_itur_p1853_attenuation_generator()
    print("All M1 tests passed!")
