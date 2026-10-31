"""Unit tests for windows.py (no leakage, off-by-one, brute-force checks)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rcdam.windows import forward_window_max, build_windows


def test_forward_window_max_brute_force():
    """Verify forward_window_max against brute-force slice max on arbitrary sequence."""
    rng = np.random.default_rng(1234)
    a = rng.uniform(0.0, 30.0, size=500)
    for H in [1, 5, 10, 30, 100]:
        z = forward_window_max(a, H)
        assert len(z) == len(a)
        # Brute-force check for interior indices
        for m in range(0, len(a) - H - 1):
            expected = np.max(a[m + 1 : m + H + 1])
            assert abs(z[m] - expected) < 1e-12, f"Mismatch at m={m}, H={H}: got {z[m]}, expected {expected}"
    print("windows.py: forward_window_max brute force check passed.")


def test_no_future_leakage():
    """Verify that features at index m depend ONLY on att[:m+1] and not on future values."""
    rng = np.random.default_rng(5678)
    att1 = rng.uniform(0.0, 20.0, size=2000)
    att2 = att1.copy()

    # Modify all values after index 1200
    m_test = 1000
    att2[m_test + 1:] = rng.uniform(50.0, 100.0, size=len(att2) - (m_test + 1))

    horizons = [10, 30, 100]
    dt = 0.1
    f1 = build_windows(att1, dt, horizons, stride_steps=10, hist_s=30.0)
    f2 = build_windows(att2, dt, horizons, stride_steps=10, hist_s=30.0)

    # For indices m <= m_test, features must be identical
    mask = f1["idx"] <= m_test
    assert np.allclose(f1["cur"][mask], f2["cur"][mask])
    assert np.allclose(f1["slope5"][mask], f2["slope5"][mask])
    assert np.allclose(f1["slope30"][mask], f2["slope30"][mask])
    assert np.allclose(f1["max60"][mask], f2["max60"][mask])

    # But targets Z must differ because future was changed
    assert not np.allclose(f1["Z"][mask], f2["Z"][mask])
    print("windows.py: No future leakage verified.")


if __name__ == "__main__":
    test_forward_window_max_brute_force()
    test_no_future_leakage()
    print("All windows tests passed.")
