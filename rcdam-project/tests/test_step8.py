"""Unit tests for Step 8: Ablation Framework (A0 - A6, B1)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcdam.ablations import run_ablation_seed, ABLATION_SYSTEMS


def test_ablation_seed_execution():
    # Run a short 3-hour seed to verify all 10 systems execute cleanly
    rows = run_ablation_seed(seed=777, hours=3.0, forecaster=None)

    assert len(rows) == len(ABLATION_SYSTEMS), f"Expected {len(ABLATION_SYSTEMS)} systems, got {len(rows)}"

    for r in rows:
        assert 0.0 <= r["miss_rate"] <= 1.0
        assert r["overhead"] >= 0.95, f"Overhead must be >= 1.0 (or near 1), got {r['overhead']}"
        assert 0.0 <= r["avoidable_miss_rate"] <= 1.0
        assert 0.0 <= r["infeasible_frac"] <= 1.0

    print(f"All {len(rows)} ablation systems executed cleanly.")


if __name__ == "__main__":
    test_ablation_seed_execution()
    print("All Step 8 unit tests passed.")
