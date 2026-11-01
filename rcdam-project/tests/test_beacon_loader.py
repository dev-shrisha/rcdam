"""Unit tests for Phase 6: Real satellite beacon data loader and dataset wrapper."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from rcdam.beacon_loader import load_beacon_csv, BeaconDataset, check_real_data_available, SIMULATION_ONLY_WARNING


def test_load_sample_beacon_csv():
    """Loads sample synthetic beacon CSV and verifies resampling and metadata."""
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_beacon_synthetic.csv")
    assert os.path.exists(sample_path), "Sample beacon CSV must exist"

    att, meta = load_beacon_csv(sample_path, dt_s=0.1)
    assert len(att) > 0
    assert isinstance(att, np.ndarray)
    assert meta["duration_h"] > 0.99
    assert meta["mean_att_db"] >= 0.0
    assert meta["n_gaps"] == 0
    print("test_load_sample_beacon_csv PASSED")


def test_beacon_dataset_wrapper():
    """BeaconDataset wrapper produces valid Scenario and matching attenuation arrays."""
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_beacon_synthetic.csv")
    ds = BeaconDataset(sample_path, dt_s=0.1)

    assert ds.duration_h > 0.9
    sc = ds.to_scenario(deadline_s=2.0)
    assert sc.dt_s == 0.1
    assert sc.deadline_s == 2.0

    att = ds.generate_attenuation_db(sc)
    assert len(att) == sc.n_steps
    assert np.all(att >= 0.0)
    print("test_beacon_dataset_wrapper PASSED")


def test_beacon_missing_file_error():
    """Missing file raises FileNotFoundError with simulation-only warning."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_beacon_csv("data/nonexistent_beacon_file.csv")
    assert "SIMULATION-ONLY" in str(exc_info.value)
    print("test_beacon_missing_file_error PASSED")


def test_real_data_check():
    """Check that check_real_data_available returns False when only sample data is present."""
    # sample_beacon_synthetic contains 'sample', so real data is correctly reported as False
    is_real = check_real_data_available("data")
    assert not is_real, "Sample synthetic beacon should not be flagged as real satellite beacon data"
    print("test_real_data_check PASSED")


if __name__ == "__main__":
    test_load_sample_beacon_csv()
    test_beacon_dataset_wrapper()
    test_beacon_missing_file_error()
    test_real_data_check()
    print("All beacon loader tests passed!")

