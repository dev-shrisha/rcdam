"""Quick correctness checks. Run:  python tests/test_step2.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rcdam.link import ModcodTable
from rcdam.arq import build_cumulative, message_delivery


def test_bler_curve():
    t = ModcodTable()
    k = 5
    p_at = t.bler(np.array([k]), np.array([t.thr[k]]))[0]
    assert abs(np.log10(p_at) + 7.0) < 1e-9, p_at
    p_low = t.bler(np.array([k]), np.array([t.thr[k] - 1.0]))[0]
    assert p_low == 1.0
    xs = np.linspace(t.thr[k] - 2, t.thr[k] + 2, 50)
    ps = t.bler(np.full(50, k), xs)
    assert np.all(np.diff(ps) <= 1e-15)          # BLER never rises with Es/N0
    print("bler curve ok")


def test_clean_channel():
    dt, n = 0.1, 2000
    c_tx, c_ok = build_cumulative(np.full(n, 10e6), np.zeros(n), dt)
    out = message_delivery(c_tx, c_ok, dt, np.array([0]), 20e6, 5, 0.27, 10.0)
    assert abs(out["latency_s"][0] - (2.0 + 0.27)) < 1e-9, out["latency_s"]
    assert not out["miss"][0]
    print("clean channel ok")


def test_lossy_channel_needs_rounds():
    dt, n = 0.1, 4000
    c_tx, c_ok = build_cumulative(np.full(n, 10e6), np.full(n, 0.5), dt)
    out = message_delivery(c_tx, c_ok, dt, np.array([0]), 20e6, 5, 0.27, 60.0)
    assert not out["miss"][0]
    assert 3.5 <= out["airtime_s"][0] <= 4.8, out["airtime_s"]       # ~ S / (rate*(1-p)) = 4 s
    assert out["latency_s"][0] > 4.0                                   # retransmission rounds cost RTTs
    print("lossy channel ok")


def test_deadline_cut():
    dt, n = 0.1, 4000
    c_tx, c_ok = build_cumulative(np.full(n, 10e6), np.full(n, 0.5), dt)
    out = message_delivery(c_tx, c_ok, dt, np.array([0]), 20e6, 5, 0.27, 2.0)
    assert out["miss"][0]
    print("deadline cut ok")


if __name__ == "__main__":
    test_bler_curve(); test_clean_channel(); test_lossy_channel_needs_rounds(); test_deadline_cut()
    print("all tests passed")
