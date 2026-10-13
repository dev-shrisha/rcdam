"""Master Test Suite Runner: Runs all unit tests from Step 2 to Step 9."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_step2 import test_bler_curve, test_clean_channel, test_lossy_channel_needs_rounds, test_deadline_cut
from tests.test_step3 import test_modcod_dvbs2_verification, test_itur_p618_attenuation, test_maseng_bakken_calibration
from tests.test_step4 import test_cost_model_overhead, test_erasure_and_interleaving
from tests.test_step5 import test_pinball_loss, test_monotonic_quantiles, test_forecaster_training_and_persistence_comparison
from tests.test_step6 import test_classify_fade_state, test_delayed_feedback_fifo, test_stratified_coverage_property
from tests.test_step7 import test_clear_sky_optimization, test_deep_fade_infeasibility_signalling, test_priority_differentiation
from tests.test_step8 import test_ablation_seed_execution
from tests.test_step9 import test_drl_agent_initialization_and_forward, test_drl_policy_generation


def run_all():
    print("========================================")
    print("   RUNNING ALL RC-DAM UNIT TEST SUITES   ")
    print("========================================\n")

    print("--- [Step 2 Tests] ---")
    test_bler_curve(); test_clean_channel(); test_lossy_channel_needs_rounds(); test_deadline_cut()
    print("-> Step 2 Passed.\n")

    print("--- [Step 3 Tests] ---")
    test_modcod_dvbs2_verification(); test_itur_p618_attenuation(); test_maseng_bakken_calibration()
    print("-> Step 3 Passed.\n")

    print("--- [Step 4 Tests] ---")
    test_cost_model_overhead(); test_erasure_and_interleaving()
    print("-> Step 4 Passed.\n")

    print("--- [Step 5 Tests] ---")
    test_pinball_loss(); test_monotonic_quantiles(); test_forecaster_training_and_persistence_comparison()
    print("-> Step 5 Passed.\n")

    print("--- [Step 6 Tests] ---")
    test_classify_fade_state(); test_delayed_feedback_fifo(); test_stratified_coverage_property()
    print("-> Step 6 Passed.\n")

    print("--- [Step 7 Tests] ---")
    test_clear_sky_optimization(); test_deep_fade_infeasibility_signalling(); test_priority_differentiation()
    print("-> Step 7 Passed.\n")

    print("--- [Step 8 Tests] ---")
    test_ablation_seed_execution()
    print("-> Step 8 Passed.\n")

    print("--- [Step 9 Tests] ---")
    test_drl_agent_initialization_and_forward(); test_drl_policy_generation()
    print("-> Step 9 Passed.\n")

    print("========================================")
    print("   ALL TESTS PASSED WITH 100% SUCCESS   ")
    print("========================================")


if __name__ == "__main__":
    run_all()
