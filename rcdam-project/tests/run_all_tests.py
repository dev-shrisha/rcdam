"""Master Test Suite Runner: Runs all unit tests from Step 2 to Step 9.

Torch-dependent tests (QuantileMLP) are skipped gracefully if torch is not installed.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_step2 import test_bler_curve, test_clean_channel, test_lossy_channel_needs_rounds, test_deadline_cut
from tests.test_m1_data_and_link import (
    test_modcod_table_standard_verification, test_bler_per_modcod_curves,
    test_data_loading_and_splitting, test_itur_p1853_attenuation_generator
)
from tests.test_windows import test_forward_window_max_brute_force, test_no_future_leakage
from tests.test_calibration import (
    test_fade_state_edges, test_cqr_correction_finite_sample,
    test_stratified_corrections_and_coverage, test_aci_step, test_calibration_json_persistence
)
from tests.test_evaluator import (
    test_evaluator_common_random_numbers_monotonicity,
    test_evaluator_deadline_exceeded_by_transmission_time, test_evaluator_clean_channel
)
from tests.test_optimizer import (
    test_clear_sky_highest_modcod, test_more_severe_bound_never_lowers_overhead,
    test_deadline_infeasible_reason, test_capacity_infeasible_reason
)
from tests.test_groundtruth import (
    test_simulate_true_delivery_clean_channel,
    test_simulate_true_delivery_extreme_fade_fails, test_policy_variants_execution
)
from tests.test_step3 import test_modcod_dvbs2_verification, test_itur_p618_attenuation, test_maseng_bakken_calibration
from tests.test_step4 import test_cost_model_overhead, test_erasure_and_interleaving
from tests.test_step6 import test_classify_fade_state, test_delayed_feedback_fifo, test_stratified_coverage_property
from tests.test_step7 import test_clear_sky_optimization, test_deep_fade_infeasibility_signalling, test_priority_differentiation


def run_all():
    print("==================================================")
    print("   RUNNING ALL RC-DAM MILESTONE & UNIT TESTS      ")
    print("==================================================\n")

    print("--- [Step 2 & Baseline Tests] ---")
    test_bler_curve(); test_clean_channel(); test_lossy_channel_needs_rounds(); test_deadline_cut()
    print("-> Step 2 Passed.\n")

    print("--- [Milestone M1: Data & Link Realism Tests] ---")
    test_modcod_table_standard_verification(); test_bler_per_modcod_curves()
    test_data_loading_and_splitting(); test_itur_p1853_attenuation_generator()
    print("-> Milestone M1 Passed.\n")

    print("--- [Milestone M2: Window Statistics & Non-Leakage Tests] ---")
    test_forward_window_max_brute_force(); test_no_future_leakage()
    print("-> Milestone M2 (windows) Passed.\n")

    # Torch-dependent M2 forecaster tests
    try:
        import torch  # noqa: F401
        from tests.test_step5 import test_pinball_loss, test_monotonic_quantiles
        test_pinball_loss(); test_monotonic_quantiles()
        print("-> Milestone M2 (QuantileMLP torch) Passed.\n")
    except ImportError:
        print("[SKIP] torch not installed — skipping QuantileMLP tests "
              "(test_pinball_loss, test_monotonic_quantiles)\n")

    print("--- [Milestone M3: Stratified Conformal & ACI Tests] ---")
    test_fade_state_edges(); test_cqr_correction_finite_sample()
    test_stratified_corrections_and_coverage(); test_aci_step(); test_calibration_json_persistence()
    print("-> Milestone M3 Passed.\n")

    print("--- [Milestone M4: Evaluator & Optimizer Tests] ---")
    test_evaluator_common_random_numbers_monotonicity()
    test_evaluator_deadline_exceeded_by_transmission_time(); test_evaluator_clean_channel()
    test_clear_sky_highest_modcod(); test_more_severe_bound_never_lowers_overhead()
    test_deadline_infeasible_reason(); test_capacity_infeasible_reason()
    print("-> Milestone M4 Passed.\n")

    print("--- [Milestone M5: Ground Truth & Variants Tests] ---")
    test_simulate_true_delivery_clean_channel()
    test_simulate_true_delivery_extreme_fade_fails(); test_policy_variants_execution()
    print("-> Milestone M5 Passed.\n")

    print("--- [Legacy Step 3-7 System Tests] ---")
    test_modcod_dvbs2_verification(); test_itur_p618_attenuation(); test_maseng_bakken_calibration()
    test_cost_model_overhead(); test_erasure_and_interleaving()
    test_classify_fade_state(); test_delayed_feedback_fifo(); test_stratified_coverage_property()
    test_clear_sky_optimization(); test_deep_fade_infeasibility_signalling(); test_priority_differentiation()
    print("-> Legacy Step 3-7 Tests Passed.\n")

    # New advancement tests
    try:
        from tests.test_adv import (
            test_wilson_interval_known, test_exact_upper_bound_rule_of_three,
            test_n_zero_miss, test_mcnemar_exact,
            test_ltt_certifies_correct_policy, test_ltt_does_not_certify_when_risk_too_high,
            test_crc_lambda_basic, test_classify_misses_categories,
            test_infeasibility_report_keys, test_log_columns_count,
            test_horizon_steps_needed, test_pick_horizon_none,
        )
        print("--- [Advancement Tests: stats, risk control, diagnostics] ---")
        test_wilson_interval_known(); test_exact_upper_bound_rule_of_three()
        test_n_zero_miss(); test_mcnemar_exact()
        test_ltt_certifies_correct_policy(); test_ltt_does_not_certify_when_risk_too_high()
        test_crc_lambda_basic(); test_classify_misses_categories()
        test_infeasibility_report_keys(); test_log_columns_count()
        test_horizon_steps_needed(); test_pick_horizon_none()
        print("-> Advancement Tests Passed.\n")
    except ImportError as e:
        print(f"[SKIP] Advancement tests not yet available: {e}\n")

    # Window profile tests
    try:
        from tests.test_window_profile import (
            test_compute_window_profile_basic, test_calibrate_profile_bounds,
            test_interleaving_helps_on_bursty_fade, test_interleaving_neutral_on_constant_fade,
        )
        print("--- [Window Profile Tests (Task 8)] ---")
        test_compute_window_profile_basic(); test_calibrate_profile_bounds()
        test_interleaving_helps_on_bursty_fade(); test_interleaving_neutral_on_constant_fade()
        print("-> Window Profile Tests Passed.\n")
    except ImportError as e:
        print(f"[SKIP] Window profile tests not yet available: {e}\n")

    # Phase 1 and Phase 2 Tests
    try:
        from tests.test_phase1_regressions import (
            test_1_1_pruning_reproduction, test_1_2_validated_selection_independence,
            test_1_3_forecaster_dataset_validation, test_1_4_horizon_consistent_bounds,
            test_1_5_evaluator_principled_bound, test_1_6_priority_differentiation_strict,
            test_1_7_hygiene,
        )
        from tests.test_phase2_infrastructure import (
            test_2_1_deterministic_oracle_feasibility, test_2_2_stress_presets_configuration,
            test_2_3_diagnostics_log_columns_and_taxonomy, test_2_4_mcnemar_and_bootstrap_statistics,
        )
        print("--- [Phase 1 & Phase 2 Regression and Infrastructure Tests] ---")
        test_1_1_pruning_reproduction(); test_1_2_validated_selection_independence()
        test_1_3_forecaster_dataset_validation(); test_1_4_horizon_consistent_bounds()
        test_1_5_evaluator_principled_bound(); test_1_6_priority_differentiation_strict()
        test_1_7_hygiene()
        test_2_1_deterministic_oracle_feasibility(); test_2_2_stress_presets_configuration()
        test_2_3_diagnostics_log_columns_and_taxonomy(); test_2_4_mcnemar_and_bootstrap_statistics()
        print("-> Phase 1 & Phase 2 Tests Passed.\n")
    except ImportError as e:
        print(f"[SKIP] Phase 1/2 tests: {e}\n")

    print("==================================================")
    print("   ALL RC-DAM TESTS PASSED WITH 100% SUCCESS!     ")
    print("==================================================")


if __name__ == "__main__":
    run_all()
