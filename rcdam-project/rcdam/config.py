"""Scenario configuration. Every experiment parameter lives here so runs are reproducible.

All numbers are PLACEHOLDERS for a GEO Ka-band downlink. Calibrate the rain model
to ITU-R P.618 / P.1853 statistics (or to real beacon data) before reporting results.
"""
from dataclasses import dataclass


@dataclass
class Scenario:
    # --- time base ---
    dt_s: float = 0.1            # simulation step
    duration_h: float = 48.0     # simulated duration
    seed: int = 7

    # --- link (GEO Ka-band downlink, placeholder numbers) ---
    rtt_s: float = 0.54          # round-trip time; feedback about the channel is this old
    symbol_rate_baud: float = 10e6
    clear_sky_esn0_db: float = 18.0

    # --- rain attenuation ---
    # "events": Poisson-arriving rain events with log-normal peak depth (good for stress tests;
    #           deep fades show up within a 2-day run)
    # "gauss_markov": simplified Maseng-Bakken process (needs weeks/months of simulated time
    #           before its deep-fade tail appears; use for calibration work)
    rain_model: str = "events"
    events_per_day: float = 4.0
    event_width_median_s: float = 600.0   # std-dev of the Gaussian event envelope
    event_peak_ln_mean: float = 1.79      # ln(dB): median peak ~6 dB
    event_peak_ln_std: float = 0.9
    event_peak_max_db: float = 0.0        # cap on event peak depth (0 = no cap)
    mod_slow_std: float = 0.25            # slow multiplicative variation inside an event
    mod_slow_tau_s: float = 60.0

    # gauss_markov model parameters
    p_rain: float = 0.05             # fraction of time it is raining
    rain_ln_mean: float = 0.18       # mean of ln(A_dB) given rain
    rain_ln_std: float = 1.2         # std of ln(A_dB) given rain
    rain_beta_per_s: float = 2e-4    # slow dynamics of the rain process
    rain_onset_width: float = 0.15   # smooth ramp when rain starts/stops
    fast_jitter_std: float = 0.15    # fast multiplicative variation during rain
    fast_jitter_tau_s: float = 5.0
    scint_std_db: float = 0.15       # additive scintillation
    scint_tau_s: float = 1.0

    # --- smooth block-error model (placeholder waterfall; replace with real LDPC curves) ---
    bler_slope_decades_per_db: float = 15.0   # steepness of the LDPC waterfall
    bler_qef_log10: float = -7.0              # log10 BLER at the QEF threshold

    # --- ARQ model (selective repeat in rounds, feedback after one RTT) ---
    arq_max_rounds: int = 8
    arq_done_tol_bits: float = 65_000.0       # remaining loss below ~1 FEC frame counts as complete

    # --- predictive ACM baseline ---
    slope_window_s: float = 5.0               # window used to estimate the Es/N0 trend

    # --- message model ---
    msg_interval_s: float = 30.0     # one message every this many seconds
    msg_bits: float = 20e6
    deadline_s: float = 2.0          # end-to-end delivery deadline

    @property
    def n_steps(self) -> int:
        return int(self.duration_h * 3600 / self.dt_s)

    @property
    def feedback_delay_steps(self) -> int:
        return int(round(self.rtt_s / self.dt_s))

    @property
    def one_way_delay_s(self) -> float:
        return self.rtt_s / 2.0
