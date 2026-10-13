"""Synthetic rain + scintillation attenuation time series (dB).

Two generators (selected by Scenario.rain_model):

  "events"        Poisson-arriving rain events. Each event is a Gaussian-shaped bump with a
                  log-normal peak depth, modulated by slow and fast multiplicative variation.
                  Easy to control, so deep fades occur inside a short run (stress testing).

  "gauss_markov"  Simplified Maseng-Bakken style process: a slow Gaussian process decides
                  WHEN it rains, a second one sets the log-normal intensity. Realistic
                  statistics but the deep-fade tail needs weeks of simulated time.

Both add fast jitter and scintillation. Both are PLACEHOLDERS: for the paper, calibrate to
ITU-R P.618 / P.1853 statistics and validate on real beacon data where you can get it.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm

from .config import Scenario


def _ar1(n: int, tau_s: float, dt_s: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary unit-variance Gaussian AR(1) process with correlation time tau_s."""
    rho = np.exp(-dt_s / tau_s)
    noise = rng.standard_normal(n)
    zi = np.array([rho * rng.standard_normal()])          # start in the stationary state
    y, _ = lfilter([np.sqrt(1.0 - rho**2)], [1.0, -rho], noise, zi=zi)
    return y


def _rain_events(sc: Scenario, rng: np.random.Generator) -> np.ndarray:
    n, dt = sc.n_steps, sc.dt_s
    total = np.zeros(n)
    n_events = rng.poisson(sc.events_per_day * sc.duration_h / 24.0)
    for _ in range(n_events):
        t0 = rng.uniform(0.0, n * dt)
        width = rng.lognormal(np.log(sc.event_width_median_s), 0.5)
        peak = rng.lognormal(sc.event_peak_ln_mean, sc.event_peak_ln_std)
        if sc.event_peak_max_db > 0:
            peak = min(peak, sc.event_peak_max_db)
        lo = max(0, int((t0 - 4 * width) / dt))
        hi = min(n, int((t0 + 4 * width) / dt))
        tt = np.arange(lo, hi) * dt
        total[lo:hi] += peak * np.exp(-0.5 * ((tt - t0) / width) ** 2)
    slow_mod = np.exp(sc.mod_slow_std * _ar1(n, sc.mod_slow_tau_s, dt, rng))
    return total * slow_mod


def _rain_gauss_markov(sc: Scenario, rng: np.random.Generator) -> np.ndarray:
    n, dt = sc.n_steps, sc.dt_s
    tau_slow = 1.0 / sc.rain_beta_per_s
    occurrence = _ar1(n, tau_slow, dt, rng)
    intensity = _ar1(n, tau_slow, dt, rng)
    threshold = norm.isf(sc.p_rain)                        # P(occurrence > threshold) = p_rain
    gate = np.clip((occurrence - threshold) / sc.rain_onset_width, 0.0, 1.0)
    return gate * np.exp(sc.rain_ln_mean + sc.rain_ln_std * intensity)


def generate_attenuation_db(sc: Scenario) -> np.ndarray:
    rng = np.random.default_rng(sc.seed)
    n, dt = sc.n_steps, sc.dt_s
    if sc.rain_model == "events":
        rain_db = _rain_events(sc, rng)
    elif sc.rain_model == "gauss_markov":
        rain_db = _rain_gauss_markov(sc, rng)
    else:
        raise ValueError(f"unknown rain_model: {sc.rain_model}")

    jitter = np.exp(sc.fast_jitter_std * _ar1(n, sc.fast_jitter_tau_s, dt, rng))
    scint = sc.scint_std_db * _ar1(n, sc.scint_tau_s, dt, rng)
    return rain_db * jitter + scint
