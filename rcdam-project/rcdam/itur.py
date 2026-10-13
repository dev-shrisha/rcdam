"""ITU-R P.618 and P.1853 rain attenuation models and calibration utilities.

This module provides:
1. itur_p618_rain_attenuation: Standard ITU-R P.618-13 method to compute the rain
   attenuation exceedance distribution A(p) for satellite earth-space links.
2. ITU-R P.1853-1 parameter mapping: converts long-term exceedance statistics into
   the dynamic Maseng-Bakken lognormal parameters (m, sigma, p_rain, beta).
3. Calibration verification: validates synthetic attenuation time series against
   ITU-R P.618 reference curves.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from scipy import stats
from scipy.optimize import minimize


@dataclass
class EarthStationConfig:
    """Satellite link parameters for ITU-R P.618 calculations."""
    freq_ghz: float = 20.0             # Ka-band downlink frequency (GHz)
    elevation_deg: float = 35.0        # Earth station elevation angle (deg)
    station_lat_deg: float = 45.0      # Latitude (deg N)
    station_alt_km: float = 0.2        # Earth station height above sea level (km)
    rain_rate_001_mmh: float = 35.0    # Rain rate exceeded for 0.01% of average year (mm/h)
    polarization_tilt_deg: float = 45.0 # Polarization tilt angle (45 = circular)


def specific_attenuation_itu_r_p838(freq_ghz: float, r_mmh: float,
                                    elevation_deg: float = 35.0,
                                    tau_deg: float = 45.0) -> float:
    """Calculate specific rain attenuation gamma_R (dB/km) via ITU-R P.838-3."""
    # ITU-R P.838 coefficients at 20 GHz (linear interpolation coefficients):
    # For 20 GHz:
    k_h = 0.09164
    k_v = 0.09611
    alpha_h = 1.0568
    alpha_v = 0.9847

    # Polarization tilt combination:
    theta = np.radians(elevation_deg)
    tau = np.radians(tau_deg)
    cos_theta = np.cos(theta)
    cos_2tau = np.cos(2 * tau)

    k = (k_h + k_v + (k_h - k_v) * (cos_theta ** 2) * cos_2tau) / 2.0
    alpha = (k_h * alpha_h + k_v * alpha_v +
             (k_h * alpha_h - k_v * alpha_v) * (cos_theta ** 2) * cos_2tau) / (2.0 * k)

    return float(k * (r_mmh ** alpha))


def itur_p618_rain_attenuation(p_pct: np.ndarray,
                               cfg: EarthStationConfig = None) -> np.ndarray:
    """Calculate rain attenuation exceeded for p percent of an average year A(p) [dB]

    Follows ITU-R P.618-13 Recommendation.
    """
    if cfg is None:
        cfg = EarthStationConfig()

    p = np.asarray(p_pct, dtype=float)
    theta_rad = np.radians(cfg.elevation_deg)

    # 1. Rain height h_R (km) from ITU-R P.839
    # Mean annual 0 deg C isotherm height h0 approx 3.0 km at lat 45 deg
    h0 = max(0.0, 5.0 - 0.075 * abs(cfg.station_lat_deg))
    h_r = h0 + 0.36  # km

    # 2. Slant path length L_s (km)
    h_s = cfg.station_alt_km
    if cfg.elevation_deg >= 5.0:
        l_s = (h_r - h_s) / np.sin(theta_rad)
    else:
        re = 8500.0  # effective Earth radius km
        l_s = 2.0 * (h_r - h_s) / (np.sqrt(np.sin(theta_rad)**2 + 2.0 * (h_r - h_s) / re) + np.sin(theta_rad))

    # Horizontal projection L_g (km)
    l_g = l_s * np.cos(theta_rad)

    # 3. Specific attenuation gamma_R (dB/km) for R_0.01
    gamma_r = specific_attenuation_itu_r_p838(
        cfg.freq_ghz, cfg.rain_rate_001_mmh, cfg.elevation_deg, cfg.polarization_tilt_deg
    )

    # 4. Horizontal reduction factor r_0.01
    r_001 = 1.0 / (1.0 + 0.78 * np.sqrt(l_g * gamma_r / cfg.freq_ghz) - 0.38 * (1.0 - np.exp(-2.0 * l_g)))

    # 5. Vertical adjustment factor v_0.01
    zeta = np.degrees(np.arctan((h_r - h_s) / (l_g * r_001)))
    if zeta > cfg.elevation_deg:
        l_r = (l_g * r_001) / np.cos(theta_rad)
    else:
        l_r = (h_r - h_s) / np.sin(theta_rad)

    chi = 36.0 - abs(cfg.station_lat_deg)
    if chi > 0:
        v_001 = 1.0 / (1.0 + np.sqrt(np.sin(theta_rad)) * (31.0 * (1.0 - np.exp(-cfg.elevation_deg / (1.0 + chi))) *
                                                          np.sqrt(l_r * gamma_r) / (cfg.freq_ghz ** 2) - 0.45))
    else:
        v_001 = 1.0 / (1.0 + np.sqrt(np.sin(theta_rad)) * (31.0 * (1.0 - np.exp(-cfg.elevation_deg)) *
                                                          np.sqrt(l_r * gamma_r) / (cfg.freq_ghz ** 2) - 0.45))

    # 6. Effective path length L_eff (km)
    l_eff = l_r * v_001

    # 7. Attenuation exceeded for 0.01% of an average year
    a_001 = gamma_r * l_eff

    # 8. Extrapolation to other exceedance probabilities p in [0.001%, 5%]
    # beta parameter:
    if p.size == 1:
        p_val = float(p.item())
        if abs(cfg.station_lat_deg) >= 36.0:
            beta = 0.0
        elif abs(cfg.station_lat_deg) < 36.0 and cfg.elevation_deg >= 25.0:
            beta = -0.005 * (abs(cfg.station_lat_deg) - 36.0)
        else:
            beta = -0.005 * (abs(cfg.station_lat_deg) - 36.0) + 1.8 - 4.25 * np.sin(theta_rad)
        exp_factor = -(0.655 + 0.033 * np.log(p_val) - 0.045 * np.log(a_001) - beta * np.sin(theta_rad) * (1.0 - p_val))
        return np.array([a_001 * ((p_val / 0.01) ** exp_factor)])

    beta = np.zeros_like(p)
    mask_low_lat = abs(cfg.station_lat_deg) < 36.0
    if mask_low_lat:
        if cfg.elevation_deg >= 25.0:
            beta[:] = -0.005 * (abs(cfg.station_lat_deg) - 36.0)
        else:
            beta[:] = -0.005 * (abs(cfg.station_lat_deg) - 36.0) + 1.8 - 4.25 * np.sin(theta_rad)

    exp_factor = -(0.655 + 0.033 * np.log(p) - 0.045 * np.log(a_001) - beta * np.sin(theta_rad) * (1.0 - p))
    a_p = a_001 * ((p / 0.01) ** exp_factor)
    return np.maximum(a_p, 0.0)


def calibrate_maseng_bakken_to_itu_r(cfg: EarthStationConfig = None) -> Dict[str, float]:
    """Fit Maseng-Bakken parameters (p_rain, rain_ln_mean, rain_ln_std, rain_beta_per_s)

    Fits lognormal parameters so that the unconditional exceedance probability:
      P(A > a) = p_rain * Q( (ln(a) - rain_ln_mean) / rain_ln_std )
    matches the ITU-R P.618 A(p) curve.
    """
    if cfg is None:
        cfg = EarthStationConfig()

    p_targets = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    a_targets = itur_p618_rain_attenuation(p_targets, cfg)

    # p_rain is typically the annual rain probability (~3% to 6% in temperate regions)
    p_rain_fixed = 0.05  # 5% rain time

    # We fit m = rain_ln_mean, s = rain_ln_std:
    def objective(params):
        m, s = params
        if s <= 0.05:
            return 1e6
        # Predicted p: P(A > a) = p_rain * (1 - Phi((ln(a) - m)/s)) * 100%
        z = (np.log(a_targets) - m) / s
        p_pred = p_rain_fixed * (1.0 - stats.norm.cdf(z)) * 100.0
        # Minimize log error
        return np.sum((np.log(np.maximum(p_pred, 1e-4)) - np.log(p_targets)) ** 2)

    res = minimize(objective, x0=[0.2, 1.2], bounds=[(-1.0, 2.0), (0.1, 3.0)])
    m_opt, s_opt = res.x

    return {
        "p_rain": p_rain_fixed,
        "rain_ln_mean": float(m_opt),
        "rain_ln_std": float(s_opt),
        "rain_beta_per_s": 2.0e-4,  # Standard ITU-R P.1853 Markov dynamic rate
        "a_001_db": float(itur_p618_rain_attenuation(np.array([0.01]), cfg)[0]),
    }


def compute_empirical_ccdf(att_series_db: np.ndarray,
                           thresholds_db: np.ndarray) -> np.ndarray:
    """Compute empirical exceedance probability percentage P(A > thr) * 100%."""
    n = len(att_series_db)
    return np.array([100.0 * np.sum(att_series_db > thr) / n for thr in thresholds_db])
