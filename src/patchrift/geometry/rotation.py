"""§5.8 sensor-frame rotation and its uncertainty propagation."""

import numpy as np


def wrap_angle_2pi(angle: float) -> float:
    """Wrap an angle to [0, 2π)."""
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    return float(angle % (2.0 * np.pi))


def pixel_north_angle(pixel_solar_azimuth: float, solar_azimuth: float) -> float:
    """Compute α = wrap[0,2π)(φ̂pix - A), as required by Eq. (42)."""
    return wrap_angle_2pi(pixel_solar_azimuth - solar_azimuth)


def inter_image_rotation(north_angle_a: float, north_angle_b: float) -> float:
    """Compute Δα = wrap[-π,π)(αB - αA)."""
    if not np.isfinite(north_angle_a) or not np.isfinite(north_angle_b):
        raise ValueError("north angles must be finite")
    return float((north_angle_b - north_angle_a + np.pi) % (2.0 * np.pi) - np.pi)


def inter_image_rotation_uncertainty(
    sigma_phi_a: float,
    sigma_phi_b: float,
    sigma_geo_a: float = 0.0,
    sigma_geo_b: float = 0.0,
) -> float:
    """Propagate Eq. (43)'s independent uncertainty contributions."""
    values = (sigma_phi_a, sigma_phi_b, sigma_geo_a, sigma_geo_b)
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("uncertainties must be finite and non-negative")
    return float(np.sqrt(sum(value**2 for value in values)))
