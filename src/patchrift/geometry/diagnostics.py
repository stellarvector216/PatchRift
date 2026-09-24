"""Non-feedback Stage II diagnostics from §5.7."""

from dataclasses import dataclass

import numpy as np

from patchrift.geometry.rotation import inter_image_rotation


@dataclass(frozen=True)
class RobustDiagnostic:
    median: float
    mad_scale: float


def robust_median_and_mad(values: np.ndarray) -> RobustDiagnostic:
    """Return the §5.7 median and 1.4826-MAD summary."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    median = float(np.median(values))
    return RobustDiagnostic(
        median=median,
        mad_scale=float(1.4826 * np.median(np.abs(values - median))),
    )


def implied_wall_slope(elevation: float, half_angle: float) -> float:
    """Invert Eq. (6): γ̂ = atan(tan β / cos δ̂)."""
    if not np.isfinite(elevation) or not np.isfinite(half_angle):
        raise ValueError("elevation and half_angle must be finite")
    cosine = float(np.cos(half_angle))
    if cosine <= 0.0:
        raise ValueError("half_angle must be strictly below 90 degrees")
    return float(np.arctan(np.tan(elevation) / cosine))


def corner_north_angle(jacobian: np.ndarray) -> float:
    """Compute αcorners from the fitted §2.3 geographic Jacobian."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.shape != (2, 2) or not np.all(np.isfinite(jacobian)):
        raise ValueError("jacobian must be a finite 2x2 matrix")
    north_in_pixels = jacobian[:, 1]
    if np.allclose(north_in_pixels, 0.0):
        raise ValueError("north Jacobian column must be non-zero")
    return float(np.arctan2(north_in_pixels[1], north_in_pixels[0]))


def corner_azimuth_disagreement(shadow_north: float, corners_north: float) -> float:
    """Return the signed wrapped shadow-vs-footprint diagnostic residual."""
    return inter_image_rotation(corners_north, shadow_north)


def elevation_from_shadow_length(rim_radius: float, wall_slope: float, shadow_length: float) -> float:
    """Invert §5.7's shadow-length relation after an external rim fit."""
    values = (rim_radius, wall_slope, shadow_length)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("shadow-length inputs must be finite")
    if rim_radius <= 0.0 or shadow_length <= 0.0:
        raise ValueError("rim_radius and shadow_length must be positive")
    tangent = 2.0 * rim_radius * np.tan(wall_slope) / shadow_length - np.tan(wall_slope)
    return float(np.arctan(tangent))
