from __future__ import annotations

import numpy as np


def wrap_angle_pi(angle: float) -> float:
    """Wrap an angle to the V3 interval [-pi, pi)."""
    angle = float(angle)

    if not np.isfinite(angle):
        raise ValueError("angle must be finite")

    wrapped = float(
        np.arctan2(
            np.sin(angle),
            np.cos(angle),
        )
    )

    if wrapped >= np.pi:
        wrapped = -np.pi

    return wrapped

def circular_median(angles: np.ndarray) -> float:
    """Return the circular median of a set of angles.

    The V3 §5.6 IRLS estimator is seeded at the circular median.
    """
    angles = np.asarray(angles, dtype=np.float64)

    if angles.ndim != 1:
        raise ValueError("angles must be a 1D array")

    if angles.size == 0:
        raise ValueError("angles must not be empty")

    if not np.all(np.isfinite(angles)):
        raise ValueError("angles must contain only finite values")

    angles = np.arctan2(
        np.sin(angles),
        np.cos(angles),
    )

    # For a finite sample on the circle, choose the candidate angle
    # minimizing the sum of absolute wrapped angular distances.
    distances = np.abs(
        np.arctan2(
            np.sin(angles[:, None] - angles[None, :]),
            np.cos(angles[:, None] - angles[None, :]),
        )
    )

    objective = np.sum(distances, axis=1)

    return float(angles[np.argmin(objective)])

def tukey_biweight_weight(
    residual: float,
    cutoff: float,
) -> float:
    """Return the Tukey biweight factor for one angular residual."""
    residual = float(residual)
    cutoff = float(cutoff)

    if not np.isfinite(residual):
        raise ValueError("residual must be finite")

    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")

    if abs(residual) >= cutoff:
        return 0.0

    scaled = residual / cutoff

    return float((1.0 - scaled**2) ** 2)

def irls_scale(
    residuals: np.ndarray,
    s_min: float,
) -> float:
    """Compute the V3 IRLS residual scale."""
    residuals = np.asarray(residuals, dtype=np.float64)

    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1D array")

    if residuals.size == 0:
        raise ValueError("residuals must not be empty")

    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values")

    s_min = float(s_min)

    if not np.isfinite(s_min) or s_min <= 0.0:
        raise ValueError("s_min must be finite and positive")

    residual_median = float(np.median(np.abs(residuals)))

    return float(
        max(
            1.4826 * residual_median,
            s_min,
        )
    )

def tukey_cutoff(
    scale: float,
    constant: float = 4.685,
) -> float:
    """Compute the V3 Tukey biweight cutoff."""
    scale = float(scale)
    constant = float(constant)

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    if not np.isfinite(constant) or constant <= 0.0:
        raise ValueError("constant must be finite and positive")

    return float(constant * scale)

def irls_weights(
    base_weights: np.ndarray,
    residuals: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Compute V3 IRLS weights from base weights and angular residuals."""
    base_weights = np.asarray(base_weights, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    if base_weights.ndim != 1:
        raise ValueError("base_weights must be a 1D array")

    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1D array")

    if base_weights.size == 0:
        raise ValueError("base_weights must not be empty")

    if base_weights.size != residuals.size:
        raise ValueError(
            "base_weights and residuals must have the same length"
        )

    if not np.all(np.isfinite(base_weights)):
        raise ValueError("base_weights must contain only finite values")

    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values")

    if np.any(base_weights < 0.0):
        raise ValueError("base_weights must be nonnegative")

    cutoff = float(cutoff)

    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")

    scaled = residuals / cutoff

    tukey = np.where(
        np.abs(scaled) < 1.0,
        (1.0 - scaled**2) ** 2,
        0.0,
    )

    return base_weights * tukey

def weighted_circular_mean(
    angles: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Compute the V3 weighted circular mean and total weight."""
    angles = np.asarray(angles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if angles.ndim != 1:
        raise ValueError("angles must be a 1D array")

    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")

    if angles.size == 0:
        raise ValueError("angles must not be empty")

    if angles.size != weights.size:
        raise ValueError(
            "angles and weights must have the same length"
        )

    if not np.all(np.isfinite(angles)):
        raise ValueError("angles must contain only finite values")

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")

    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")

    sine_sum = float(np.sum(weights * np.sin(angles)))
    cosine_sum = float(np.sum(weights * np.cos(angles)))
    total_weight = float(np.sum(weights))

    if total_weight <= 0.0:
        raise ValueError("total weight must be positive")

    angle = wrap_angle_pi(
        float(np.arctan2(sine_sum, cosine_sum))
    )

    return angle, total_weight

def irls_iteration(
    angles: np.ndarray,
    base_weights: np.ndarray,
    previous_angle: float,
    previous_weights: np.ndarray,
    s_min: float,
) -> tuple[float, np.ndarray, float, bool]:
    """Perform one V3 robust circular-fusion IRLS iteration.

    Returns:
        updated_angle,
        updated_weights,
        scale,
        low_confidence
    """
    angles = np.asarray(angles, dtype=np.float64)
    base_weights = np.asarray(base_weights, dtype=np.float64)
    previous_weights = np.asarray(previous_weights, dtype=np.float64)

    if angles.ndim != 1:
        raise ValueError("angles must be a 1D array")

    if base_weights.ndim != 1:
        raise ValueError("base_weights must be a 1D array")

    if previous_weights.ndim != 1:
        raise ValueError("previous_weights must be a 1D array")

    if angles.size == 0:
        raise ValueError("angles must not be empty")

    if angles.size != base_weights.size:
        raise ValueError(
            "angles and base_weights must have the same length"
        )

    if angles.size != previous_weights.size:
        raise ValueError(
            "angles and previous_weights must have the same length"
        )

    if not np.all(np.isfinite(angles)):
        raise ValueError("angles must contain only finite values")

    if not np.all(np.isfinite(base_weights)):
        raise ValueError(
            "base_weights must contain only finite values"
        )

    if not np.all(np.isfinite(previous_weights)):
        raise ValueError(
            "previous_weights must contain only finite values"
        )

    if np.any(base_weights < 0.0):
        raise ValueError("base_weights must be nonnegative")

    if np.any(previous_weights < 0.0):
        raise ValueError("previous_weights must be nonnegative")

    previous_angle = wrap_angle_pi(previous_angle)

    residuals = np.array(
        [
            wrap_angle_pi(angle - previous_angle)
            for angle in angles
        ],
        dtype=np.float64,
    )

    scale = irls_scale(residuals, s_min)

    cutoff = tukey_cutoff(scale)

    weights = irls_weights(
        base_weights,
        residuals,
        cutoff,
    )

    total_weight = float(np.sum(weights))

    if total_weight <= 0.0:
        return (
            previous_angle,
            previous_weights.copy(),
            scale,
            True,
        )

    updated_angle, _ = weighted_circular_mean(
        angles,
        weights,
    )

    return (
        updated_angle,
        weights,
        scale,
        False,
    )

def robust_circular_fusion(
    angles: np.ndarray,
    base_weights: np.ndarray,
    s_min: float,
    max_iterations: int = 5,
    convergence_tolerance: float = np.deg2rad(0.01),
) -> tuple[float, np.ndarray, int, bool]:
    """Perform the V3 robust circular IRLS fusion.

    Returns:
        fused_angle,
        final_weights,
        iterations_used,
        low_confidence
    """
    angles = np.asarray(angles, dtype=np.float64)
    base_weights = np.asarray(base_weights, dtype=np.float64)

    if angles.ndim != 1:
        raise ValueError("angles must be a 1D array")

    if base_weights.ndim != 1:
        raise ValueError("base_weights must be a 1D array")

    if angles.size == 0:
        raise ValueError("angles must not be empty")

    if angles.size != base_weights.size:
        raise ValueError(
            "angles and base_weights must have the same length"
        )

    if not np.all(np.isfinite(angles)):
        raise ValueError("angles must contain only finite values")

    if not np.all(np.isfinite(base_weights)):
        raise ValueError(
            "base_weights must contain only finite values"
        )

    if np.any(base_weights < 0.0):
        raise ValueError("base_weights must be nonnegative")

    if not np.isfinite(s_min) or s_min <= 0.0:
        raise ValueError("s_min must be finite and positive")

    max_iterations = int(max_iterations)

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    convergence_tolerance = float(convergence_tolerance)

    if (
        not np.isfinite(convergence_tolerance)
        or convergence_tolerance <= 0.0
    ):
        raise ValueError(
            "convergence_tolerance must be finite and positive"
        )

    theta = circular_median(angles)

    previous_weights = base_weights.copy()

    for iteration in range(1, max_iterations + 1):
        updated_theta, updated_weights, _, low_confidence = (
            irls_iteration(
                angles,
                base_weights,
                previous_angle=theta,
                previous_weights=previous_weights,
                s_min=s_min,
            )
        )

        if low_confidence:
            return (
                theta,
                updated_weights,
                iteration,
                True,
            )

        change = abs(
            wrap_angle_pi(updated_theta - theta)
        )

        theta = updated_theta
        previous_weights = updated_weights

        if change < convergence_tolerance:
            return (
                theta,
                previous_weights,
                iteration,
                False,
            )

    return (
        theta,
        previous_weights,
        max_iterations,
        False,
    )

def effective_sample_size(weights: np.ndarray) -> float:
    """Compute the V3 effective light/component sample size."""
    weights = np.asarray(weights, dtype=np.float64)

    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")

    if weights.size == 0:
        raise ValueError("weights must not be empty")

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")

    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")

    weight_sum = float(np.sum(weights))
    squared_weight_sum = float(np.sum(weights**2))

    if weight_sum <= 0.0:
        raise ValueError("sum of weights must be positive")

    if squared_weight_sum <= 0.0:
        raise ValueError("sum of squared weights must be positive")

    return float(
        weight_sum**2 / squared_weight_sum
    )

def residual_variance(
    residuals: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the V3 weighted residual variance."""
    residuals = np.asarray(residuals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1D array")

    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")

    if residuals.size == 0:
        raise ValueError("residuals must not be empty")

    if residuals.size != weights.size:
        raise ValueError(
            "residuals and weights must have the same length"
        )

    if not np.all(np.isfinite(residuals)):
        raise ValueError(
            "residuals must contain only finite values"
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain only finite values"
        )

    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")

    weight_sum = float(np.sum(weights))

    if weight_sum <= 0.0:
        raise ValueError("sum of weights must be positive")

    return float(
        np.sum(weights * residuals**2) / weight_sum
    )

def fused_azimuth_uncertainty(
    residuals: np.ndarray,
    weights: np.ndarray,
    sigma_sys: float = np.deg2rad(0.5),
) -> float:
    """Compute the V3 fused solar-azimuth uncertainty."""
    residuals = np.asarray(residuals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    sigma_sys = float(sigma_sys)

    if not np.isfinite(sigma_sys) or sigma_sys <= 0.0:
        raise ValueError(
            "sigma_sys must be finite and positive"
        )

    n_eff = effective_sample_size(weights)

    s_res_squared = residual_variance(
        residuals,
        weights,
    )

    s_res = float(np.sqrt(s_res_squared))

    statistical_error = s_res / np.sqrt(n_eff)

    return float(
        max(
            statistical_error,
            sigma_sys,
        )
    )

def fuse_solar_azimuths(
    angles: np.ndarray,
    base_weights: np.ndarray,
    s_min: float,
    sigma_sys: float = np.deg2rad(0.5),
    max_iterations: int = 5,
    convergence_tolerance: float = np.deg2rad(0.01),
) -> tuple[float, np.ndarray, float, float, float, int, bool]:
    """Perform the complete V3 §5.6 robust circular fusion.

    Returns:
        fused_angle,
        final_weights,
        effective_sample_size,
        residual_scale,
        azimuth_uncertainty,
        iterations_used,
        low_confidence
    """
    angles = np.asarray(angles, dtype=np.float64)
    base_weights = np.asarray(base_weights, dtype=np.float64)

    fused_angle, final_weights, iterations, low_confidence = (
        robust_circular_fusion(
            angles,
            base_weights,
            s_min=s_min,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
        )
    )

    if low_confidence:
        return (
            fused_angle,
            final_weights,
            0.0,
            0.0,
            float(sigma_sys),
            iterations,
            True,
        )

    final_residuals = np.array(
        [
            wrap_angle_pi(angle - fused_angle)
            for angle in angles
        ],
        dtype=np.float64,
    )

    n_eff = effective_sample_size(final_weights)

    residual_variance_value = residual_variance(
        final_residuals,
        final_weights,
    )

    residual_scale = float(
        np.sqrt(residual_variance_value)
    )

    sigma_phi = fused_azimuth_uncertainty(
        final_residuals,
        final_weights,
        sigma_sys=sigma_sys,
    )

    return (
        fused_angle,
        final_weights,
        n_eff,
        residual_scale,
        sigma_phi,
        iterations,
        False,
    )