from __future__ import annotations

import numpy as np


SCHARR_X = np.array(
    [
        [-3.0, 0.0, 3.0],
        [-10.0, 0.0, 10.0],
        [-3.0, 0.0, 3.0],
    ],
    dtype=np.float64,
) / 32.0

SCHARR_Y = SCHARR_X.T.copy()


def scharr_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Return the V3 3x3 Scharr gradient kernels."""
    return SCHARR_X.copy(), SCHARR_Y.copy()

def scharr_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute V3 Scharr gradients using the 1/32-scaled kernels."""
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    image = np.asarray(image, dtype=np.float64)

    gx_kernel, gy_kernel = scharr_kernels()

    from scipy.ndimage import correlate

    gx = correlate(image, gx_kernel, mode="constant", cval=0.0)
    gy = correlate(image, gy_kernel, mode="constant", cval=0.0)
    return gx, gy

def gradient_magnitude(
    gx: np.ndarray,
    gy: np.ndarray,
) -> np.ndarray:
    """Compute Scharr gradient magnitude M."""
    if gx.shape != gy.shape:
        raise ValueError("gx and gy must have the same shape")

    return np.hypot(gx, gy)

def gradient_orientation(
    gx: np.ndarray,
    gy: np.ndarray,
) -> np.ndarray:
    """Compute the V3 shadow-directed gradient orientation.

    V3 Eq. 12:
        theta = atan2(-Gy, -Gx)

    The resulting angle is in radians in the standard NumPy
    [-pi, pi) convention.
    """
    if gx.shape != gy.shape:
        raise ValueError("gx and gy must have the same shape")

    return np.arctan2(-gy, -gx)

def scharr_azimuth_resultant(
    magnitude: np.ndarray,
    orientation: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute the V3 Eq. 34 Scharr circular resultant.

    Returns:
        C: weighted cosine sum
        S: weighted sine sum
        phi: resultant azimuth
        rho_bar: mean resultant length
    """
    if magnitude.shape != orientation.shape:
        raise ValueError("magnitude and orientation must have the same shape")

    magnitude = np.asarray(magnitude, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)

    if not np.all(np.isfinite(magnitude)):
        raise ValueError("magnitude must contain only finite values")

    if not np.all(np.isfinite(orientation)):
        raise ValueError("orientation must contain only finite values")

    if np.any(magnitude < 0.0):
        raise ValueError("magnitude must be non-negative")

    total_magnitude = float(np.sum(magnitude))

    if total_magnitude <= 0.0:
        raise ValueError("total gradient magnitude must be positive")

    C = float(np.sum(magnitude * np.cos(orientation)))
    S = float(np.sum(magnitude * np.sin(orientation)))

    phi = float(np.arctan2(S, C))
    rho_bar = float(np.hypot(C, S) / total_magnitude)

    return C, S, phi, rho_bar

def select_scharr_samples(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    boundary_region: np.ndarray,
    valid_mask: np.ndarray,
    saturated_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select valid Scharr samples from the existing V3 boundary region R.

    V3 §5.5 requires gradients to contribute only from pixels that:
    - lie inside R,
    - are not no-data/invalid,
    - are not saturated.
    """
    if not (
        magnitude.shape
        == orientation.shape
        == boundary_region.shape
        == valid_mask.shape
        == saturated_mask.shape
    ):
        raise ValueError(
            "magnitude, orientation, and all masks must have the same shape"
        )

    magnitude = np.asarray(magnitude, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    boundary_region = np.asarray(boundary_region, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    saturated_mask = np.asarray(saturated_mask, dtype=bool)

    if not np.all(np.isfinite(magnitude)):
        raise ValueError("magnitude must contain only finite values")

    if not np.all(np.isfinite(orientation)):
        raise ValueError("orientation must contain only finite values")

    if np.any(magnitude < 0.0):
        raise ValueError("magnitude must be non-negative")

    selection = (
        boundary_region
        & valid_mask
        & ~saturated_mask
    )

    return magnitude[selection], orientation[selection]

def scharr_region_status(
    boundary_region: np.ndarray,
    selected_magnitude: np.ndarray,
) -> tuple[bool, bool]:
    """Classify the V3 §5.5 boundary-gradient region.

    Returns:
        region_nonempty:
            True when R contains at least one pixel.

        samples_available:
            True when at least one pixel from R survives the
            valid/no-data/saturation selection.
    """
    boundary_region = np.asarray(boundary_region, dtype=bool)
    selected_magnitude = np.asarray(selected_magnitude)

    if boundary_region.ndim != 2:
        raise ValueError("boundary_region must be a 2D array")

    region_nonempty = bool(np.any(boundary_region))
    samples_available = bool(selected_magnitude.size > 0)

    return region_nonempty, samples_available

def scharr_fallback_uncertainty(
    rho_bar: float,
) -> float:
    """Compute the V3 §5.5 Scharr-only azimuth uncertainty.

    V3:
        sigma_phi = sqrt(-2 ln(rho_bar_1))
    """
    rho_bar = float(rho_bar)

    if not np.isfinite(rho_bar):
        raise ValueError("rho_bar must be finite")

    if rho_bar < 0.0 or rho_bar > 1.0:
        raise ValueError("rho_bar must lie in [0, 1]")

    return float(np.sqrt(-2.0 * np.log(rho_bar)))

def evaluate_scharr_region(
    boundary_region: np.ndarray,
    selected_magnitude: np.ndarray,
    selected_orientation: np.ndarray,
) -> tuple[bool, bool, float | None, float | None, float | None]:
    """Evaluate the §5.5 Scharr region and resultant.

    Returns:
        region_nonempty:
            Whether the existing V3 boundary region R contains pixels.

        samples_available:
            Whether usable Scharr samples remain after masking.

        phi:
            Scharr azimuth estimate, or None when no usable samples exist.

        rho_bar:
            Mean resultant length, or None when no usable samples exist.

        sigma_phi:
            Scharr-only uncertainty, or None when no usable samples exist.
    """
    region_nonempty, samples_available = scharr_region_status(
        boundary_region,
        selected_magnitude,
    )

    if not samples_available:
        return (
            region_nonempty,
            False,
            None,
            None,
            None,
        )

    if selected_orientation.shape != selected_magnitude.shape:
        raise ValueError(
            "selected_magnitude and selected_orientation "
            "must have the same shape"
        )

    _, _, phi, rho_bar = scharr_azimuth_resultant(
        selected_magnitude,
        selected_orientation,
    )

    sigma_phi = scharr_fallback_uncertainty(rho_bar)

    return (
        region_nonempty,
        True,
        phi,
        rho_bar,
        sigma_phi,
    )

def scharr_azimuth_cross_check(
    scharr_azimuth: float,
    fused_azimuth: float,
) -> tuple[float, bool]:
    """Compare the Scharr azimuth with the fused azimuth.

    V3 §5.5 flags the independent Scharr check when the wrapped
    angular disagreement exceeds 90 degrees.

    Returns:
        angular_difference:
            Signed wrapped difference in [-pi, pi).

        flagged:
            True when the absolute disagreement is greater than pi/2.
    """
    scharr_azimuth = float(scharr_azimuth)
    fused_azimuth = float(fused_azimuth)

    if not np.isfinite(scharr_azimuth):
        raise ValueError("scharr_azimuth must be finite")

    if not np.isfinite(fused_azimuth):
        raise ValueError("fused_azimuth must be finite")

    difference = scharr_azimuth - fused_azimuth

    # Wrap to [-pi, pi).
    difference = float(
        np.arctan2(
            np.sin(difference),
            np.cos(difference),
        )
    )

    flagged = abs(difference) > (np.pi / 2.0)

    return difference, flagged