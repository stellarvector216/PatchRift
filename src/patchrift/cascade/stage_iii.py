"""Stage III geometric and photometric conditioning (§6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import convolve2d


@dataclass(frozen=True)
class StageIIIResult:
    detection_image: np.ndarray
    masks: dict[str, np.ndarray]
    native_masks: dict[str, np.ndarray]
    valid_mask: np.ndarray
    conditioned_native: np.ndarray
    gsd_target: float
    scale_ratio: float
    native_coordinate_scale: tuple[float, float]


def stage3_geometric_conditioning(
    image: np.ndarray,
    gsd_current: float,
    gsd_target: float,
    masks: dict[str, np.ndarray],
    phi_pix: float,
    lambda_max: float,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Compatibility wrapper returning (detection image, masks, native conditioned image)."""
    result = condition_image(
        image, gsd_current, gsd_target, masks, phi_pix, lambda_max, valid_mask
    )
    return result.detection_image, result.masks, result.conditioned_native


def condition_image(
    image: np.ndarray,
    gsd_current: float,
    gsd_target: float,
    masks: dict[str, np.ndarray],
    phi_pix: float,
    lambda_max: float,
    valid_mask: np.ndarray,
) -> StageIIIResult:
    """Apply Eq. (44), transport masks, and remove shading using §6.2/B5."""
    image = np.asarray(image, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if image.ndim != 2 or valid.shape != image.shape:
        raise ValueError("image and valid_mask must be matching 2D arrays")
    if np.any(valid & ~np.isfinite(image)):
        raise ValueError("valid image samples must be finite")
    if not np.any(valid):
        raise ValueError("image has no valid samples")
    if not np.isfinite(gsd_current) or gsd_current <= 0.0:
        raise ValueError("gsd_current must be finite and positive")
    if not np.isfinite(gsd_target) or gsd_target < gsd_current:
        raise ValueError("gsd_target must be finite and no finer than gsd_current")
    if not np.isfinite(phi_pix) or not np.isfinite(lambda_max) or lambda_max <= 0.0:
        raise ValueError("phi_pix must be finite and lambda_max must be positive")
    if any(np.asarray(mask).shape != image.shape for mask in masks.values()):
        raise ValueError("all masks must match the image shape")

    sigma = float(gsd_target / gsd_current)
    if sigma > 1.0:
        numerator = gaussian_filter(np.where(valid, image, 0.0), 0.5 * np.sqrt(sigma**2 - 1.0))
        support = gaussian_filter(valid.astype(float), 0.5 * np.sqrt(sigma**2 - 1.0))
        preblurred = np.divide(numerator, support, out=np.zeros_like(numerator), where=support > 1e-8)
        scaled_image = _resample_at_physical_scale(preblurred, sigma, order=3, mode="nearest")
        scaled_valid = _resample_at_physical_scale(valid.astype(np.uint8), sigma, order=0, mode="constant").astype(bool)
        scaled_masks = {
            name: _resample_at_physical_scale(np.asarray(mask, dtype=np.uint8), sigma, order=0, mode="constant").astype(bool)
            for name, mask in masks.items()
        }
    else:
        preblurred = image.copy()
        scaled_image = image.copy()
        scaled_valid = valid.copy()
        scaled_masks = {name: np.asarray(mask, dtype=bool).copy() for name, mask in masks.items()}

    non_shadow = scaled_valid & ~np.asarray(scaled_masks.get("S_int", np.zeros_like(scaled_valid)), dtype=bool)
    if not np.any(non_shadow):
        raise ValueError("photometric conditioning requires valid non-shadow pixels for epsilon")
    median = float(np.median(scaled_image[non_shadow]))
    if median <= 0.0:
        raise ValueError("median valid non-shadow intensity must be positive")
    epsilon = 0.02 * median

    detection = _condition_with_solar_lowpass(
        scaled_image, scaled_valid, non_shadow, phi_pix, lambda_max, epsilon
    )
    native = _condition_with_solar_lowpass(
        preblurred, valid, valid & ~np.asarray(masks.get("S_int", np.zeros_like(valid)), dtype=bool),
        phi_pix, lambda_max * sigma, epsilon,
    )
    return StageIIIResult(
        detection_image=detection,
        masks=scaled_masks,
        native_masks={**{name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}, "valid": valid.copy()},
        valid_mask=scaled_valid,
        conditioned_native=native,
        gsd_target=float(gsd_target),
        scale_ratio=sigma,
        native_coordinate_scale=(sigma, sigma),
    )


def _resample_at_physical_scale(array, scale, *, order, mode):
    """Sample output index ``u`` at source coordinate ``scale*u``.

    This preserves the requested physical sampling interval when dimensions
    need rounding. The final output sample stays inside the source footprint.
    """
    array = np.asarray(array)
    if array.ndim != 2 or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("array must be 2D and scale finite and positive")
    out_rows = int(np.floor((array.shape[0] - 1) / scale)) + 1
    out_cols = int(np.floor((array.shape[1] - 1) / scale)) + 1
    rows, cols = np.mgrid[:out_rows, :out_cols]
    coordinates = np.stack((rows * scale, cols * scale))
    return map_coordinates(
        array, coordinates, order=order, mode=mode, cval=0.0, prefilter=order > 1
    )


def _condition_with_solar_lowpass(image, valid, non_shadow, phi_pix, lambda_max, epsilon):
    # Addendum B5: major-axis support rho=8 lambda_max and minor-axis support
    # lambda_max, a fixed 8:1 extent ratio. The elliptical Gaussian is
    # explicitly zero outside those supports.
    kernel = solar_lowpass_kernel(lambda_max, phi_pix)
    safe_image = np.where(valid, image, 0.0)
    trend_numerator = convolve2d(safe_image, kernel, mode="same", boundary="symm")
    trend_support = convolve2d(valid.astype(float), kernel, mode="same", boundary="symm")
    trend = np.divide(trend_numerator, trend_support, out=np.zeros_like(image), where=trend_support > 1e-8)
    conditioned = np.clip(image / (trend + epsilon), 0.0, 4.0)
    conditioned[~valid] = 0.0
    if not np.all(np.isfinite(conditioned[valid])):
        raise ValueError("photometric conditioning produced non-finite valid samples")
    return conditioned


def solar_lowpass_kernel(lambda_max, phi_pix):
    """Return the normalized, explicitly truncated B5 anisotropic kernel."""
    if not np.isfinite(lambda_max) or lambda_max <= 0.0 or not np.isfinite(phi_pix):
        raise ValueError("lambda_max must be positive and phi_pix finite")
    support_major = 8.0 * lambda_max
    support_minor = lambda_max
    radius = int(np.ceil(support_major))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    along = x * np.cos(phi_pix) + y * np.sin(phi_pix)
    across = -x * np.sin(phi_pix) + y * np.cos(phi_pix)
    kernel = np.exp(-0.5 * ((along / (support_major / 2.0)) ** 2 + (across / (support_minor / 2.0)) ** 2))
    kernel[(np.abs(along) > support_major) | (np.abs(across) > support_minor)] = 0.0
    kernel /= np.sum(kernel)
    return kernel
