from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

from patchrift.io.iirs import IIRSProduct


def tmc_spectral_overlap_weights(
    product: IIRSProduct,
    response: Callable[[float], float] | tuple[np.ndarray, np.ndarray],
    response_domain: tuple[float, float] | None = None,
) -> np.ndarray:
    """Integrate the TMC spectral response over each IIRS band.

    A callable response must be accompanied by its finite wavelength
    domain. A lookup table is a pair of wavelength and response arrays;
    linear interpolation is used only between its first and last sample.
    Portions of IIRS bands outside that domain contribute exactly zero.
    """
    if product.band_bounds is None:
        raise ValueError("IIRS band_bounds are required to integrate response overlap")

    if callable(response):
        if response_domain is None:
            raise ValueError("response_domain is required for a callable response")
        domain = np.asarray(response_domain, dtype=float)
        if domain.shape != (2,) or not np.all(np.isfinite(domain)) or domain[0] >= domain[1]:
            raise ValueError("response_domain must be finite increasing wavelength limits")

        def response_at(wavelength: float) -> float:
            value = float(response(wavelength))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("spectral response must be finite and non-negative")
            return value

        evaluate = response_at
    else:
        if response_domain is not None:
            raise ValueError("response_domain is inferred from a lookup table")
        if not isinstance(response, tuple) or len(response) != 2:
            raise TypeError("response must be callable or a (wavelengths, values) tuple")
        wavelengths = np.asarray(response[0], dtype=float)
        values = np.asarray(response[1], dtype=float)
        if (
            wavelengths.ndim != 1 or values.shape != wavelengths.shape
            or wavelengths.size < 2 or not np.all(np.isfinite(wavelengths))
            or not np.all(np.isfinite(values)) or np.any(np.diff(wavelengths) <= 0.0)
            or np.any(values < 0.0)
        ):
            raise ValueError("response lookup arrays must be finite, increasing, and non-negative")
        domain = np.array([wavelengths[0], wavelengths[-1]])
        evaluate = lambda wavelength: float(np.interp(wavelength, wavelengths, values))

    weights = np.zeros(product.cube.shape[0], dtype=float)
    for index, (band_low, band_high) in enumerate(product.band_bounds):
        low = max(float(band_low), float(domain[0]))
        high = min(float(band_high), float(domain[1]))
        if high > low:
            weights[index] = quad(evaluate, low, high, epsabs=1e-10, epsrel=1e-8)[0]
    return weights


def reduce_iirs_to_panchromatic(
    product: IIRSProduct,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reduce an IIRS hyperspectral cube to a single scalar image
    according to Version 3 §4.2.

    Only bands in the 0.8–2.5 µm interval are eligible.

    Parameters
    ----------
    product:
        Raw IIRS hyperspectral product.

    weights:
        Optional spectral weights. When supplied, these represent the
        TMC spectral-response overlap weights. When omitted, the band's
        SNR is used as the fallback weight.

    Returns
    -------
    np.ndarray
        Reduced 2D float32 image.
    """

    cube = product.cube
    wavelengths = product.wavelengths
    bad_bands = product.bad_bands

    if weights is None:
        weights = product.snr
    else:
        weights = np.asarray(weights, dtype=float)

        if weights.shape != wavelengths.shape:
            raise ValueError(
                "weights must contain one value per band"
            )

        if np.any(weights < 0):
            raise ValueError(
                "weights cannot be negative"
            )

    eligible = (
        (~bad_bands)
        & (wavelengths >= 0.8)
        & (wavelengths <= 2.5)
        & (weights > 0)
    )

    if not np.any(eligible):
        raise ValueError(
            "no valid IIRS bands remain after wavelength and quality filtering"
        )

    selected_cube = cube[eligible]
    selected_weights = weights[eligible]
    selected_valid = product.valid_mask[eligible]

    weighted_valid = selected_valid * selected_weights[:, None, None]
    denominator = np.sum(weighted_valid, axis=0)

    numerator = np.sum(
        np.where(selected_valid, selected_cube, 0.0)
        * selected_weights[:, None, None],
        axis=0,
    )

    reduced = np.full(
        cube.shape[1:],
        np.nan,
        dtype=np.float32,
    )
    np.divide(
        numerator,
        denominator,
        out=reduced,
        where=denominator > 0.0,
    )

    return reduced.astype(np.float32)
