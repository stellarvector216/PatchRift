import numpy as np

from patchrift.io.iirs import IIRSProduct


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

    denominator = np.sum(selected_weights)

    reduced = np.sum(
        selected_cube * selected_weights[:, None, None],
        axis=0,
    ) / denominator

    return reduced.astype(np.float32)