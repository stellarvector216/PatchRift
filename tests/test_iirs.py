import numpy as np
import pytest

from patchrift.io.iirs import IIRSProduct
from patchrift.preprocessing.iirs import reduce_iirs_to_panchromatic


def test_iirs_reduction_uses_only_valid_spectral_range():
    cube = np.array(
        [
            [[10.0, 10.0], [10.0, 10.0]],  # 0.7 µm — excluded
            [[20.0, 20.0], [20.0, 20.0]],  # 1.0 µm — included
            [[30.0, 30.0], [30.0, 30.0]],  # 3.0 µm — excluded
        ],
        dtype=np.float32,
    )

    product = IIRSProduct(
        cube=cube,
        wavelengths=np.array([0.7, 1.0, 3.0]),
        bad_bands=np.array([False, False, False]),
        snr=np.array([1.0, 1.0, 1.0]),
    )

    result = reduce_iirs_to_panchromatic(product)

    np.testing.assert_allclose(
        result,
        np.full((2, 2), 20.0, dtype=np.float32),
    )


def test_iirs_reduction_excludes_bad_bands():
    cube = np.array(
        [
            [[10.0, 10.0], [10.0, 10.0]],
            [[100.0, 100.0], [100.0, 100.0]],
        ],
        dtype=np.float32,
    )

    product = IIRSProduct(
        cube=cube,
        wavelengths=np.array([1.0, 1.5]),
        bad_bands=np.array([False, True]),
        snr=np.array([1.0, 1.0]),
    )

    result = reduce_iirs_to_panchromatic(product)

    np.testing.assert_allclose(
        result,
        np.full((2, 2), 10.0, dtype=np.float32),
    )


def test_iirs_reduction_uses_snr_as_fallback_weight():
    cube = np.array(
        [
            [[10.0, 10.0], [10.0, 10.0]],
            [[30.0, 30.0], [30.0, 30.0]],
        ],
        dtype=np.float32,
    )

    product = IIRSProduct(
        cube=cube,
        wavelengths=np.array([1.0, 2.0]),
        bad_bands=np.array([False, False]),
        snr=np.array([1.0, 3.0]),
    )

    result = reduce_iirs_to_panchromatic(product)

    expected = (10.0 * 1.0 + 30.0 * 3.0) / 4.0

    np.testing.assert_allclose(
        result,
        np.full((2, 2), expected, dtype=np.float32),
    )


def test_iirs_reduction_accepts_explicit_spectral_weights():
    cube = np.array(
        [
            [[10.0, 10.0], [10.0, 10.0]],
            [[30.0, 30.0], [30.0, 30.0]],
        ],
        dtype=np.float32,
    )

    product = IIRSProduct(
        cube=cube,
        wavelengths=np.array([1.0, 2.0]),
        bad_bands=np.array([False, False]),
        snr=np.array([1.0, 1.0]),
    )

    weights = np.array([3.0, 1.0])

    result = reduce_iirs_to_panchromatic(
        product,
        weights=weights,
    )

    expected = (10.0 * 3.0 + 30.0 * 1.0) / 4.0

    np.testing.assert_allclose(
        result,
        np.full((2, 2), expected, dtype=np.float32),
    )


def test_iirs_reduction_fails_when_no_valid_bands_remain():
    cube = np.ones((2, 2, 2), dtype=np.float32)

    product = IIRSProduct(
        cube=cube,
        wavelengths=np.array([0.5, 3.0]),
        bad_bands=np.array([False, False]),
        snr=np.array([1.0, 1.0]),
    )

    with pytest.raises(ValueError, match="no valid IIRS bands"):
        reduce_iirs_to_panchromatic(product)


def test_iirs_reduction_ignores_invalid_band_samples_per_pixel():
    product = IIRSProduct(
        cube=np.array(
            [
                [[10.0, 10.0]],
                [[30.0, np.nan]],
            ],
            dtype=np.float32,
        ),
        wavelengths=np.array([1.0, 2.0]),
        bad_bands=np.array([False, False]),
        snr=np.array([1.0, 1.0]),
    )

    result = reduce_iirs_to_panchromatic(product)

    np.testing.assert_allclose(result[0, 0], 20.0)
    np.testing.assert_allclose(result[0, 1], 10.0)


@pytest.mark.parametrize("snr", [np.nan, np.inf, -np.inf])
def test_iirs_product_rejects_nonfinite_snr(snr):
    with pytest.raises(ValueError, match="snr must be finite"):
        IIRSProduct(
            cube=np.ones((1, 2, 2)),
            wavelengths=np.array([1.0]),
            bad_bands=np.array([False]),
            snr=np.array([snr]),
        )
