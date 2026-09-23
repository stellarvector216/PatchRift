import numpy as np
import pytest

from patchrift.shadows.variance import (
    subtile_has_sufficient_variance,
)


def test_constant_subtile_is_excluded():
    image = np.ones(
        (8, 8),
        dtype=np.float32,
    )

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    assert not subtile_has_sufficient_variance(
        image,
        valid_mask,
    )


def test_varying_subtile_is_qualified():
    image = np.zeros(
        (8, 8),
        dtype=np.float32,
    )

    image[:, 4:] = 1.0

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    assert subtile_has_sufficient_variance(
        image,
        valid_mask,
    )


def test_invalid_pixels_are_excluded_from_variance():
    image = np.ones(
        (4, 4),
        dtype=np.float32,
    )

    image[0, 0] = 1000.0

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    valid_mask[0, 0] = False

    assert not subtile_has_sufficient_variance(
        image,
        valid_mask,
    )


def test_no_valid_pixels_are_excluded():
    image = np.ones(
        (4, 4),
        dtype=np.float32,
    )

    valid_mask = np.zeros(
        image.shape,
        dtype=bool,
    )

    assert not subtile_has_sufficient_variance(
        image,
        valid_mask,
    )


def test_normalized_variance_threshold_is_inclusive():
    image = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    values = image[valid_mask]
    variance = np.var(values)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    normalized_variance = variance / (iqr ** 2)

    assert subtile_has_sufficient_variance(
        image,
        valid_mask,
        epsilon_var=normalized_variance,
    )

def test_rejects_negative_variance_threshold():
    image = np.ones(
        (4, 4),
        dtype=np.float32,
    )

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    with pytest.raises(ValueError):
        subtile_has_sufficient_variance(
            image,
            valid_mask,
            epsilon_var=-1.0,
        )
def test_subtile_variance_uses_iqr_normalization():
    image = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )

    valid_mask = np.ones_like(
        image,
        dtype=bool,
    )

    result = subtile_has_sufficient_variance(
        image,
        valid_mask,
        epsilon_var=0.01,
    )

    assert result is True
def test_subtile_zero_iqr_is_excluded():
    image = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ]
    )

    valid_mask = np.ones_like(
        image,
        dtype=bool,
    )

    result = subtile_has_sufficient_variance(
        image,
        valid_mask,
        epsilon_var=1e-3,
    )

    assert result is False