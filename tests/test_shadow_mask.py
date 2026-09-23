import numpy as np
import pytest

from patchrift.shadows.mask import build_shadow_mask


def test_build_shadow_mask_uses_defined_thresholds():
    image = np.array(
        [
            [1.0, 5.0],
            [3.0, 8.0],
        ]
    )

    threshold_field = np.array(
        [
            [2.0, 4.0],
            [3.0, 7.0],
        ]
    )

    defined_mask = np.array(
        [
            [True, True],
            [True, True],
        ]
    )

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    result = build_shadow_mask(
        image,
        threshold_field,
        defined_mask,
        valid_mask,
    )

    expected = np.array(
        [
            [True, False],
            [True, False],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_undefined_threshold_cannot_enter_shadow_mask():
    image = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    threshold_field = np.array(
        [
            [2.0, 2.0],
            [2.0, 2.0],
        ]
    )

    defined_mask = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    result = build_shadow_mask(
        image,
        threshold_field,
        defined_mask,
        valid_mask,
    )

    expected = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_invalid_pixels_cannot_enter_shadow_mask():
    image = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    threshold_field = np.full(
        image.shape,
        2.0,
    )

    defined_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    valid_mask = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    result = build_shadow_mask(
        image,
        threshold_field,
        defined_mask,
        valid_mask,
    )

    expected = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_shadow_mask_rejects_shape_mismatch():
    image = np.ones((2, 2))
    threshold_field = np.ones((2, 3))
    defined_mask = np.ones((2, 2), dtype=bool)
    valid_mask = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError):
        build_shadow_mask(
            image,
            threshold_field,
            defined_mask,
            valid_mask,
        )