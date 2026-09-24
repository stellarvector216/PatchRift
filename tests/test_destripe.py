import numpy as np
import pytest

from patchrift.preprocessing.destripe import destripe


def test_destripe_removes_column_offsets():
    image = np.array(
        [
            [11.0, 22.0, 33.0],
            [11.0, 22.0, 33.0],
            [11.0, 22.0, 33.0],
            [11.0, 22.0, 33.0],
            [11.0, 22.0, 33.0],
        ],
        dtype=np.float32,
    )

    valid_mask = np.ones_like(image, dtype=bool)

    result = destripe(image, valid_mask)

    # The column offsets are removed. The global median is 22.
    expected = np.full_like(image, 22.0)

    np.testing.assert_allclose(result, expected)


def test_destripe_uses_valid_pixels_only():
    image = np.array(
        [
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 300.0],
        ],
        dtype=np.float32,
    )

    valid_mask = np.array(
        [
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]
    )

    result = destripe(image, valid_mask)

    # The invalid 300 must not affect the column median or global median.
    # Valid global values are [10, 20, 30, 10, 20, 30, 10, 20].
    # Their median is 20.
    #
    # The valid pixels in each column already have medians
    # 10, 20, and 30, so all valid pixels become 20 before filtering.
    expected = np.full_like(image, 20.0)

    np.testing.assert_allclose(
        result[valid_mask],
        expected[valid_mask],
    )


def test_destripe_does_not_filter_invalid_fill_values_into_neighbors():
    image = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.0, -999.0, 10.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    valid_mask = image != -999.0

    result = destripe(image, valid_mask)

    np.testing.assert_allclose(result[valid_mask], 10.0)
    assert result[1, 1] == -999.0


def test_destripe_preserves_float32_output():
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    valid_mask = np.ones_like(image, dtype=bool)

    result = destripe(image, valid_mask)

    assert result.dtype == np.float32


def test_destripe_rejects_empty_valid_mask():
    image = np.ones((4, 4), dtype=np.float32)
    valid_mask = np.zeros_like(image, dtype=bool)

    with pytest.raises(ValueError, match="no valid pixels"):
        destripe(image, valid_mask)
