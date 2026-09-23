import numpy as np
import pytest

from patchrift.shadows.segmentation import (
    otsu_threshold,
    shadow_threshold,
)


def test_otsu_threshold_uses_valid_pixels_only():
    image = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 10.0, 10.0],
            [10.0, 10.0, 999.0],
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

    threshold = otsu_threshold(
        image,
        valid_mask,
    )

    assert threshold == 5.5


def test_otsu_threshold_rejects_no_valid_pixels():
    image = np.ones(
        (4, 4),
        dtype=np.float32,
    )

    valid_mask = np.zeros(
        (4, 4),
        dtype=bool,
    )

    with pytest.raises(ValueError):
        otsu_threshold(
            image,
            valid_mask,
        )


def test_shadow_threshold_applies_downward_percentile_floor():
    image = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
        ],
        dtype=np.float32,
    )

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    q = 0.05

    threshold = shadow_threshold(
        image,
        valid_mask,
        q,
    )

    otsu = otsu_threshold(
        image,
        valid_mask,
    )

    percentile = np.quantile(
        image[valid_mask],
        q,
    )

    assert threshold == min(
        otsu,
        percentile,
    )


@pytest.mark.parametrize(
    "q",
    [0.05, 0.075, 0.10],
)
def test_shadow_threshold_accepts_version_3_q_range(q):
    image = np.arange(
        100,
        dtype=np.float32,
    ).reshape(10, 10)

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    threshold = shadow_threshold(
        image,
        valid_mask,
        q,
    )

    assert np.isfinite(threshold)


@pytest.mark.parametrize(
    "q",
    [0.049, 0.101],
)
def test_shadow_threshold_rejects_q_outside_version_3_range(q):
    image = np.arange(
        100,
        dtype=np.float32,
    ).reshape(10, 10)

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    with pytest.raises(ValueError):
        shadow_threshold(
            image,
            valid_mask,
            q,
        )
def test_otsu_threshold_uses_midpoint_between_distinct_values():
    image = np.array([
        [1.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    ])

    valid_mask = np.ones_like(
        image,
        dtype=bool,
    )

    threshold = otsu_threshold(
        image,
        valid_mask,
    )

    assert threshold == 3.0