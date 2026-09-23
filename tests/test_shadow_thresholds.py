from patchrift.shadows.thresholds import (
    SubtileThreshold,
)


def test_subtile_threshold_stores_bounds_and_value():
    threshold = SubtileThreshold(
        row_start=0,
        row_end=1024,
        col_start=1024,
        col_end=2048,
        threshold=12.5,
    )

    assert threshold.row_start == 0
    assert threshold.row_end == 1024
    assert threshold.col_start == 1024
    assert threshold.col_end == 2048
    assert threshold.threshold == 12.5
import numpy as np

from patchrift.shadows.thresholds import (
    compute_subtile_thresholds,
)


def test_compute_subtile_thresholds_returns_one_threshold_per_qualified_tile():
    image = np.zeros(
        (2048, 2048),
        dtype=np.float32,
    )

    image[:1024, :1024] = np.arange(
        1024,
        dtype=np.float32,
    )[:, None]

    image[1024:, :1024] = np.arange(
        1024,
        dtype=np.float32,
    )[:, None]

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    thresholds = compute_subtile_thresholds(
        image,
        valid_mask,
        q=0.05,
    )

    assert len(thresholds) == 2

    assert thresholds[0].row_start == 0
    assert thresholds[0].row_end == 1024
    assert thresholds[0].col_start == 0
    assert thresholds[0].col_end == 1024

    assert thresholds[1].row_start == 1024
    assert thresholds[1].row_end == 2048
    assert thresholds[1].col_start == 0
    assert thresholds[1].col_end == 1024

def test_compute_subtile_thresholds_excludes_low_variance_tiles():
    image = np.zeros(
        (2048, 2048),
        dtype=np.float32,
    )

    image[:1024, 1024:] = np.arange(
        1024,
        dtype=np.float32,
    )[None, :]

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    thresholds = compute_subtile_thresholds(
        image,
        valid_mask,
        q=0.05,
    )

    assert len(thresholds) == 1

    assert thresholds[0].row_start == 0
    assert thresholds[0].row_end == 1024
    assert thresholds[0].col_start == 1024
    assert thresholds[0].col_end == 2048

def test_compute_subtile_thresholds_returns_empty_when_no_tile_qualifies():
    image = np.ones(
        (2048, 2048),
        dtype=np.float32,
    )

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    thresholds = compute_subtile_thresholds(
        image,
        valid_mask,
        q=0.05,
    )

    assert thresholds == []


def test_compute_subtile_thresholds_respects_valid_mask():
    image = np.ones(
        (1024, 1024),
        dtype=np.float32,
    )

    image[0, 0] = 1000.0

    valid_mask = np.ones(
        image.shape,
        dtype=bool,
    )

    valid_mask[0, 0] = False

    thresholds = compute_subtile_thresholds(
        image,
        valid_mask,
        q=0.05,
    )

    assert thresholds == []