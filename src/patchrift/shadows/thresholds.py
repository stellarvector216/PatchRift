from dataclasses import dataclass

import numpy as np

from patchrift.shadows.segmentation import shadow_threshold
from patchrift.shadows.tiles import generate_subtiles
from patchrift.shadows.variance import (
    subtile_has_sufficient_variance,
)


@dataclass(frozen=True)
class SubtileThreshold:
    """
    Threshold associated with one qualified subtile.
    """

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    threshold: float


def compute_subtile_thresholds(
    image: np.ndarray,
    valid_mask: np.ndarray,
    q: float,
    tile_size: int = 1024,
    epsilon_var: float = 1e-3,
) -> list[SubtileThreshold]:
    """
    Compute Version 3 shadow thresholds for all qualifying subtiles.

    Each subtile is:
        1. extracted from the image,
        2. checked for sufficient valid-pixel variance,
        3. assigned an Otsu + downward-percentile threshold.

    Low-variance subtiles are excluded.

    If no subtile qualifies, an empty list is returned.
    """

    image = np.asarray(image, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if valid_mask.shape != image.shape:
        raise ValueError(
            "valid_mask must have the same shape as image"
        )

    subtiles = generate_subtiles(
        image,
        tile_size=tile_size,
    )

    thresholds: list[SubtileThreshold] = []

    for subtile in subtiles:
        image_tile = image[
            subtile.row_start:subtile.row_end,
            subtile.col_start:subtile.col_end,
        ]

        valid_mask_tile = valid_mask[
            subtile.row_start:subtile.row_end,
            subtile.col_start:subtile.col_end,
        ]

        if not subtile_has_sufficient_variance(
            image_tile,
            valid_mask_tile,
            epsilon_var=epsilon_var,
        ):
            continue

        threshold = shadow_threshold(
            image_tile,
            valid_mask_tile,
            q=q,
        )

        thresholds.append(
            SubtileThreshold(
                row_start=subtile.row_start,
                row_end=subtile.row_end,
                col_start=subtile.col_start,
                col_end=subtile.col_end,
                threshold=threshold,
            )
        )

    return thresholds