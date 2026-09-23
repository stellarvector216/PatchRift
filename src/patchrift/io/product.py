from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class ProductMetadata:
    """
    Metadata required by the Version 3 correspondence pipeline.

    footprint_corners has shape (N, 2), with columns:
        [longitude, latitude]
    in degrees.
    """

    timestamp: datetime
    footprint_corners: np.ndarray
    footprint_pixel_coords: np.ndarray
    gsd: float
    emission_angle: float
    hmax: float | None = None

    def __post_init__(self) -> None:
        if self.footprint_corners.ndim != 2:
            raise ValueError("footprint_corners must be a 2D array")
        if self.footprint_pixel_coords.ndim != 2:
            raise ValueError(
                "footprint_pixel_coords must be a 2D array"
            )
        if self.footprint_pixel_coords.shape[1] != 2:
            raise ValueError(
                "footprint_pixel_coords must have shape (N, 2)"
            )
        if self.footprint_pixel_coords.shape[0] != self.footprint_corners.shape[0]:
            raise ValueError(
                "footprint_pixel_coords and footprint_corners must have the same number of points"
            )
        if self.footprint_corners.shape[1] != 2:
            raise ValueError(
                "footprint_corners must have shape (N, 2)"
            )

        if self.gsd <= 0:
            raise ValueError("gsd must be positive")

        if self.hmax is not None and self.hmax < 0:
            raise ValueError("hmax cannot be negative")

@dataclass
class ImageProduct:
    """
    An image product after the initial ingest step.

    The image is always stored as float32.
    valid_mask is True for valid data pixels and False for no-data/fill pixels.
    was_reflected records whether handedness normalization reflected the
    delivered image array during ingest.
    """
    image: np.ndarray
    valid_mask: np.ndarray
    metadata: ProductMetadata
    was_reflected: bool = False