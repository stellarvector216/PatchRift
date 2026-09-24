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
    # Emission angle is in radians, consistent with §2.2's internal convention.
    emission_angle: float
    hmax: float | None = None
    # Optional 1-sigma geolocation uncertainties, in radians, for Eq. (43).
    latitude_uncertainty: float | None = None
    longitude_uncertainty: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")

        self.footprint_corners = np.asarray(
            self.footprint_corners,
            dtype=float,
        )
        self.footprint_pixel_coords = np.asarray(
            self.footprint_pixel_coords,
            dtype=float,
        )

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

        if self.footprint_corners.shape[0] < 3:
            raise ValueError(
                "at least three footprint points are required"
            )

        if not np.all(np.isfinite(self.footprint_corners)):
            raise ValueError("footprint_corners must be finite")

        if not np.all(np.isfinite(self.footprint_pixel_coords)):
            raise ValueError("footprint_pixel_coords must be finite")

        if not np.isfinite(self.gsd) or self.gsd <= 0:
            raise ValueError("gsd must be positive")

        if not np.isfinite(self.emission_angle):
            raise ValueError("emission_angle must be finite")

        if (
            self.hmax is not None
            and (not np.isfinite(self.hmax) or self.hmax < 0)
        ):
            raise ValueError("hmax cannot be negative")
        for name in ("latitude_uncertainty", "longitude_uncertainty"):
            uncertainty = getattr(self, name)
            if uncertainty is not None and (
                not np.isfinite(uncertainty) or uncertainty < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative radians")

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

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image, dtype=np.float32)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)

        if self.image.ndim != 2:
            raise ValueError("image must be a 2D array")

        if self.valid_mask.shape != self.image.shape:
            raise ValueError(
                "valid_mask must have the same shape as image"
            )

        if np.any(self.valid_mask & ~np.isfinite(self.image)):
            raise ValueError(
                "valid image pixels must contain only finite values"
            )
