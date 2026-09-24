from collections.abc import Collection

import numpy as np

from patchrift.geometry.handedness import (
    fit_geolocation_jacobian,
    normalize_handedness,
)
from .product import ImageProduct, ProductMetadata


def ingest_array(
    array: np.ndarray,
    fill_value: float | int | Collection[float | int] | None,
    metadata: ProductMetadata,
) -> ImageProduct:
    """
    Ingest an image array according to the Version 3 specification.

    The image is converted to float32, a valid-pixel mask is created,
    and the image frame is handedness-normalized once at ingest.
    """

    image = np.asarray(array, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError("array must be a 2D scalar image")

    # The specification requires every subsequent reduction to use
    # valid pixels only.  Non-finite samples and every declared fill
    # value are therefore excluded at ingest, including NaN fills.
    valid_mask = np.isfinite(image)

    if fill_value is not None:
        if isinstance(fill_value, Collection) and not isinstance(
            fill_value,
            (str, bytes),
        ):
            fill_values = tuple(fill_value)
        else:
            fill_values = (fill_value,)

        for value in fill_values:
            if np.isnan(value):
                valid_mask &= ~np.isnan(image)
            else:
                valid_mask &= image != value

    jacobian = fit_geolocation_jacobian(
        metadata.footprint_corners,
        metadata.footprint_pixel_coords,
    )

    (
        image,
        valid_mask,
        footprint_pixel_coords,
        was_reflected,
    ) = normalize_handedness(
        image,
        valid_mask,
        metadata.footprint_pixel_coords,
        jacobian,
    )

    if was_reflected:
        metadata = ProductMetadata(
            timestamp=metadata.timestamp,
            footprint_corners=metadata.footprint_corners,
            footprint_pixel_coords=footprint_pixel_coords,
            gsd=metadata.gsd,
            emission_angle=metadata.emission_angle,
            hmax=metadata.hmax,
            latitude_uncertainty=metadata.latitude_uncertainty,
            longitude_uncertainty=metadata.longitude_uncertainty,
        )

    return ImageProduct(
        image=image,
        valid_mask=valid_mask,
        metadata=metadata,
        was_reflected=was_reflected,
    )
