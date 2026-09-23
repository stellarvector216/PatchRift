import numpy as np

from patchrift.geometry.handedness import (
    fit_geolocation_jacobian,
    normalize_handedness,
)
from .product import ImageProduct, ProductMetadata


def ingest_array(
    array: np.ndarray,
    fill_value: float | int,
    metadata: ProductMetadata,
) -> ImageProduct:
    """
    Ingest an image array according to the Version 3 specification.

    The image is converted to float32, a valid-pixel mask is created,
    and the image frame is handedness-normalized once at ingest.
    """

    image = np.asarray(array, dtype=np.float32)
    valid_mask = image != fill_value

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
        )

    return ImageProduct(
        image=image,
        valid_mask=valid_mask,
        metadata=metadata,
        was_reflected=was_reflected,
    )