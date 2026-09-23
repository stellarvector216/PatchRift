import numpy as np


def fit_geolocation_jacobian(
    footprint_corners: np.ndarray,
    footprint_pixel_coords: np.ndarray,
) -> np.ndarray:
    """
    Fit the local affine Jacobian from geographic coordinates to
    image pixel coordinates.

    Geographic coordinates are supplied as:
        [longitude, latitude] in degrees.

    Pixel coordinates are supplied as:
        [column, row].

    Longitude is scaled by cos(latitude), as specified by the
    Version 3 handedness-normalization procedure.

    Returns
    -------
    np.ndarray
        A 2x2 Jacobian matrix J such that, locally,

            [column]       [J00 J01] [scaled longitude]
            [row]    ~=    [J10 J11] [latitude]
    """

    footprint_corners = np.asarray(footprint_corners, dtype=float)
    footprint_pixel_coords = np.asarray(
        footprint_pixel_coords,
        dtype=float,
    )

    if footprint_corners.shape != footprint_pixel_coords.shape:
        raise ValueError(
            "footprint_corners and footprint_pixel_coords "
            "must have the same shape"
        )

    if footprint_corners.ndim != 2 or footprint_corners.shape[1] != 2:
        raise ValueError(
            "footprint coordinates must have shape (N, 2)"
        )

    if footprint_corners.shape[0] < 3:
        raise ValueError(
            "at least three footprint points are required"
        )

    longitude = footprint_corners[:, 0]
    latitude = footprint_corners[:, 1]

    latitude_rad = np.deg2rad(latitude)

    scaled_longitude = longitude * np.cos(latitude_rad)

    geographic = np.column_stack(
        [scaled_longitude, latitude]
    )

    # Add an affine intercept:
    #
    # pixel = A @ geographic + b
    #
    # where A is the 2x2 Jacobian we need.
    design = np.column_stack(
        [geographic, np.ones(geographic.shape[0])]
    )

    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        footprint_pixel_coords,
        rcond=None,
    )

    if rank < 3:
        raise ValueError(
            "footprint points do not define a valid affine mapping"
        )

    jacobian = coefficients[:2, :].T

    return jacobian
def is_mirrored(jacobian: np.ndarray) -> bool:
    """
    Return True when the fitted geographic-to-pixel mapping is mirrored.
    """
    return np.linalg.det(jacobian) < 0.0
def normalize_handedness(
    image: np.ndarray,
    valid_mask: np.ndarray,
    footprint_pixel_coords: np.ndarray,
    jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Normalize a mirrored image frame by reflecting the image, valid mask,
    and footprint pixel coordinates once at ingest.

    Returns
    -------
    image:
        Handedness-normalized image.
    valid_mask:
        Correspondingly reflected validity mask.
    footprint_pixel_coords:
        Correspondingly reflected pixel coordinates.
    was_reflected:
        True if a reflection was performed.
    """

    if image.shape != valid_mask.shape:
        raise ValueError(
            "image and valid_mask must have the same shape"
        )

    if footprint_pixel_coords.ndim != 2:
        raise ValueError(
            "footprint_pixel_coords must be a 2D array"
        )

    if footprint_pixel_coords.shape[1] != 2:
        raise ValueError(
            "footprint_pixel_coords must have shape (N, 2)"
        )

    if not is_mirrored(jacobian):
        return (
            image,
            valid_mask,
            footprint_pixel_coords,
            False,
        )

    width = image.shape[1]

    reflected_image = np.fliplr(image)
    reflected_mask = np.fliplr(valid_mask)

    reflected_coords = footprint_pixel_coords.copy()
    reflected_coords[:, 0] = (
        (width - 1) - reflected_coords[:, 0]
    )

    return (
        reflected_image,
        reflected_mask,
        reflected_coords,
        True,
    )