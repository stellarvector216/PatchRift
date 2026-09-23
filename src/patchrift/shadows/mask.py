import numpy as np


def build_shadow_mask(
    image: np.ndarray,
    threshold_field: np.ndarray,
    defined_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Build the raw shadow mask from a per-pixel threshold field.

    A pixel belongs to the shadow mask only when:
    - its threshold is defined,
    - the pixel itself is valid,
    - its intensity is less than or equal to the threshold.

    Undefined threshold locations are never included in the mask.
    """

    if image.shape != threshold_field.shape:
        raise ValueError(
            "image and threshold_field must have the same shape"
        )

    if image.shape != defined_mask.shape:
        raise ValueError(
            "image and defined_mask must have the same shape"
        )

    if image.shape != valid_mask.shape:
        raise ValueError(
            "image and valid_mask must have the same shape"
        )

    shadow_mask = (
        defined_mask
        & valid_mask
        & (image <= threshold_field)
    )

    return shadow_mask