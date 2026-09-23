import numpy as np
from scipy import ndimage


def cross_structuring_element() -> np.ndarray:
    """
    Return the specified 3x3 cross structuring element B3.
    """
    return np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )


def apply_opening_and_closing(
    shadow_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply morphological opening followed by closing
    using the specified 3x3 cross structuring element.
    """

    if shadow_mask.ndim != 2:
        raise ValueError(
            "shadow_mask must be a 2D array"
        )

    structure = cross_structuring_element()

    opened = ndimage.binary_opening(
        shadow_mask,
        structure=structure,
    )

    closed = ndimage.binary_closing(
        opened,
        structure=structure,
    )

    return closed
def fill_shadow_holes(
    shadow_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill holes in a binary shadow mask using 8-connectivity.
    """

    if shadow_mask.ndim != 2:
        raise ValueError(
            "shadow_mask must be a 2D array"
        )

    connectivity_8 = np.ones(
        (3, 3),
        dtype=bool,
    )

    return ndimage.binary_fill_holes(
        shadow_mask,
        structure=connectivity_8,
    )
def label_shadow_components(
    shadow_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Label connected shadow components using 8-connectivity.

    Background pixels receive label 0.
    Shadow components receive positive integer labels.
    """

    if shadow_mask.ndim != 2:
        raise ValueError(
            "shadow_mask must be a 2D array"
        )

    connectivity_8 = np.ones(
        (3, 3),
        dtype=bool,
    )

    labels, num_components = ndimage.label(
        shadow_mask,
        structure=connectivity_8,
    )

    return labels, num_components
def filter_shadow_components(
    shadow_mask: np.ndarray,
    labels: np.ndarray,
    num_components: int,
    amin: int = 25,
    epsilon_ar: float = 8.0,
) -> np.ndarray:
    """
    Remove shadow components that fail the area or bounding-box
    aspect-ratio criteria.
    """

    if shadow_mask.ndim != 2:
        raise ValueError(
            "shadow_mask must be a 2D array"
        )

    if labels.ndim != 2:
        raise ValueError(
            "labels must be a 2D array"
        )

    if shadow_mask.shape != labels.shape:
        raise ValueError(
            "shadow_mask and labels must have the same shape"
        )

    if amin < 0:
        raise ValueError(
            "amin must be non-negative"
        )

    if epsilon_ar <= 0:
        raise ValueError(
            "epsilon_ar must be positive"
        )

    filtered_mask = np.zeros_like(
        shadow_mask,
        dtype=bool,
    )

    for component_id in range(
        1,
        num_components + 1,
    ):
        component = labels == component_id

        area = int(np.count_nonzero(component))

        if area < amin:
            continue

        rows, cols = np.nonzero(component)

        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1

        aspect_ratio = max(
            width,
            height,
        ) / min(
            width,
            height,
        )

        if aspect_ratio > epsilon_ar:
            continue

        filtered_mask[component] = True

    return filtered_mask
def build_shadow_regions(
    shadow_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the filtered shadow region S, morphological ring R,
    and interior shadow region Sint.

    S    = shadow_mask
    R    = dilate(S, B3) \\ erode(S, B3)
    Sint = erode(S, B3)

    B3 is the specified 3x3 cross structuring element.
    """

    if shadow_mask.ndim != 2:
        raise ValueError(
            "shadow_mask must be a 2D array"
        )

    structure = cross_structuring_element()

    eroded = ndimage.binary_erosion(
        shadow_mask,
        structure=structure,
    )

    dilated = ndimage.binary_dilation(
        shadow_mask,
        structure=structure,
    )

    shadow_region = shadow_mask.copy()

    ring = dilated & ~eroded

    interior = eroded

    return shadow_region, ring, interior