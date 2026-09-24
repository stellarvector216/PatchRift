import numpy as np


def destripe(
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply Version 3 §4.3 column-median destriping followed by
    a 3x3 median filter.

    The column median is computed independently for each column
    using valid pixels only.

    Parameters
    ----------
    image:
        2D scalar image.

    valid_mask:
        Boolean mask with True for valid pixels.

    Returns
    -------
    np.ndarray
        Destriped and median-filtered image.
    """

    image = np.asarray(image, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if valid_mask.shape != image.shape:
        raise ValueError(
            "valid_mask must have the same shape as image"
        )

    result = image.copy()

    global_valid = image[valid_mask]

    if global_valid.size == 0:
        raise ValueError("no valid pixels available")

    global_median = np.median(global_valid)

    for column in range(image.shape[1]):
        column_valid = valid_mask[:, column]

        if not np.any(column_valid):
            continue

        column_median = np.median(
            image[column_valid, column]
        )

        result[column_valid, column] = (
            image[column_valid, column]
            - column_median
            + global_median
        )

    result = _masked_median_filter_3x3(
        result,
        valid_mask,
    )

    return result.astype(np.float32)


def _masked_median_filter_3x3(
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Apply the specified 3x3 median using valid samples only.

    The array edge uses nearest-neighbour extension, matching the
    former filtering behaviour.  Invalid pixels are never treated as
    terrain measurements and remain unchanged in the returned array.
    """
    padded_image = np.pad(image, 1, mode="edge")
    padded_valid = np.pad(valid_mask, 1, mode="edge")
    result = image.copy()

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if not valid_mask[row, col]:
                continue

            local_valid = padded_valid[row:row + 3, col:col + 3]
            local_values = padded_image[row:row + 3, col:col + 3]
            result[row, col] = np.median(local_values[local_valid])

    return result
