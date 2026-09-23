import numpy as np


def otsu_threshold(
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """
    Compute the exact Otsu threshold from valid pixels only.

    Version 3 + Addendum A3 specify that the candidate thresholds
    are the midpoints between consecutive sorted distinct
    intensity values. No histogram binning is used.

    Parameters
    ----------
    image:
        2D prepared image from Stage I.

    valid_mask:
        Boolean mask. True pixels participate in the threshold
        calculation; False pixels are ignored.

    Returns
    -------
    float
        Otsu threshold tau*.
    """

    image = np.asarray(image, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if valid_mask.shape != image.shape:
        raise ValueError(
            "valid_mask must have the same shape as image"
        )

    values = image[valid_mask]

    if values.size == 0:
        raise ValueError("no valid pixels available")

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "valid pixels must contain only finite values"
        )

    unique_values, counts = np.unique(
        values,
        return_counts=True,
    )

    if unique_values.size < 2:
        return float(unique_values[0])

    # Version 3 Addendum A3:
    # candidate thresholds are the midpoints between
    # consecutive sorted distinct intensity values.
    candidate_thresholds = (
        unique_values[:-1]
        + unique_values[1:]
    ) / 2.0

    cumulative_counts = np.cumsum(counts)
    cumulative_sums = np.cumsum(
        unique_values * counts
    )

    total_count = cumulative_counts[-1]
    total_sum = cumulative_sums[-1]

    class0_count = cumulative_counts[:-1]
    class1_count = total_count - class0_count

    class0_sum = cumulative_sums[:-1]
    class1_sum = total_sum - class0_sum

    omega0 = class0_count / total_count
    omega1 = class1_count / total_count

    mu0 = class0_sum / class0_count
    mu1 = class1_sum / class1_count

    between_class_variance = (
        omega0
        * omega1
        * (mu0 - mu1) ** 2
    )

    best_index = int(
        np.argmax(between_class_variance)
    )

    return float(
        candidate_thresholds[best_index]
    )

def shadow_threshold(
    image: np.ndarray,
    valid_mask: np.ndarray,
    q: float,
) -> float:
    """
    Compute the Version 3 §5.2 shadow-segmentation threshold.

    The final threshold is:

        tau = min(tau*, Q_q(I))

    where tau* is the Otsu threshold and q is the
    downward-percentile floor parameter.

    Parameters
    ----------
    image:
        2D prepared image from Stage I.

    valid_mask:
        Boolean mask. True pixels participate in the calculation.

    q:
        Percentile parameter. Version 3 specifies q in [0.05, 0.10].

    Returns
    -------
    float
        Final shadow threshold tau.
    """

    if not 0.05 <= q <= 0.10:
        raise ValueError(
            "q must lie in the Version 3 range [0.05, 0.10]"
        )

    image = np.asarray(image, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if valid_mask.shape != image.shape:
        raise ValueError(
            "valid_mask must have the same shape as image"
        )

    values = image[valid_mask]

    if values.size == 0:
        raise ValueError("no valid pixels available")

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "valid pixels must contain only finite values"
        )

    tau_otsu = otsu_threshold(
        image,
        valid_mask,
    )

    percentile = np.quantile(
        values,
        q,
    )

    return float(
        min(tau_otsu, percentile)
    )