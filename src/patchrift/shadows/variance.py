import numpy as np


def subtile_has_sufficient_variance(
    image: np.ndarray,
    valid_mask: np.ndarray,
    epsilon_var: float = 1e-3,
) -> bool:
    """
    Determine whether a subtile has sufficient normalized variance
    for Version 3 shadow segmentation.

    The Specification Addendum B2 defines the variance guard as:

        normalized_variance =
            Var(I_valid) / IQR(I_valid)^2

    A subtile qualifies when:

        normalized_variance >= epsilon_var

    Invalid pixels are excluded from the calculation.

    A zero-IQR subtile is treated as degenerate and does not qualify.
    This zero-IQR behaviour is an implementation convention because
    the normalization is otherwise undefined.
    """

    image = np.asarray(image, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if valid_mask.shape != image.shape:
        raise ValueError(
            "valid_mask must have the same shape as image"
        )

    if epsilon_var < 0:
        raise ValueError(
            "epsilon_var cannot be negative"
        )

    values = image[valid_mask]

    if values.size == 0:
        return False

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "valid pixels must contain only finite values"
        )

    variance = np.var(values)

    q25, q75 = np.percentile(
        values,
        [25.0, 75.0],
    )

    iqr = q75 - q25

    if iqr == 0.0:
        return False

    normalized_variance = variance / (iqr ** 2)

    return bool(
        normalized_variance >= epsilon_var
    )