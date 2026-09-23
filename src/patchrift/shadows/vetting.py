import numpy as np
from scipy.spatial import ConvexHull


def vet_components_by_area(
    labels: np.ndarray,
    num_components: int,
    min_area: int = 60,
) -> np.ndarray:
    """
    Return a boolean array indicating which labeled components
    satisfy the minimum area criterion.
    """

    if labels.ndim != 2:
        raise ValueError(
            "labels must be a 2D array"
        )

    if min_area < 0:
        raise ValueError(
            "min_area must be non-negative"
        )

    qualified = np.zeros(
        num_components + 1,
        dtype=bool,
    )

    for component_id in range(
        1,
        num_components + 1,
    ):
        area = np.count_nonzero(
            labels == component_id
        )

        if area >= min_area:
            qualified[component_id] = True

    return qualified
def vet_components_by_border(
    labels: np.ndarray,
    num_components: int,
) -> np.ndarray:
    """
    Return a boolean array indicating which labeled components
    do not touch the image border.
    """

    if labels.ndim != 2:
        raise ValueError(
            "labels must be a 2D array"
        )

    rows, cols = labels.shape

    qualified = np.ones(
        num_components + 1,
        dtype=bool,
    )

    qualified[0] = False

    border_labels = np.unique(
        np.concatenate(
            (
                labels[0, :],
                labels[rows - 1, :],
                labels[:, 0],
                labels[:, cols - 1],
            )
        )
    )

    qualified[border_labels] = False

    return qualified
def vet_components_by_aspect_ratio(
    labels: np.ndarray,
    num_components: int,
    epsilon_ar: float = 8.0,
) -> np.ndarray:
    """
    Return a boolean array indicating which labeled components
    satisfy the bounding-box aspect-ratio criterion.
    """

    if labels.ndim != 2:
        raise ValueError(
            "labels must be a 2D array"
        )

    if epsilon_ar <= 0:
        raise ValueError(
            "epsilon_ar must be positive"
        )

    qualified = np.ones(
        num_components + 1,
        dtype=bool,
    )

    qualified[0] = False

    for component_id in range(
        1,
        num_components + 1,
    ):
        rows, cols = np.nonzero(
            labels == component_id
        )

        if rows.size == 0:
            qualified[component_id] = False
            continue

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
            qualified[component_id] = False

    return qualified
def component_eccentricity(
    labels: np.ndarray,
    component_id: int,
) -> float:
    """
    Compute the eccentricity of one labeled component
    from its second central moments.
    """

    if labels.ndim != 2:
        raise ValueError(
            "labels must be a 2D array"
        )

    rows, cols = np.nonzero(
        labels == component_id
    )

    if rows.size < 2:
        return 0.0

    row_centered = rows.astype(
        np.float64
    ) - rows.mean()

    col_centered = cols.astype(
        np.float64
    ) - cols.mean()

    covariance = np.array(
        [
            [
                np.mean(row_centered * row_centered),
                np.mean(row_centered * col_centered),
            ],
            [
                np.mean(row_centered * col_centered),
                np.mean(col_centered * col_centered),
            ],
        ]
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    lambda_min = max(
        float(eigenvalues[0]),
        0.0,
    )
    lambda_max = max(
        float(eigenvalues[1]),
        0.0,
    )

    if lambda_max == 0.0:
        return 0.0

    eccentricity_squared = (
        1.0
        - lambda_min / lambda_max
    )

    eccentricity_squared = min(
        max(eccentricity_squared, 0.0),
        1.0,
    )

    return float(
        np.sqrt(eccentricity_squared)
    )
def vet_components_by_eccentricity(
    labels: np.ndarray,
    num_components: int,
    max_eccentricity: float = 0.97,
) -> np.ndarray:
    """
    Return a boolean array indicating which labeled components
    satisfy the eccentricity criterion.
    """

    if labels.ndim != 2:
        raise ValueError(
            "labels must be a 2D array"
        )

    if not 0.0 <= max_eccentricity <= 1.0:
        raise ValueError(
            "max_eccentricity must be between 0 and 1"
        )

    qualified = np.ones(
        num_components + 1,
        dtype=bool,
    )

    qualified[0] = False

    for component_id in range(
        1,
        num_components + 1,
    ):
        eccentricity = component_eccentricity(
            labels,
            component_id,
        )

        if eccentricity > max_eccentricity:
            qualified[component_id] = False

    return qualified
def component_solidity(
    labels: np.ndarray,
    component_id: int,
) -> float:
    """
    Compute the solidity of one labeled component.

    Solidity = Ak / Ahull.

    Ak is the number of pixels in the component.

    Ahull is the number of pixels whose centres lie inside or
    on the boundary of the continuous convex hull of the
    component's pixel centres.
    """

    if labels.ndim != 2:
        raise ValueError(
            "labels must be a 2D array"
        )

    rows, cols = np.nonzero(
        labels == component_id
    )

    if rows.size == 0:
        return 0.0

    component_area = rows.size

    # Degenerate components have solidity 1 by definition.
    if rows.size < 3:
        return 1.0

    points = np.column_stack(
        (
            cols.astype(np.float64),
            rows.astype(np.float64),
        )
    )

    # All-collinear point sets have no 2D hull area.
    centered = points - points.mean(axis=0)
    rank = np.linalg.matrix_rank(centered)

    if rank < 2:
        return 1.0

    hull = ConvexHull(points)

    hull_vertices = points[hull.vertices]

    # Determine the bounding box of the hull in pixel coordinates.
    min_col = int(np.floor(hull_vertices[:, 0].min()))
    max_col = int(np.ceil(hull_vertices[:, 0].max()))
    min_row = int(np.floor(hull_vertices[:, 1].min()))
    max_row = int(np.ceil(hull_vertices[:, 1].max()))

    # Candidate pixel centres.
    candidate_rows, candidate_cols = np.mgrid[
        min_row:max_row + 1,
        min_col:max_col + 1,
    ]

    candidates = np.column_stack(
        (
            candidate_cols.ravel().astype(np.float64),
            candidate_rows.ravel().astype(np.float64),
        )
    )

    # Test whether each candidate point lies inside OR on the
    # boundary of the convex hull.
    #
    # For a convex polygon, every point inside the hull satisfies
    # all edge half-plane inequalities.
    hull_points = hull_vertices
    inside = np.ones(
        candidates.shape[0],
        dtype=bool,
    )

    for i in range(hull_points.shape[0]):
        p1 = hull_points[i]
        p2 = hull_points[
            (i + 1) % hull_points.shape[0]
        ]

        edge = p2 - p1
        relative = candidates - p1

        cross = (
            edge[0] * relative[:, 1]
            - edge[1] * relative[:, 0]
        )

        # Include the boundary. A small tolerance avoids rejecting
        # points because of floating-point roundoff.
        tolerance = 1e-12

        inside &= cross >= -tolerance

    hull_area = int(
        np.count_nonzero(inside)
    )

    if hull_area < component_area:
        raise RuntimeError(
            "Convex-hull rasterisation produced "
            "Ahull < Ak"
        )

    return component_area / hull_area

def vet_components_by_solidity(
    labels: np.ndarray,
    num_components: int,
    min_solidity: float = 0.80,
) -> np.ndarray:
    """
    Return a boolean array indicating which labeled components
    satisfy the minimum solidity criterion.
    """

    if labels.ndim != 2:
        raise ValueError(
            "labels must be a 2D array"
        )

    if not 0.0 <= min_solidity <= 1.0:
        raise ValueError(
            "min_solidity must be between 0 and 1"
        )

    qualified = np.ones(
        num_components + 1,
        dtype=bool,
    )

    qualified[0] = False

    for component_id in range(
        1,
        num_components + 1,
    ):
        solidity = component_solidity(
            labels,
            component_id,
        )

        if solidity < min_solidity:
            qualified[component_id] = False

    return qualified
def shadow_centroid(labels: np.ndarray, component_id: int) -> tuple[float, float]:
    """Return the raw-moment centroid (x, y) of a labelled shadow component."""
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    rows, cols = np.nonzero(labels == component_id)

    if rows.size == 0:
        raise ValueError(f"component_id {component_id} does not exist")

    m00 = float(rows.size)
    m10 = float(np.sum(cols))
    m01 = float(np.sum(rows))

    x_centroid = m10 / m00
    y_centroid = m01 / m00

    return x_centroid, y_centroid
def shadow_dmax(labels: np.ndarray, component_id: int) -> float:
    """Return the maximum distance from the shadow centroid to a component pixel."""
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    rows, cols = np.nonzero(labels == component_id)

    if rows.size == 0:
        raise ValueError(f"component_id {component_id} does not exist")

    centroid_x, centroid_y = shadow_centroid(labels, component_id)

    distances = np.sqrt(
        (cols.astype(float) - centroid_x) ** 2
        + (rows.astype(float) - centroid_y) ** 2
    )

    return float(np.max(distances))
def shadow_context_radius(dmax: float) -> float:
    """Return the pairing/context radius defined by V3."""
    if not np.isfinite(dmax):
        raise ValueError("dmax must be finite")

    if dmax < 0:
        raise ValueError("dmax must be non-negative")

    return 3.0 * dmax

def build_context_region(
    shadow_mask: np.ndarray,
    centroid: tuple[float, float],
    context_radius: float,
) -> np.ndarray:
    """Build the photometric context region B_k around a shadow centroid."""
    if shadow_mask.ndim != 2:
        raise ValueError("shadow_mask must be a 2D array")

    if context_radius < 0:
        raise ValueError("context_radius must be non-negative")

    centroid_x, centroid_y = centroid

    if not np.isfinite(centroid_x) or not np.isfinite(centroid_y):
        raise ValueError("centroid coordinates must be finite")

    if not np.isfinite(context_radius):
        raise ValueError("context_radius must be finite")

    rows, cols = np.indices(shadow_mask.shape, dtype=float)

    inside_radius = (
        (cols - centroid_x) ** 2
        + (rows - centroid_y) ** 2
        < context_radius ** 2
    )

    return inside_radius & ~shadow_mask.astype(bool)

def local_background_level(
    image: np.ndarray,
    context_mask: np.ndarray,
) -> float:
    """Return the median image intensity over the context region B_k."""
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if context_mask.ndim != 2:
        raise ValueError("context_mask must be a 2D array")

    if image.shape != context_mask.shape:
        raise ValueError("image and context_mask must have the same shape")

    values = image[context_mask.astype(bool)]

    if values.size == 0:
        raise ValueError("context region contains no pixels")

    return float(np.median(values))

def raised_cosine_taper(t: float) -> float:
    """Return the V3 raised-cosine radial taper T(t)."""
    if not np.isfinite(t):
        raise ValueError("t must be finite")

    if t < 0:
        raise ValueError("t must be non-negative")

    if t < 0.5:
        return 1.0

    if t <= 1.0:
        return 0.5 * (1.0 + np.cos(np.pi * (2.0 * t - 1.0)))

    return 0.0

def build_excess_brightness_weights(
    image: np.ndarray,
    context_mask: np.ndarray,
    centroid: tuple[float, float],
    background_level: float,
    context_radius: float,
) -> np.ndarray:
    """Build the V3 excess-brightness weight field u(p)."""
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if context_mask.ndim != 2:
        raise ValueError("context_mask must be a 2D array")

    if image.shape != context_mask.shape:
        raise ValueError("image and context_mask must have the same shape")

    if not np.isfinite(background_level):
        raise ValueError("background_level must be finite")

    if not np.isfinite(context_radius) or context_radius <= 0:
        raise ValueError("context_radius must be finite and positive")

    centroid_x, centroid_y = centroid

    if not np.isfinite(centroid_x) or not np.isfinite(centroid_y):
        raise ValueError("centroid coordinates must be finite")

    rows, cols = np.indices(image.shape, dtype=float)

    distance = np.sqrt(
        (cols - centroid_x) ** 2
        + (rows - centroid_y) ** 2
    )

    normalized_distance = distance / context_radius

    taper = np.zeros(image.shape, dtype=float)

    inner = normalized_distance < 0.5
    transition = (
        (normalized_distance >= 0.5)
        & (normalized_distance <= 1.0)
    )

    taper[inner] = 1.0

    taper[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi * (2.0 * normalized_distance[transition] - 1.0)
        )
    )

    excess = np.maximum(0.0, image - background_level)

    weights = excess * taper

    weights[~context_mask.astype(bool)] = 0.0

    return weights

def light_centroid(
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return the brightness-weighted light centroid c_l,k."""
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D array")

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")

    total_weight = float(np.sum(weights))

    if total_weight <= 0.0:
        raise ValueError("total weight must be positive")

    rows, cols = np.indices(weights.shape, dtype=float)

    x_centroid = float(np.sum(weights * cols) / total_weight)
    y_centroid = float(np.sum(weights * rows) / total_weight)

    return x_centroid, y_centroid

def centroid_separation(
    shadow_centroid: tuple[float, float],
    light_centroid: tuple[float, float],
) -> float:
    """Return the Euclidean separation between shadow and light centroids."""
    shadow_x, shadow_y = shadow_centroid
    light_x, light_y = light_centroid

    if not all(
        np.isfinite(value)
        for value in (shadow_x, shadow_y, light_x, light_y)
    ):
        raise ValueError("centroid coordinates must be finite")

    return float(
        np.hypot(
            shadow_x - light_x,
            shadow_y - light_y,
        )
    )

def minimum_centroid_separation(area: float) -> float:
    """Return the V3 minimum centroid separation r_min."""
    if not np.isfinite(area):
        raise ValueError("area must be finite")

    if area <= 0:
        raise ValueError("area must be positive")

    return float(0.15 * np.sqrt(area / np.pi))

def is_centroid_separation_usable(
    separation: float,
    minimum_separation: float,
) -> bool:
    """Return whether the centroid separation passes the V3 minimum."""
    if not np.isfinite(separation):
        raise ValueError("separation must be finite")

    if not np.isfinite(minimum_separation):
        raise ValueError("minimum_separation must be finite")

    if separation < 0:
        raise ValueError("separation must be non-negative")

    if minimum_separation < 0:
        raise ValueError("minimum_separation must be non-negative")

    return separation >= minimum_separation

def solar_azimuth_from_centroids(
    shadow_centroid: tuple[float, float],
    light_centroid: tuple[float, float],
) -> float:
    """Return the solar azimuth estimate from the shadow/light centroids."""
    shadow_x, shadow_y = shadow_centroid
    light_x, light_y = light_centroid

    if not all(
        np.isfinite(value)
        for value in (shadow_x, shadow_y, light_x, light_y)
    ):
        raise ValueError("centroid coordinates must be finite")

    dx = shadow_x - light_x
    dy = shadow_y - light_y

    if dx == 0.0 and dy == 0.0:
        raise ValueError("centroids must not coincide")

    return float(np.arctan2(dy, dx))

def effective_light_count(weights: np.ndarray) -> float:
    """Return the V3 effective light-centroid count N_L."""
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D array")

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")

    total_weight = float(np.sum(weights))

    if total_weight <= 0.0:
        raise ValueError("total weight must be positive")

    squared_weight_sum = float(np.sum(weights ** 2))

    if squared_weight_sum <= 0.0:
        raise ValueError("sum of squared weights must be positive")

    return float((total_weight ** 2) / squared_weight_sum)

def shadow_sample_count(labels: np.ndarray, component_id: int) -> int:
    """Return N_S,k = A_k, the pixel area of a shadow component."""
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    if component_id <= 0:
        raise ValueError("component_id must be positive")

    return int(np.count_nonzero(labels == component_id))

def positional_uncertainty(
    shadow_sample_count: float,
    light_sample_count: float,
    segmentation_factor: float = 3.0,
) -> float:
    """Return V3 positional uncertainty sigma_pos in pixels."""
    if not np.isfinite(shadow_sample_count):
        raise ValueError("shadow_sample_count must be finite")

    if not np.isfinite(light_sample_count):
        raise ValueError("light_sample_count must be finite")

    if not np.isfinite(segmentation_factor):
        raise ValueError("segmentation_factor must be finite")

    if shadow_sample_count <= 0.0:
        raise ValueError("shadow_sample_count must be positive")

    if light_sample_count <= 0.0:
        raise ValueError("light_sample_count must be positive")

    if segmentation_factor <= 0.0:
        raise ValueError("segmentation_factor must be positive")

    return float(
        segmentation_factor
        * np.sqrt(
            1.0 / (12.0 * shadow_sample_count)
            + 1.0 / (12.0 * light_sample_count)
        )
    )

def angular_uncertainty(
    positional_sigma: float,
    centroid_separation: float,
) -> float:
    """Return V3 angular uncertainty sigma_theta in radians."""
    if not np.isfinite(positional_sigma):
        raise ValueError("positional_sigma must be finite")

    if not np.isfinite(centroid_separation):
        raise ValueError("centroid_separation must be finite")

    if positional_sigma < 0.0:
        raise ValueError("positional_sigma must be nonnegative")

    if centroid_separation <= 0.0:
        raise ValueError("centroid_separation must be positive")

    return float(positional_sigma / centroid_separation)
def component_shape_moments(
    labels: np.ndarray,
    component_id: int,
) -> tuple[float, float, float]:
    """Return (lambda_max, lambda_min, psi_maj) for a shadow component.

    Coordinates use x = column and y = row, matching the V3 convention.
    psi_maj is measured in radians from +x toward +y.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    if component_id <= 0:
        raise ValueError("component_id must be positive")

    rows, cols = np.nonzero(labels == component_id)

    if rows.size == 0:
        raise ValueError("component_id is not present in labels")

    x = cols.astype(float)
    y = rows.astype(float)

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    m_xx = float(np.mean(x_centered * x_centered))
    m_yy = float(np.mean(y_centered * y_centered))
    m_xy = float(np.mean(x_centered * y_centered))

    moment_matrix = np.array(
        [
            [m_xx, m_xy],
            [m_xy, m_yy],
        ],
        dtype=float,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(moment_matrix)

    lambda_min = float(eigenvalues[0])
    lambda_max = float(eigenvalues[1])

    major_axis = eigenvectors[:, 1]

    psi_maj = float(np.arctan2(major_axis[1], major_axis[0]))

    return lambda_max, lambda_min, psi_maj

def eccentricity_from_eigenvalues(
    lambda_max: float,
    lambda_min: float,
) -> float:
    """Return V3 eccentricity from central-moment eigenvalues."""
    if not np.isfinite(lambda_max):
        raise ValueError("lambda_max must be finite")

    if not np.isfinite(lambda_min):
        raise ValueError("lambda_min must be finite")

    if lambda_max <= 0.0:
        raise ValueError("lambda_max must be positive")

    if lambda_min < 0.0:
        raise ValueError("lambda_min must be nonnegative")

    if lambda_min > lambda_max:
        raise ValueError("lambda_min must not exceed lambda_max")

    ratio = lambda_min / lambda_max

    return float(np.sqrt(1.0 - ratio))

def axis_angle(
    axis_angle_a: float,
    axis_angle_b: float,
) -> float:
    """Return the unsigned angle between two axes in [0, pi/2]."""
    if not np.isfinite(axis_angle_a):
        raise ValueError("axis_angle_a must be finite")

    if not np.isfinite(axis_angle_b):
        raise ValueError("axis_angle_b must be finite")

    difference = float(
        (axis_angle_a - axis_angle_b + np.pi) % (2.0 * np.pi) - np.pi
    )

    difference = abs(difference)

    if difference > np.pi / 2.0:
        difference = np.pi - difference

    return float(difference)

def shape_ratio_from_eccentricity(
    eccentricity: float,
    omega: float,
) -> float:
    """Return V3 rho_k from eccentricity and axis-angle omega."""
    if not np.isfinite(eccentricity):
        raise ValueError("eccentricity must be finite")

    if eccentricity < 0.0 or eccentricity > 1.0:
        raise ValueError("eccentricity must be in [0, 1]")

    if not np.isfinite(omega):
        raise ValueError("omega must be finite")

    if omega < 0.0 or omega > np.pi / 2.0:
        raise ValueError("omega must be in [0, pi/2]")

    base_ratio = 1.0 - eccentricity**2

    if omega < np.pi / 4.0:
        return float(base_ratio)

    return float(1.0 / base_ratio)

def sector_shape_ratio(delta: float) -> float:
    """Return the V3 normalized second-moment ratio rho(delta)."""
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")

    if delta <= 0.0:
        raise ValueError("delta must be positive")

    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    mu_yy = 0.25 * (delta - sin_delta * cos_delta)

    mu_xx = (
        0.25 * (delta + sin_delta * cos_delta)
        - 4.0 * sin_delta**2 / (9.0 * delta)
    )

    if mu_xx <= 0.0:
        raise ValueError("mu_xx must be positive")

    return float(mu_yy / mu_xx)

def build_sector_shape_table() -> tuple[np.ndarray, np.ndarray]:
    """Build the V3 rho(delta) lookup table at 0.5-degree spacing."""
    delta_degrees = np.arange(5.0, 78.0 + 0.5, 0.5)
    delta_radians = np.deg2rad(delta_degrees)

    rho_values = np.array(
        [sector_shape_ratio(delta) for delta in delta_radians],
        dtype=float,
    )

    return delta_degrees, rho_values

def invert_sector_shape_ratio(rho: float) -> float:
    """Invert the V3 sector-shape ratio using the 0.5-degree lookup table.

    Returns the inferred sector half-angle delta in radians.
    No extrapolation is permitted.
    """
    if not np.isfinite(rho):
        raise ValueError("rho must be finite")

    delta_degrees, rho_values = build_sector_shape_table()

    if rho < rho_values[0] or rho > rho_values[-1]:
        raise ValueError("rho is outside the tabulated V3 domain")

    delta_hat_degrees = np.interp(rho, rho_values, delta_degrees)

    return float(np.deg2rad(delta_hat_degrees))

def is_half_angle_usable(
    half_angle: float,
    minimum_angle: float = np.deg2rad(5.0),
    maximum_angle: float = np.deg2rad(78.0),
) -> bool:
    """Return whether a V3 sector half-angle lies in the valid range."""
    if not np.isfinite(half_angle):
        raise ValueError("half_angle must be finite")

    if not np.isfinite(minimum_angle):
        raise ValueError("minimum_angle must be finite")

    if not np.isfinite(maximum_angle):
        raise ValueError("maximum_angle must be finite")

    if minimum_angle < 0.0:
        raise ValueError("minimum_angle must be non-negative")

    if maximum_angle < minimum_angle:
        raise ValueError("maximum_angle must be at least minimum_angle")

    return minimum_angle <= half_angle <= maximum_angle

def infer_sector_half_angle(eccentricity: float, omega: float) -> float:
    """Infer the V3 sector half-angle from eccentricity and axis angle.

    Returns delta_hat in radians.
    """
    rho = shape_ratio_from_eccentricity(eccentricity, omega)
    delta_hat = invert_sector_shape_ratio(rho)

    return delta_hat

def is_sector_half_angle_usable(delta_hat: float) -> bool:
    """Return whether delta_hat satisfies the V3 half-angle validity range."""
    if not np.isfinite(delta_hat):
        return False

    delta_min = np.deg2rad(5.0)
    delta_max = np.deg2rad(78.0)

    return delta_min <= delta_hat <= delta_max

def infer_sector_half_angles(
    eccentricities: np.ndarray,
    omegas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer V3 sector half-angles for an ensemble of components.

    Components whose inferred half-angle lies outside the operational
    [5°, 78°] range are marked invalid.

    Returns
    -------
    delta_hats
        Array of inferred sector half-angles in radians. Invalid entries
        are represented by NaN.

    valid
        Boolean array indicating which components satisfy the V3
        half-angle validity gate.
    """
    eccentricities = np.asarray(eccentricities, dtype=float)
    omegas = np.asarray(omegas, dtype=float)

    if eccentricities.ndim != 1 or omegas.ndim != 1:
        raise ValueError(
            "eccentricities and omegas must be one-dimensional"
        )

    if eccentricities.size == 0:
        raise ValueError(
            "eccentricities and omegas must not be empty"
        )

    if eccentricities.size != omegas.size:
        raise ValueError(
            "eccentricities and omegas must have the same length"
        )

    delta_hats = np.full(
        eccentricities.shape,
        np.nan,
        dtype=float,
    )

    valid = np.zeros(
        eccentricities.shape,
        dtype=bool,
    )

    for i in range(eccentricities.size):
        try:
            delta_hat = infer_sector_half_angle(
                eccentricities[i],
                omegas[i],
            )
        except ValueError:
            continue

        if is_sector_half_angle_usable(delta_hat):
            delta_hats[i] = delta_hat
            valid[i] = True

    return delta_hats, valid

def precision_weight(
    sigma_theta: float,
    sigma0: float = np.deg2rad(0.5),
) -> float:
    """Compute the V3 precision weight from Eq. 29."""
    if not np.isfinite(sigma_theta):
        raise ValueError("sigma_theta must be finite")

    if sigma_theta < 0.0:
        raise ValueError("sigma_theta must be non-negative")

    if not np.isfinite(sigma0):
        raise ValueError("sigma0 must be finite")

    if sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive")

    return float(
        sigma0**2 / (sigma0**2 + sigma_theta**2)
    )

def symmetry_weight(
    eccentricity: float,
    omega: float,
    eaxis: float = 0.40,
) -> float:
    """Compute the V3 symmetry/axis weight from Eq. 30."""
    if not np.isfinite(eccentricity):
        raise ValueError("eccentricity must be finite")

    if not np.isfinite(omega):
        raise ValueError("omega must be finite")

    if not 0.0 <= eccentricity <= 1.0:
        raise ValueError("eccentricity must be in [0, 1]")

    if not 0.0 <= omega <= np.pi / 2.0:
        raise ValueError("omega must be in [0, pi/2]")

    if not np.isfinite(eaxis):
        raise ValueError("eaxis must be finite")

    if not 0.0 <= eaxis <= 1.0:
        raise ValueError("eaxis must be in [0, 1]")

    if eccentricity < eaxis:
        return 1.0

    return float(np.cos(2.0 * omega) ** 2)

def shape_scale(
    delta_hats: np.ndarray,
    sdelta_min: float = np.deg2rad(5.0),
) -> float:
    """Compute the V3 robust shape scale s_delta."""
    values = np.asarray(delta_hats, dtype=float)

    if values.ndim != 1:
        raise ValueError("delta_hats must be one-dimensional")

    if values.size == 0:
        raise ValueError("delta_hats must not be empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("delta_hats must be finite")

    if not np.isfinite(sdelta_min):
        raise ValueError("sdelta_min must be finite")

    if sdelta_min <= 0.0:
        raise ValueError("sdelta_min must be positive")

    delta_med = float(np.median(values))
    mad = float(np.median(np.abs(values - delta_med)))

    return float(max(1.4826 * mad, sdelta_min))

def shape_median(
    delta_hats: np.ndarray,
) -> float:
    """Compute the V3 ensemble median sector half-angle."""
    values = np.asarray(delta_hats, dtype=float)

    if values.ndim != 1:
        raise ValueError("delta_hats must be one-dimensional")

    if values.size == 0:
        raise ValueError("delta_hats must not be empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("delta_hats must be finite")

    return float(np.median(values))

def shape_ensemble_statistics(
    delta_hats: np.ndarray,
    sdelta_min: float = np.deg2rad(5.0),
) -> tuple[float, float]:
    """Return the V3 ensemble median and robust shape scale."""
    delta_med = shape_median(delta_hats)
    s_delta = shape_scale(
        delta_hats,
        sdelta_min=sdelta_min,
    )

    return delta_med, s_delta

def shape_weight(
    delta_hat: float,
    delta_med: float,
    s_delta: float,
) -> float:
    """Compute the V3 Gaussian shape weight from Eq. 31."""
    if not np.isfinite(delta_hat):
        raise ValueError("delta_hat must be finite")

    if not np.isfinite(delta_med):
        raise ValueError("delta_med must be finite")

    if not np.isfinite(s_delta):
        raise ValueError("s_delta must be finite")

    if s_delta <= 0.0:
        raise ValueError("s_delta must be positive")

    return float(
        np.exp(
            -((delta_hat - delta_med) ** 2)
            / (2.0 * s_delta**2)
        )
    )

def base_weight(
    w_prec: float,
    w_sym: float,
    w_shape: float,
    w_floor: float = 1e-3,
) -> float:
    """Compute the V3 unnormalized base reliability weight."""
    weights = (w_prec, w_sym, w_shape)

    if not all(np.isfinite(value) for value in weights):
        raise ValueError("component weights must be finite")

    if any(value < 0.0 for value in weights):
        raise ValueError("component weights must be non-negative")

    if not np.isfinite(w_floor):
        raise ValueError("w_floor must be finite")

    if w_floor < 0.0:
        raise ValueError("w_floor must be non-negative")

    return float(
        max(
            w_prec * w_sym * w_shape,
            w_floor,
        )
    )

def compute_component_base_weight(
    sigma_theta: float,
    eccentricity: float,
    omega: float,
    delta_hat: float,
    delta_med: float,
    s_delta: float,
    sigma0: float = np.deg2rad(0.5),
    eaxis: float = 0.40,
    w_floor: float = 1e-3,
) -> float:
    """Compute the V3 unnormalized base weight for one component."""
    w_prec = precision_weight(
        sigma_theta,
        sigma0=sigma0,
    )

    w_sym = symmetry_weight(
        eccentricity,
        omega,
        eaxis=eaxis,
    )

    w_shape = shape_weight(
        delta_hat,
        delta_med,
        s_delta,
    )

    return base_weight(
        w_prec,
        w_sym,
        w_shape,
        w_floor=w_floor,
    )

def compute_normalized_component_weights(
    sigma_theta: np.ndarray,
    eccentricity: np.ndarray,
    omega: np.ndarray,
    delta_hat: np.ndarray,
    delta_med: float,
    s_delta: float,
    sigma0: float = np.deg2rad(0.5),
    eaxis: float = 0.40,
    w_floor: float = 1e-3,
) -> np.ndarray:
    """Compute normalized V3 base weights for an ensemble of components."""
    sigma_theta = np.asarray(sigma_theta, dtype=float)
    eccentricity = np.asarray(eccentricity, dtype=float)
    omega = np.asarray(omega, dtype=float)
    delta_hat = np.asarray(delta_hat, dtype=float)

    arrays = (
        sigma_theta,
        eccentricity,
        omega,
        delta_hat,
    )

    if any(values.ndim != 1 for values in arrays):
        raise ValueError("all component inputs must be one-dimensional")

    n = sigma_theta.size

    if n == 0:
        raise ValueError("component inputs must not be empty")

    if any(values.size != n for values in arrays):
        raise ValueError(
            "all component inputs must have the same length"
        )

    base_weights = np.array(
        [
            compute_component_base_weight(
                sigma_theta=sigma_theta[i],
                eccentricity=eccentricity[i],
                omega=omega[i],
                delta_hat=delta_hat[i],
                delta_med=delta_med,
                s_delta=s_delta,
                sigma0=sigma0,
                eaxis=eaxis,
                w_floor=w_floor,
            )
            for i in range(n)
        ],
        dtype=float,
    )

    return normalize_base_weights(base_weights)
def normalize_base_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize V3 base weights so the ensemble maximum is one."""
    values = np.asarray(weights, dtype=float)

    if values.ndim != 1:
        raise ValueError("weights must be one-dimensional")

    if values.size == 0:
        raise ValueError("weights must not be empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite")

    if np.any(values < 0.0):
        raise ValueError("weights must be non-negative")

    maximum = float(np.max(values))

    if maximum <= 0.0:
        raise ValueError("maximum weight must be positive")

    return values / maximum

def vet_components_by_pairing_window(
    centroids: np.ndarray,
    context_radii: np.ndarray,
) -> np.ndarray:
    """Apply the symmetric Addendum B3 pairing-window isolation gate.

    Components are excluded when another gate-passing component lies
    strictly within the larger of the two context radii.
    """
    centers = np.asarray(centroids, dtype=float)
    radii = np.asarray(context_radii, dtype=float)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centroids must have shape (N, 2)")

    if radii.ndim != 1:
        raise ValueError("context_radii must be one-dimensional")

    if centers.shape[0] != radii.size:
        raise ValueError(
            "centroids and context_radii must have the same length"
        )

    if not np.all(np.isfinite(centers)):
        raise ValueError("centroids must be finite")

    if not np.all(np.isfinite(radii)):
        raise ValueError("context_radii must be finite")

    if np.any(radii < 0.0):
        raise ValueError("context_radii must be non-negative")

    n = centers.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            exclusion_radius = max(radii[i], radii[j])

            if distance < exclusion_radius:
                keep[i] = False
                keep[j] = False

    return keep

def combine_component_vetting_gates(
    area_pass: np.ndarray,
    border_pass: np.ndarray,
    aspect_pass: np.ndarray,
    eccentricity_pass: np.ndarray,
    solidity_pass: np.ndarray,
    centroids: np.ndarray,
    context_radii: np.ndarray,
) -> np.ndarray:
    """Combine the V3 §5.3 component-vetting gates.

    Pairing-window isolation is evaluated only among components that
    pass all preceding gates.
    """
    gates = (
        np.asarray(area_pass, dtype=bool),
        np.asarray(border_pass, dtype=bool),
        np.asarray(aspect_pass, dtype=bool),
        np.asarray(eccentricity_pass, dtype=bool),
        np.asarray(solidity_pass, dtype=bool),
    )

    n = gates[0].size

    if any(gate.ndim != 1 or gate.size != n for gate in gates):
        raise ValueError("all gate arrays must have the same 1-D shape")

    passing = np.logical_and.reduce(gates)

    centers = np.asarray(centroids, dtype=float)
    radii = np.asarray(context_radii, dtype=float)

    if centers.shape != (n, 2):
        raise ValueError("centroids must have shape (N, 2)")

    if radii.shape != (n,):
        raise ValueError("context_radii must have shape (N,)")

    isolated = np.ones(n, dtype=bool)

    passing_indices = np.flatnonzero(passing)

    if passing_indices.size > 1:
        isolated_subset = vet_components_by_pairing_window(
            centers[passing_indices],
            radii[passing_indices],
        )
        isolated[passing_indices] = isolated_subset

    return passing & isolated

def compute_centroid_azimuth_measurement(
    image: np.ndarray,
    labels: np.ndarray,
    component_id: int,
    shadow_mask: np.ndarray,
    fseg: float = 3.0,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the V3 centroid-based azimuth measurement for one component.

    Returns:
        shadow_centroid_x,
        shadow_centroid_y,
        light_centroid_x,
        light_centroid_y,
        centroid_separation,
        solar_azimuth,
        positional_uncertainty,
        angular_uncertainty
    """
    if not np.isfinite(fseg) or fseg <= 0.0:
        raise ValueError("fseg must be positive and finite")

    shadow_centroid_xy = shadow_centroid(
        labels,
        component_id,
    )

    dmax = shadow_dmax(
        labels,
        component_id,
    )

    rctx = shadow_context_radius(dmax)

    context_mask = build_context_region(
        shadow_mask,
        shadow_centroid_xy,
        rctx,
    )

    background = local_background_level(
        image,
        context_mask,
    )

    weights = build_excess_brightness_weights(
        image,
        context_mask,
        shadow_centroid_xy,
        background,
        rctx,    
        )

    light_centroid_xy = light_centroid(weights)

    separation = centroid_separation(
        shadow_centroid_xy,
        light_centroid_xy,
    )

    area = shadow_sample_count(
        labels,
        component_id,
    )

    minimum_separation = minimum_centroid_separation(
        area,
    )

    if not is_centroid_separation_usable(
        separation,
        minimum_separation,
    ):
        raise ValueError(
            "centroid separation is below the V3 minimum"
        )

    azimuth = solar_azimuth_from_centroids(
        shadow_centroid_xy,
        light_centroid_xy,
    )

    n_light = effective_light_count(weights)

    sigma_pos = positional_uncertainty(
        area,
        n_light,
        segmentation_factor=fseg,
    )
    sigma_theta = angular_uncertainty(
        sigma_pos,
        separation,
    )

    return (
        float(shadow_centroid_xy[0]),
        float(shadow_centroid_xy[1]),
        float(light_centroid_xy[0]),
        float(light_centroid_xy[1]),
        float(separation),
        float(azimuth),
        float(sigma_pos),
        float(sigma_theta),
    )

