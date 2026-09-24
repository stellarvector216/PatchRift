import numpy as np
import pytest

from patchrift.shadows.morphology import (
    label_shadow_components,
)
from patchrift.shadows.vetting import (
    vet_components_by_area,
    vet_components_by_border,
    vet_components_by_aspect_ratio,
    component_eccentricity,
    vet_components_by_eccentricity,
    component_solidity,
    vet_components_by_solidity,
    shadow_centroid,
    shadow_dmax,
    shadow_context_radius,
    build_context_region,
    local_background_level,
    raised_cosine_taper,
    build_excess_brightness_weights,
    light_centroid,
    centroid_separation,
    minimum_centroid_separation,
    is_centroid_separation_usable,
    solar_azimuth_from_centroids,
    effective_light_count,
    shadow_sample_count,
    positional_uncertainty,
    angular_uncertainty,
    component_shape_moments,
    eccentricity_from_eigenvalues,
    axis_angle,
    shape_ratio_from_eccentricity,
    sector_shape_ratio,
    build_sector_shape_table,
    invert_sector_shape_ratio,
    infer_sector_half_angle,
    is_sector_half_angle_usable,
    precision_weight,
    symmetry_weight,
    shape_scale,
    shape_weight,
    base_weight,
    normalize_base_weights,
    vet_components_by_pairing_window,
    combine_component_vetting_gates,
    compute_centroid_azimuth_measurement,
    is_half_angle_usable,
    shape_median,
    shape_ensemble_statistics,
    compute_component_base_weight,
    compute_normalized_component_weights,
    infer_sector_half_angles
    )

def test_vet_components_by_area():
    shadow_mask = np.zeros(
        (12, 12),
        dtype=bool,
    )

    shadow_mask[1:5, 1:5] = True
    shadow_mask[7:9, 7:9] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    qualified = vet_components_by_area(
        labels,
        num_components,
        min_area=10,
    )

    assert qualified[1]
    assert not qualified[2]


def test_vet_components_by_area_inclusive_boundary():
    shadow_mask = np.zeros(
        (10, 10),
        dtype=bool,
    )

    shadow_mask[1:5, 1:5] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    qualified = vet_components_by_area(
        labels,
        num_components,
        min_area=16,
    )

    assert qualified[1]


def test_vet_components_by_area_rejects_non_2d_labels():
    labels = np.zeros(
        (5, 5, 1),
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        vet_components_by_area(
            labels,
            0,
        )


def test_vet_components_by_area_rejects_negative_min_area():
    labels = np.zeros(
        (5, 5),
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        vet_components_by_area(
            labels,
            0,
            min_area=-1,
        )
def test_vet_components_by_border():
    shadow_mask = np.zeros(
        (10, 10),
        dtype=bool,
    )

    shadow_mask[1:4, 1:4] = True
    shadow_mask[0:3, 6:9] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    qualified = vet_components_by_border(
        labels,
        num_components,
    )

    interior_label = labels[2, 2]
    border_label = labels[0, 7]

    assert qualified[interior_label]
    assert not qualified[border_label]

def test_vet_components_by_border_rejects_non_2d_labels():
    labels = np.zeros(
        (5, 5, 1),
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        vet_components_by_border(
            labels,
            0,
        )
def test_vet_components_by_aspect_ratio():
    shadow_mask = np.zeros(
        (12, 20),
        dtype=bool,
    )

    shadow_mask[1:5, 1:5] = True
    shadow_mask[6:7, 6:20] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    qualified = vet_components_by_aspect_ratio(
        labels,
        num_components,
    )

    square_label = labels[2, 2]
    elongated_label = labels[6, 10]

    assert qualified[square_label]
    assert not qualified[elongated_label]


def test_vet_components_by_aspect_ratio_boundary():
    shadow_mask = np.zeros(
        (12, 12),
        dtype=bool,
    )

    shadow_mask[2:3, 2:11] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    qualified = vet_components_by_aspect_ratio(
        labels,
        num_components,
        epsilon_ar=9.0,
    )

    component_label = labels[2, 5]

    assert qualified[component_label]


def test_vet_components_by_aspect_ratio_rejects_invalid_threshold():
    labels = np.zeros(
        (5, 5),
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        vet_components_by_aspect_ratio(
            labels,
            0,
            epsilon_ar=0,
        )


def test_vet_components_by_aspect_ratio_rejects_non_2d_labels():
    labels = np.zeros(
        (5, 5, 1),
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        vet_components_by_aspect_ratio(
            labels,
            0,
        )
def test_component_eccentricity_square():
    shadow_mask = np.zeros(
        (10, 10),
        dtype=bool,
    )

    shadow_mask[2:6, 2:6] = True

    labels, _ = label_shadow_components(
        shadow_mask
    )

    component_label = labels[3, 3]

    eccentricity = component_eccentricity(
        labels,
        component_label,
    )

    assert eccentricity < 0.1


def test_component_eccentricity_elongated_component():
    shadow_mask = np.zeros(
        (20, 20),
        dtype=bool,
    )

    shadow_mask[8:10, 2:18] = True

    labels, _ = label_shadow_components(
        shadow_mask
    )

    component_label = labels[8, 10]

    eccentricity = component_eccentricity(
        labels,
        component_label,
    )

    assert eccentricity > 0.95


def test_vet_components_by_eccentricity():
    shadow_mask = np.zeros(
        (20, 20),
        dtype=bool,
    )

    shadow_mask[2:6, 2:6] = True
    shadow_mask[8:10, 2:18] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    qualified = vet_components_by_eccentricity(
        labels,
        num_components,
        max_eccentricity=0.97,
    )

    square_label = labels[3, 3]
    elongated_label = labels[8, 10]

    assert qualified[square_label]
    assert not qualified[elongated_label]


def test_vet_components_by_eccentricity_rejects_invalid_threshold():
    labels = np.zeros(
        (5, 5),
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        vet_components_by_eccentricity(
            labels,
            0,
            max_eccentricity=1.1,
        )
def test_component_solidity_convex_component():
    shadow_mask = np.zeros(
        (10, 10),
        dtype=bool,
    )

    shadow_mask[2:6, 2:6] = True

    labels, _ = label_shadow_components(
        shadow_mask
    )

    component_label = labels[3, 3]

    solidity = component_solidity(
        labels,
        component_label,
    )

    assert solidity > 0.95


def test_component_solidity_concave_component():
    shadow_mask = np.zeros(
        (10, 10),
        dtype=bool,
    )

    shadow_mask[2:6, 2:6] = True
    shadow_mask[4, 4] = False
    shadow_mask[5, 5] = False

    labels, _ = label_shadow_components(
        shadow_mask
    )

    component_label = labels[2, 2]

    solidity = component_solidity(
        labels,
        component_label,
    )

    assert 0.80 < solidity < 1.0


def test_vet_components_by_solidity():
    shadow_mask = np.zeros(
        (12, 12),
        dtype=bool,
    )

    shadow_mask[1:5, 1:5] = True

    # L-shaped component.
    shadow_mask[7:11, 7] = True
    shadow_mask[10, 7:11] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    qualified = vet_components_by_solidity(
        labels,
        num_components,
        min_solidity=0.80,
    )

    square_label = labels[2, 2]
    l_shape_label = labels[10, 8]

    assert qualified[square_label]
    assert not qualified[l_shape_label]


def test_vet_components_by_solidity_rejects_invalid_threshold():
    labels = np.zeros(
        (5, 5),
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        vet_components_by_solidity(
            labels,
            0,
            min_solidity=1.1,
        )
def test_shadow_centroid_single_pixel():
    labels = np.zeros((5, 5), dtype=int)
    labels[2, 3] = 1

    centroid = shadow_centroid(labels, 1)

    assert centroid == (3.0, 2.0)


def test_shadow_centroid_multiple_pixels():
    labels = np.zeros((5, 6), dtype=int)

    labels[1, 2] = 1
    labels[1, 3] = 1
    labels[2, 2] = 1
    labels[2, 3] = 1

    centroid = shadow_centroid(labels, 1)

    assert centroid == (2.5, 1.5)


def test_shadow_centroid_uses_requested_component():
    labels = np.zeros((5, 6), dtype=int)

    labels[1, 1] = 1
    labels[1, 2] = 1

    labels[3, 4] = 2
    labels[3, 5] = 2

    centroid = shadow_centroid(labels, 2)

    assert centroid == (4.5, 3.0)


def test_shadow_centroid_rejects_missing_component():
    labels = np.zeros((5, 5), dtype=int)

    with pytest.raises(ValueError):
        shadow_centroid(labels, 7)

def test_shadow_dmax_single_pixel():
    labels = np.zeros((5, 5), dtype=int)
    labels[2, 3] = 1

    dmax = shadow_dmax(labels, 1)

    assert dmax == 0.0


def test_shadow_dmax_two_pixels():
    labels = np.zeros((5, 5), dtype=int)

    labels[2, 2] = 1
    labels[2, 4] = 1

    dmax = shadow_dmax(labels, 1)

    assert dmax == 1.0


def test_shadow_dmax_square():
    labels = np.zeros((5, 5), dtype=int)

    labels[1, 1] = 1
    labels[1, 2] = 1
    labels[2, 1] = 1
    labels[2, 2] = 1

    dmax = shadow_dmax(labels, 1)

    assert dmax == np.sqrt(0.5)


def test_shadow_dmax_uses_requested_component():
    labels = np.zeros((6, 6), dtype=int)

    labels[1, 1] = 1
    labels[1, 2] = 1

    labels[4, 4] = 2
    labels[4, 5] = 2

    dmax = shadow_dmax(labels, 2)

    assert dmax == 0.5


def test_shadow_dmax_rejects_missing_component():
    labels = np.zeros((5, 5), dtype=int)

    with pytest.raises(ValueError):
        shadow_dmax(labels, 7)

def test_shadow_context_radius():
    assert shadow_context_radius(2.0) == 6.0


def test_shadow_context_radius_zero():
    assert shadow_context_radius(0.0) == 0.0


def test_shadow_context_radius_rejects_nonfinite():
    with pytest.raises(ValueError):
        shadow_context_radius(np.inf)


def test_shadow_context_radius_rejects_negative():
    with pytest.raises(ValueError):
        shadow_context_radius(-1.0)
def test_build_context_region_excludes_shadow_pixels():
    shadow_mask = np.zeros((7, 7), dtype=bool)
    shadow_mask[3, 3] = True

    context = build_context_region(
        shadow_mask,
        centroid=(3.0, 3.0),
        context_radius=2.0,
    )

    assert not context[3, 3]
    assert context[3, 4]
    assert context[3, 2]
    assert context[2, 3]
    assert context[4, 3]


def test_build_context_region_excludes_all_shadow_components():
    shadow_mask = np.zeros((7, 7), dtype=bool)

    # Current component.
    shadow_mask[3, 3] = True

    # A different shadow component inside the context radius.
    shadow_mask[3, 4] = True

    context = build_context_region(
        shadow_mask,
        centroid=(3.0, 3.0),
        context_radius=3.0,
    )

    assert not context[3, 3]
    assert not context[3, 4]


def test_build_context_region_uses_strict_radius():
    shadow_mask = np.zeros((7, 7), dtype=bool)

    context = build_context_region(
        shadow_mask,
        centroid=(3.0, 3.0),
        context_radius=2.0,
    )

    # Distance exactly 2: excluded.
    assert not context[3, 5]

    # Distance less than 2: included.
    assert context[3, 4]


def test_build_context_region_returns_boolean_mask():
    shadow_mask = np.zeros((5, 5), dtype=bool)

    context = build_context_region(
        shadow_mask,
        centroid=(2.0, 2.0),
        context_radius=2.0,
    )

    assert context.dtype == bool
    assert context.shape == shadow_mask.shape


def test_build_context_region_rejects_invalid_radius():
    shadow_mask = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError):
        build_context_region(
            shadow_mask,
            centroid=(2.0, 2.0),
            context_radius=-1.0,
        )

def test_local_background_level():
    image = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 100.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    context_mask = np.zeros((3, 3), dtype=bool)
    context_mask[0, 0] = True
    context_mask[0, 1] = True
    context_mask[1, 0] = True
    context_mask[2, 0] = True
    context_mask[2, 1] = True

    background = local_background_level(image, context_mask)

    assert background == 4.0


def test_local_background_level_uses_only_context():
    image = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 1000.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    context_mask = np.zeros((3, 3), dtype=bool)
    context_mask[0, 0] = True
    context_mask[0, 1] = True
    context_mask[1, 0] = True

    background = local_background_level(image, context_mask)

    assert background == 2.0


def test_local_background_level_rejects_empty_context():
    image = np.ones((3, 3), dtype=float)
    context_mask = np.zeros((3, 3), dtype=bool)

    with pytest.raises(ValueError):
        local_background_level(image, context_mask)


def test_local_background_level_rejects_shape_mismatch():
    image = np.ones((3, 3), dtype=float)
    context_mask = np.ones((4, 4), dtype=bool)

    with pytest.raises(ValueError):
        local_background_level(image, context_mask)
def test_raised_cosine_taper_inside_inner_half():
    assert raised_cosine_taper(0.0) == 1.0
    assert raised_cosine_taper(0.25) == 1.0
    assert raised_cosine_taper(0.499999) == 1.0


def test_raised_cosine_taper_transition():
    assert raised_cosine_taper(0.5) == 1.0
    assert np.isclose(raised_cosine_taper(0.75), 0.5)


def test_raised_cosine_taper_at_outer_boundary():
    assert np.isclose(raised_cosine_taper(1.0), 0.0)


def test_raised_cosine_taper_outside_window():
    assert raised_cosine_taper(1.01) == 0.0
    assert raised_cosine_taper(2.0) == 0.0


def test_raised_cosine_taper_rejects_invalid_input():
    with pytest.raises(ValueError):
        raised_cosine_taper(-0.1)

    with pytest.raises(ValueError):
        raised_cosine_taper(np.inf)

def test_excess_brightness_weights_use_positive_excess_only():
    image = np.array([
        [5.0, 10.0, 15.0],
        [20.0, 25.0, 30.0],
        [35.0, 40.0, 45.0],
    ])

    context_mask = np.ones((3, 3), dtype=bool)

    weights = build_excess_brightness_weights(
        image,
        context_mask,
        centroid=(1.0, 1.0),
        background_level=25.0,
        context_radius=10.0,
    )

    assert weights[0, 0] == 0.0
    assert weights[1, 1] == 0.0
    assert weights[2, 2] > 0.0


def test_excess_brightness_weights_respect_context_mask():
    image = np.full((5, 5), 20.0)
    context_mask = np.ones((5, 5), dtype=bool)
    context_mask[2, 4] = False

    weights = build_excess_brightness_weights(
        image,
        context_mask,
        centroid=(2.0, 2.0),
        background_level=10.0,
        context_radius=10.0,
    )

    assert weights[2, 4] == 0.0
    assert weights[2, 3] > 0.0


def test_excess_brightness_weights_apply_taper():
    image = np.full((5, 5), 20.0)
    context_mask = np.ones((5, 5), dtype=bool)

    weights = build_excess_brightness_weights(
        image,
        context_mask,
        centroid=(2.0, 2.0),
        background_level=10.0,
        context_radius=2.0,
    )

    # At the centroid: T(0) = 1.
    assert weights[2, 2] == 10.0

    # At distance 1 = r_ctx/2: T(0.5) = 1.
    assert weights[2, 3] == 10.0

    # At distance 2 = r_ctx: T(1) = 0.
    assert weights[2, 4] == 0.0


def test_excess_brightness_weights_have_zero_outside_context():
    image = np.full((5, 5), 20.0)
    context_mask = np.ones((5, 5), dtype=bool)

    weights = build_excess_brightness_weights(
        image,
        context_mask,
        centroid=(2.0, 2.0),
        background_level=10.0,
        context_radius=1.0,
    )

    assert weights[0, 0] == 0.0
    assert weights[2, 2] == 10.0


def test_excess_brightness_weights_reject_invalid_radius():
    image = np.ones((3, 3), dtype=float)
    context_mask = np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError):
        build_excess_brightness_weights(
            image,
            context_mask,
            centroid=(1.0, 1.0),
            background_level=0.0,
            context_radius=0.0,
        )

def test_light_centroid_single_weighted_pixel():
    weights = np.zeros((5, 5), dtype=float)
    weights[2, 3] = 10.0

    centroid = light_centroid(weights)

    assert centroid == (3.0, 2.0)


def test_light_centroid_uniform_square():
    weights = np.zeros((5, 5), dtype=float)
    weights[1:4, 1:4] = 1.0

    centroid = light_centroid(weights)

    assert centroid == (2.0, 2.0)


def test_light_centroid_weighted_position():
    weights = np.zeros((5, 5), dtype=float)

    weights[2, 2] = 1.0
    weights[2, 4] = 3.0

    centroid = light_centroid(weights)

    assert centroid == (3.5, 2.0)


def test_light_centroid_ignores_zero_weight_pixels():
    weights = np.zeros((5, 5), dtype=float)

    weights[0, 0] = 0.0
    weights[3, 4] = 5.0

    centroid = light_centroid(weights)

    assert centroid == (4.0, 3.0)


def test_light_centroid_rejects_non_positive_total_weight():
    weights = np.zeros((5, 5), dtype=float)

    with pytest.raises(ValueError):
        light_centroid(weights)

def test_centroid_separation_same_point():
    separation = centroid_separation(
        (2.0, 3.0),
        (2.0, 3.0),
    )

    assert separation == 0.0


def test_centroid_separation_horizontal():
    separation = centroid_separation(
        (2.0, 3.0),
        (5.0, 3.0),
    )

    assert separation == 3.0


def test_centroid_separation_vertical():
    separation = centroid_separation(
        (2.0, 3.0),
        (2.0, 7.0),
    )

    assert separation == 4.0


def test_centroid_separation_diagonal():
    separation = centroid_separation(
        (1.0, 1.0),
        (4.0, 5.0),
    )

    assert separation == 5.0


def test_centroid_separation_rejects_nonfinite_coordinates():
    with pytest.raises(ValueError):
        centroid_separation(
            (np.nan, 1.0),
            (2.0, 3.0),
        )

def test_minimum_centroid_separation():
    area = np.pi

    r_min = minimum_centroid_separation(area)

    assert np.isclose(r_min, 0.15)


def test_minimum_centroid_separation_known_area():
    area = 100.0

    r_min = minimum_centroid_separation(area)

    assert np.isclose(r_min, 0.15 * np.sqrt(100.0 / np.pi))


def test_minimum_centroid_separation_scales_with_sqrt_area():
    r_min_1 = minimum_centroid_separation(25.0)
    r_min_2 = minimum_centroid_separation(100.0)

    assert np.isclose(r_min_2 / r_min_1, 2.0)


def test_minimum_centroid_separation_rejects_zero_area():
    with pytest.raises(ValueError):
        minimum_centroid_separation(0.0)


def test_minimum_centroid_separation_rejects_negative_area():
    with pytest.raises(ValueError):
        minimum_centroid_separation(-1.0)


def test_minimum_centroid_separation_rejects_nonfinite_area():
    with pytest.raises(ValueError):
        minimum_centroid_separation(np.inf)

def test_centroid_separation_usable_above_threshold():
    assert is_centroid_separation_usable(2.0, 1.0)


def test_centroid_separation_usable_at_threshold():
    assert is_centroid_separation_usable(1.0, 1.0)


def test_centroid_separation_rejected_below_threshold():
    assert not is_centroid_separation_usable(0.99, 1.0)


def test_centroid_separation_usable_zero_values():
    assert is_centroid_separation_usable(0.0, 0.0)


def test_centroid_separation_usable_rejects_invalid_values():
    with pytest.raises(ValueError):
        is_centroid_separation_usable(np.nan, 1.0)

    with pytest.raises(ValueError):
        is_centroid_separation_usable(1.0, np.inf)

    with pytest.raises(ValueError):
        is_centroid_separation_usable(-1.0, 1.0)

    with pytest.raises(ValueError):
        is_centroid_separation_usable(1.0, -1.0)

def test_solar_azimuth_from_centroids_positive_x():
    angle = solar_azimuth_from_centroids(
        shadow_centroid=(5.0, 2.0),
        light_centroid=(2.0, 2.0),
    )

    assert np.isclose(angle, 0.0)


def test_solar_azimuth_from_centroids_positive_y():
    angle = solar_azimuth_from_centroids(
        shadow_centroid=(2.0, 5.0),
        light_centroid=(2.0, 2.0),
    )

    assert np.isclose(angle, np.pi / 2.0)


def test_solar_azimuth_from_centroids_negative_x():
    angle = solar_azimuth_from_centroids(
        shadow_centroid=(1.0, 2.0),
        light_centroid=(4.0, 2.0),
    )

    assert np.isclose(angle, np.pi)


def test_solar_azimuth_from_centroids_negative_y():
    angle = solar_azimuth_from_centroids(
        shadow_centroid=(2.0, 1.0),
        light_centroid=(2.0, 4.0),
    )

    assert np.isclose(angle, -np.pi / 2.0)


def test_solar_azimuth_from_centroids_diagonal():
    angle = solar_azimuth_from_centroids(
        shadow_centroid=(4.0, 5.0),
        light_centroid=(1.0, 1.0),
    )

    assert np.isclose(angle, np.arctan2(4.0, 3.0))


def test_solar_azimuth_from_centroids_rejects_coincident_centroids():
    with pytest.raises(ValueError):
        solar_azimuth_from_centroids(
            shadow_centroid=(2.0, 2.0),
            light_centroid=(2.0, 2.0),
        )


def test_solar_azimuth_from_centroids_rejects_nonfinite_coordinates():
    with pytest.raises(ValueError):
        solar_azimuth_from_centroids(
            shadow_centroid=(np.nan, 2.0),
            light_centroid=(1.0, 1.0),
        )

def test_effective_light_count_uniform_weights():
    weights = np.ones((2, 2), dtype=float)

    count = effective_light_count(weights)

    assert np.isclose(count, 4.0)


def test_effective_light_count_single_weight():
    weights = np.zeros((3, 3), dtype=float)
    weights[1, 1] = 5.0

    count = effective_light_count(weights)

    assert np.isclose(count, 1.0)


def test_effective_light_count_two_equal_weights():
    weights = np.zeros((3, 3), dtype=float)
    weights[1, 1] = 2.0
    weights[1, 2] = 2.0

    count = effective_light_count(weights)

    assert np.isclose(count, 2.0)


def test_effective_light_count_concentrated_weights():
    weights = np.zeros((3, 3), dtype=float)
    weights[1, 1] = 10.0
    weights[1, 2] = 1.0

    count = effective_light_count(weights)

    assert np.isclose(
        count,
        121.0 / 101.0,
    )


def test_effective_light_count_rejects_zero_total():
    weights = np.zeros((3, 3), dtype=float)

    with pytest.raises(ValueError):
        effective_light_count(weights)

def test_shadow_sample_count():
    labels = np.array(
        [
            [0, 1, 1],
            [0, 1, 2],
            [2, 2, 0],
        ],
        dtype=int,
    )

    assert shadow_sample_count(labels, 1) == 3
    assert shadow_sample_count(labels, 2) == 3


def test_shadow_sample_count_missing_component():
    labels = np.zeros((3, 3), dtype=int)

    assert shadow_sample_count(labels, 1) == 0


def test_shadow_sample_count_rejects_background():
    labels = np.zeros((3, 3), dtype=int)

    with pytest.raises(ValueError):
        shadow_sample_count(labels, 0)

def test_positional_uncertainty_uses_v3_formula():
    result = positional_uncertainty(
        shadow_sample_count=100.0,
        light_sample_count=100.0,
    )

    expected = 3.0 * np.sqrt(
        1.0 / (12.0 * 100.0)
        + 1.0 / (12.0 * 100.0)
    )

    assert np.isclose(result, expected)


def test_positional_uncertainty_default_segmentation_factor_is_three():
    result = positional_uncertainty(
        shadow_sample_count=100.0,
        light_sample_count=100.0,
    )

    result_without_inflation = positional_uncertainty(
        shadow_sample_count=100.0,
        light_sample_count=100.0,
        segmentation_factor=1.0,
    )

    assert np.isclose(result, 3.0 * result_without_inflation)


def test_positional_uncertainty_rejects_nonpositive_counts():
    with pytest.raises(ValueError):
        positional_uncertainty(0.0, 10.0)

    with pytest.raises(ValueError):
        positional_uncertainty(10.0, 0.0)


def test_positional_uncertainty_rejects_invalid_segmentation_factor():
    with pytest.raises(ValueError):
        positional_uncertainty(10.0, 10.0, 0.0)

    with pytest.raises(ValueError):
        positional_uncertainty(10.0, 10.0, -1.0)

def test_angular_uncertainty_uses_v3_formula():
    result = angular_uncertainty(
        positional_sigma=0.6,
        centroid_separation=3.0,
    )

    assert np.isclose(result, 0.2)


def test_angular_uncertainty_scales_inversely_with_separation():
    sigma = 0.6

    result_near = angular_uncertainty(sigma, 2.0)
    result_far = angular_uncertainty(sigma, 4.0)

    assert np.isclose(result_near, 2.0 * result_far)


def test_angular_uncertainty_zero_positional_sigma():
    result = angular_uncertainty(
        positional_sigma=0.0,
        centroid_separation=5.0,
    )

    assert np.isclose(result, 0.0)


def test_angular_uncertainty_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        angular_uncertainty(-1.0, 5.0)

    with pytest.raises(ValueError):
        angular_uncertainty(1.0, 0.0)

    with pytest.raises(ValueError):
        angular_uncertainty(1.0, -2.0)

def test_component_shape_moments_horizontal_component():
    labels = np.zeros((5, 7), dtype=int)

    labels[2, 1:6] = 1

    lambda_max, lambda_min, psi_maj = component_shape_moments(
        labels,
        1,
    )

    assert lambda_max > 0.0
    assert np.isclose(lambda_min, 0.0)
    assert np.isclose(abs(psi_maj), 0.0)


def test_component_shape_moments_vertical_component():
    labels = np.zeros((7, 5), dtype=int)

    labels[1:6, 2] = 1

    lambda_max, lambda_min, psi_maj = component_shape_moments(
        labels,
        1,
    )

    assert lambda_max > 0.0
    assert np.isclose(lambda_min, 0.0)
    assert np.isclose(abs(abs(psi_maj) - np.pi / 2), 0.0)


def test_component_shape_moments_diagonal_component():
    labels = np.zeros((7, 7), dtype=int)

    for i in range(1, 6):
        labels[i, i] = 1

    lambda_max, lambda_min, psi_maj = component_shape_moments(
        labels,
        1,
    )

    assert lambda_max > 0.0
    assert np.isclose(lambda_min, 0.0)
    assert np.isclose(abs(psi_maj), np.pi / 4)


def test_component_shape_moments_rejects_missing_component():
    labels = np.zeros((5, 5), dtype=int)

    with pytest.raises(ValueError):
        component_shape_moments(labels, 1)

def test_eccentricity_from_eigenvalues():
    result = eccentricity_from_eigenvalues(
        lambda_max=4.0,
        lambda_min=1.0,
    )

    assert np.isclose(result, np.sqrt(0.75))


def test_eccentricity_from_eigenvalues_isotropic():
    result = eccentricity_from_eigenvalues(
        lambda_max=5.0,
        lambda_min=5.0,
    )

    assert np.isclose(result, 0.0)


def test_eccentricity_from_eigenvalues_degenerate():
    result = eccentricity_from_eigenvalues(
        lambda_max=5.0,
        lambda_min=0.0,
    )

    assert np.isclose(result, 1.0)


def test_eccentricity_from_eigenvalues_rejects_invalid_order():
    with pytest.raises(ValueError):
        eccentricity_from_eigenvalues(
            lambda_max=1.0,
            lambda_min=2.0,
        )

def test_axis_angle_same_direction():
    result = axis_angle(0.0, 0.0)

    assert np.isclose(result, 0.0)


def test_axis_angle_perpendicular():
    result = axis_angle(0.0, np.pi / 2.0)

    assert np.isclose(result, np.pi / 2.0)


def test_axis_angle_opposite_axes():
    result = axis_angle(0.0, np.pi)

    assert np.isclose(result, 0.0)


def test_axis_angle_folds_obtuse_difference():
    result = axis_angle(np.deg2rad(10.0), np.deg2rad(170.0))

    assert np.isclose(result, np.deg2rad(20.0))


def test_axis_angle_wraps():
    result = axis_angle(np.deg2rad(179.0), np.deg2rad(-179.0))

    assert np.isclose(result, np.deg2rad(2.0))

def test_shape_ratio_from_eccentricity_parallel():
    result = shape_ratio_from_eccentricity(
        eccentricity=0.6,
        omega=0.0,
    )

    assert np.isclose(result, 0.64)


def test_shape_ratio_from_eccentricity_perpendicular():
    result = shape_ratio_from_eccentricity(
        eccentricity=0.6,
        omega=np.pi / 2.0,
    )

    assert np.isclose(result, 1.0 / 0.64)


def test_shape_ratio_from_eccentricity_boundary():
    result = shape_ratio_from_eccentricity(
        eccentricity=0.6,
        omega=np.pi / 4.0,
    )

    assert np.isclose(result, 1.0 / 0.64)


def test_shape_ratio_from_eccentricity_isotropic():
    result = shape_ratio_from_eccentricity(
        eccentricity=0.0,
        omega=0.0,
    )

    assert np.isclose(result, 1.0)


def test_shape_ratio_from_eccentricity_rejects_invalid_eccentricity():
    with pytest.raises(ValueError):
        shape_ratio_from_eccentricity(
            eccentricity=1.1,
            omega=0.0,
        )

def test_sector_shape_ratio_matches_v3_table_at_5_degrees():
    result = sector_shape_ratio(np.deg2rad(5.0))

    assert np.isclose(result, 0.023, atol=0.001)


def test_sector_shape_ratio_matches_v3_table_at_30_degrees():
    result = sector_shape_ratio(np.deg2rad(30.0))

    assert np.isclose(result, 0.840, atol=0.001)


def test_sector_shape_ratio_matches_v3_table_at_45_degrees():
    result = sector_shape_ratio(np.deg2rad(45.0))

    assert np.isclose(result, 1.858, atol=0.001)


def test_sector_shape_ratio_matches_v3_table_at_78_degrees():
    result = sector_shape_ratio(np.deg2rad(78.0))

    assert np.isclose(result, 3.673, atol=0.001)


def test_sector_shape_ratio_rejects_nonpositive_delta():
    with pytest.raises(ValueError):
        sector_shape_ratio(0.0)

def test_build_sector_shape_table_spacing_and_bounds():
    delta_degrees, rho_values = build_sector_shape_table()

    assert np.isclose(delta_degrees[0], 5.0)
    assert np.isclose(delta_degrees[-1], 78.0)

    assert len(delta_degrees) == 147
    assert len(rho_values) == 147

    assert np.allclose(np.diff(delta_degrees), 0.5)


def test_build_sector_shape_table_is_strictly_increasing():
    _, rho_values = build_sector_shape_table()

    assert np.all(np.diff(rho_values) > 0.0)


def test_build_sector_shape_table_matches_v3_endpoints():
    delta_degrees, rho_values = build_sector_shape_table()

    assert np.isclose(delta_degrees[0], 5.0)
    assert np.isclose(rho_values[0], 0.023, atol=0.001)

    assert np.isclose(delta_degrees[-1], 78.0)
    assert np.isclose(rho_values[-1], 3.673, atol=0.001)

def test_invert_sector_shape_ratio_recovers_table_endpoints():
    delta_degrees, rho_values = build_sector_shape_table()

    assert np.isclose(
        invert_sector_shape_ratio(rho_values[0]),
        np.deg2rad(delta_degrees[0]),
    )

    assert np.isclose(
        invert_sector_shape_ratio(rho_values[-1]),
        np.deg2rad(delta_degrees[-1]),
    )


def test_invert_sector_shape_ratio_interpolates():
    delta_degrees, rho_values = build_sector_shape_table()

    midpoint_rho = 0.5 * (rho_values[20] + rho_values[21])
    expected_delta_degrees = 0.5 * (delta_degrees[20] + delta_degrees[21])

    assert np.isclose(
        invert_sector_shape_ratio(midpoint_rho),
        np.deg2rad(expected_delta_degrees),
    )


def test_invert_sector_shape_ratio_rejects_out_of_domain():
    delta_degrees, rho_values = build_sector_shape_table()

    with pytest.raises(ValueError):
        invert_sector_shape_ratio(rho_values[0] - 1e-6)

    with pytest.raises(ValueError):
        invert_sector_shape_ratio(rho_values[-1] + 1e-6)


def test_invert_sector_shape_ratio_rejects_nonfinite():
    with pytest.raises(ValueError):
        invert_sector_shape_ratio(np.nan)

    with pytest.raises(ValueError):
        invert_sector_shape_ratio(np.inf)

def test_infer_sector_half_angle_parallel_branch():
    # omega < pi/4 uses rho = 1 - e^2.
    # e chosen so that rho is exactly the table value at 30 degrees.
    _, rho_values = build_sector_shape_table()
    rho = rho_values[50]  # 30 degrees

    eccentricity = np.sqrt(1.0 - rho)

    delta_hat = infer_sector_half_angle(
        eccentricity,
        0.0,
    )

    assert np.isclose(delta_hat, np.deg2rad(30.0), atol=1e-10)


def test_infer_sector_half_angle_perpendicular_branch():
    _, rho_values = build_sector_shape_table()

    # Use a rho > 1 so the perpendicular branch is physically valid.
    rho = rho_values[80]  # 45 degrees

    eccentricity = np.sqrt(1.0 - 1.0 / rho)

    delta_hat = infer_sector_half_angle(
        eccentricity,
        np.pi / 4,
    )

    assert np.isclose(delta_hat, np.deg2rad(45.0), atol=1e-10)


def test_infer_sector_half_angle_pi_over_four_uses_second_branch():
    _, rho_values = build_sector_shape_table()

    # omega = pi/4 must use the second branch.
    rho = rho_values[80]  # 45 degrees

    eccentricity = np.sqrt(1.0 - 1.0 / rho)

    delta_hat = infer_sector_half_angle(
        eccentricity,
        np.pi / 4,
    )

    assert np.isclose(delta_hat, np.deg2rad(45.0), atol=1e-10)

def test_is_sector_half_angle_usable_accepts_boundaries():
    assert is_sector_half_angle_usable(np.deg2rad(5.0))
    assert is_sector_half_angle_usable(np.deg2rad(78.0))


def test_is_sector_half_angle_usable_accepts_inside_range():
    assert is_sector_half_angle_usable(np.deg2rad(30.0))


def test_is_sector_half_angle_usable_rejects_outside_range():
    assert not is_sector_half_angle_usable(np.deg2rad(4.999))
    assert not is_sector_half_angle_usable(np.deg2rad(78.001))


def test_is_sector_half_angle_usable_rejects_nonfinite():
    assert not is_sector_half_angle_usable(np.nan)
    assert not is_sector_half_angle_usable(np.inf)
    assert not is_sector_half_angle_usable(-np.inf)

def test_precision_weight_zero_uncertainty_is_one():
    assert np.isclose(precision_weight(0.0), 1.0)
    assert np.isclose(precision_weight(angular_uncertainty=0.0), 1.0)


def test_precision_weight_decreases_with_uncertainty():
    w_small = precision_weight(np.deg2rad(0.5))
    w_large = precision_weight(np.deg2rad(2.0))

    assert w_small > w_large


def test_precision_weight_matches_v3_equation():
    sigma0 = np.deg2rad(0.5)
    sigma_theta = np.deg2rad(1.0)

    expected = sigma0**2 / (
        sigma0**2 + sigma_theta**2
    )

    assert np.isclose(
        precision_weight(sigma_theta),
        expected,
    )


def test_precision_weight_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        precision_weight(-1.0)

    with pytest.raises(ValueError):
        precision_weight(np.nan)

    with pytest.raises(ValueError):
        precision_weight(1.0, sigma0=0.0)

def test_symmetry_weight_low_eccentricity_is_one():
    assert np.isclose(
        symmetry_weight(0.39, np.pi / 4.0),
        1.0,
    )


def test_symmetry_weight_at_axis_threshold_uses_cosine_branch():
    assert np.isclose(
        symmetry_weight(0.40, 0.0),
        1.0,
    )

    assert np.isclose(
        symmetry_weight(0.40, np.pi / 4.0),
        0.0,
    )


def test_symmetry_weight_matches_v3_equation():
    eccentricity = 0.8
    omega = np.deg2rad(20.0)

    expected = np.cos(2.0 * omega) ** 2

    assert np.isclose(
        symmetry_weight(eccentricity, omega),
        expected,
    )


def test_symmetry_weight_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        symmetry_weight(np.nan, 0.0)

    with pytest.raises(ValueError):
        symmetry_weight(0.5, np.nan)

    with pytest.raises(ValueError):
        symmetry_weight(-0.1, 0.0)

    with pytest.raises(ValueError):
        symmetry_weight(0.5, -0.1)

    with pytest.raises(ValueError):
        symmetry_weight(0.5, np.pi / 2.0 + 0.1)

def test_shape_scale_uses_minimum_floor():
    values = np.deg2rad(np.array([20.0, 20.0, 20.0]))

    assert np.isclose(
        shape_scale(values),
        np.deg2rad(5.0),
    )


def test_shape_scale_uses_scaled_mad_when_larger_than_floor():
    values = np.deg2rad(np.array([10.0, 20.0, 30.0]))

    median = np.deg2rad(20.0)
    mad = np.deg2rad(10.0)
    expected = 1.4826 * mad

    assert np.isclose(
        shape_scale(values),
        expected,
    )


def test_shape_scale_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        shape_scale(np.array([]))

    with pytest.raises(ValueError):
        shape_scale(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError):
        shape_scale(np.array([1.0, np.nan]))

    with pytest.raises(ValueError):
        shape_scale(np.array([1.0]), sdelta_min=0.0)

def test_shape_weight_at_median_is_one():
    delta_med = np.deg2rad(30.0)
    s_delta = np.deg2rad(5.0)

    assert np.isclose(
        shape_weight(delta_med, delta_med, s_delta),
        1.0,
    )


def test_shape_weight_matches_v3_equation():
    delta_hat = np.deg2rad(40.0)
    delta_med = np.deg2rad(30.0)
    s_delta = np.deg2rad(5.0)

    expected = np.exp(
        -((delta_hat - delta_med) ** 2)
        / (2.0 * s_delta**2)
    )

    assert np.isclose(
        shape_weight(delta_hat, delta_med, s_delta),
        expected,
    )


def test_shape_weight_is_symmetric_about_median():
    delta_med = np.deg2rad(30.0)
    s_delta = np.deg2rad(5.0)

    lower = shape_weight(
        np.deg2rad(25.0),
        delta_med,
        s_delta,
    )

    upper = shape_weight(
        np.deg2rad(35.0),
        delta_med,
        s_delta,
    )

    assert np.isclose(lower, upper)


def test_shape_weight_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        shape_weight(np.nan, 0.0, 1.0)

    with pytest.raises(ValueError):
        shape_weight(0.0, np.nan, 1.0)

    with pytest.raises(ValueError):
        shape_weight(0.0, 0.0, np.nan)

    with pytest.raises(ValueError):
        shape_weight(0.0, 0.0, 0.0)

def test_base_weight_matches_product():
    result = base_weight(
        0.8,
        0.5,
        0.25,
    )

    assert np.isclose(
        result,
        0.8 * 0.5 * 0.25,
    )


def test_base_weight_applies_floor():
    result = base_weight(
        0.1,
        0.1,
        0.1,
    )

    assert np.isclose(result, 1e-3)


def test_base_weight_all_ones_is_one():
    assert np.isclose(
        base_weight(1.0, 1.0, 1.0),
        1.0,
    )


def test_base_weight_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        base_weight(np.nan, 1.0, 1.0)

    with pytest.raises(ValueError):
        base_weight(-0.1, 1.0, 1.0)

    with pytest.raises(ValueError):
        base_weight(1.0, 1.0, 1.0, w_floor=-1.0)

    with pytest.raises(ValueError):
        base_weight(1.0, 1.0, 1.0, w_floor=np.nan)

def test_normalize_base_weights_sets_maximum_to_one():
    weights = np.array([0.2, 0.5, 0.8])

    normalized = normalize_base_weights(weights)

    assert np.allclose(
        normalized,
        np.array([0.25, 0.625, 1.0]),
    )


def test_normalize_base_weights_preserves_relative_ratios():
    weights = np.array([0.2, 0.4, 0.8])

    normalized = normalize_base_weights(weights)

    assert np.isclose(
        normalized[1] / normalized[0],
        weights[1] / weights[0],
    )


def test_normalize_base_weights_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        normalize_base_weights(np.array([]))

    with pytest.raises(ValueError):
        normalize_base_weights(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError):
        normalize_base_weights(np.array([1.0, np.nan]))

    with pytest.raises(ValueError):
        normalize_base_weights(np.array([1.0, -0.1]))

    with pytest.raises(ValueError):
        normalize_base_weights(np.array([0.0, 0.0]))

def test_pairing_window_keeps_isolated_components():
    centroids = np.array([
        [0.0, 0.0],
        [100.0, 100.0],
    ])

    radii = np.array([10.0, 10.0])

    keep = vet_components_by_pairing_window(
        centroids,
        radii,
    )

    assert np.array_equal(
        keep,
        np.array([True, True]),
    )


def test_pairing_window_excludes_both_components_when_overlapping():
    centroids = np.array([
        [0.0, 0.0],
        [15.0, 0.0],
    ])

    radii = np.array([20.0, 10.0])

    keep = vet_components_by_pairing_window(
        centroids,
        radii,
    )

    assert np.array_equal(
        keep,
        np.array([False, False]),
    )


def test_pairing_window_uses_larger_context_radius():
    # Distance = 15, radii = 10 and 20.
    # max(r1,r2) = 20, so the pair is rejected.
    centroids = np.array([
        [0.0, 0.0],
        [15.0, 0.0],
    ])

    radii = np.array([10.0, 20.0])

    keep = vet_components_by_pairing_window(
        centroids,
        radii,
    )

    assert np.array_equal(
        keep,
        np.array([False, False]),
    )


def test_pairing_window_boundary_is_not_excluded():
    # Distance == max radius.
    # The Addendum uses strict "<", so this pair survives.
    centroids = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
    ])

    radii = np.array([20.0, 10.0])

    keep = vet_components_by_pairing_window(
        centroids,
        radii,
    )

    assert np.array_equal(
        keep,
        np.array([True, True]),
    )


def test_pairing_window_is_order_independent():
    centroids = np.array([
        [0.0, 0.0],
        [15.0, 0.0],
        [100.0, 100.0],
    ])

    radii = np.array([20.0, 10.0, 5.0])

    keep_original = vet_components_by_pairing_window(
        centroids,
        radii,
    )

    permutation = np.array([2, 0, 1])

    keep_permuted = vet_components_by_pairing_window(
        centroids[permutation],
        radii[permutation],
    )

    keep_restored = np.empty_like(keep_permuted)
    keep_restored[permutation] = keep_permuted

    assert np.array_equal(
        keep_original,
        keep_restored,
    )


def test_pairing_window_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        vet_components_by_pairing_window(
            np.array([0.0, 1.0]),
            np.array([1.0]),
        )

    with pytest.raises(ValueError):
        vet_components_by_pairing_window(
            np.array([[0.0, np.nan]]),
            np.array([1.0]),
        )

    with pytest.raises(ValueError):
        vet_components_by_pairing_window(
            np.array([[0.0, 0.0]]),
            np.array([-1.0]),
        )

def test_combine_component_vetting_gates_requires_all_preceding_gates():
    n = 3

    all_pass = np.ones(n, dtype=bool)

    centroids = np.array([
        [0.0, 0.0],
        [100.0, 100.0],
        [200.0, 200.0],
    ])

    radii = np.array([10.0, 10.0, 10.0])

    area_pass = all_pass.copy()
    border_pass = all_pass.copy()
    aspect_pass = all_pass.copy()
    eccentricity_pass = all_pass.copy()
    solidity_pass = all_pass.copy()

    # Component 1 fails an earlier gate.
    border_pass[1] = False

    result = combine_component_vetting_gates(
        area_pass,
        border_pass,
        aspect_pass,
        eccentricity_pass,
        solidity_pass,
        centroids,
        radii,
    )

    assert np.array_equal(
        result,
        np.array([True, False, True]),
    )


def test_combine_component_vetting_gates_pairing_only_considers_passing_components():
    # Components 0 and 1 are close enough to trigger pairing isolation,
    # but component 1 has already failed the border gate.
    centroids = np.array([
        [0.0, 0.0],
        [15.0, 0.0],
        [100.0, 100.0],
    ])

    radii = np.array([20.0, 10.0, 5.0])

    all_pass = np.ones(3, dtype=bool)

    border_pass = np.array([True, False, True])

    result = combine_component_vetting_gates(
        all_pass,
        border_pass,
        all_pass,
        all_pass,
        all_pass,
        centroids,
        radii,
    )

    # Component 0 must survive because component 1 is not a
    # gate-passing component and therefore cannot pair-exclude it.
    assert np.array_equal(
        result,
        np.array([True, False, True]),
    )


def test_combine_component_vetting_gates_applies_pairing_to_gate_passing_components():
    centroids = np.array([
        [0.0, 0.0],
        [15.0, 0.0],
        [100.0, 100.0],
    ])

    radii = np.array([20.0, 10.0, 5.0])

    all_pass = np.ones(3, dtype=bool)

    result = combine_component_vetting_gates(
        all_pass,
        all_pass,
        all_pass,
        all_pass,
        all_pass,
        centroids,
        radii,
    )

    assert np.array_equal(
        result,
        np.array([False, False, True]),
    )


def test_combine_component_vetting_gates_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        combine_component_vetting_gates(
            np.array([True, True]),
            np.array([True]),
            np.array([True, True]),
            np.array([True, True]),
            np.array([True, True]),
            np.zeros((2, 2)),
            np.ones(2),
        )

def test_compute_centroid_azimuth_measurement_returns_expected_structure():
    labels = np.zeros((21, 21), dtype=int)
    shadow_mask = np.zeros((21, 21), dtype=bool)
    image = np.ones((21, 21), dtype=float) * 10.0

    # Compact shadow component centred near (10, 10).
    labels[7:14, 7:14] = 1
    shadow_mask[7:14, 7:14] = True
    image[7:14, 7:14] = 1.0

    # Brighter region inside the context radius and away from the shadow.
    image[8:13, 17:20] = 20.0
    result = compute_centroid_azimuth_measurement(
        image,
        labels,
        1,
        shadow_mask,
    )

    assert len(result) == 8
    assert np.all(np.isfinite(result))


def test_compute_centroid_azimuth_measurement_rejects_nonpositive_fseg():
    labels = np.zeros((5, 5), dtype=int)
    shadow_mask = np.zeros((5, 5), dtype=bool)
    image = np.ones((5, 5), dtype=float)

    with pytest.raises(ValueError):
        compute_centroid_azimuth_measurement(
            image,
            labels,
            1,
            shadow_mask,
            fseg=0.0,
        )
def test_is_half_angle_usable_accepts_inclusive_v3_bounds():
    assert is_half_angle_usable(np.deg2rad(5.0))
    assert is_half_angle_usable(np.deg2rad(78.0))


def test_is_half_angle_usable_rejects_outside_v3_bounds():
    assert not is_half_angle_usable(np.deg2rad(4.999))
    assert not is_half_angle_usable(np.deg2rad(78.001))


def test_is_half_angle_usable_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        is_half_angle_usable(np.nan)

    with pytest.raises(ValueError):
        is_half_angle_usable(np.inf)


def test_is_half_angle_usable_rejects_invalid_range():
    with pytest.raises(ValueError):
        is_half_angle_usable(
            np.deg2rad(10.0),
            minimum_angle=np.deg2rad(78.0),
            maximum_angle=np.deg2rad(5.0),
        )

def test_precision_weight_is_one_at_zero_angular_uncertainty():
    assert precision_weight(0.0) == pytest.approx(1.0)


def test_precision_weight_matches_v3_formula():
    sigma0 = np.deg2rad(0.5)
    sigma_theta = sigma0

    expected = 0.5

    assert precision_weight(
        sigma_theta,
        sigma0=sigma0,
    ) == pytest.approx(expected)


def test_precision_weight_decreases_with_angular_uncertainty():
    sigma0 = np.deg2rad(0.5)

    low = precision_weight(
        sigma0 / 2.0,
        sigma0=sigma0,
    )

    high = precision_weight(
        2.0 * sigma0,
        sigma0=sigma0,
    )

    assert low > high


def test_precision_weight_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        precision_weight(np.nan)

    with pytest.raises(ValueError):
        precision_weight(np.inf)

    with pytest.raises(ValueError):
        precision_weight(-1.0)

    with pytest.raises(ValueError):
        precision_weight(0.1, sigma0=0.0)

def test_symmetry_weight_is_one_below_eaxis():
    assert symmetry_weight(
        eccentricity=0.39,
        omega=np.deg2rad(30.0),
    ) == pytest.approx(1.0)


def test_symmetry_weight_uses_cos_squared_above_eaxis():
    omega = np.deg2rad(30.0)

    expected = np.cos(2.0 * omega) ** 2

    assert symmetry_weight(
        eccentricity=0.40,
        omega=omega,
    ) == pytest.approx(expected)


def test_symmetry_weight_at_eaxis_is_axis_penalized():
    omega = np.deg2rad(30.0)

    expected = np.cos(2.0 * omega) ** 2

    assert symmetry_weight(
        eccentricity=0.40,
        omega=omega,
    ) == pytest.approx(expected)


def test_symmetry_weight_is_one_for_zero_axis_angle():
    assert symmetry_weight(
        eccentricity=0.80,
        omega=0.0,
    ) == pytest.approx(1.0)


def test_symmetry_weight_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        symmetry_weight(np.nan, 0.1)

    with pytest.raises(ValueError):
        symmetry_weight(0.5, np.nan)

    with pytest.raises(ValueError):
        symmetry_weight(-0.1, 0.1)

    with pytest.raises(ValueError):
        symmetry_weight(0.5, -0.1)

    with pytest.raises(ValueError):
        symmetry_weight(0.5, np.pi / 2.0 + 0.01)

    with pytest.raises(ValueError):
        symmetry_weight(0.5, 0.1, eaxis=-0.1)

    with pytest.raises(ValueError):
        symmetry_weight(0.5, 0.1, eaxis=1.1)

def test_shape_scale_uses_minimum_scale_floor():
    delta_hats = np.deg2rad(
        np.array([20.0, 20.1, 19.9, 20.0])
    )

    result = shape_scale(delta_hats)

    assert result == pytest.approx(np.deg2rad(5.0))


def test_shape_scale_uses_scaled_mad_when_above_floor():
    delta_hats = np.deg2rad(
        np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    )

    median = np.deg2rad(30.0)
    mad = np.deg2rad(10.0)
    expected = 1.4826 * mad

    assert shape_scale(delta_hats) == pytest.approx(expected)


def test_shape_scale_accepts_custom_minimum():
    delta_hats = np.deg2rad(
        np.array([20.0, 20.1, 19.9, 20.0])
    )

    minimum = np.deg2rad(2.0)

    assert shape_scale(
        delta_hats,
        sdelta_min=minimum,
    ) == pytest.approx(minimum)


def test_shape_scale_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        shape_scale(np.array([]))

    with pytest.raises(ValueError):
        shape_scale(np.array([[0.1, 0.2]]))

    with pytest.raises(ValueError):
        shape_scale(np.array([0.1, np.nan]))

    with pytest.raises(ValueError):
        shape_scale(
            np.array([0.1, 0.2]),
            sdelta_min=0.0,
        )

    with pytest.raises(ValueError):
        shape_scale(
            np.array([0.1, 0.2]),
            sdelta_min=np.nan,
        )

def test_shape_weight_is_one_at_median():
    delta = np.deg2rad(30.0)

    assert shape_weight(
        delta_hat=delta,
        delta_med=delta,
        s_delta=np.deg2rad(5.0),
    ) == pytest.approx(1.0)


def test_shape_weight_matches_v3_gaussian():
    delta_hat = np.deg2rad(40.0)
    delta_med = np.deg2rad(30.0)
    s_delta = np.deg2rad(5.0)

    expected = np.exp(
        -((delta_hat - delta_med) ** 2)
        / (2.0 * s_delta**2)
    )

    assert shape_weight(
        delta_hat,
        delta_med,
        s_delta,
    ) == pytest.approx(expected)


def test_shape_weight_decreases_with_distance_from_median():
    delta_med = np.deg2rad(30.0)
    s_delta = np.deg2rad(5.0)

    near = shape_weight(
        np.deg2rad(32.0),
        delta_med,
        s_delta,
    )

    far = shape_weight(
        np.deg2rad(50.0),
        delta_med,
        s_delta,
    )

    assert near > far


def test_shape_weight_is_symmetric_about_median():
    delta_med = np.deg2rad(30.0)
    s_delta = np.deg2rad(5.0)

    lower = shape_weight(
        np.deg2rad(25.0),
        delta_med,
        s_delta,
    )

    upper = shape_weight(
        np.deg2rad(35.0),
        delta_med,
        s_delta,
    )

    assert lower == pytest.approx(upper)


def test_shape_weight_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        shape_weight(np.nan, 0.5, 0.1)

    with pytest.raises(ValueError):
        shape_weight(0.5, np.nan, 0.1)

    with pytest.raises(ValueError):
        shape_weight(0.5, 0.5, np.nan)

    with pytest.raises(ValueError):
        shape_weight(0.5, 0.5, 0.0)

def test_base_weight_returns_product_when_above_floor():
    result = base_weight(
        w_prec=0.8,
        w_sym=0.9,
        w_shape=0.95,
        w_floor=1e-3,
    )

    expected = 0.8 * 0.9 * 0.95

    assert result == pytest.approx(expected)


def test_base_weight_applies_floor():
    result = base_weight(
        w_prec=0.01,
        w_sym=0.02,
        w_shape=0.03,
        w_floor=1e-3,
    )

    assert result == pytest.approx(1e-3)


def test_base_weight_uses_custom_floor():
    result = base_weight(
        w_prec=0.01,
        w_sym=0.02,
        w_shape=0.03,
        w_floor=0.1,
    )

    assert result == pytest.approx(0.1)


def test_base_weight_accepts_zero_component_weights():
    assert base_weight(
        w_prec=0.0,
        w_sym=1.0,
        w_shape=1.0,
    ) == pytest.approx(1e-3)


def test_base_weight_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        base_weight(np.nan, 1.0, 1.0)

    with pytest.raises(ValueError):
        base_weight(1.0, np.nan, 1.0)

    with pytest.raises(ValueError):
        base_weight(1.0, 1.0, np.nan)

    with pytest.raises(ValueError):
        base_weight(-0.1, 1.0, 1.0)

    with pytest.raises(ValueError):
        base_weight(1.0, -0.1, 1.0)

    with pytest.raises(ValueError):
        base_weight(1.0, 1.0, -0.1)

    with pytest.raises(ValueError):
        base_weight(1.0, 1.0, 1.0, w_floor=-0.1)

    with pytest.raises(ValueError):
        base_weight(1.0, 1.0, 1.0, w_floor=np.nan)

def test_normalize_base_weights_scales_by_ensemble_maximum():
    weights = np.array([0.2, 0.5, 1.0, 0.25])

    result = normalize_base_weights(weights)

    expected = np.array([0.2, 0.5, 1.0, 0.25])

    assert np.allclose(result, expected)


def test_normalize_base_weights_handles_nonunit_maximum():
    weights = np.array([2.0, 4.0, 8.0])

    result = normalize_base_weights(weights)

    expected = np.array([0.25, 0.5, 1.0])

    assert np.allclose(result, expected)


def test_normalize_base_weights_preserves_relative_ratios():
    weights = np.array([0.3, 0.6, 1.2])

    result = normalize_base_weights(weights)

    assert result[1] / result[0] == pytest.approx(2.0)
    assert result[2] / result[1] == pytest.approx(2.0)


def test_normalize_base_weights_rejects_empty_input():
    with pytest.raises(ValueError):
        normalize_base_weights(np.array([]))


def test_normalize_base_weights_rejects_non_1d_input():
    with pytest.raises(ValueError):
        normalize_base_weights(np.array([[1.0, 2.0]]))


def test_normalize_base_weights_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        normalize_base_weights(np.array([1.0, np.nan]))

    with pytest.raises(ValueError):
        normalize_base_weights(np.array([1.0, np.inf]))


def test_normalize_base_weights_rejects_negative_values():
    with pytest.raises(ValueError):
        normalize_base_weights(np.array([1.0, -0.1]))


def test_normalize_base_weights_rejects_all_zero_weights():
    with pytest.raises(ValueError):
        normalize_base_weights(np.array([0.0, 0.0, 0.0]))

def test_shape_median_matches_v3_ensemble_median():
    delta_hats = np.deg2rad(
        np.array([10.0, 20.0, 40.0, 50.0, 60.0])
    )

    expected = np.deg2rad(40.0)

    assert shape_median(delta_hats) == pytest.approx(expected)


def test_shape_median_handles_even_number_of_samples():
    delta_hats = np.deg2rad(
        np.array([10.0, 20.0, 30.0, 40.0])
    )

    expected = np.deg2rad(25.0)

    assert shape_median(delta_hats) == pytest.approx(expected)


def test_shape_median_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        shape_median(np.array([]))

    with pytest.raises(ValueError):
        shape_median(np.array([[0.1, 0.2]]))

    with pytest.raises(ValueError):
        shape_median(np.array([0.1, np.nan]))

    with pytest.raises(ValueError):
        shape_median(np.array([0.1, np.inf]))

def test_shape_scale_uses_same_median_as_shape_median():
    delta_hats = np.deg2rad(
        np.array([10.0, 20.0, 30.0, 40.0, 80.0])
    )

    median = shape_median(delta_hats)

    mad = np.median(
        np.abs(delta_hats - median)
    )

    expected = max(
        1.4826 * mad,
        np.deg2rad(5.0),
    )

    assert shape_scale(delta_hats) == pytest.approx(expected)

def test_shape_ensemble_statistics_matches_v3():
    delta_hats = np.deg2rad(
        np.array([10.0, 20.0, 30.0, 40.0, 80.0])
    )

    expected_median = np.deg2rad(30.0)

    mad = np.median(
        np.abs(delta_hats - expected_median)
    )

    expected_scale = max(
        1.4826 * mad,
        np.deg2rad(5.0),
    )

    median, scale = shape_ensemble_statistics(
        delta_hats
    )

    assert median == pytest.approx(expected_median)
    assert scale == pytest.approx(expected_scale)


def test_shape_ensemble_statistics_respects_custom_floor():
    delta_hats = np.deg2rad(
        np.array([20.0, 20.1, 19.9, 20.0])
    )

    minimum = np.deg2rad(10.0)

    median, scale = shape_ensemble_statistics(
        delta_hats,
        sdelta_min=minimum,
    )

    assert median == pytest.approx(np.deg2rad(20.0))
    assert scale == pytest.approx(minimum)

def test_compute_component_base_weight_matches_v3_chain():
    sigma_theta = np.deg2rad(0.5)
    sigma0 = np.deg2rad(0.5)

    eccentricity = 0.2
    omega = np.deg2rad(10.0)

    delta_hat = np.deg2rad(30.0)
    delta_med = np.deg2rad(30.0)
    s_delta = np.deg2rad(5.0)

    expected_precision = precision_weight(
        sigma_theta,
        sigma0=sigma0,
    )

    expected_symmetry = symmetry_weight(
        eccentricity,
        omega,
    )

    expected_shape = shape_weight(
        delta_hat,
        delta_med,
        s_delta,
    )

    expected = base_weight(
        expected_precision,
        expected_symmetry,
        expected_shape,
    )

    result = compute_component_base_weight(
        sigma_theta=sigma_theta,
        eccentricity=eccentricity,
        omega=omega,
        delta_hat=delta_hat,
        delta_med=delta_med,
        s_delta=s_delta,
        sigma0=sigma0,
    )

    assert result == pytest.approx(expected)


def test_compute_component_base_weight_applies_floor():
    result = compute_component_base_weight(
        sigma_theta=1.0,
        eccentricity=0.9,
        omega=np.deg2rad(30.0),
        delta_hat=np.deg2rad(60.0),
        delta_med=np.deg2rad(30.0),
        s_delta=np.deg2rad(5.0),
    )

    assert result == pytest.approx(1e-3)

def test_compute_normalized_component_weights_matches_ensemble_chain():
    sigma_theta = np.deg2rad(
        np.array([0.5, 1.0, 2.0])
    )

    eccentricity = np.array([
        0.2,
        0.5,
        0.8,
    ])

    omega = np.deg2rad(
        np.array([5.0, 20.0, 35.0])
    )

    delta_hat = np.deg2rad(
        np.array([25.0, 30.0, 35.0])
    )

    delta_med = shape_median(delta_hat)
    s_delta = shape_scale(delta_hat)

    expected_base = np.array([
        compute_component_base_weight(
            sigma_theta=sigma_theta[i],
            eccentricity=eccentricity[i],
            omega=omega[i],
            delta_hat=delta_hat[i],
            delta_med=delta_med,
            s_delta=s_delta,
        )
        for i in range(3)
    ])

    expected = normalize_base_weights(expected_base)

    result = compute_normalized_component_weights(
        sigma_theta=sigma_theta,
        eccentricity=eccentricity,
        omega=omega,
        delta_hat=delta_hat,
        delta_med=delta_med,
        s_delta=s_delta,
    )

    assert np.allclose(result, expected)


def test_compute_normalized_component_weights_has_unit_maximum():
    result = compute_normalized_component_weights(
        sigma_theta=np.deg2rad(
            np.array([0.5, 1.0, 2.0])
        ),
        eccentricity=np.array([
            0.2,
            0.5,
            0.8,
        ]),
        omega=np.deg2rad(
            np.array([5.0, 20.0, 35.0])
        ),
        delta_hat=np.deg2rad(
            np.array([25.0, 30.0, 35.0])
        ),
        delta_med=np.deg2rad(30.0),
        s_delta=np.deg2rad(5.0),
    )

    assert np.max(result) == pytest.approx(1.0)


def test_compute_normalized_component_weights_rejects_empty_ensemble():
    with pytest.raises(ValueError):
        compute_normalized_component_weights(
            sigma_theta=np.array([]),
            eccentricity=np.array([]),
            omega=np.array([]),
            delta_hat=np.array([]),
            delta_med=0.5,
            s_delta=0.1,
        )


def test_compute_normalized_component_weights_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_normalized_component_weights(
            sigma_theta=np.array([0.1, 0.2]),
            eccentricity=np.array([0.2]),
            omega=np.array([0.1, 0.2]),
            delta_hat=np.array([0.5, 0.6]),
            delta_med=0.5,
            s_delta=0.1,
        )


def test_compute_normalized_component_weights_rejects_non_1d_inputs():
    with pytest.raises(ValueError):
        compute_normalized_component_weights(
            sigma_theta=np.array([[0.1, 0.2]]),
            eccentricity=np.array([0.2, 0.3]),
            omega=np.array([0.1, 0.2]),
            delta_hat=np.array([0.5, 0.6]),
            delta_med=0.5,
            s_delta=0.1,
        )

def test_precision_weight_is_one_for_zero_uncertainty():
    weight = precision_weight(
        sigma_theta=0.0,
    )

    assert np.isclose(weight, 1.0)


def test_precision_weight_matches_v3_formula():
    sigma0 = np.deg2rad(0.5)
    sigma_theta = np.deg2rad(1.0)

    expected = sigma0**2 / (sigma0**2 + sigma_theta**2)

    weight = precision_weight(
        sigma_theta=sigma_theta,
        sigma0=sigma0,
    )

    assert np.isclose(weight, expected)


def test_precision_weight_decreases_with_uncertainty():
    w_small = precision_weight(np.deg2rad(0.5))
    w_large = precision_weight(np.deg2rad(2.0))

    assert w_small > w_large


def test_precision_weight_rejects_invalid_uncertainty():
    with pytest.raises(ValueError):
        precision_weight(np.nan)

    with pytest.raises(ValueError):
        precision_weight(-1.0)


def test_precision_weight_rejects_invalid_sigma0():
    with pytest.raises(ValueError):
        precision_weight(0.1, sigma0=0.0)

    with pytest.raises(ValueError):
        precision_weight(0.1, sigma0=-1.0)

    with pytest.raises(ValueError):
        precision_weight(0.1, sigma0=np.nan)

def test_infer_sector_half_angles_returns_valid_ensemble():
    _, rho_values = build_sector_shape_table()

    rho_30 = rho_values[50]  # 30 degrees
    rho_45 = rho_values[80]  # 45 degrees

    eccentricities = np.array([
        np.sqrt(1.0 - rho_30),
        np.sqrt(1.0 - 1.0 / rho_45),
    ])

    omegas = np.array([
        0.0,
        np.pi / 4.0,
    ])

    delta_hats, valid = infer_sector_half_angles(
        eccentricities,
        omegas,
    )

    assert np.all(valid)
    assert np.allclose(
        delta_hats,
        np.deg2rad([30.0, 45.0]),
        atol=1e-10,
    )


def test_infer_sector_half_angles_rejects_out_of_domain_component():
    eccentricities = np.array([
        0.99,
        0.5,
    ])

    omegas = np.array([
        0.0,
        0.0,
    ])

    delta_hats, valid = infer_sector_half_angles(
        eccentricities,
        omegas,
    )

    assert not valid[0]
    assert np.isnan(delta_hats[0])

    assert valid[1]
    assert np.isfinite(delta_hats[1])

def test_infer_sector_half_angles_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        infer_sector_half_angles(
            np.array([0.2, 0.3]),
            np.array([0.1]),
        )


def test_infer_sector_half_angles_rejects_empty_input():
    with pytest.raises(ValueError):
        infer_sector_half_angles(
            np.array([]),
            np.array([]),
        )


@pytest.mark.parametrize("eccentricity", [0.0, 1.0, 1.1, np.nan, np.inf, -np.inf])
def test_infer_sector_half_angles_marks_invalid_eccentricities_without_nan_leak(eccentricity):
    values, valid = infer_sector_half_angles(
        np.array([0.2, 0.5, eccentricity]),
        np.array([0.0, 0.0, 0.0]),
    )
    assert valid[:2].all()
    assert not valid[2]
    assert np.isnan(values[2])
    assert np.isfinite(values[valid]).all()


def test_shape_ratio_rejects_eccentricity_one_boundary():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        shape_ratio_from_eccentricity(1.0, np.pi / 2.0)
