import numpy as np
import pytest
from scipy import ndimage

from patchrift.shadows.morphology import (
    apply_opening_and_closing,
    cross_structuring_element,
    fill_shadow_holes,
    label_shadow_components,
    filter_shadow_components,
    build_shadow_regions,
)

def test_cross_structuring_element_is_3x3_cross():
    expected = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    result = cross_structuring_element()

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_opening_and_closing_returns_boolean_mask():
    shadow_mask = np.zeros(
        (7, 7),
        dtype=bool,
    )

    shadow_mask[2:5, 2:5] = True

    result = apply_opening_and_closing(
        shadow_mask,
    )

    assert result.dtype == np.bool_
    assert result.shape == shadow_mask.shape


def test_opening_removes_single_pixel_island():
    shadow_mask = np.zeros(
        (7, 7),
        dtype=bool,
    )

    shadow_mask[3, 3] = True

    result = apply_opening_and_closing(
        shadow_mask,
    )

    assert not result.any()


def test_cross_shaped_region_survives_opening():
    shadow_mask = np.zeros(
        (7, 7),
        dtype=bool,
    )

    shadow_mask[3, 2:5] = True
    shadow_mask[2:5, 3] = True

    result = apply_opening_and_closing(
        shadow_mask,
    )

    expected = shadow_mask

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_rejects_non_2d_mask():
    shadow_mask = np.zeros(
        (3, 3, 1),
        dtype=bool,
    )

    with pytest.raises(ValueError):
        apply_opening_and_closing(
            shadow_mask,
        )
def test_fill_shadow_holes_fills_enclosed_hole():
    shadow_mask = np.ones(
        (7, 7),
        dtype=bool,
    )

    shadow_mask[3, 3] = False

    result = fill_shadow_holes(
        shadow_mask,
    )

    expected = np.ones(
        (7, 7),
        dtype=bool,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_fill_shadow_holes_does_not_fill_boundary_connected_background():
    shadow_mask = np.ones(
        (7, 7),
        dtype=bool,
    )

    shadow_mask[3, 0:4] = False

    result = fill_shadow_holes(
        shadow_mask,
    )

    expected = shadow_mask

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_fill_shadow_holes_uses_8_connectivity():
    shadow_mask = np.ones(
        (5, 5),
        dtype=bool,
    )

    shadow_mask[1:4, 1:4] = False

    # Connect the interior background diagonally to the outside.
    shadow_mask[0, 0] = False
    shadow_mask[1, 1] = False

    result = fill_shadow_holes(
        shadow_mask,
    )

    assert not result[1, 1]
    assert not result[2, 2]


def test_fill_shadow_holes_rejects_non_2d_mask():
    shadow_mask = np.zeros(
        (3, 3, 1),
        dtype=bool,
    )

    with pytest.raises(ValueError):
        fill_shadow_holes(
            shadow_mask,
        )
def test_label_shadow_components_uses_8_connectivity():
    shadow_mask = np.zeros(
        (5, 5),
        dtype=bool,
    )

    shadow_mask[1, 1] = True
    shadow_mask[2, 2] = True

    labels, num_components = label_shadow_components(
        shadow_mask,
    )

    assert num_components == 1
    assert labels[1, 1] == labels[2, 2]
    assert labels[1, 1] > 0


def test_label_shadow_components_separates_disconnected_regions():
    shadow_mask = np.zeros(
        (5, 5),
        dtype=bool,
    )

    shadow_mask[1, 1] = True
    shadow_mask[3, 3] = True

    labels, num_components = label_shadow_components(
        shadow_mask,
    )

    assert num_components == 2
    assert labels[1, 1] > 0
    assert labels[3, 3] > 0
    assert labels[1, 1] != labels[3, 3]


def test_background_receives_label_zero():
    shadow_mask = np.zeros(
        (4, 4),
        dtype=bool,
    )

    shadow_mask[1, 1] = True

    labels, num_components = label_shadow_components(
        shadow_mask,
    )

    assert num_components == 1
    assert labels[0, 0] == 0


def test_label_shadow_components_rejects_non_2d_mask():
    shadow_mask = np.zeros(
        (3, 3, 1),
        dtype=bool,
    )

    with pytest.raises(ValueError):
        label_shadow_components(
            shadow_mask,
        )
def test_filter_shadow_components_removes_small_components():
    shadow_mask = np.zeros(
        (10, 10),
        dtype=bool,
    )

    shadow_mask[1:3, 1:3] = True
    shadow_mask[5:9, 5:9] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    filtered = filter_shadow_components(
        shadow_mask,
        labels,
        num_components,
        amin=5,
    )

    assert np.count_nonzero(filtered) == 16
    assert not filtered[1:3, 1:3].any()
    assert filtered[5:9, 5:9].all()


def test_filter_shadow_components_removes_high_aspect_ratio_components():
    shadow_mask = np.zeros(
        (10, 15),
        dtype=bool,
    )

    shadow_mask[2:3, 1:12] = True
    shadow_mask[5:9, 5:9] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    filtered = filter_shadow_components(
        shadow_mask,
        labels,
        num_components,
        amin=1,
        epsilon_ar=8.0,
    )

    assert not filtered[2:3, 1:12].any()
    assert filtered[5:9, 5:9].all()


def test_filter_shadow_components_keeps_area_and_aspect_ratio_boundary():
    shadow_mask = np.zeros(
        (10, 10),
        dtype=bool,
    )

    shadow_mask[2:6, 2:6] = True

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    filtered = filter_shadow_components(
        shadow_mask,
        labels,
        num_components,
        amin=16,
        epsilon_ar=1.0,
    )

    assert filtered[2:6, 2:6].all()


def test_filter_shadow_components_rejects_invalid_parameters():
    shadow_mask = np.ones(
        (4, 4),
        dtype=bool,
    )

    labels, num_components = label_shadow_components(
        shadow_mask
    )

    with pytest.raises(ValueError):
        filter_shadow_components(
            shadow_mask,
            labels,
            num_components,
            amin=-1,
        )

    with pytest.raises(ValueError):
        filter_shadow_components(
            shadow_mask,
            labels,
            num_components,
            epsilon_ar=0,
        )
def test_build_shadow_regions():
    shadow_mask = np.zeros(
        (7, 7),
        dtype=bool,
    )

    shadow_mask[2:5, 2:5] = True

    shadow_region, ring, interior = build_shadow_regions(
        shadow_mask
    )

    structure = cross_structuring_element()

    expected_eroded = ndimage.binary_erosion(
        shadow_mask,
        structure=structure,
    )

    expected_dilated = ndimage.binary_dilation(
        shadow_mask,
        structure=structure,
    )

    expected_ring = expected_dilated & ~expected_eroded

    assert np.array_equal(
        shadow_region,
        shadow_mask,
    )

    assert np.array_equal(
        interior,
        expected_eroded,
    )

    assert np.array_equal(
        ring,
        expected_ring,
    )


def test_build_shadow_regions_empty_mask():
    shadow_mask = np.zeros(
        (5, 5),
        dtype=bool,
    )

    shadow_region, ring, interior = build_shadow_regions(
        shadow_mask
    )

    assert not shadow_region.any()
    assert not ring.any()
    assert not interior.any()


def test_build_shadow_regions_rejects_non_2d_input():
    shadow_mask = np.zeros(
        (5, 5, 1),
        dtype=bool,
    )

    with pytest.raises(ValueError):
        build_shadow_regions(shadow_mask)