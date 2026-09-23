import numpy as np
import pytest

from patchrift.shadows.scharr import scharr_kernels
from patchrift.shadows.scharr import scharr_gradients
from patchrift.shadows.scharr import gradient_magnitude
from patchrift.shadows.scharr import gradient_orientation
from patchrift.shadows.scharr import scharr_azimuth_resultant
from patchrift.shadows.scharr import select_scharr_samples
from patchrift.shadows.scharr import scharr_region_status
from patchrift.shadows.scharr import scharr_fallback_uncertainty
from patchrift.shadows.scharr import evaluate_scharr_region
from patchrift.shadows.scharr import scharr_azimuth_cross_check

def test_scharr_kernels_match_v3():
    expected_x = np.array(
        [
            [-3.0, 0.0, 3.0],
            [-10.0, 0.0, 10.0],
            [-3.0, 0.0, 3.0],
        ]
    ) / 32.0

    expected_y = expected_x.T

    gx, gy = scharr_kernels()

    np.testing.assert_allclose(gx, expected_x)
    np.testing.assert_allclose(gy, expected_y)

def test_scharr_gradients_on_linear_x_ramp():
    image = np.tile(np.arange(9, dtype=float), (9, 1))

    gx, gy = scharr_gradients(image)

    np.testing.assert_allclose(gx[2:-2, 2:-2], 1.0)
    np.testing.assert_allclose(gy[2:-2, 2:-2], 0.0)

def test_gradient_magnitude():
    gx = np.array([[3.0, 0.0], [0.0, -5.0]])
    gy = np.array([[4.0, 7.0], [0.0, 12.0]])

    magnitude = gradient_magnitude(gx, gy)

    expected = np.array(
        [
            [5.0, 7.0],
            [0.0, 13.0],
        ]
    )

    np.testing.assert_allclose(magnitude, expected)

def test_gradient_orientation_points_into_shadow():
    gx = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
    )

    gy = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    theta = gradient_orientation(gx, gy)

    expected = np.array(
        [
            [-np.pi, -np.pi / 2.0, 0.0],
            [-np.pi, np.pi / 2.0, 0.0],
        ]
    )

    np.testing.assert_allclose(theta, expected)

def test_scharr_azimuth_resultant_single_direction():
    magnitude = np.ones(4)
    orientation = np.full(4, np.pi / 4.0)

    C, S, phi, rho_bar = scharr_azimuth_resultant(
        magnitude,
        orientation,
    )

    np.testing.assert_allclose(C, 4.0 * np.cos(np.pi / 4.0))
    np.testing.assert_allclose(S, 4.0 * np.sin(np.pi / 4.0))
    np.testing.assert_allclose(phi, np.pi / 4.0)
    np.testing.assert_allclose(rho_bar, 1.0)


def test_scharr_azimuth_resultant_cancels_opposite_directions():
    magnitude = np.ones(2)
    orientation = np.array([0.0, np.pi])

    C, S, phi, rho_bar = scharr_azimuth_resultant(
        magnitude,
        orientation,
    )

    np.testing.assert_allclose(C, 0.0, atol=1e-15)
    np.testing.assert_allclose(S, 0.0, atol=1e-15)
    np.testing.assert_allclose(rho_bar, 0.0, atol=1e-15)

def test_select_scharr_samples_applies_r_valid_and_saturation_masks():
    magnitude = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    orientation = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    boundary_region = np.array(
        [
            [True, True],
            [False, True],
        ]
    )

    valid_mask = np.array(
        [
            [True, False],
            [True, True],
        ]
    )

    saturated_mask = np.array(
        [
            [False, False],
            [False, True],
        ]
    )

    selected_magnitude, selected_orientation = select_scharr_samples(
        magnitude,
        orientation,
        boundary_region,
        valid_mask,
        saturated_mask,
    )

    np.testing.assert_allclose(
        selected_magnitude,
        np.array([1.0]),
    )

    np.testing.assert_allclose(
        selected_orientation,
        np.array([0.1]),
    )


def test_select_scharr_samples_returns_empty_when_r_is_empty():
    magnitude = np.ones((2, 2))
    orientation = np.zeros((2, 2))

    boundary_region = np.zeros((2, 2), dtype=bool)
    valid_mask = np.ones((2, 2), dtype=bool)
    saturated_mask = np.zeros((2, 2), dtype=bool)

    selected_magnitude, selected_orientation = select_scharr_samples(
        magnitude,
        orientation,
        boundary_region,
        valid_mask,
        saturated_mask,
    )

    assert selected_magnitude.size == 0
    assert selected_orientation.size == 0

def test_scharr_region_status_detects_empty_r():
    boundary_region = np.zeros(
        (3, 3),
        dtype=bool,
    )

    selected_magnitude = np.array(
        [],
        dtype=float,
    )

    region_nonempty, samples_available = scharr_region_status(
        boundary_region,
        selected_magnitude,
    )

    assert region_nonempty is False
    assert samples_available is False


def test_scharr_region_status_detects_r_with_no_usable_samples():
    boundary_region = np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ]
    )

    selected_magnitude = np.array(
        [],
        dtype=float,
    )

    region_nonempty, samples_available = scharr_region_status(
        boundary_region,
        selected_magnitude,
    )

    assert region_nonempty is True
    assert samples_available is False


def test_scharr_region_status_detects_usable_samples():
    boundary_region = np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ]
    )

    selected_magnitude = np.array(
        [2.5],
        dtype=float,
    )

    region_nonempty, samples_available = scharr_region_status(
        boundary_region,
        selected_magnitude,
    )

    assert region_nonempty is True
    assert samples_available is True

def test_scharr_fallback_uncertainty():
    rho_bar = np.exp(-0.5)

    sigma = scharr_fallback_uncertainty(rho_bar)

    np.testing.assert_allclose(sigma, 1.0)


def test_scharr_fallback_uncertainty_perfect_coherence():
    sigma = scharr_fallback_uncertainty(1.0)

    np.testing.assert_allclose(sigma, 0.0)


def test_scharr_fallback_uncertainty_rejects_invalid_rho():
    with pytest.raises(ValueError):
        scharr_fallback_uncertainty(-0.1)

    with pytest.raises(ValueError):
        scharr_fallback_uncertainty(1.1)

def test_evaluate_scharr_region_empty_r():
    boundary_region = np.zeros(
        (2, 2),
        dtype=bool,
    )

    selected_magnitude = np.array([], dtype=float)
    selected_orientation = np.array([], dtype=float)

    result = evaluate_scharr_region(
        boundary_region,
        selected_magnitude,
        selected_orientation,
    )

    assert result == (
        False,
        False,
        None,
        None,
        None,
    )


def test_evaluate_scharr_region_nonempty_r_without_samples():
    boundary_region = np.array(
        [
            [True, False],
            [False, False],
        ]
    )

    selected_magnitude = np.array([], dtype=float)
    selected_orientation = np.array([], dtype=float)

    result = evaluate_scharr_region(
        boundary_region,
        selected_magnitude,
        selected_orientation,
    )

    assert result == (
        True,
        False,
        None,
        None,
        None,
    )


def test_evaluate_scharr_region_with_samples():
    boundary_region = np.array(
        [
            [True, False],
            [False, False],
        ]
    )

    selected_magnitude = np.ones(4)
    selected_orientation = np.full(
        4,
        np.pi / 4.0,
    )

    region_nonempty, samples_available, phi, rho_bar, sigma_phi = (
        evaluate_scharr_region(
            boundary_region,
            selected_magnitude,
            selected_orientation,
        )
    )

    assert region_nonempty is True
    assert samples_available is True

    np.testing.assert_allclose(
        phi,
        np.pi / 4.0,
    )

    np.testing.assert_allclose(
        rho_bar,
        1.0,
    )

    np.testing.assert_allclose(
        sigma_phi,
        0.0,
    )

def test_scharr_azimuth_cross_check_small_difference():
    difference, flagged = scharr_azimuth_cross_check(
        np.deg2rad(10.0),
        np.deg2rad(20.0),
    )

    np.testing.assert_allclose(
        difference,
        np.deg2rad(-10.0),
    )

    assert flagged is False


def test_scharr_azimuth_cross_check_flags_more_than_90_degrees():
    difference, flagged = scharr_azimuth_cross_check(
        np.deg2rad(100.0),
        np.deg2rad(0.0),
    )

    np.testing.assert_allclose(
        difference,
        np.deg2rad(100.0),
    )

    assert flagged is True


def test_scharr_azimuth_cross_check_does_not_flag_exactly_90_degrees():
    difference, flagged = scharr_azimuth_cross_check(
        np.deg2rad(90.0),
        np.deg2rad(0.0),
    )

    np.testing.assert_allclose(
        difference,
        np.deg2rad(90.0),
    )

    assert flagged is False


def test_scharr_azimuth_cross_check_wraps_across_pi_boundary():
    difference, flagged = scharr_azimuth_cross_check(
        np.deg2rad(179.0),
        np.deg2rad(-179.0),
    )

    np.testing.assert_allclose(
        difference,
        np.deg2rad(-2.0),
    )

    assert flagged is False