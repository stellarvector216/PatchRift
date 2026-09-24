import numpy as np
import pytest

from patchrift.geometry.rotation import (
    inter_image_rotation,
    inter_image_rotation_uncertainty,
    pixel_north_angle,
)


def test_pixel_north_angle_wraps_to_zero_to_two_pi():
    assert pixel_north_angle(0.1, 0.2) == pytest.approx(2.0 * np.pi - 0.1)


def test_inter_image_rotation_uses_signed_wrapped_difference():
    assert inter_image_rotation(np.deg2rad(350.0), np.deg2rad(10.0)) == pytest.approx(np.deg2rad(20.0))


def test_rotation_uncertainty_propagates_independent_terms():
    assert inter_image_rotation_uncertainty(3.0, 4.0) == pytest.approx(5.0)


def test_canonical_frame_rotation_sign_round_trips_native_displacement():
    from patchrift.features.stage_iv import compute_sampling_matrix
    from patchrift.matching.stage_v import map_to_canonical_frame

    canonical = np.array([3.0, -2.0])
    for alpha in (0.0, np.deg2rad(35.0), np.deg2rad(170.0)):
        matrix = compute_sampling_matrix(2.5, alpha)
        native = matrix @ canonical
        recovered = map_to_canonical_frame(native[0], native[1], matrix)
        np.testing.assert_allclose(recovered, canonical, atol=1e-12)
