import numpy as np

from patchrift.geometry.handedness import fit_geolocation_jacobian


def test_jacobian_for_non_mirrored_mapping():
    footprint_corners = np.array([
        [85.0, 20.0],
        [85.1, 20.0],
        [85.1, 20.1],
        [85.0, 20.1],
    ])

    footprint_pixel_coords = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0],
    ])

    jacobian = fit_geolocation_jacobian(
        footprint_corners,
        footprint_pixel_coords,
    )

    assert jacobian.shape == (2, 2)
    assert np.linalg.det(jacobian) > 0


def test_jacobian_for_mirrored_mapping():
    footprint_corners = np.array([
        [85.0, 20.0],
        [85.1, 20.0],
        [85.1, 20.1],
        [85.0, 20.1],
    ])

    footprint_pixel_coords = np.array([
        [100.0, 0.0],
        [0.0, 0.0],
        [0.0, 100.0],
        [100.0, 100.0],
    ])

    jacobian = fit_geolocation_jacobian(
        footprint_corners,
        footprint_pixel_coords,
    )

    assert jacobian.shape == (2, 2)
    assert np.linalg.det(jacobian) < 0
def test_non_mirrored_mapping_is_unchanged():
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    valid_mask = np.ones((3, 4), dtype=bool)

    footprint_pixel_coords = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 2.0],
        [0.0, 2.0],
    ])

    jacobian = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    from patchrift.geometry.handedness import normalize_handedness

    result_image, result_mask, result_coords, was_reflected = (
        normalize_handedness(
            image,
            valid_mask,
            footprint_pixel_coords,
            jacobian,
        )
    )

    np.testing.assert_array_equal(result_image, image)
    np.testing.assert_array_equal(result_mask, valid_mask)
    np.testing.assert_array_equal(result_coords, footprint_pixel_coords)
    assert was_reflected is False


def test_mirrored_mapping_reflects_image_mask_and_coordinates():
    image = np.arange(12, dtype=np.float32).reshape(3, 4)

    valid_mask = np.array([
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
    ])

    footprint_pixel_coords = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 2.0],
        [0.0, 2.0],
    ])

    jacobian = np.array([
        [-1.0, 0.0],
        [0.0, 1.0],
    ])

    from patchrift.geometry.handedness import normalize_handedness

    result_image, result_mask, result_coords, was_reflected = (
        normalize_handedness(
            image,
            valid_mask,
            footprint_pixel_coords,
            jacobian,
        )
    )

    expected_image = np.fliplr(image)
    expected_mask = np.fliplr(valid_mask)

    expected_coords = np.array([
        [3.0, 0.0],
        [0.0, 0.0],
        [0.0, 2.0],
        [3.0, 2.0],
    ])

    np.testing.assert_array_equal(result_image, expected_image)
    np.testing.assert_array_equal(result_mask, expected_mask)
    np.testing.assert_array_equal(result_coords, expected_coords)
    assert was_reflected is True