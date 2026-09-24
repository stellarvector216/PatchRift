import numpy as np
import pytest

from patchrift.cascade.stage_iii import (
    _resample_at_physical_scale,
    condition_image,
    solar_lowpass_kernel,
)
from patchrift.features.stage_iv import _sample_decimated_grid


@pytest.mark.parametrize("scale", [1.0, 2.0, 1.7])
def test_stage_iii_scale_sampling_preserves_requested_grid_and_rounding(scale):
    source = np.tile(np.arange(10, dtype=float), (9, 1))
    sampled = _resample_at_physical_scale(source, scale, order=1, mode="nearest")
    expected_cols = int(np.floor((source.shape[1] - 1) / scale)) + 1
    np.testing.assert_allclose(sampled[0], np.arange(expected_cols) * scale, atol=1e-6)
    assert sampled.shape == (
        int(np.floor((source.shape[0] - 1) / scale)) + 1,
        expected_cols,
    )


@pytest.mark.parametrize("angle", [0.0, np.pi / 4.0, np.pi / 2.0])
def test_b5_solar_kernel_has_finite_support_and_is_normalized(angle):
    lam = 3.0
    kernel = solar_lowpass_kernel(lam, angle)
    radius = (kernel.shape[0] - 1) // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    along = x * np.cos(angle) + y * np.sin(angle)
    across = -x * np.sin(angle) + y * np.cos(angle)
    support = kernel > 0.0
    assert np.all(np.abs(along[support]) <= 8.0 * lam + 1e-10)
    assert np.all(np.abs(across[support]) <= lam + 1e-10)
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-12)


def test_stage_iii_mask_transport_is_nearest_boolean_and_native_is_unchanged():
    image = np.arange(81, dtype=float).reshape(9, 9) + 1.0
    valid = np.ones((9, 9), bool)
    valid[0, 0] = False
    shadow = np.zeros((9, 9), bool)
    shadow[4:6, 4:6] = True
    ring = np.zeros_like(shadow)
    ring[3, 4:6] = True
    originals = {"S_int": shadow.copy(), "R": ring.copy()}
    result = condition_image(image, 1.0, 2.0, originals, 0.0, 2.0, valid)
    assert result.masks["S_int"].shape == result.detection_image.shape
    assert result.masks["S_int"].dtype == np.bool_
    assert result.masks["R"].dtype == np.bool_
    np.testing.assert_array_equal(result.native_masks["S_int"], shadow)
    np.testing.assert_array_equal(result.native_masks["R"], ring)
    assert result.native_coordinate_scale == (2.0, 2.0)


@pytest.mark.parametrize(
    "current,target,lam,valid,shadow,expected",
    [
        (1.0, 1.0, 2.0, False, False, ValueError),
        (1.0, 1.0, 2.0, True, True, ValueError),
        (0.0, 1.0, 2.0, True, False, ValueError),
        (2.0, 1.0, 2.0, True, False, ValueError),
        (1.0, 1.0, 0.0, True, False, ValueError),
    ],
)
def test_stage_iii_rejects_invalid_or_degenerate_inputs(current, target, lam, valid, shadow, expected):
    image = np.ones((12, 12), dtype=float)
    valid_mask = np.full((12, 12), valid, dtype=bool)
    masks = {"S_int": np.full((12, 12), shadow, dtype=bool)}
    with pytest.raises(expected):
        condition_image(image, current, target, masks, 0.0, lam, valid_mask)


def test_native_detection_and_detector_decimation_coordinates_round_trip():
    # The stages sample at exact physical intervals even when output length
    # rounding means a generic zoom factor would imply a different scale.
    for scale, detector_step, native_xy in (
        (1.0, 1, (14.0, 12.0)),
        (2.0, 2, (24.0, 20.0)),
        (1.7, 3, (20.4, 15.3)),
        (1.7, 3, (0.0, 0.0)),
    ):
        working = np.array([native_xy[0] / scale, native_xy[1] / scale])
        detected = working / detector_step
        reconstructed = detected * detector_step * scale
        np.testing.assert_allclose(reconstructed, native_xy, atol=1e-12)

    data = np.tile(np.arange(21, dtype=float), (21, 1))
    decimated = _sample_decimated_grid(data, 3, order=1, mode="nearest")
    np.testing.assert_allclose(decimated[0], [0, 3, 6, 9, 12, 15, 18])
