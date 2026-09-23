import numpy as np

from patchrift.shadows.fusion import (
    circular_median,
    effective_sample_size,
    fuse_solar_azimuths,
    fused_azimuth_uncertainty,
    irls_iteration,
    irls_scale,
    irls_weights,
    residual_variance,
    robust_circular_fusion,
    tukey_biweight_weight,
    tukey_cutoff,
    weighted_circular_mean,
    wrap_angle_pi,
)

def test_wrap_angle_pi_basic():
    np.testing.assert_allclose(
        wrap_angle_pi(np.deg2rad(190.0)),
        np.deg2rad(-170.0),
    )

    np.testing.assert_allclose(
        wrap_angle_pi(np.deg2rad(-190.0)),
        np.deg2rad(170.0),
    )


def test_wrap_angle_pi_preserves_pi_boundary_convention():
    np.testing.assert_allclose(
        wrap_angle_pi(np.pi),
        -np.pi,
    )


def test_circular_median_simple_cluster():
    angles = np.deg2rad(
        np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    )

    result = circular_median(angles)

    np.testing.assert_allclose(
        result,
        np.deg2rad(12.0),
    )


def test_circular_median_wraps_across_boundary():
    angles = np.deg2rad(
        np.array([179.0, -179.0, 178.0, -178.0, 180.0])
    )

    result = circular_median(angles)

    # The circular median should lie at the +/-180-degree direction.
    np.testing.assert_allclose(
        abs(abs(result) - np.pi),
        0.0,
        atol=1e-12,
    )

def test_tukey_biweight_weight_at_zero():
    np.testing.assert_allclose(
        tukey_biweight_weight(0.0, 4.685),
        1.0,
    )


def test_tukey_biweight_weight_inside_cutoff():
    cutoff = 4.685
    residual = cutoff / 2.0

    expected = (1.0 - 0.5**2) ** 2

    np.testing.assert_allclose(
        tukey_biweight_weight(residual, cutoff),
        expected,
    )


def test_tukey_biweight_weight_at_cutoff_is_zero():
    cutoff = 4.685

    np.testing.assert_allclose(
        tukey_biweight_weight(cutoff, cutoff),
        0.0,
    )


def test_tukey_biweight_weight_outside_cutoff_is_zero():
    cutoff = 4.685

    np.testing.assert_allclose(
        tukey_biweight_weight(2.0 * cutoff, cutoff),
        0.0,
    )

def test_irls_scale_uses_median_absolute_residual():
    residuals = np.deg2rad(
        np.array([-1.0, 2.0, 3.0, 4.0, 5.0])
    )

    expected = 1.4826 * np.deg2rad(3.0)

    np.testing.assert_allclose(
        irls_scale(residuals, s_min=np.deg2rad(0.1)),
        expected,
    )


def test_irls_scale_respects_scale_floor():
    residuals = np.deg2rad(
        np.array([-0.01, 0.01, 0.02, -0.02, 0.0])
    )

    s_min = np.deg2rad(0.3)

    np.testing.assert_allclose(
        irls_scale(residuals, s_min),
        s_min,
    )


def test_irls_scale_uses_absolute_residuals_not_signed_median():
    residuals = np.deg2rad(
        np.array([-10.0, -8.0, 8.0, 10.0])
    )

    expected = 1.4826 * np.deg2rad(9.0)

    np.testing.assert_allclose(
        irls_scale(residuals, s_min=np.deg2rad(0.1)),
        expected,
    )


def test_irls_scale_rejects_empty_residuals():
    with np.testing.assert_raises(ValueError):
        irls_scale(
            np.array([], dtype=np.float64),
            s_min=np.deg2rad(0.3),
        )

def test_tukey_cutoff_uses_v3_constant():
    scale = np.deg2rad(0.5)

    expected = 4.685 * scale

    np.testing.assert_allclose(
        tukey_cutoff(scale),
        expected,
    )


def test_tukey_cutoff_accepts_custom_constant():
    scale = 2.0
    constant = 3.0

    np.testing.assert_allclose(
        tukey_cutoff(scale, constant),
        6.0,
    )


def test_tukey_cutoff_rejects_nonpositive_scale():
    with np.testing.assert_raises(ValueError):
        tukey_cutoff(0.0)


def test_tukey_cutoff_rejects_nonpositive_constant():
    with np.testing.assert_raises(ValueError):
        tukey_cutoff(1.0, 0.0)

def test_irls_weights_multiply_base_weights_by_tukey_factor():
    base_weights = np.array([1.0, 0.5, 0.25])
    residuals = np.array([0.0, 0.5, 1.0])
    cutoff = 2.0

    expected = np.array([
        1.0,
        0.5 * (1.0 - 0.25**2) ** 2,
        0.25 * (1.0 - 0.5**2) ** 2,
    ])

    np.testing.assert_allclose(
        irls_weights(base_weights, residuals, cutoff),
        expected,
    )


def test_irls_weights_zero_at_cutoff_and_beyond():
    base_weights = np.array([1.0, 1.0, 1.0])
    residuals = np.array([0.5, 1.0, 1.5])
    cutoff = 1.0

    expected = np.array([0.5625, 0.0, 0.0])

    np.testing.assert_allclose(
        irls_weights(base_weights, residuals, cutoff),
        expected,
    )


def test_irls_weights_preserves_zero_base_weights():
    base_weights = np.array([1.0, 0.0, 0.5])
    residuals = np.array([0.0, 0.0, 0.0])
    cutoff = 1.0

    expected = np.array([1.0, 0.0, 0.5])

    np.testing.assert_allclose(
        irls_weights(base_weights, residuals, cutoff),
        expected,
    )


def test_irls_weights_rejects_negative_base_weights():
    with np.testing.assert_raises(ValueError):
        irls_weights(
            np.array([1.0, -0.1]),
            np.array([0.0, 0.0]),
            1.0,
        )

def test_weighted_circular_mean_simple_cluster():
    angles = np.deg2rad(
        np.array([10.0, 20.0, 30.0])
    )
    weights = np.array([1.0, 1.0, 1.0])

    result, total_weight = weighted_circular_mean(
        angles,
        weights,
    )

    np.testing.assert_allclose(
        result,
        np.deg2rad(20.0),
    )

    np.testing.assert_allclose(
        total_weight,
        3.0,
    )


def test_weighted_circular_mean_respects_weights():
    angles = np.deg2rad(
        np.array([0.0, 90.0])
    )
    weights = np.array([3.0, 1.0])

    result, total_weight = weighted_circular_mean(
        angles,
        weights,
    )

    expected = np.arctan2(
        1.0,
        3.0,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )

    np.testing.assert_allclose(
        total_weight,
        4.0,
    )


def test_weighted_circular_mean_handles_wrap_boundary():
    angles = np.deg2rad(
        np.array([179.0, -179.0])
    )
    weights = np.array([1.0, 1.0])

    result, total_weight = weighted_circular_mean(
        angles,
        weights,
    )

    np.testing.assert_allclose(
        abs(abs(result) - np.pi),
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        total_weight,
        2.0,
    )


def test_weighted_circular_mean_rejects_zero_total_weight():
    with np.testing.assert_raises(ValueError):
        weighted_circular_mean(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
        )
def test_irls_iteration_moves_toward_weighted_cluster():
    angles = np.deg2rad(
        np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    )
    base_weights = np.ones(5)

    updated_angle, weights, scale, low_confidence = irls_iteration(
        angles,
        base_weights,
        previous_angle=np.deg2rad(10.0),
        previous_weights=np.ones(5),
        s_min=np.deg2rad(0.3),
    )

    assert not low_confidence
    assert np.all(weights > 0.0)
    assert scale >= np.deg2rad(0.3)

    assert np.deg2rad(10.0) < updated_angle < np.deg2rad(14.0)


def test_irls_iteration_downweights_large_outlier():
    angles = np.deg2rad(
        np.array([10.0, 11.0, 12.0, 13.0, 100.0])
    )
    base_weights = np.ones(5)

    updated_angle, weights, _, low_confidence = irls_iteration(
        angles,
        base_weights,
        previous_angle=np.deg2rad(12.0),
        previous_weights=np.ones(5),
        s_min=np.deg2rad(0.3),
    )

    assert not low_confidence

    # The 100-degree observation should receive zero Tukey weight.
    np.testing.assert_allclose(
        weights[-1],
        0.0,
    )

    assert np.deg2rad(10.0) < updated_angle < np.deg2rad(14.0)


def test_irls_iteration_retains_previous_state_when_weights_zero():
    angles = np.deg2rad(
        np.array([10.0, 20.0])
    )

    base_weights = np.zeros(2)
    previous_weights = np.array([0.7, 0.3])

    previous_angle = np.deg2rad(45.0)

    updated_angle, weights, _, low_confidence = irls_iteration(
        angles,
        base_weights,
        previous_angle=previous_angle,
        previous_weights=previous_weights,
        s_min=np.deg2rad(0.3),
    )

    assert low_confidence

    np.testing.assert_allclose(
        updated_angle,
        previous_angle,
    )

    np.testing.assert_allclose(
        weights,
        previous_weights,
    )

def test_irls_iteration_rejects_mismatched_previous_weights():
    with np.testing.assert_raises(ValueError):
        irls_iteration(
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
            previous_angle=0.0,
            previous_weights=np.array([1.0]),
            s_min=0.1,
        )

def test_robust_circular_fusion_converges_on_cluster():
    angles = np.deg2rad(
        np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    )
    base_weights = np.ones(5)

    result, weights, iterations, low_confidence = (
        robust_circular_fusion(
            angles,
            base_weights,
            s_min=np.deg2rad(0.3),
        )
    )

    assert not low_confidence
    assert iterations <= 5
    assert np.all(weights > 0.0)

    assert np.deg2rad(10.0) <= result <= np.deg2rad(14.0)

def test_robust_circular_fusion_rejects_large_outlier():
    angles = np.deg2rad(
        np.array([10.0, 11.0, 12.0, 13.0, 100.0])
    )
    base_weights = np.ones(5)

    result, weights, _, low_confidence = robust_circular_fusion(
        angles,
        base_weights,
        s_min=np.deg2rad(0.3),
    )

    assert not low_confidence

    np.testing.assert_allclose(
        weights[-1],
        0.0,
    )

    assert np.deg2rad(10.0) <= result <= np.deg2rad(13.0)

def test_robust_circular_fusion_handles_wrap_boundary():
    angles = np.deg2rad(
        np.array([178.0, 179.0, -179.0, -178.0])
    )
    base_weights = np.ones(4)

    result, weights, _, low_confidence = robust_circular_fusion(
        angles,
        base_weights,
        s_min=np.deg2rad(0.3),
    )

    assert not low_confidence
    assert np.all(weights > 0.0)

    np.testing.assert_allclose(
        abs(abs(result) - np.pi),
        0.0,
        atol=np.deg2rad(1.0),
    )

def test_robust_circular_fusion_handles_zero_weight_state():
    angles = np.deg2rad(
        np.array([10.0, 20.0])
    )
    base_weights = np.zeros(2)

    result, weights, iterations, low_confidence = (
        robust_circular_fusion(
            angles,
            base_weights,
            s_min=np.deg2rad(0.3),
        )
    )

    assert low_confidence
    assert iterations == 1

    np.testing.assert_allclose(
        result,
        circular_median(angles),
    )

    np.testing.assert_allclose(
        weights,
        base_weights,
    )

def test_effective_sample_size_equal_weights():
    weights = np.ones(4)

    np.testing.assert_allclose(
        effective_sample_size(weights),
        4.0,
    )


def test_effective_sample_size_downweights_unequal_distribution():
    weights = np.array([1.0, 1.0, 0.0])

    expected = 2.0

    np.testing.assert_allclose(
        effective_sample_size(weights),
        expected,
    )

def test_residual_variance_equal_weights():
    residuals = np.array([
        -1.0,
        0.0,
        1.0,
    ])

    weights = np.ones(3)

    expected = 2.0 / 3.0

    np.testing.assert_allclose(
        residual_variance(residuals, weights),
        expected,
    )


def test_residual_variance_respects_weights():
    residuals = np.array([
        0.0,
        2.0,
    ])

    weights = np.array([
        3.0,
        1.0,
    ])

    expected = 1.0

    np.testing.assert_allclose(
        residual_variance(residuals, weights),
        expected,
    )

def test_fused_azimuth_uncertainty_uses_systematic_floor():
    residuals = np.array([
        0.0,
        0.0,
        0.0,
    ])

    weights = np.ones(3)

    result = fused_azimuth_uncertainty(
        residuals,
        weights,
    )

    np.testing.assert_allclose(
        result,
        np.deg2rad(0.5),
    )


def test_fused_azimuth_uncertainty_uses_statistical_error_when_larger():
    residuals = np.array([
        -1.0,
        1.0,
    ])

    weights = np.ones(2)

    # N_eff = 2
    # s_res = 1
    # statistical error = 1/sqrt(2)
    expected = 1.0 / np.sqrt(2.0)

    result = fused_azimuth_uncertainty(
        residuals,
        weights,
        sigma_sys=0.1,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_fuse_solar_azimuths_returns_complete_result():
    angles = np.deg2rad(
        np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    )
    base_weights = np.ones(5)

    (
        fused_angle,
        weights,
        n_eff,
        residual_scale,
        sigma_phi,
        iterations,
        low_confidence,
    ) = fuse_solar_azimuths(
        angles,
        base_weights,
        s_min=np.deg2rad(0.3),
    )

    assert not low_confidence
    assert iterations <= 5
    assert n_eff > 0.0
    assert residual_scale >= 0.0
    assert sigma_phi >= np.deg2rad(0.5)

    assert np.deg2rad(10.0) <= fused_angle <= np.deg2rad(14.0)
    assert np.all(weights >= 0.0)

def test_fuse_solar_azimuths_rejects_outlier():
    angles = np.deg2rad(
        np.array([10.0, 11.0, 12.0, 13.0, 100.0])
    )
    base_weights = np.ones(5)

    (
        fused_angle,
        weights,
        n_eff,
        _,
        _,
        _,
        low_confidence,
    ) = fuse_solar_azimuths(
        angles,
        base_weights,
        s_min=np.deg2rad(0.3),
    )

    assert not low_confidence

    np.testing.assert_allclose(
        weights[-1],
        0.0,
    )

    assert n_eff > 0.0
    assert np.deg2rad(10.0) <= fused_angle <= np.deg2rad(13.0)

def test_fuse_solar_azimuths_handles_wrap_boundary():
    angles = np.deg2rad(
        np.array([178.0, 179.0, -179.0, -178.0])
    )
    base_weights = np.ones(4)

    (
        fused_angle,
        weights,
        n_eff,
        _,
        _,
        _,
        low_confidence,
    ) = fuse_solar_azimuths(
        angles,
        base_weights,
        s_min=np.deg2rad(0.3),
    )

    assert not low_confidence
    assert n_eff > 0.0
    assert np.all(weights > 0.0)

    np.testing.assert_allclose(
        abs(abs(fused_angle) - np.pi),
        0.0,
        atol=np.deg2rad(1.0),
    )

def test_fuse_solar_azimuths_propagates_low_confidence():
    angles = np.deg2rad(
        np.array([10.0, 20.0])
    )
    base_weights = np.zeros(2)

    (
        fused_angle,
        weights,
        n_eff,
        residual_scale,
        sigma_phi,
        iterations,
        low_confidence,
    ) = fuse_solar_azimuths(
        angles,
        base_weights,
        s_min=np.deg2rad(0.3),
    )

    assert low_confidence
    assert iterations == 1

    np.testing.assert_allclose(
        fused_angle,
        circular_median(angles),
    )

    np.testing.assert_allclose(
        weights,
        base_weights,
    )

    np.testing.assert_allclose(n_eff, 0.0)
    np.testing.assert_allclose(residual_scale, 0.0)
    np.testing.assert_allclose(
        sigma_phi,
        np.deg2rad(0.5),
    )

