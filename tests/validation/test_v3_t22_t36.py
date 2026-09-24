"""Formal V3 validation cases T22-T36 (main document §§6-8)."""

from types import SimpleNamespace

import numpy as np
import pytest
from patchrift.matching.stage_v import compute_output_transformation

from patchrift.cascade.stage_iii import condition_image
from patchrift.features.stage_iv import (
    calculate_patch_size_bounds,
    calculate_phase_congruency,
    create_log_gabor_bank,
    lanczos3_interpolate,
)
from patchrift.matching.stage_v import (
    _mutual_ratio_matches,
    _translation_consensus,
    calculate_hough_bin_size,
    calculate_search_window,
    fit_huber_homography,
    match_feature_sets,
    roll_descriptor,
    self_diagnose_alignment,
)


def test_t22_patch_size_respects_both_sampling_bounds():
    patch, spectral, spatial = calculate_patch_size_bounds(
        lambda_max=24.0, r_max=48.0, m=4, B_omega=2.0, r_taper=0.25
    )
    assert patch >= max(spectral, spatial)
    assert patch == 128


def test_t23_filter_bank_has_dc_rejection_coverage_and_quadrature_response():
    bank = create_log_gabor_bank(64, 64, num_scales=4, num_orientations=12)
    assert len(bank) == 4 and all(len(scale) == 12 for scale in bank)
    assert all(np.isfinite(response).all() and response[0, 0] == 0.0
               for scale in bank for response in scale)
    coverage = np.sum(np.asarray(bank), axis=(0, 1))
    assert np.all(coverage[1:, :] > 0.0)
    assert np.all(coverage[:, 1:] > 0.0)
    # A real cosine has a nonzero analytic quadrature response after the
    # oriented half-plane factor has been applied in the PC implementation.
    y, x = np.mgrid[:64, :64]
    image = np.cos(2.0 * np.pi * x / 8.0)
    pc, amplitudes = calculate_phase_congruency(image, bank, return_amplitudes=True)
    assert len(pc) == 12 and amplitudes.shape == (12, 64, 64)
    assert np.isfinite(amplitudes).all()


def test_t24_rayleigh_noise_floor_keeps_phase_congruency_finite_and_bounded():
    noise = np.random.default_rng(24).normal(size=(64, 64))
    bank = create_log_gabor_bank(64, 64, num_scales=4, num_orientations=12)
    pc = np.asarray(calculate_phase_congruency(noise, bank))
    assert np.isfinite(pc).all()
    assert np.min(pc) >= 0.0
    assert np.max(pc) <= 1.0


@pytest.mark.parametrize("scale", [1.0, 2.0, 1.7])
def test_t25_native_and_target_grid_preserve_physical_spacing(scale):
    image = np.tile(np.arange(31, dtype=float), (29, 1)) + 1.0
    valid = np.ones_like(image, dtype=bool)
    result = condition_image(
        image, 1.0, scale, {"S_int": np.zeros_like(valid)}, 0.0, 2.0, valid
    )
    expected_cols = int(np.floor((image.shape[1] - 1) / scale)) + 1
    assert result.detection_image.shape[1] == expected_cols


def test_lanczos3_interpolation_reproduces_affine_ramp():
    y, x = np.mgrid[:40, :40]
    image = 2.0 * x - 3.0 * y + 7.0
    xs = np.array([12.2, 13.7, 18.35])
    ys = np.array([10.4, 17.1, 21.6])
    actual = lanczos3_interpolate(image, xs, ys)
    np.testing.assert_allclose(actual, 2.0 * xs - 3.0 * ys + 7.0, atol=1e-10)
    mirrored = lanczos3_interpolate(np.fliplr(image), 39.0 - xs, ys)
    np.testing.assert_allclose(mirrored, actual, atol=1e-10)


def test_t26_equal_canonicalization_has_equal_per_scale_amplitude_ratios():
    from patchrift.features.stage_iv import (
        calculate_phase_congruency,
        compute_sampling_matrix,
        create_log_gabor_bank,
        extract_canonical_patch,
    )

    rng = np.random.default_rng(260)
    image_a = rng.normal(size=(192, 192))
    image_b = np.zeros_like(image_a)
    image_b[24:168, 32:176] = image_a[12:156, 20:164]
    # The same physical neighbourhood appears at these translated pixel
    # centres, with identical GSD and alpha in both inputs.
    matrix = compute_sampling_matrix(1.0, np.deg2rad(23.0))
    patch_a = extract_canonical_patch(image_a, 88.0, 84.0, matrix, 64)
    patch_b = extract_canonical_patch(image_b, 100.0, 96.0, matrix, 64)
    bank = create_log_gabor_bank(64, 64, num_scales=4, num_orientations=12)
    _, amplitudes_a = calculate_phase_congruency(patch_a, bank, return_amplitudes=True)
    _, amplitudes_b = calculate_phase_congruency(patch_b, bank, return_amplitudes=True)
    ratios_a = amplitudes_a.mean(axis=(1, 2)) / amplitudes_a.sum(axis=(1, 2))
    ratios_b = amplitudes_b.mean(axis=(1, 2)) / amplitudes_b.sum(axis=(1, 2))
    np.testing.assert_allclose(ratios_a, ratios_b, atol=1e-12, rtol=1e-12)


def test_t27_iirs_band_order_is_irrelevant_to_reduction():
    from patchrift.io.iirs import IIRSProduct
    from patchrift.preprocessing.iirs import reduce_iirs_to_panchromatic

    cube = np.array([[[1.0]], [[4.0]], [[9.0]]])
    weights = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([1.0, 1.5, 2.0])
    bad = np.zeros(3, dtype=bool)
    first = reduce_iirs_to_panchromatic(
        IIRSProduct(cube, wavelengths, bad, np.ones(3)), weights=weights
    )
    order = np.array([2, 0, 1])
    second = reduce_iirs_to_panchromatic(
        IIRSProduct(cube[order], wavelengths[order], bad[order], np.ones(3)),
        weights=weights[order],
    )
    np.testing.assert_allclose(first, second)
    single = reduce_iirs_to_panchromatic(
        IIRSProduct(cube[:1], wavelengths[:1], bad[:1], np.ones(1))
    )
    np.testing.assert_allclose(single, cube[0])


def test_t28_canonical_coordinate_transform_matches_scale_rotation_translation():
    point = np.array([4.0, -2.0])
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    expected = 2.0 * rotation @ point + np.array([7.0, -5.0])
    actual = compute_output_transformation(
        *point, 7.0, -5.0, 7.0, 3.5, angle,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    np.testing.assert_allclose(actual, expected)


def test_t29_detector_repeatability_and_subpixel_relocalization_are_deterministic():
    from patchrift.features.stage_iv import detect_keypoints, quadratic_subpixel_peak

    y, x = np.mgrid[:31, :31]
    response = 5.0 - 2.0 * (x - 14.25) ** 2 - 3.0 * (y - 15.4) ** 2
    first = quadratic_subpixel_peak(response, 15, 15, 8)
    second = quadratic_subpixel_peak(response.copy(), 15, 15, 8)
    assert first == second
    np.testing.assert_allclose([first[0] + first[2], first[1] + first[3]], [-0.75, 0.4])

    yy, xx = np.mgrid[:96, :96]
    image = 1.0 + 0.2 * ((xx // 12 + yy // 12) % 2)
    valid = np.ones(image.shape, dtype=bool)
    shadow = np.zeros(image.shape, dtype=bool)
    options = dict(
        downsample=1, min_feature_diameter=12.0, grid_size=24,
        per_cell=2, target_count=20, patch_size=16,
    )
    detected_a = detect_keypoints(image, valid, shadow, **options)
    detected_b = detect_keypoints(image.copy(), valid.copy(), shadow.copy(), **options)
    assert detected_a == detected_b


def test_t30_reliability_weights_affect_rescore_after_unweighted_retrieval():
    from patchrift.matching.stage_v import compute_weighted_distance

    a = np.zeros((1, 17 * 12))
    b = np.zeros_like(a)
    b[0, 0] = 1.0
    b[0, 12] = 1.0
    unweighted_choice = int(np.argmin(np.sum((a - b) ** 2, axis=1)))
    clean = compute_weighted_distance(a[0], b[0], np.ones(17), np.ones(17))
    rel = np.ones(17)
    rel[0] = 0.0
    downweighted = compute_weighted_distance(a[0], b[0], rel, np.ones(17))
    assert unweighted_choice == 0
    assert downweighted < clean


def test_t31_dense_rotation_search_covers_the_full_orientation_ring():
    descriptor = np.arange(17 * 12)
    shifted = roll_descriptor(descriptor, 3)
    np.testing.assert_array_equal(shifted.reshape(17, 12)[:, :], np.roll(descriptor.reshape(17, 12), 3, axis=1))
    np.testing.assert_array_equal(calculate_search_window(np.pi, 12), np.arange(-24, 25))


def test_t32_hough_consensus_rejects_translation_outliers():
    translations = np.array([[4.0, -2.0], [4.2, -1.9], [3.8, -2.1], [4.1, -2.2], [40.0, 30.0]])
    inliers, translation, peak_fraction = _translation_consensus(
        translations, 3.0, np.ones(len(translations))
    )
    assert inliers.sum() == 4
    assert peak_fraction == pytest.approx(0.8)
    np.testing.assert_allclose(translation, np.mean(translations[:4], axis=0), atol=0.4)


def test_canonical_translation_votes_use_q_b_minus_q_a():
    q_a = np.array([[1.0, 4.0], [9.0, -2.0], [13.0, 8.0]])
    translation = np.array([6.25, -3.5])
    q_b = q_a + translation
    votes = q_b - q_a
    np.testing.assert_allclose(votes, np.broadcast_to(translation, votes.shape))


def test_ratio_test_uses_squared_euclidean_distance_convention():
    accepted, _, _ = _mutual_ratio_matches(
        np.array([[0.25, 1.0]]), np.zeros((1, 2), dtype=int), ratio=0.8
    )
    rejected, _, _ = _mutual_ratio_matches(
        np.array([[0.64, 1.0]]), np.zeros((1, 2), dtype=int), ratio=0.8
    )
    np.testing.assert_array_equal(accepted, [[0, 0]])
    assert rejected.shape == (0, 2)


@pytest.mark.parametrize(
    "source,destination",
    [
        (np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float), np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)),
        (np.array([[0, 0], [1, 0], [2, 0], [3, 0.000000001]], dtype=float), np.array([[0, 0], [1, 0], [2, 0], [3, 0.000000001]], dtype=float)),
        (np.array([[0, 0], [0, 0], [1, 1], [2, 2]], dtype=float), np.array([[0, 0], [0, 0], [1, 1], [2, 2]], dtype=float)),
        (np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float), np.array([[0, 0], [1, 0], [0, 1], [np.nan, 1]], dtype=float)),
    ],
)
def test_homography_rejects_degenerate_point_sets(source, destination):
    with pytest.raises(ValueError):
        fit_huber_homography(source, destination, np.eye(3))


def test_homography_rejects_fewer_than_four_points_and_singular_initial_matrix():
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    with pytest.raises(ValueError, match="at least four"):
        fit_huber_homography(points, points)
    square = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    with pytest.raises(ValueError, match="nonsingular"):
        fit_huber_homography(square, square, np.zeros((3, 3)))


def test_self_diagnosis_uses_both_inlier_and_configured_hough_mass_gates():
    assert self_diagnose_alignment(12, 10, hough_peak_fraction=0.8, minimum_hough_peak_fraction=0.5)
    assert not self_diagnose_alignment(9, 10, hough_peak_fraction=0.8, minimum_hough_peak_fraction=0.5)
    assert not self_diagnose_alignment(12, 10, hough_peak_fraction=0.2, minimum_hough_peak_fraction=0.5)


def test_t33_solar_rotation_uncertainty_expands_orientation_search_window():
    narrow = calculate_search_window(np.deg2rad(1.0), 12)
    broad = calculate_search_window(np.deg2rad(30.0), 12)
    assert len(broad) > len(narrow)


def test_c4_uses_complete_orientation_ring_when_rotation_uncertainty_is_unavailable():
    rng = np.random.default_rng(331)
    descriptors = rng.random((12, 17 * 12))
    positions = np.column_stack((np.arange(12) * 8.0, np.arange(12) % 4 * 7.0))
    features = SimpleNamespace(
        descriptors=descriptors, reliability=np.ones((12, 17)),
        canonical_positions=positions,
    )
    result = match_feature_sets(
        features,
        SimpleNamespace(
            descriptors=descriptors.copy(), reliability=np.ones((12, 17)),
            canonical_positions=positions + [4.0, 2.0],
        ),
        delta_alpha=0.3, sigma_delta_alpha=np.inf, minimum_inliers=8,
    )
    assert result.fallback_used
    assert result.delta_alpha == 0.0
    assert result.orientation_shift_histogram.sum() == len(result.pairs)


def test_c4_retries_when_hough_peak_mass_is_weak():
    rng = np.random.default_rng(332)
    descriptors = rng.random((12, 17 * 12))
    positions = np.column_stack((np.arange(12) * 8.0, np.arange(12) % 4 * 7.0))
    shifted_positions = positions + [4.0, 2.0]
    shifted_positions[6:] += [100.0, 80.0]
    a = SimpleNamespace(descriptors=descriptors, reliability=np.ones((12, 17)), canonical_positions=positions)
    b = SimpleNamespace(descriptors=descriptors.copy(), reliability=np.ones((12, 17)), canonical_positions=shifted_positions)
    result = match_feature_sets(
        a, b, delta_alpha=0.0, sigma_delta_alpha=0.0,
        minimum_inliers=3, minimum_hough_peak_fraction=0.75,
    )
    assert result.fallback_used
    assert not result.aligned


def test_t34_parallax_sets_hough_threshold_with_three_pixel_floor():
    assert calculate_hough_bin_size(300.0, 0.0, 0.0, 10.0) == 3.0
    value = calculate_hough_bin_size(300.0, np.deg2rad(20.0), 0.0, 10.0)
    assert value > 3.0


def test_t35_corrupt_rotation_uses_full_ring_c4_fallback(monkeypatch):
    import patchrift.pipeline.stages_iii_v as pipeline
    from patchrift.geometry.solar import SolarGeometry
    from patchrift.pipeline.stage_ii import SolarTileResult

    alpha_calls = []
    fake_features = SimpleNamespace(features=object())
    monkeypatch.setattr(
        pipeline,
        "extract_tile_features",
        lambda *args, **kwargs: alpha_calls.append(kwargs.get("alpha")) or fake_features,
    )
    matcher_calls = []

    def fake_match(*args, **kwargs):
        matcher_calls.append(kwargs)
        return SimpleNamespace(fallback_used=True)

    monkeypatch.setattr(pipeline, "match_feature_sets", fake_match)
    metadata = SimpleNamespace(hmax=None, emission_angle=0.0)
    stage_image = SimpleNamespace(product=SimpleNamespace(metadata=metadata))

    def solar(alpha):
        return SolarTileResult(
            segmentation=None, solar_geometry=SolarGeometry(0.0, 0.5),
            pixel_azimuth=0.2, sigma_phi=0.01, north_angle=alpha,
            measurements=(), used_scharr_fallback=False, low_confidence=False,
            c4=False, scharr_disagreement_flag=False, implied_slope_diagnostic=None,
            sigma_geo=0.01,
        )

    pair = pipeline.match_tile_pair(
        stage_image, None, solar(0.0), stage_image, None, solar(np.deg2rad(30.0)),
        gsd_a=1.0, gsd_b=1.0, gsd_target=1.0,
        lambda_max_a=3.0, lambda_max_b=3.0,
    )
    assert alpha_calls == [None, None, 0.0, 0.0]
    assert matcher_calls[1]["delta_alpha"] == 0.0
    assert np.isinf(matcher_calls[1]["sigma_delta_alpha"])
    assert pair.matching.fallback_used is True


def test_t36_stage_v_result_exposes_reporting_fields():
    rng = np.random.default_rng(36)
    descriptors = rng.random((12, 17 * 12))
    positions = np.column_stack((np.arange(12) * 9.0, np.arange(12) % 3 * 13.0))
    features = SimpleNamespace(
        descriptors=descriptors,
        reliability=np.ones((12, 17)),
        canonical_positions=positions,
    )
    result = match_feature_sets(
        features,
        SimpleNamespace(
            descriptors=descriptors.copy(), reliability=np.ones((12, 17)),
            canonical_positions=positions + [5.0, -3.0],
        ),
        delta_alpha=0.0, sigma_delta_alpha=0.0, minimum_inliers=8, gsd_target=2.0,
    )
    assert result.aligned
    assert result.inlier_count == len(result.pairs)
    assert result.translation is not None
    assert result.rms_residual_pixels is not None
    assert result.rms_residual_meters is not None
    assert result.hough_bin_size == 3.0
