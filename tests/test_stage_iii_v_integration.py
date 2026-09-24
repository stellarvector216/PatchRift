from types import SimpleNamespace

import numpy as np
import pytest

from patchrift.features.stage_iv import _analytic_half_plane, quadratic_subpixel_peak
from patchrift.matching.stage_v import (
    build_unweighted_ann_shortlist,
    compute_weighted_distance,
    match_feature_sets,
)


def test_analytic_half_plane_breaks_pi_symmetry_and_removes_dc():
    factor = _analytic_half_plane(32, 32, 0.0)
    assert factor[0, 0] == 0.0
    assert factor[0, 2] == 2.0
    assert factor[0, -2] == 0.0


def test_stage_v_full_correspondence_path_finds_translation_and_homography():
    rng = np.random.default_rng(17)
    descriptors = rng.random((12, 17 * 12))
    positions = np.column_stack((np.arange(12) * 9.0, np.arange(12) % 3 * 13.0))
    features_a = SimpleNamespace(
        descriptors=descriptors,
        reliability=np.ones((12, 17)),
        canonical_positions=positions,
    )
    features_b = SimpleNamespace(
        descriptors=descriptors.copy(),
        reliability=np.ones((12, 17)),
        canonical_positions=positions + np.array([5.0, -3.0]),
    )

    result = match_feature_sets(
        features_a,
        features_b,
        delta_alpha=0.0,
        sigma_delta_alpha=0.0,
        minimum_inliers=8,
    )

    assert result.aligned
    assert not result.fallback_used
    assert result.homography is not None
    assert len(result.pairs) == 12
    np.testing.assert_allclose(result.homography, [[1, 0, 5], [0, 1, -3], [0, 0, 1]], atol=1e-5)


def test_public_pair_pipeline_connects_tile_features_to_stage_v(monkeypatch):
    import patchrift.pipeline.stages_iii_v as pipeline
    from patchrift.geometry.solar import SolarGeometry
    from patchrift.pipeline.stage_ii import SolarTileResult

    descriptors = np.random.default_rng(42).random((12, 17 * 12))
    positions = np.column_stack((np.arange(12) * 9.0, np.arange(12) % 3 * 13.0))
    feature_set = SimpleNamespace(
        descriptors=descriptors, reliability=np.ones((12, 17)),
        canonical_positions=positions,
    )
    fake_features = SimpleNamespace(features=feature_set)
    calls = []
    monkeypatch.setattr(pipeline, "extract_tile_features", lambda *args, **kwargs: calls.append(args) or fake_features)
    def solar(alpha):
        return SolarTileResult(
            segmentation=None, solar_geometry=SolarGeometry(0.0, 0.5),
            pixel_azimuth=0.2, sigma_phi=0.01, north_angle=alpha,
            measurements=(), used_scharr_fallback=False, low_confidence=False,
            c4=False, scharr_disagreement_flag=False, implied_slope_diagnostic=None,
            sigma_geo=0.01,
        )

    result = pipeline.match_tile_pair(
        None, None, solar(0.1), None, None, solar(0.1),
        gsd_a=1.0, gsd_b=1.0, gsd_target=1.0,
        lambda_max_a=3.0, lambda_max_b=3.0,
        matching_options={"minimum_inliers": 8},
    )
    assert len(calls) == 2
    assert result.matching.aligned
    np.testing.assert_allclose(result.matching.delta_alpha, 0.0)


def test_quadratic_subpixel_peak_recovers_known_offset():
    y, x = np.mgrid[:31, :31]
    peak_x, peak_y = 15.2, 14.35
    response = 10.0 - 2.0 * (x - peak_x) ** 2 - 3.0 * (y - peak_y) ** 2 - 0.3 * (x - peak_x) * (y - peak_y)
    dx, dy, sub_x, sub_y = quadratic_subpixel_peak(response, 15, 15, 8)
    np.testing.assert_allclose([dx + sub_x, dy + sub_y], [0.2, -0.65], atol=1e-10)


def test_quadratic_subpixel_peak_rejects_boundary_flat_and_nonfinite_neighborhoods():
    flat = np.ones((15, 15))
    assert quadratic_subpixel_peak(flat, 7, 7, 8) is None
    boundary = np.zeros((15, 15))
    boundary[7, 0] = 1.0
    assert quadratic_subpixel_peak(boundary, 2, 7, 8) is None
    invalid = np.zeros((15, 15))
    invalid[7, 7] = np.inf
    assert quadratic_subpixel_peak(invalid, 7, 7, 8) is None


def test_stage_iii_native_validity_rejects_stage_iv_patch_with_no_data(monkeypatch):
    import patchrift.features.stage_iv as stage_iv
    from patchrift.cascade.stage_iii import condition_image

    image = np.ones((96, 96), dtype=float)
    valid = np.ones((96, 96), dtype=bool)
    valid[48, 48] = False
    stage_iii = condition_image(
        image, 1.0, 1.0, {"R": np.zeros_like(valid), "S_int": np.zeros_like(valid)},
        0.0, 3.0, valid,
    )
    phase_calls = []
    monkeypatch.setattr(
        stage_iv,
        "calculate_phase_congruency",
        lambda *args, **kwargs: phase_calls.append(True),
    )
    features = stage_iv.describe_keypoints(
        [stage_iv.Keypoint(48.0, 48.0, 1.0)],
        stage_iii.conditioned_native,
        stage_iii.native_masks,
        0.0,
        stage_iii.scale_ratio,
        patch_size=32,
        num_scales=1,
        num_orientations=4,
        r1=3.0,
        r2=6.0,
        rmax=10.0,
    )
    assert not phase_calls
    assert features.descriptors.shape == (0, 68)


def test_stage_iv_rejects_canonical_patch_crossing_native_frame_boundary():
    from patchrift.features.stage_iv import (
        _canonical_descriptor_support_is_valid,
        compute_sampling_matrix,
    )

    valid = np.ones((100, 100), dtype=bool)
    matrix = compute_sampling_matrix(1.0, 0.0)
    assert not _canonical_descriptor_support_is_valid(valid, 5.0, 50.0, matrix, 16.0)
    assert _canonical_descriptor_support_is_valid(valid, 50.0, 50.0, matrix, 16.0)


def test_dense_keypoint_path_uses_full_image_formulation(monkeypatch):
    import patchrift.features.stage_iv as stage_iv

    marker = object()
    monkeypatch.setattr(stage_iv, "_describe_keypoints_full_image", lambda *args, **kwargs: marker)
    points = [stage_iv.Keypoint(float(x), float(y), 1.0) for x, y in
              [(12, 12), (16, 12), (20, 12), (12, 16), (16, 16), (20, 16),
               (12, 20), (16, 20), (20, 20)]]
    image = np.ones((32, 32), dtype=float)
    masks = {"valid": np.ones_like(image, dtype=bool)}

    result = stage_iv.describe_keypoints(
        points, image, masks, 0.0, 1.0, patch_size=8,
        r1=1.0, r2=2.0, rmax=3.0,
    )
    assert result is marker


def test_full_image_subpixel_recentering_does_not_double_add_first_offset(monkeypatch):
    import patchrift.features.stage_iv as stage_iv

    monkeypatch.setattr(stage_iv, "lanczos3_interpolate", lambda image, x, y: np.zeros_like(x, dtype=float))
    monkeypatch.setattr(stage_iv, "_canonical_descriptor_support_is_valid", lambda *args: True)
    monkeypatch.setattr(stage_iv, "create_log_gabor_bank", lambda *args, **kwargs: [[None]])
    monkeypatch.setattr(
        stage_iv, "calculate_phase_congruency",
        lambda image, bank, return_amplitudes=False: (
            [np.zeros_like(image)], np.ones((1, *image.shape))
        ),
    )
    monkeypatch.setattr(stage_iv, "calculate_moment_analysis", lambda pc: (None, np.zeros_like(pc[0])))
    monkeypatch.setattr(stage_iv, "compute_maximum_index_map", lambda pc: np.zeros_like(pc[0], dtype=np.uint8))
    refinements = iter([(2, 0, 0.0, 0.0), (1, 0, 0.1, 0.0)])
    monkeypatch.setattr(stage_iv, "quadratic_subpixel_peak", lambda *args: next(refinements))
    monkeypatch.setattr(stage_iv, "extract_canonical_mask", lambda *args: np.ones((8, 8), dtype=bool))
    monkeypatch.setattr(
        stage_iv, "extract_descriptor_and_weights",
        lambda *args, **kwargs: (np.zeros(68), np.ones(17)),
    )

    feature = stage_iv._describe_keypoints_full_image(
        [stage_iv.Keypoint(20.0, 20.0, 1.0)], np.ones((40, 40)),
        {"valid": np.ones((40, 40), dtype=bool), "R": np.ones((40, 40), dtype=bool),
         "S_int": np.ones((40, 40), dtype=bool)},
        np.eye(2), 1.0, 1.0, 1.0, patch_size=8, lambda_min=3.0,
        scale_multiplier=2.0, num_scales=1, num_orientations=1,
        cell_map=None, r1=1.0, r2=2.0, rmax=3.0, taper=0.25,
        v_min=0.2, downsample=8,
    )
    assert len(feature.keypoints) == 1
    assert feature.keypoints[0].x == pytest.approx(23.1)
    np.testing.assert_allclose(feature.canonical_positions[0], [23.1, 20.0])


def test_unweighted_ann_shortlist_uses_only_descriptor_vectors():
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = np.array([[10.1, 10.0], [0.1, 0.0]])
    np.testing.assert_array_equal(
        build_unweighted_ann_shortlist(a, b, top_m=1, ann_eps=0.0),
        [[0, 1], [1, 0]],
    )


def test_weighted_rescore_changes_scores_after_unweighted_retrieval():
    a = np.zeros(17 * 12)
    b = np.zeros_like(a)
    b[0] = 1.0
    b[12] = 1.0
    retrieved = build_unweighted_ann_shortlist(a[None, :], b[None, :], top_m=1, ann_eps=0.0)
    np.testing.assert_array_equal(retrieved, [[0, 0]])

    clean = compute_weighted_distance(a, b, np.ones(17), np.ones(17))
    suppress_first_cell = np.ones(17)
    suppress_first_cell[0] = 0.0
    rescored = compute_weighted_distance(a, b, suppress_first_cell, np.ones(17))
    assert clean > rescored


def test_stage_v_empty_and_invalid_descriptor_cases_fail_closed():
    empty = SimpleNamespace(
        descriptors=np.empty((0, 17 * 12)), reliability=np.empty((0, 17)),
        canonical_positions=np.empty((0, 2)),
    )
    result = match_feature_sets(empty, empty, delta_alpha=0.0, sigma_delta_alpha=0.0)
    assert not result.aligned
    assert len(result.pairs) == 0

    invalid = SimpleNamespace(
        descriptors=np.full((1, 17 * 12), np.nan), reliability=np.ones((1, 17)),
        canonical_positions=np.zeros((1, 2)),
    )
    try:
        match_feature_sets(invalid, invalid, delta_alpha=0.0, sigma_delta_alpha=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite descriptors must be rejected")


def test_adjacent_hop_composition_preserves_hops_and_propagates_end_to_end_error():
    from patchrift.pipeline.cascade import compose_cascade_correspondences

    first = SimpleNamespace(
        matching=SimpleNamespace(
            aligned=True, homography=np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=float),
            rms_residual_meters=3.0,
        )
    )
    second = SimpleNamespace(
        matching=SimpleNamespace(
            aligned=True, homography=np.array([[1, 0, 2], [0, 1, -1], [0, 0, 1]], dtype=float),
            rms_residual_meters=4.0,
        )
    )
    result = compose_cascade_correspondences(
        first, second, hop1_gsd_target=2.0, hop2_gsd_target=4.0,
        second_hop_extent_meters=100.0, first_hop_scale_error=0.1,
    )
    assert result.source_to_intermediate is first
    assert result.intermediate_to_target is second
    expected = second.matching.homography @ np.diag([0.5, 0.5, 1.0]) @ first.matching.homography
    expected /= expected[2, 2]
    np.testing.assert_allclose(result.composed_homography, expected)
    assert result.hop1_residual_meters == 3.0
    assert result.hop2_residual_meters == 4.0
    assert result.end_to_end_residual_meters == pytest.approx(np.sqrt(125.0))


def test_adjacent_hop_composition_fails_closed_if_either_hop_is_unaligned():
    from patchrift.pipeline.cascade import compose_cascade_correspondences

    hop = SimpleNamespace(matching=SimpleNamespace(
        aligned=False, homography=None, rms_residual_meters=None
    ))
    result = compose_cascade_correspondences(
        hop, hop, hop1_gsd_target=1.0, hop2_gsd_target=2.0,
        second_hop_extent_meters=100.0,
    )
    assert not result.aligned
    assert result.composed_homography is None
    assert result.end_to_end_residual_meters is None
