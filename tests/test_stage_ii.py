import numpy as np
import pytest
from unittest.mock import patch

from patchrift.geometry.diagnostics import implied_wall_slope
from patchrift.geometry.solar import SolarGeometry
from patchrift.pipeline.stage_ii import (
    ShadowSegmentation,
    StageIIConfig,
    _centroid_measurements,
    process_solar_tile,
)


def test_stage_ii_routes_unusable_elevation_to_c4_before_segmentation():
    result = process_solar_tile(
        np.ones((8, 8), dtype=np.float32),
        np.ones((8, 8), dtype=bool),
        SolarGeometry(azimuth=0.0, elevation=np.deg2rad(81.0)),
        StageIIConfig(q=0.05),
    )

    assert result.c4
    assert result.segmentation is None
    assert result.pixel_azimuth is None
    assert np.isinf(result.sigma_phi)


def test_stage_ii_routes_empty_shadow_and_boundary_to_c4():
    result = process_solar_tile(
        np.ones((8, 8), dtype=np.float32),
        np.ones((8, 8), dtype=bool),
        SolarGeometry(azimuth=0.0, elevation=np.deg2rad(20.0)),
        StageIIConfig(q=0.05, subtile_size=4),
    )

    assert result.c4
    assert result.segmentation is not None
    assert not np.any(result.segmentation.shadow_mask)
    assert not np.any(result.segmentation.boundary_ring)
    assert result.pixel_azimuth is None
    assert np.isinf(result.sigma_phi)


def test_stage_ii_uses_scharr_only_when_crater_ensemble_is_empty():
    segmentation = ShadowSegmentation(
        threshold_field=np.zeros((4, 4)),
        defined_mask=np.ones((4, 4), dtype=bool),
        shadow_mask=np.ones((4, 4), dtype=bool),
        boundary_ring=np.ones((4, 4), dtype=bool),
        shadow_interior=np.zeros((4, 4), dtype=bool),
        labels=np.zeros((4, 4), dtype=np.int32),
        num_components=0,
    )
    with (
        patch("patchrift.pipeline.stage_ii.segment_shadows", return_value=segmentation),
        patch(
            "patchrift.pipeline.stage_ii._scharr_samples",
            return_value=(np.array([1.0]), np.array([0.4])),
        ),
        patch(
            "patchrift.pipeline.stage_ii.evaluate_scharr_region",
            return_value=(True, True, 0.4, 1.0, 0.0),
        ),
        patch("patchrift.pipeline.stage_ii._centroid_measurements", return_value=[]),
    ):
        result = process_solar_tile(
            np.ones((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=bool),
            SolarGeometry(azimuth=0.2, elevation=np.deg2rad(20.0)),
            StageIIConfig(q=0.05),
        )

    assert not result.c4
    assert result.used_scharr_fallback
    np.testing.assert_allclose(result.pixel_azimuth, 0.4)
    np.testing.assert_allclose(result.sigma_phi, 0.0)
    np.testing.assert_allclose(result.north_angle, 0.2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"subtile_size": 0},
        {"subtile_size": 1.5},
        {"epsilon_var": -1.0},
        {"amin": -1},
        {"avet": 60.5},
        {"avet": -1},
        {"epsilon_ar": 0.0},
        {"emax": 1.1},
        {"solidity_min": -0.1},
        {"fseg": 0.0},
        {"sigma_sys": 0.0},
        {"eaxis": 1.1},
        {"sdelta_min": 0.0},
        {"wfloor": 0.0},
        {"smin_fallback": 0.0},
        {"q": float("nan")},
    ],
)
def test_stage_ii_config_rejects_invalid_parameters(overrides):
    with pytest.raises(ValueError):
        StageIIConfig(**{"q": 0.05, **overrides})


def test_segment_shadows_rejects_mismatched_or_nonfinite_valid_pixels():
    from patchrift.pipeline.stage_ii import segment_shadows

    config = StageIIConfig(q=0.05)
    with pytest.raises(ValueError, match="matching 2D arrays"):
        segment_shadows(np.ones((2, 2)), np.ones((2, 3), dtype=bool), config)

    image = np.ones((2, 2), dtype=np.float32)
    image[0, 0] = np.nan
    with pytest.raises(ValueError, match="valid pixels must be finite"):
        segment_shadows(image, np.ones((2, 2), dtype=bool), config)


def test_solar_geometry_rejects_nonfinite_angles():
    with pytest.raises(ValueError, match="must be finite"):
        SolarGeometry(azimuth=float("nan"), elevation=0.2)


def test_crater_slope_diagnostic_uses_tile_solar_elevation():
    labels = np.ones((5, 5), dtype=np.int32)
    segmentation = ShadowSegmentation(
        threshold_field=np.zeros((5, 5)),
        defined_mask=np.ones((5, 5), dtype=bool),
        shadow_mask=np.ones((5, 5), dtype=bool),
        boundary_ring=np.zeros((5, 5), dtype=bool),
        shadow_interior=np.zeros((5, 5), dtype=bool),
        labels=labels,
        num_components=1,
    )
    gate = np.array([False, True])
    elevation = np.deg2rad(20.0)
    delta = np.deg2rad(30.0)

    with (
        patch("patchrift.pipeline.stage_ii.vet_components_by_area", return_value=gate),
        patch("patchrift.pipeline.stage_ii.vet_components_by_border", return_value=gate),
        patch("patchrift.pipeline.stage_ii.vet_components_by_aspect_ratio", return_value=gate),
        patch("patchrift.pipeline.stage_ii.vet_components_by_eccentricity", return_value=gate),
        patch("patchrift.pipeline.stage_ii.vet_components_by_solidity", return_value=gate),
        patch(
            "patchrift.pipeline.stage_ii.compute_centroid_azimuth_measurement",
            return_value=(2.0, 2.0, 3.0, 2.0, 1.0, 0.4, 0.1, 0.2),
        ),
        patch(
            "patchrift.pipeline.stage_ii.component_shape_moments",
            return_value=(2.0, 1.0, 0.0),
        ),
        patch("patchrift.pipeline.stage_ii.component_eccentricity", return_value=0.5),
        patch("patchrift.pipeline.stage_ii.axis_angle", return_value=0.1),
        patch("patchrift.pipeline.stage_ii.infer_sector_half_angle", return_value=delta),
    ):
        records = _centroid_measurements(
            np.ones((5, 5)),
            np.ones((5, 5), dtype=bool),
            segmentation,
            StageIIConfig(q=0.05),
            elevation,
        )

    assert len(records) == 1
    np.testing.assert_allclose(
        records[0].implied_wall_slope,
        implied_wall_slope(elevation, delta),
    )
