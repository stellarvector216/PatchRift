from datetime import datetime
import numpy as np
import pytest

from patchrift.io import IIRSProduct, ImageProduct
from patchrift.io.product import ProductMetadata
from patchrift.pipeline.stage_i import prepare_iirs_product, prepare_image_product
from patchrift.pipeline.stage_ii import SolarTileResult, StageIIConfig
from patchrift.pipeline.workflow import process_stage_ii_image
from patchrift.geometry.solar import SolarGeometry


def _metadata(emission_angle=0.0):
    return ProductMetadata(
        timestamp=datetime(2025, 1, 1),
        footprint_corners=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        footprint_pixel_coords=np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=float),
        gsd=10.0,
        emission_angle=emission_angle,
        latitude_uncertainty=1e-5,
        longitude_uncertainty=2e-5,
    )


def test_planarity_flag_exposes_oblique_geometry():
    product = ImageProduct(np.ones((8, 8)), np.ones((8, 8), bool), _metadata(np.deg2rad(15.01)))
    assert prepare_image_product(product).planarity_flag is True
    product.metadata.emission_angle = np.deg2rad(15.0)
    assert prepare_image_product(product).planarity_flag is False


def test_iirs_band_overlap_is_clipped_and_snr_fallback_is_retained():
    cube = np.stack([np.full((2, 2), value) for value in (1.0, 3.0, 9.0)])
    product = IIRSProduct(
        cube=cube,
        wavelengths=np.array([0.9, 1.1, 1.3]),
        bad_bands=np.zeros(3, dtype=bool),
        snr=np.array([1.0, 3.0, 1.0]),
        band_bounds=np.array([[0.8, 1.0], [1.0, 1.2], [1.2, 1.4]]),
    )
    meta = _metadata()
    lookup = (np.array([0.9, 1.1]), np.array([1.0, 1.0]))
    result = prepare_iirs_product(product, fill_value=None, metadata=meta, spectral_response=lookup)
    np.testing.assert_allclose(result.prepared_image, 2.0)

    fallback = prepare_iirs_product(product, fill_value=None, metadata=meta)
    np.testing.assert_allclose(fallback.prepared_image, 3.8)


def test_iirs_callable_is_integrated_only_inside_declared_domain():
    from patchrift.preprocessing.iirs import tmc_spectral_overlap_weights

    product = IIRSProduct(
        cube=np.ones((3, 1, 1)),
        wavelengths=np.array([0.9, 1.1, 1.3]),
        bad_bands=np.zeros(3, dtype=bool),
        snr=np.ones(3),
        band_bounds=np.array([[0.8, 1.0], [1.0, 1.2], [1.2, 1.4]]),
    )
    weights = tmc_spectral_overlap_weights(product, lambda wavelength: 1.0, (0.95, 1.05))
    np.testing.assert_allclose(weights, [0.05, 0.05, 0.0], atol=1e-8)


def test_stage_ii_pipeline_plans_memory_bounded_tiles_and_propagates_geometry(monkeypatch):
    import patchrift.pipeline.workflow as workflow

    product = ImageProduct(np.ones((8, 8)), np.ones((8, 8), bool), _metadata())
    stage_i = prepare_image_product(product)
    seen = []
    derivative_calls = []

    def fake_process(image, valid, solar, config, **kwargs):
        seen.append(image.shape)
        return SolarTileResult(
            segmentation=None, solar_geometry=solar, pixel_azimuth=0.2,
            sigma_phi=0.01, north_angle=0.3, measurements=(),
            used_scharr_fallback=False, low_confidence=False, c4=False,
            scharr_disagreement_flag=False, implied_slope_diagnostic=None,
        )

    def fake_derivatives(*args, **kwargs):
        derivative_calls.append((kwargs["latitude_step"], kwargs["longitude_step"]))
        return 2.0, 3.0

    monkeypatch.setattr(workflow, "process_solar_tile", fake_process)
    result = process_stage_ii_image(
        stage_i,
        StageIIConfig(q=0.05, sigma_sys=0.3),
        max_pixels_per_tile=16,
        pixel_to_location=lambda row, col: (row * 1e-4, col * 1e-4),
        solar_geometry_at=lambda timestamp, lat, lon: SolarGeometry(0.1, 0.5),
        solar_azimuth_at=lambda lat, lon: lon,
        solar_derivatives_at=fake_derivatives,
    )

    assert result.tiles
    assert all(rows * cols <= 16 for rows, cols in seen)
    assert len(result.tile_results) == len(result.tiles)
    assert len(derivative_calls) == len(result.tiles)
    assert all(lat_step > 0 and lon_step > 0 for lat_step, lon_step in derivative_calls)
    assert all(tile.d_azimuth_d_latitude == 2.0 for tile in result.tile_results)
    assert all(tile.d_azimuth_d_longitude == 3.0 for tile in result.tile_results)
    assert all(tile.sigma_geo is not None and tile.sigma_geo > 0 for tile in result.tile_results)
    assert all(tile.alpha_corners is not None for tile in result.tile_results)
    assert all(tile.shadow_footprint_disagreement is not None for tile in result.tile_results)


def test_pair_solar_geometry_requires_angles_and_propagates_variance():
    from patchrift.pipeline.stage_ii import pair_solar_geometry

    def tile(alpha, sigma_geo):
        return SolarTileResult(
            segmentation=None, solar_geometry=SolarGeometry(0.0, 0.5),
            pixel_azimuth=0.2, sigma_phi=0.03, north_angle=alpha,
            measurements=(), used_scharr_fallback=False, low_confidence=False,
            c4=False, scharr_disagreement_flag=False, implied_slope_diagnostic=None,
            sigma_geo=sigma_geo,
        )

    pair = pair_solar_geometry(tile(0.2, 0.04), tile(0.5, 0.05))
    np.testing.assert_allclose(pair.alpha_a, 0.2)
    np.testing.assert_allclose(pair.alpha_b, 0.5)
    np.testing.assert_allclose(pair.delta_alpha, 0.3)
    np.testing.assert_allclose(pair.sigma_delta_alpha, np.sqrt(2 * 0.03**2 + 0.04**2 + 0.05**2))


def test_a5_tile_default_steps_are_used_when_not_overridden():
    from patchrift.geometry.spice import default_azimuth_derivative_steps

    latitude_step, longitude_step = default_azimuth_derivative_steps(
        np.deg2rad(0.2), np.deg2rad(0.4), np.deg2rad(60.0)
    )
    np.testing.assert_allclose(latitude_step, np.deg2rad(0.02))
    np.testing.assert_allclose(longitude_step, np.deg2rad(0.08))


def test_stage_ii_shadow_length_elevation_uses_radius_provider(monkeypatch):
    import patchrift.pipeline.stage_ii as stage_ii
    from patchrift.pipeline.stage_ii import CraterAzimuthMeasurement, ShadowSegmentation

    segmentation = ShadowSegmentation(
        threshold_field=np.zeros((5, 5)), defined_mask=np.ones((5, 5), bool),
        shadow_mask=np.ones((5, 5), bool), boundary_ring=np.ones((5, 5), bool),
        shadow_interior=np.zeros((5, 5), bool), labels=np.ones((5, 5), np.int32),
        num_components=1,
    )
    record = CraterAzimuthMeasurement(1, 0.4, 0.01, 0.5, 1.0, 0.2, 0.2)
    monkeypatch.setattr(stage_ii, "segment_shadows", lambda *args: segmentation)
    monkeypatch.setattr(stage_ii, "_scharr_samples", lambda *args: (np.array([1.0]), np.array([0.4])))
    monkeypatch.setattr(stage_ii, "evaluate_scharr_region", lambda *args: (True, True, 0.4, 1.0, 0.01))
    monkeypatch.setattr(stage_ii, "_centroid_measurements", lambda *args: [record])

    result = stage_ii.process_solar_tile(
        np.ones((5, 5)), np.ones((5, 5), bool), SolarGeometry(0.2, 0.5),
        StageIIConfig(q=0.05), rim_radius_provider=lambda *args: 3.0,
    )
    assert result.shadow_length_elevation_diagnostic is not None
    median, mad = result.shadow_length_elevation_diagnostic
    assert np.isfinite(median)
    assert mad == 0.0
