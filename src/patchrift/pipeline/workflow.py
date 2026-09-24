"""Image-level Stage I/II orchestration and Stage III handoff products."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from patchrift.geometry.diagnostics import corner_north_angle
from patchrift.geometry.handedness import fit_geolocation_jacobian
from patchrift.geometry.rotation import wrap_angle_2pi
from patchrift.geometry.solar import SolarGeometry
from patchrift.geometry.spice import (
    default_azimuth_derivative_steps,
    solar_azimuth_derivatives,
    solar_geometry_at_location,
)
from patchrift.geometry.tiling import (
    PixelTile,
    bisect_tiles_for_solar_azimuth,
    footprint_geolocation_model,
    footprint_tile_corners,
)
from patchrift.pipeline.stage_i import StageIResult
from patchrift.pipeline.stage_ii import SolarTileResult, StageIIConfig, process_solar_tile


@dataclass(frozen=True)
class StageIIImageResult:
    """Adaptive tile plan and solar-geometry outputs for one Stage I image."""

    stage_i: StageIResult
    tiles: tuple[PixelTile, ...]
    tile_results: tuple[SolarTileResult, ...]

    @property
    def planarity_flag(self) -> bool:
        """True when Stage I metadata warns that planar geometry is limited."""
        return self.stage_i.planarity_flag


def process_stage_ii_image(
    stage_i: StageIResult,
    config: StageIIConfig,
    *,
    max_pixels_per_tile: int,
    pixel_to_location: Callable[[float, float], tuple[float, float]] | None = None,
    solar_geometry_at: Callable[[str, float, float], SolarGeometry] | None = None,
    solar_azimuth_at: Callable[[float, float], float] | None = None,
    solar_derivatives_at: Callable[..., tuple[float, float]] | None = None,
    derivative_steps: tuple[float, float] | None = None,
    rim_radius_provider: Callable | None = None,
) -> StageIIImageResult:
    """Plan tiles adaptively, then run Stage II for each tile.

    Geographic callbacks use radians and [latitude, longitude] ordering. If
    omitted, the §2.3 affine model is fitted from the ingested footprint
    corners. SPICE callbacks require kernels to have been loaded by the caller.
    """
    product = stage_i.product
    metadata = product.metadata
    if max_pixels_per_tile <= 0:
        raise ValueError("max_pixels_per_tile must be positive")
    uncertainty_pair = (metadata.latitude_uncertainty, metadata.longitude_uncertainty)
    if (uncertainty_pair[0] is None) != (uncertainty_pair[1] is None):
        raise ValueError("latitude and longitude uncertainty must either both be supplied or both be omitted")
    if pixel_to_location is None:
        pixel_to_location = footprint_geolocation_model(metadata)
    if solar_geometry_at is None:
        solar_geometry_at = solar_geometry_at_location
    if solar_derivatives_at is None:
        solar_derivatives_at = solar_azimuth_derivatives
    if solar_azimuth_at is None:
        solar_azimuth_at = lambda lat, lon: solar_geometry_at(
            _spice_timestamp(metadata.timestamp), lat, lon
        ).azimuth

    def tile_corners(tile: PixelTile) -> np.ndarray:
        return footprint_tile_corners(tile, pixel_to_location)

    tiles = bisect_tiles_for_solar_azimuth(
        stage_i.prepared_image.shape,
        tile_corners,
        solar_azimuth_at,
        sigma_sys=config.sigma_sys,
        max_pixels_per_tile=max_pixels_per_tile,
    )

    jacobian = fit_geolocation_jacobian(
        metadata.footprint_corners,
        metadata.footprint_pixel_coords,
    )
    alpha_corners = wrap_angle_2pi(corner_north_angle(jacobian))
    results: list[SolarTileResult] = []
    for tile in tiles:
        corners = tile_corners(tile)
        center = np.mean(corners, axis=0)
        latitude, longitude = float(center[0]), float(center[1])
        timestamp = _spice_timestamp(metadata.timestamp)
        solar = solar_geometry_at(timestamp, latitude, longitude)

        latitude_half_extent = float(np.max(np.abs(corners[:, 0] - latitude)))
        longitude_half_extent = float(np.max(np.abs(corners[:, 1] - longitude)))
        steps = derivative_steps or default_azimuth_derivative_steps(
            latitude_half_extent,
            longitude_half_extent,
            latitude,
        )
        if len(steps) != 2 or not np.all(np.isfinite(steps)) or min(steps) <= 0.0:
            raise ValueError("derivative_steps must contain two finite positive angular steps")
        d_a_d_lat, d_a_d_lon = solar_derivatives_at(
            timestamp,
            latitude,
            longitude,
            latitude_step=steps[0],
            longitude_step=steps[1],
        )
        sigma_geo = None
        if (
            metadata.latitude_uncertainty is not None
            and metadata.longitude_uncertainty is not None
        ):
            sigma_geo = float(np.hypot(
                d_a_d_lat * metadata.latitude_uncertainty,
                d_a_d_lon * metadata.longitude_uncertainty,
            ))

        row_slice = slice(tile.row_start, tile.row_end)
        col_slice = slice(tile.col_start, tile.col_end)
        result = process_solar_tile(
            stage_i.prepared_image[row_slice, col_slice],
            product.valid_mask[row_slice, col_slice],
            solar,
            config,
            rim_radius_provider=rim_radius_provider,
        )
        discrepancy = None
        if result.north_angle is not None:
            discrepancy = float(
                (result.north_angle - alpha_corners + np.pi) % (2.0 * np.pi) - np.pi
            )
        results.append(replace(
            result,
            d_azimuth_d_latitude=d_a_d_lat,
            d_azimuth_d_longitude=d_a_d_lon,
            sigma_geo=sigma_geo,
            alpha_corners=alpha_corners,
            shadow_footprint_disagreement=discrepancy,
        ))
    return StageIIImageResult(stage_i, tuple(tiles), tuple(results))


def _spice_timestamp(timestamp) -> str:
    """Format metadata timestamps in a SPICE-readable UTC form."""
    if timestamp.tzinfo is not None:
        from datetime import timezone

        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.strftime("%Y %b %d %H:%M:%S UTC")
