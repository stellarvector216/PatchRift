"""Executable tile-level bridge from Stage II geometry through Stage V matches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from patchrift.cascade.stage_iii import StageIIIResult, condition_image
from patchrift.features.stage_iv import FeatureSet, describe_keypoints, detect_keypoints
from patchrift.matching.stage_v import StageVResult, calculate_hough_bin_size, match_feature_sets
from patchrift.pipeline.stage_i import StageIResult
from patchrift.pipeline.stage_ii import SolarTileResult
from patchrift.geometry.tiling import PixelTile


@dataclass(frozen=True)
class TileFeatureResult:
    conditioning: StageIIIResult
    features: FeatureSet


@dataclass(frozen=True)
class PairCorrespondenceResult:
    features_a: TileFeatureResult
    features_b: TileFeatureResult
    matching: StageVResult


def extract_tile_features(
    stage_i: StageIResult,
    tile: PixelTile,
    solar: SolarTileResult,
    *,
    gsd_current: float,
    gsd_target: float,
    lambda_max: float,
    alpha: float | None = None,
    detector_options: dict | None = None,
    descriptor_options: dict | None = None,
) -> TileFeatureResult:
    """Run Stage III and IV on the exact adaptive tile produced by Stage II."""
    rows = slice(tile.row_start, tile.row_end)
    cols = slice(tile.col_start, tile.col_end)
    image = stage_i.prepared_image[rows, cols]
    valid = stage_i.product.valid_mask[rows, cols]
    segmentation = solar.segmentation
    if segmentation is None:
        shape = image.shape
        masks = {"R": np.zeros(shape, dtype=bool), "S_int": np.zeros(shape, dtype=bool)}
    else:
        masks = {"R": segmentation.boundary_ring, "S_int": segmentation.shadow_interior}
        if segmentation.shadow_mask.shape != image.shape:
            raise ValueError("Stage II segmentation dimensions do not match its adaptive tile")
    # An explicit alpha is used by §8.4 C4 to rebuild canonical patches in
    # the unrotated frame. Otherwise prefer the Stage II measurement, then
    # the footprint-derived angle.
    angle = alpha if alpha is not None else solar.north_angle
    if angle is None:
        angle = solar.alpha_corners
    if angle is None:
        raise ValueError("Stage III/IV require a measured or footprint-derived north angle")

    conditioned = condition_image(
        image, gsd_current, gsd_target, masks,
        solar.pixel_azimuth if solar.pixel_azimuth is not None else solar.solar_geometry.azimuth,
        lambda_max, valid,
    )
    options = dict(detector_options or {})
    keypoints = detect_keypoints(
        conditioned.detection_image,
        conditioned.valid_mask,
        conditioned.masks.get("S_int", np.zeros_like(conditioned.valid_mask)),
        **options,
    )
    descriptor_options = dict(descriptor_options or {})
    descriptor_downsample = descriptor_options.pop("downsample", options.get("downsample", 8))
    descriptor_options.pop("native_coordinate_scale", None)
    features = describe_keypoints(
        keypoints, conditioned.conditioned_native, conditioned.native_masks,
        float(angle), conditioned.scale_ratio, **descriptor_options,
        downsample=descriptor_downsample,
        native_coordinate_scale=conditioned.native_coordinate_scale,
    )
    return TileFeatureResult(conditioned, features)


def match_tile_pair(
    stage_i_a: StageIResult,
    tile_a: PixelTile,
    solar_a: SolarTileResult,
    stage_i_b: StageIResult,
    tile_b: PixelTile,
    solar_b: SolarTileResult,
    *,
    gsd_a: float,
    gsd_b: float,
    gsd_target: float,
    lambda_max_a: float,
    lambda_max_b: float,
    alpha_a: float | None = None,
    alpha_b: float | None = None,
    detector_options: dict | None = None,
    descriptor_options: dict | None = None,
    matching_options: dict | None = None,
) -> PairCorrespondenceResult:
    """Execute the tile-level Stage III→IV→V pair workflow."""
    def extract_pair_features(a_alpha, b_alpha):
        return (
            extract_tile_features(
                stage_i_a, tile_a, solar_a, gsd_current=gsd_a, gsd_target=gsd_target,
                lambda_max=lambda_max_a, alpha=a_alpha, detector_options=detector_options,
                descriptor_options=descriptor_options,
            ),
            extract_tile_features(
                stage_i_b, tile_b, solar_b, gsd_current=gsd_b, gsd_target=gsd_target,
                lambda_max=lambda_max_b, alpha=b_alpha, detector_options=detector_options,
                descriptor_options=descriptor_options,
            ),
        )

    fa, fb = extract_pair_features(alpha_a, alpha_b)
    from patchrift.pipeline.stage_ii import pair_solar_geometry

    pair_geometry = pair_solar_geometry(solar_a, solar_b)
    matching_config = dict(matching_options or {})
    metadata_a = getattr(getattr(stage_i_a, "product", None), "metadata", None)
    metadata_b = getattr(getattr(stage_i_b, "product", None), "metadata", None)
    hmax_values = [
        getattr(metadata, "hmax", None)
        for metadata in (metadata_a, metadata_b)
        if getattr(metadata, "hmax", None) is not None
    ]
    hmax = max(hmax_values) if hmax_values else 300.0
    emission_a = getattr(metadata_a, "emission_angle", 0.0)
    emission_b = getattr(metadata_b, "emission_angle", 0.0)
    matching_config.setdefault(
        "hough_bin",
        calculate_hough_bin_size(
            hmax, emission_a, emission_b, gsd_target
        ),
    )
    matching = match_feature_sets(
        fa.features, fb.features,
        delta_alpha=pair_geometry.delta_alpha,
        sigma_delta_alpha=pair_geometry.sigma_delta_alpha,
        **matching_config,
    )
    def needs_unrotated_retry(explicit, solar):
        angle = explicit
        if angle is None:
            angle = solar.north_angle if solar.north_angle is not None else solar.alpha_corners
        return angle is not None and not np.isclose(float(angle) % (2.0 * np.pi), 0.0)

    if matching.fallback_used and (
        needs_unrotated_retry(alpha_a, solar_a) or needs_unrotated_retry(alpha_b, solar_b)
    ):
        # §8.4 C4 redoes conditioning from the original Stage-I/II products;
        # changing only the matcher would leave the incorrectly rotated
        # canonical patches in place.
        fa, fb = extract_pair_features(0.0, 0.0)
        matching = match_feature_sets(
            fa.features,
            fb.features,
            delta_alpha=0.0,
            sigma_delta_alpha=float("inf"),
            **matching_config,
        )
    return PairCorrespondenceResult(fa, fb, matching)
