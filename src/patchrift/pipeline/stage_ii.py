"""Stage II - solar geometry and shadow-derived pixel-frame azimuth (§5)."""

from dataclasses import dataclass

import numpy as np

from patchrift.geometry.solar import SolarGeometry, elevation_is_usable
from patchrift.geometry.diagnostics import (
    elevation_from_shadow_length,
    implied_wall_slope,
    robust_median_and_mad,
)
from patchrift.shadows.fusion import fuse_solar_azimuths
from patchrift.shadows.interpolation import (
    build_complete_interpolation_cells,
    populate_threshold_field,
)
from patchrift.shadows.mask import build_shadow_mask
from patchrift.shadows.morphology import (
    apply_opening_and_closing,
    build_shadow_regions,
    fill_shadow_holes,
    filter_shadow_components,
    label_shadow_components,
)
from patchrift.shadows.scharr import (
    evaluate_scharr_region,
    gradient_magnitude,
    gradient_orientation,
    scharr_azimuth_cross_check,
    scharr_gradients,
    select_scharr_samples,
)
from patchrift.shadows.thresholds import compute_subtile_thresholds
from patchrift.shadows.vetting import (
    axis_angle,
    combine_component_vetting_gates,
    component_eccentricity,
    component_shape_moments,
    compute_centroid_azimuth_measurement,
    compute_normalized_component_weights,
    infer_sector_half_angle,
    shape_ensemble_statistics,
    shadow_centroid,
    shadow_context_radius,
    shadow_dmax,
    vet_components_by_area,
    vet_components_by_aspect_ratio,
    vet_components_by_border,
    vet_components_by_eccentricity,
    vet_components_by_solidity,
)


@dataclass(frozen=True)
class StageIIConfig:
    """Fixed §5 parameters; ``q`` remains an explicit project choice."""

    q: float
    subtile_size: int = 1024
    epsilon_var: float = 1e-3
    amin: int = 25
    avet: int = 60
    epsilon_ar: float = 8.0
    emax: float = 0.97
    solidity_min: float = 0.80
    fseg: float = 3.0
    sigma_sys: float = np.deg2rad(0.5)
    eaxis: float = 0.40
    sdelta_min: float = np.deg2rad(5.0)
    wfloor: float = 1e-3
    smin_fallback: float = np.deg2rad(0.3)

    def __post_init__(self) -> None:
        if not 0.05 <= self.q <= 0.10:
            raise ValueError("q must lie in the specified interval [0.05, 0.10]")
        if not isinstance(self.subtile_size, (int, np.integer)):
            raise ValueError("subtile_size must be an integer")
        if self.subtile_size <= 0:
            raise ValueError("subtile_size must be positive")
        if self.epsilon_var < 0.0:
            raise ValueError("epsilon_var must be non-negative")
        if not isinstance(self.amin, (int, np.integer)) or not isinstance(
            self.avet, (int, np.integer)
        ):
            raise ValueError("amin and avet must be integers")
        if self.amin < 0 or self.avet < 0:
            raise ValueError("amin and avet must be non-negative")
        if self.epsilon_ar <= 0.0:
            raise ValueError("epsilon_ar must be positive")
        if not 0.0 <= self.emax <= 1.0:
            raise ValueError("emax must lie in [0, 1]")
        if not 0.0 <= self.solidity_min <= 1.0:
            raise ValueError("solidity_min must lie in [0, 1]")
        if self.fseg <= 0.0:
            raise ValueError("fseg must be positive")
        if self.sigma_sys <= 0.0:
            raise ValueError("sigma_sys must be positive")
        if not 0.0 <= self.eaxis <= 1.0:
            raise ValueError("eaxis must lie in [0, 1]")
        if self.sdelta_min <= 0.0 or self.wfloor <= 0.0:
            raise ValueError("sdelta_min and wfloor must be positive")
        if self.smin_fallback <= 0.0:
            raise ValueError("smin_fallback must be positive")
        numeric = (
            self.q, self.epsilon_var, self.epsilon_ar, self.emax,
            self.solidity_min, self.fseg, self.sigma_sys, self.eaxis,
            self.sdelta_min, self.wfloor, self.smin_fallback,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("all floating-point configuration values must be finite")


@dataclass(frozen=True)
class ShadowSegmentation:
    """Mandatory §5.2 mask products and their threshold provenance."""

    threshold_field: np.ndarray
    defined_mask: np.ndarray
    shadow_mask: np.ndarray
    boundary_ring: np.ndarray
    shadow_interior: np.ndarray
    labels: np.ndarray
    num_components: int


@dataclass(frozen=True)
class CraterAzimuthMeasurement:
    component_id: int
    azimuth: float
    sigma_theta: float
    eccentricity: float
    omega: float
    delta_hat: float
    implied_wall_slope: float


@dataclass(frozen=True)
class SolarTileResult:
    """Complete Stage II output for one processing tile."""

    segmentation: ShadowSegmentation | None
    solar_geometry: SolarGeometry
    pixel_azimuth: float | None
    sigma_phi: float
    north_angle: float | None
    measurements: tuple[CraterAzimuthMeasurement, ...]
    used_scharr_fallback: bool
    low_confidence: bool
    c4: bool
    scharr_disagreement_flag: bool
    implied_slope_diagnostic: tuple[float, float] | None
    d_azimuth_d_latitude: float | None = None
    d_azimuth_d_longitude: float | None = None
    sigma_geo: float | None = None
    alpha_corners: float | None = None
    shadow_footprint_disagreement: float | None = None
    shadow_length_elevation_diagnostic: tuple[float, float] | None = None


@dataclass(frozen=True)
class PairSolarGeometry:
    """§5.8 pair-level rotation estimate and uncertainty."""

    alpha_a: float
    alpha_b: float
    delta_alpha: float
    sigma_delta_alpha: float


def pair_solar_geometry(tile_a: SolarTileResult, tile_b: SolarTileResult) -> PairSolarGeometry:
    """Combine the two per-image Stage II outputs according to Eq. (43)."""
    from patchrift.geometry.rotation import inter_image_rotation

    alpha_a = tile_a.north_angle if tile_a.north_angle is not None else tile_a.alpha_corners
    alpha_b = tile_b.north_angle if tile_b.north_angle is not None else tile_b.alpha_corners
    if alpha_a is None or alpha_b is None:
        raise ValueError("pair-level rotation requires measured or footprint-derived north angles for both images")
    alpha_a, alpha_b = float(alpha_a), float(alpha_b)
    sigma_a = tile_a.sigma_phi if tile_a.pixel_azimuth is not None else float("inf")
    sigma_b = tile_b.sigma_phi if tile_b.pixel_azimuth is not None else float("inf")
    sigma_geo_a = float("inf") if tile_a.sigma_geo is None else tile_a.sigma_geo
    sigma_geo_b = float("inf") if tile_b.sigma_geo is None else tile_b.sigma_geo
    sigma_delta = float(np.sqrt(sigma_a**2 + sigma_b**2 + sigma_geo_a**2 + sigma_geo_b**2))
    return PairSolarGeometry(
        alpha_a=alpha_a,
        alpha_b=alpha_b,
        delta_alpha=inter_image_rotation(alpha_a, alpha_b),
        sigma_delta_alpha=sigma_delta,
    )


def segment_shadows(
    prepared_image: np.ndarray,
    valid_mask: np.ndarray,
    config: StageIIConfig,
) -> ShadowSegmentation:
    """Execute §5.2 including every mandatory addendum rule."""
    prepared_image = np.asarray(prepared_image, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if prepared_image.ndim != 2 or valid_mask.shape != prepared_image.shape:
        raise ValueError("prepared_image and valid_mask must be matching 2D arrays")
    if np.any(valid_mask & ~np.isfinite(prepared_image)):
        raise ValueError("valid pixels must be finite")

    thresholds = compute_subtile_thresholds(
        prepared_image,
        valid_mask,
        q=config.q,
        tile_size=config.subtile_size,
        epsilon_var=config.epsilon_var,
    )
    cells = build_complete_interpolation_cells(thresholds)
    threshold_field, defined_mask = populate_threshold_field(
        prepared_image.shape,
        cells,
    )
    raw_mask = build_shadow_mask(
        prepared_image,
        threshold_field,
        defined_mask,
        valid_mask,
    )
    cleaned = fill_shadow_holes(apply_opening_and_closing(raw_mask))
    labels, count = label_shadow_components(cleaned)
    filtered = filter_shadow_components(
        cleaned,
        labels,
        count,
        amin=config.amin,
        epsilon_ar=config.epsilon_ar,
    )
    labels, count = label_shadow_components(filtered)
    shadow, ring, interior = build_shadow_regions(filtered)
    return ShadowSegmentation(
        threshold_field=threshold_field,
        defined_mask=defined_mask,
        shadow_mask=shadow,
        boundary_ring=ring,
        shadow_interior=interior,
        labels=labels,
        num_components=count,
    )


def process_solar_tile(
    prepared_image: np.ndarray,
    valid_mask: np.ndarray,
    solar_geometry: SolarGeometry,
    config: StageIIConfig,
    saturated_mask: np.ndarray | None = None,
    rim_radius_provider=None,
) -> SolarTileResult:
    """Execute §5 for one prepared image tile.

    The caller supplies ephemeris geometry at this tile's geographic
    centre.  An unusable solar elevation is immediately routed to C4.
    """
    prepared_image = np.asarray(prepared_image, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if prepared_image.ndim != 2 or valid_mask.shape != prepared_image.shape:
        raise ValueError("prepared_image and valid_mask must be matching 2D arrays")
    if np.any(valid_mask & ~np.isfinite(prepared_image)):
        raise ValueError("valid pixels must be finite")
    if saturated_mask is None:
        saturated_mask = np.zeros_like(valid_mask)
    saturated_mask = np.asarray(saturated_mask, dtype=bool)
    if saturated_mask.shape != prepared_image.shape:
        raise ValueError("saturated_mask must match prepared_image")

    if not elevation_is_usable(solar_geometry.elevation):
        return SolarTileResult(
            segmentation=None,
            solar_geometry=solar_geometry,
            pixel_azimuth=None,
            sigma_phi=float("inf"),
            north_angle=None,
            measurements=(),
            used_scharr_fallback=False,
            low_confidence=True,
            c4=True,
            scharr_disagreement_flag=False,
            implied_slope_diagnostic=None,
        )

    segmentation = segment_shadows(prepared_image, valid_mask, config)
    selected_magnitude, selected_orientation = _scharr_samples(
        prepared_image,
        segmentation.boundary_ring,
        valid_mask,
        saturated_mask,
    )
    ring_nonempty, scharr_available, scharr_phi, _, scharr_sigma = evaluate_scharr_region(
        segmentation.boundary_ring,
        selected_magnitude,
        selected_orientation,
    )
    measurements = _centroid_measurements(
        prepared_image,
        valid_mask,
        segmentation,
        config,
        solar_geometry.elevation,
    )

    if not measurements:
        if not ring_nonempty or not scharr_available:
            return SolarTileResult(
                segmentation=segmentation,
                solar_geometry=solar_geometry,
                pixel_azimuth=None,
                sigma_phi=float("inf"),
                north_angle=None,
                measurements=(),
                used_scharr_fallback=False,
                low_confidence=True,
                c4=True,
                scharr_disagreement_flag=False,
                implied_slope_diagnostic=None,
            )
        north_angle = _north_angle(scharr_phi, solar_geometry.azimuth)
        return SolarTileResult(
            segmentation=segmentation,
            solar_geometry=solar_geometry,
            pixel_azimuth=scharr_phi,
            sigma_phi=scharr_sigma,
            north_angle=north_angle,
            measurements=(),
            used_scharr_fallback=True,
            low_confidence=False,
            c4=False,
            scharr_disagreement_flag=False,
            implied_slope_diagnostic=None,
        )

    angles = np.array([record.azimuth for record in measurements])
    sigma_theta = np.array([record.sigma_theta for record in measurements])
    eccentricity = np.array([record.eccentricity for record in measurements])
    omega = np.array([record.omega for record in measurements])
    delta = np.array([record.delta_hat for record in measurements])
    delta_med, s_delta = shape_ensemble_statistics(delta, config.sdelta_min)
    base_weights = compute_normalized_component_weights(
        sigma_theta, eccentricity, omega, delta, delta_med, s_delta,
        sigma0=config.sigma_sys, eaxis=config.eaxis, w_floor=config.wfloor,
    )
    # §5.6 uses the measured median positional-precision angle whenever
    # centroid samples exist. The configured fallback is reserved for a
    # path where those per-sample uncertainties are unavailable.
    s_min = float(np.median(sigma_theta))
    phi, _, _, _, sigma_phi, _, low_confidence = fuse_solar_azimuths(
        angles, base_weights, s_min=s_min, sigma_sys=config.sigma_sys,
    )
    disagreement = False
    if scharr_available:
        _, disagreement = scharr_azimuth_cross_check(scharr_phi, phi)
    slope_summary = robust_median_and_mad(
        np.array([record.implied_wall_slope for record in measurements])
    )
    shadow_elevation_diagnostic = None
    if rim_radius_provider is not None:
        estimates = []
        for record in measurements:
            rows, cols = np.nonzero(segmentation.labels == record.component_id)
            along_sun = cols * np.cos(record.azimuth) + rows * np.sin(record.azimuth)
            shadow_length = float(np.max(along_sun) - np.min(along_sun))
            rim_radius = float(rim_radius_provider(
                prepared_image,
                segmentation.labels == record.component_id,
                record.component_id,
            ))
            try:
                estimates.append(elevation_from_shadow_length(
                    rim_radius,
                    record.implied_wall_slope,
                    shadow_length,
                ))
            except ValueError:
                continue
        if estimates:
            elevation_summary = robust_median_and_mad(np.asarray(estimates))
            shadow_elevation_diagnostic = (
                elevation_summary.median,
                elevation_summary.mad_scale,
            )
    return SolarTileResult(
        segmentation=segmentation,
        solar_geometry=solar_geometry,
        pixel_azimuth=phi,
        sigma_phi=sigma_phi,
        north_angle=_north_angle(phi, solar_geometry.azimuth),
        measurements=tuple(measurements),
        used_scharr_fallback=False,
        low_confidence=low_confidence,
        c4=False,
        scharr_disagreement_flag=disagreement,
        implied_slope_diagnostic=(slope_summary.median, slope_summary.mad_scale),
        shadow_length_elevation_diagnostic=shadow_elevation_diagnostic,
    )


def _scharr_samples(image, ring, valid_mask, saturated_mask):
    gradients = scharr_gradients(image)
    magnitude = gradient_magnitude(*gradients)
    orientation = gradient_orientation(*gradients)
    return select_scharr_samples(magnitude, orientation, ring, valid_mask, saturated_mask)


def _centroid_measurements(
    image,
    valid_mask,
    segmentation,
    config,
    solar_elevation,
):
    labels = segmentation.labels
    count = segmentation.num_components
    if count == 0:
        return []
    gates = (
        vet_components_by_area(labels, count, config.avet),
        vet_components_by_border(labels, count),
        vet_components_by_aspect_ratio(labels, count, config.epsilon_ar),
        vet_components_by_eccentricity(labels, count, config.emax),
        vet_components_by_solidity(labels, count, config.solidity_min),
    )
    centroids = np.array([shadow_centroid(labels, i) for i in range(1, count + 1)])
    radii = np.array([shadow_context_radius(shadow_dmax(labels, i)) for i in range(1, count + 1)])
    keep = combine_component_vetting_gates(
        *(gate[1:] for gate in gates), centroids, radii,
    )
    records = []
    for component_id in np.flatnonzero(keep) + 1:
        try:
            _, _, _, _, _, azimuth, _, sigma_theta = compute_centroid_azimuth_measurement(
                image, labels, int(component_id), segmentation.shadow_mask,
                valid_mask=valid_mask, fseg=config.fseg,
            )
            lam_max, lam_min, psi_maj = component_shape_moments(labels, int(component_id))
            eccentricity = component_eccentricity(labels, int(component_id))
            omega = axis_angle(psi_maj, azimuth)
            delta_hat = infer_sector_half_angle(eccentricity, omega)
        except ValueError:
            continue
        records.append(CraterAzimuthMeasurement(
            component_id=int(component_id), azimuth=azimuth, sigma_theta=sigma_theta,
            eccentricity=eccentricity, omega=omega, delta_hat=delta_hat,
            implied_wall_slope=implied_wall_slope(solar_elevation, delta_hat),
        ))
    return records


def _north_angle(pixel_azimuth: float, solar_azimuth: float) -> float:
    from patchrift.geometry.rotation import pixel_north_angle
    return pixel_north_angle(pixel_azimuth, solar_azimuth)
