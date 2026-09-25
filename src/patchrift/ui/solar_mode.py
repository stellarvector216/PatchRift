"""UI-level choice of SPICE or scene-wide metadata solar geometry."""

from __future__ import annotations

import numpy as np

from patchrift.geometry.solar import SolarGeometry

SPICE_MODE = "SPICE ephemeris (default)"
METADATA_MODE = "Use metadata geometry · skip SPICE"
COMPARE_MODE = "Metadata plus SPICE comparison"


def uses_spice(mode: str) -> bool:
    """Whether the selected UI mode requires SPICE geometry evaluation."""
    if mode not in {SPICE_MODE, METADATA_MODE, COMPARE_MODE}:
        raise ValueError("unknown solar geometry mode")
    return mode != METADATA_MODE


def metadata_geometry_kwargs(mode: str, azimuth: float | None, elevation: float | None) -> dict:
    """Return Stage-II callbacks for metadata-only mode, otherwise no overrides.

    The metadata angles are scene-wide, so this callback intentionally returns
    constant geometry at every tile. Its derivative is unavailable, not zero.
    """
    if mode not in {SPICE_MODE, METADATA_MODE, COMPARE_MODE}:
        raise ValueError("unknown solar geometry mode")
    if mode != METADATA_MODE:
        return {}
    if azimuth is None or elevation is None:
        raise ValueError("metadata-only mode requires solar azimuth and elevation")
    geometry = SolarGeometry(float(azimuth), float(elevation))
    return {
        "solar_geometry_at": lambda timestamp, latitude, longitude: geometry,
        "solar_azimuth_at": lambda latitude, longitude: geometry.azimuth,
        "solar_derivatives_at": lambda *args, **kwargs: None,
    }


def degrees_to_geometry(azimuth_degrees: float, elevation_degrees: float) -> SolarGeometry:
    """Validate user-entered scene-wide angles and convert degrees to radians."""
    return SolarGeometry(
        float(np.deg2rad(azimuth_degrees)), float(np.deg2rad(elevation_degrees))
    )


def metadata_minus_spice(
    metadata_azimuth: float,
    metadata_elevation: float,
    spice_azimuth: float,
    spice_elevation: float,
) -> tuple[float, float]:
    """Return wrapped azimuth and ordinary elevation differences in radians."""
    azimuth_difference = float(np.arctan2(
        np.sin(metadata_azimuth - spice_azimuth),
        np.cos(metadata_azimuth - spice_azimuth),
    ))
    return azimuth_difference, float(metadata_elevation - spice_elevation)
