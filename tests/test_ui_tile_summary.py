from types import SimpleNamespace

import numpy as np

from patchrift.geometry.solar import SolarGeometry
from patchrift.geometry.tiling import PixelTile
from patchrift.ui.tile_summary import tile_summary_rows


def _stage_ii_tile(*, solar_azimuth, solar_elevation, pixel_azimuth, c4=False, low_confidence=False):
    tile = PixelTile(0, 8, 0, 9)
    solar = SimpleNamespace(
        solar_geometry=SolarGeometry(solar_azimuth, solar_elevation),
        pixel_azimuth=pixel_azimuth,
        north_angle=None,
        low_confidence=low_confidence,
        c4=c4,
    )
    return SimpleNamespace(
        tiles=(tile,), tile_results=(solar,), planarity_flag=False
    )


def test_solar_ephemeris_azimuth_is_shown_even_when_shadow_azimuth_is_unavailable():
    result = _stage_ii_tile(
        solar_azimuth=np.deg2rad(137.0), solar_elevation=np.deg2rad(25.0),
        pixel_azimuth=None, c4=True, low_confidence=True,
    )

    row = tile_summary_rows(result)[0]
    assert row["Solar azimuth (°)"] == 137.0
    assert row["Solar elevation (°)"] == 25.0
    assert row["Shadow azimuth (°)"] is None
    assert row["Shadow azimuth status"] == "unavailable: C4 fallback"


def test_shadow_azimuth_is_reported_separately_when_estimated():
    result = _stage_ii_tile(
        solar_azimuth=np.deg2rad(137.0), solar_elevation=np.deg2rad(25.0),
        pixel_azimuth=np.deg2rad(42.5),
    )

    row = tile_summary_rows(result)[0]
    assert row["Solar azimuth (°)"] == 137.0
    assert row["Shadow azimuth (°)"] == 42.5
    assert row["Shadow azimuth status"] == "estimated"
