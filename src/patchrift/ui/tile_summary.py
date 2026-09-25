"""Stage-II summaries for the user interface, separated from Streamlit widgets."""

from __future__ import annotations

import numpy as np

from patchrift.ui.solar_mode import metadata_minus_spice


def tile_summary_rows(stage_ii_result, metadata_geometry=None) -> list[dict]:
    """Build readable per-tile solar and shadow azimuth diagnostics.

    Solar azimuth is the ephemeris/metadata Sun direction and is available even
    when Stage II cannot estimate an image-frame shadow azimuth. The latter has
    a separate status field so the two quantities cannot be confused.
    """
    rows = []
    for index, (tile, solar) in enumerate(
        zip(stage_ii_result.tiles, stage_ii_result.tile_results), start=1
    ):
        pixel_azimuth = solar.pixel_azimuth
        if pixel_azimuth is not None:
            shadow_status = "estimated"
        elif solar.c4:
            shadow_status = "unavailable: C4 fallback"
        elif solar.low_confidence:
            shadow_status = "unavailable: low confidence"
        else:
            shadow_status = "unavailable: no valid shadow estimate"
        row = {
            "Tile": index,
            "Rows": f"{tile.row_start}:{tile.row_end}",
            "Columns": f"{tile.col_start}:{tile.col_end}",
            "Solar azimuth (°)": round(float(np.rad2deg(solar.solar_geometry.azimuth)), 3),
            "Solar elevation (°)": round(float(np.rad2deg(solar.solar_geometry.elevation)), 3),
            "Shadow azimuth (°)": None if pixel_azimuth is None else round(float(np.rad2deg(pixel_azimuth)), 3),
            "Shadow azimuth status": shadow_status,
            "North angle (°)": None if solar.north_angle is None else round(float(np.rad2deg(solar.north_angle)), 3),
            "Planarity warning": stage_ii_result.planarity_flag,
            "Low confidence": solar.low_confidence,
            "C4 fallback": solar.c4,
        }
        if metadata_geometry is not None:
            metadata_azimuth, metadata_elevation = metadata_geometry
            azimuth_difference, elevation_difference = metadata_minus_spice(
                metadata_azimuth, metadata_elevation,
                solar.solar_geometry.azimuth, solar.solar_geometry.elevation,
            )
            row["Metadata azimuth (°)"] = round(float(np.rad2deg(metadata_azimuth)), 3)
            row["Metadata − SPICE azimuth (°)"] = round(float(np.rad2deg(azimuth_difference)), 3)
            row["Metadata − SPICE elevation (°)"] = round(float(np.rad2deg(elevation_difference)), 3)
        rows.append(row)
    return rows
