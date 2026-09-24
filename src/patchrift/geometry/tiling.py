"""§4.1 adaptive tile sizing before any photometric estimate exists."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from patchrift.io.product import ProductMetadata


@dataclass(frozen=True)
class PixelTile:
    """Half-open pixel rectangle, expressed as (row, col) bounds."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.row_end - self.row_start, self.col_end - self.col_start


def footprint_geolocation_model(metadata: "ProductMetadata") -> Callable[[float, float], tuple[float, float]]:
    """Fit the §2.3 local affine map and return pixel-to-[lat, lon] in radians.

    This uses the product's footprint control points and their delivered pixel
    coordinates. Longitude is unwrapped around the footprint centre and scaled
    by cos(latitude), as required by the handedness fit.
    """
    geographic = np.asarray(metadata.footprint_corners, dtype=float)
    pixels = np.asarray(metadata.footprint_pixel_coords, dtype=float)
    lon0 = float(np.mean(np.unwrap(np.deg2rad(geographic[:, 0]))))
    lat0 = float(np.deg2rad(np.mean(geographic[:, 1])))
    cos_lat = abs(float(np.cos(lat0)))
    if cos_lat <= np.finfo(float).eps:
        raise ValueError("footprint affine longitude is undefined at a lunar pole")
    longitude = np.unwrap(np.deg2rad(geographic[:, 0]))
    local = np.column_stack(((longitude - lon0) * cos_lat, np.deg2rad(geographic[:, 1]) - lat0))
    design = np.column_stack((local, np.ones(local.shape[0])))
    coefficients, _, rank, _ = np.linalg.lstsq(design, pixels, rcond=None)
    if rank < 3:
        raise ValueError("footprint points do not define a valid affine geolocation model")
    jacobian = coefficients[:2, :].T
    if abs(float(np.linalg.det(jacobian))) <= np.finfo(float).eps:
        raise ValueError("footprint affine Jacobian is singular")
    inverse = np.linalg.inv(jacobian)
    pixel_origin = coefficients[2, :]

    def location(row: float, col: float) -> tuple[float, float]:
        east_scaled, north = inverse @ (np.array([col, row], dtype=float) - pixel_origin)
        latitude = lat0 + north
        longitude = lon0 + east_scaled / cos_lat
        return float(latitude), float(longitude)

    return location


def footprint_tile_corners(
    tile: PixelTile,
    pixel_to_location: Callable[[float, float], tuple[float, float]],
) -> np.ndarray:
    """Return tile corner locations in [latitude, longitude] radians."""
    last_row = tile.row_end - 1
    last_col = tile.col_end - 1
    pixels = (
        (tile.row_start, tile.col_start),
        (tile.row_start, last_col),
        (last_row, tile.col_start),
        (last_row, last_col),
    )
    return np.asarray([pixel_to_location(row, col) for row, col in pixels], dtype=float)


def tile_azimuth_span(
    tile: PixelTile,
    corner_location: Callable[[PixelTile], np.ndarray],
    solar_azimuth_at: Callable[[float, float], float],
) -> float:
    """Return the largest wrapped azimuth separation over tile corners.

    ``corner_location`` is deliberately an interface: the specification
    permits either footprint corners or a product geolocation model and
    does not authorize inventing a geolocation interpolation method.
    It returns [latitude, longitude] in radians for the four corners.
    """
    locations = np.asarray(corner_location(tile), dtype=float)
    if locations.shape != (4, 2) or not np.all(np.isfinite(locations)):
        raise ValueError("corner_location must return four finite [lat, lon] pairs")
    azimuths = np.array([solar_azimuth_at(lat, lon) for lat, lon in locations], dtype=float)
    if not np.all(np.isfinite(azimuths)):
        raise ValueError("solar_azimuth_at must return finite angles")
    differences = azimuths[:, None] - azimuths[None, :]
    return float(np.max(np.abs(np.arctan2(np.sin(differences), np.cos(differences)))))


def bisect_tiles_for_solar_azimuth(
    image_shape: tuple[int, int],
    corner_location: Callable[[PixelTile], np.ndarray],
    solar_azimuth_at: Callable[[float, float], float],
    sigma_sys: float = np.deg2rad(0.5),
    max_pixels_per_tile: int | None = None,
) -> list[PixelTile]:
    """Apply §4.1 with Addendum B1's σsys bootstrap criterion."""
    rows, cols = image_shape
    if rows <= 0 or cols <= 0:
        raise ValueError("image_shape dimensions must be positive")
    if not np.isfinite(sigma_sys) or sigma_sys <= 0.0:
        raise ValueError("sigma_sys must be finite and positive")
    if max_pixels_per_tile is not None and max_pixels_per_tile <= 0:
        raise ValueError("max_pixels_per_tile must be positive")

    pending = [PixelTile(0, rows, 0, cols)]
    accepted: list[PixelTile] = []
    limit = sigma_sys / 3.0
    while pending:
        tile = pending.pop()
        tile_rows, tile_cols = tile.shape
        memory_ok = max_pixels_per_tile is None or tile_rows * tile_cols <= max_pixels_per_tile
        azimuth_ok = tile_azimuth_span(tile, corner_location, solar_azimuth_at) < limit
        if (memory_ok and azimuth_ok) or (tile_rows == 1 and tile_cols == 1):
            accepted.append(tile)
            continue
        if tile_rows >= tile_cols and tile_rows > 1:
            mid = tile.row_start + tile_rows // 2
            pending.extend((PixelTile(tile.row_start, mid, tile.col_start, tile.col_end), PixelTile(mid, tile.row_end, tile.col_start, tile.col_end)))
        elif tile_cols > 1:
            mid = tile.col_start + tile_cols // 2
            pending.extend((PixelTile(tile.row_start, tile.row_end, tile.col_start, mid), PixelTile(tile.row_start, tile.row_end, mid, tile.col_end)))
        else:
            accepted.append(tile)
    return accepted
