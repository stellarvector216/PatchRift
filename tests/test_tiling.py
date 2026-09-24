import numpy as np

from patchrift.geometry.tiling import bisect_tiles_for_solar_azimuth


def _corners(tile):
    return np.array([
        [tile.row_start, tile.col_start],
        [tile.row_start, tile.col_end],
        [tile.row_end, tile.col_start],
        [tile.row_end, tile.col_end],
    ], dtype=float)


def test_tile_bisection_uses_sigma_sys_before_live_sigma_phi_exists():
    tiles = bisect_tiles_for_solar_azimuth(
        (16, 16),
        _corners,
        lambda latitude, longitude: 0.01 * longitude,
        sigma_sys=0.12,
    )

    assert len(tiles) > 1
    assert all(tile.row_end > tile.row_start for tile in tiles)
    assert all(tile.col_end > tile.col_start for tile in tiles)
