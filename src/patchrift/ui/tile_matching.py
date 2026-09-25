"""Automatic tile-pair planning for the prototype UI."""

from __future__ import annotations

import numpy as np

from patchrift.geometry.tiling import footprint_geolocation_model, footprint_tile_corners


def tile_footprints(stage_ii_result) -> tuple[np.ndarray, ...]:
    """Return every adaptive tile's four geolocated corners as [lat, lon] radians."""
    metadata = stage_ii_result.stage_i.product.metadata
    pixel_to_location = footprint_geolocation_model(metadata)
    polygons = []
    for tile in stage_ii_result.tiles:
        corners = footprint_tile_corners(tile, pixel_to_location)
        # footprint_tile_corners is [top-left, top-right, bottom-left,
        # bottom-right], while polygon clipping needs cyclic corner order.
        polygons.append(corners[[0, 1, 3, 2]])
    return tuple(polygons)


def find_overlapping_tile_pairs(
    footprints_a,
    footprints_b,
    *,
    minimum_overlap_area: float = 1e-15,
) -> list[tuple[int, int]]:
    """Find all tile pairs with positive geographic footprint overlap.

    Input polygons are four [latitude, longitude] corner pairs in radians.
    A local equirectangular plane is used for this small-footprint intersection
    test. It selects candidate work pairs only; Stage V still verifies visual
    correspondence from image features.
    """
    a = _validate_footprints(footprints_a, "footprints_a")
    b = _validate_footprints(footprints_b, "footprints_b")
    if not np.isfinite(minimum_overlap_area) or minimum_overlap_area < 0.0:
        raise ValueError("minimum_overlap_area must be finite and non-negative")
    if not a or not b:
        return []

    all_points = np.concatenate((*a, *b), axis=0)
    reference_latitude = float(np.mean(all_points[:, 0]))
    reference_longitude = float(np.angle(np.mean(np.exp(1j * all_points[:, 1]))))
    longitude_scale = max(abs(float(np.cos(reference_latitude))), 1e-6)
    a_xy = tuple(_local_xy(poly, reference_latitude, reference_longitude, longitude_scale) for poly in a)
    b_xy = tuple(_local_xy(poly, reference_latitude, reference_longitude, longitude_scale) for poly in b)

    pairs = []
    for index_a, polygon_a in enumerate(a_xy):
        for index_b, polygon_b in enumerate(b_xy):
            if _polygon_intersection_area(polygon_a, polygon_b) > minimum_overlap_area:
                pairs.append((index_a, index_b))
    return pairs


def all_tile_pairs(tile_count_a: int, tile_count_b: int) -> list[tuple[int, int]]:
    """Return the exhaustive Cartesian product of the two tile lists."""
    if not isinstance(tile_count_a, (int, np.integer)) or not isinstance(tile_count_b, (int, np.integer)):
        raise ValueError("tile counts must be integers")
    if tile_count_a < 0 or tile_count_b < 0:
        raise ValueError("tile counts must be non-negative")
    return [(index_a, index_b) for index_a in range(tile_count_a) for index_b in range(tile_count_b)]


def plan_tile_pairs(stage_ii_a, stage_ii_b, mode: str) -> list[tuple[int, int]]:
    """Plan automatic tile work for geographic-overlap or exhaustive matching."""
    if mode == "overlap":
        return find_overlapping_tile_pairs(
            tile_footprints(stage_ii_a), tile_footprints(stage_ii_b)
        )
    if mode == "all":
        return all_tile_pairs(len(stage_ii_a.tiles), len(stage_ii_b.tiles))
    raise ValueError("mode must be 'overlap' or 'all'")


def match_tile_pairs_incrementally(pairs, match_pair, on_result=None):
    """Yield each tile match result immediately and continue after pair errors.

    ``on_result`` is called before the next pair begins, allowing a UI to save
    and render each completed result while the remaining pairs continue.
    Exceptions from one pair are recorded and do not stop later work.
    """
    for tile_a, tile_b in pairs:
        try:
            record = {
                "tile_a": tile_a,
                "tile_b": tile_b,
                "result": match_pair(tile_a, tile_b),
                "error": None,
            }
        except Exception as exc:  # one difficult tile must not discard the run
            record = {
                "tile_a": tile_a,
                "tile_b": tile_b,
                "result": None,
                "error": str(exc),
            }
        if on_result is not None:
            on_result(record)
        yield record


def _validate_footprints(footprints, name):
    result = []
    for index, footprint in enumerate(footprints):
        polygon = np.asarray(footprint, dtype=float)
        if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
            raise ValueError(f"{name}[{index}] must be four finite [latitude, longitude] corners")
        if np.any(np.abs(polygon[:, 0]) > np.pi / 2.0 + 1e-12):
            raise ValueError(f"{name}[{index}] latitude must be in radians within [-pi/2, pi/2]")
        result.append(polygon)
    return result


def _local_xy(polygon, reference_latitude, reference_longitude, longitude_scale):
    latitude = polygon[:, 0]
    longitude_delta = (polygon[:, 1] - reference_longitude + np.pi) % (2.0 * np.pi) - np.pi
    return np.column_stack((longitude_delta * longitude_scale, latitude - reference_latitude))


def _polygon_intersection_area(subject, clip):
    """Sutherland–Hodgman clipping for two convex quadrilaterals."""
    output = [np.asarray(point, dtype=float) for point in subject]
    signed_area = _signed_area(clip)
    if abs(signed_area) <= 1e-24:
        return 0.0
    orientation = 1.0 if signed_area > 0.0 else -1.0
    for edge_index in range(len(clip)):
        edge_start = clip[edge_index]
        edge_end = clip[(edge_index + 1) % len(clip)]
        edge = edge_end - edge_start
        input_points = output
        output = []
        if not input_points:
            return 0.0
        previous = input_points[-1]
        previous_inside = orientation * _cross(edge, previous - edge_start) >= -1e-14
        for current in input_points:
            current_inside = orientation * _cross(edge, current - edge_start) >= -1e-14
            if current_inside != previous_inside:
                segment = current - previous
                denominator = _cross(segment, edge)
                if abs(denominator) > 1e-24:
                    fraction = _cross(edge_start - previous, edge) / denominator
                    output.append(previous + fraction * segment)
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
    if len(output) < 3:
        return 0.0
    return abs(_signed_area(np.asarray(output)))


def _signed_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])
