from types import SimpleNamespace

import numpy as np
import pytest

from patchrift.ui.tile_matching import (
    all_tile_pairs,
    find_overlapping_tile_pairs,
    match_tile_pairs_incrementally,
    plan_tile_pairs,
)
from patchrift.ui.solar_mode import (
    COMPARE_MODE,
    METADATA_MODE,
    SPICE_MODE,
    degrees_to_geometry,
    metadata_geometry_kwargs,
    metadata_minus_spice,
    uses_spice,
)


def _rectangle(lat0, lon0, lat1, lon1):
    return np.deg2rad([
        [lat0, lon0], [lat0, lon1], [lat1, lon1], [lat1, lon0]
    ])


def test_overlap_planner_returns_only_geographically_intersecting_tile_pairs():
    footprints_a = [_rectangle(0, 0, 1, 1), _rectangle(0, 2, 1, 3)]
    footprints_b = [_rectangle(0.5, 0.5, 1.5, 1.5), _rectangle(0, 4, 1, 5)]

    assert find_overlapping_tile_pairs(footprints_a, footprints_b) == [(0, 0)]


def test_touching_edges_do_not_count_as_positive_area_overlap():
    assert find_overlapping_tile_pairs(
        [_rectangle(0, 0, 1, 1)], [_rectangle(0, 1, 1, 2)]
    ) == []


def test_overlap_planner_handles_antimeridian_longitude_wrap():
    footprint_a = np.deg2rad([
        [-1, 179.0], [-1, -179.0], [1, -179.0], [1, 179.0]
    ])
    footprint_b = np.deg2rad([
        [-0.5, 179.5], [-0.5, -179.5], [0.5, -179.5], [0.5, 179.5]
    ])
    assert find_overlapping_tile_pairs([footprint_a], [footprint_b]) == [(0, 0)]


def test_exhaustive_planner_returns_cartesian_product_without_manual_tile_choice():
    stage_a = SimpleNamespace(tiles=[object(), object()])
    stage_b = SimpleNamespace(tiles=[object(), object(), object()])

    assert plan_tile_pairs(stage_a, stage_b, "all") == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    ]
    assert all_tile_pairs(0, 3) == []


def test_pair_planner_rejects_unknown_mode_and_invalid_polygons():
    with pytest.raises(ValueError, match="mode"):
        plan_tile_pairs(SimpleNamespace(tiles=[]), SimpleNamespace(tiles=[]), "manual")
    with pytest.raises(ValueError, match="four finite"):
        find_overlapping_tile_pairs([np.zeros((3, 2))], [])


def test_metadata_geometry_mode_bypasses_spice_and_is_constant_scene_wide():
    geometry = degrees_to_geometry(123.0, 24.0)
    callbacks = metadata_geometry_kwargs(
        METADATA_MODE, geometry.azimuth, geometry.elevation
    )

    assert not uses_spice(METADATA_MODE)
    assert uses_spice(SPICE_MODE) and uses_spice(COMPARE_MODE)
    first = callbacks["solar_geometry_at"]("ignored", 0.1, 0.2)
    second = callbacks["solar_geometry_at"]("ignored", -0.3, 2.0)
    assert first == second == geometry
    assert callbacks["solar_azimuth_at"](0.1, 0.2) == geometry.azimuth
    assert callbacks["solar_derivatives_at"]("ignored") is None


def test_metadata_geometry_requires_both_angles_and_valid_elevation():
    with pytest.raises(ValueError, match="requires"):
        metadata_geometry_kwargs(METADATA_MODE, 1.0, None)
    with pytest.raises(ValueError, match="elevation"):
        degrees_to_geometry(180.0, 95.0)
    assert metadata_geometry_kwargs(COMPARE_MODE, None, None) == {}


def test_metadata_vs_spice_azimuth_difference_wraps_at_zero_degrees():
    azimuth_delta, elevation_delta = metadata_minus_spice(
        np.deg2rad(1.0), np.deg2rad(30.0),
        np.deg2rad(359.0), np.deg2rad(25.0),
    )
    np.testing.assert_allclose(np.rad2deg(azimuth_delta), 2.0)
    np.testing.assert_allclose(np.rad2deg(elevation_delta), 5.0)


def test_tile_matches_are_saved_before_next_pair_and_failures_do_not_stop_run():
    events = []
    pairs = [(0, 0), (0, 1), (1, 0)]

    def match_pair(tile_a, tile_b):
        events.append(("match", tile_a, tile_b))
        if (tile_a, tile_b) == (0, 1):
            raise RuntimeError("synthetic pair failure")
        return tile_a + tile_b

    def save_result(record):
        events.append(("saved", record["tile_a"], record["tile_b"]))

    results = list(match_tile_pairs_incrementally(pairs, match_pair, save_result))

    assert events == [
        ("match", 0, 0), ("saved", 0, 0),
        ("match", 0, 1), ("saved", 0, 1),
        ("match", 1, 0), ("saved", 1, 0),
    ]
    assert [record["result"] for record in results] == [0, None, 1]
    assert results[1]["error"] == "synthetic pair failure"
