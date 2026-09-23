import numpy as np
from patchrift.shadows.interpolation import (
    InterpolationCell,
    build_complete_interpolation_cells,
)
from patchrift.shadows.thresholds import (
    SubtileThreshold,
)
from patchrift.shadows.interpolation import (
    bilinear_threshold,
)
from patchrift.shadows.interpolation import (
    bilinear_threshold_field,
    initialize_threshold_field,
    populate_threshold_field
)

def make_threshold(
    row_start,
    row_end,
    col_start,
    col_end,
    value,
):
    return SubtileThreshold(
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        threshold=value,
    )


def test_complete_2x2_threshold_grid_produces_one_cell():
    thresholds = [
        make_threshold(0, 1024, 0, 1024, 1.0),
        make_threshold(0, 1024, 1024, 2048, 2.0),
        make_threshold(1024, 2048, 0, 1024, 3.0),
        make_threshold(1024, 2048, 1024, 2048, 4.0),
    ]

    cells = build_complete_interpolation_cells(
        thresholds
    )

    assert len(cells) == 1

    assert cells[0] == InterpolationCell(
        top_left=thresholds[0],
        top_right=thresholds[1],
        bottom_left=thresholds[2],
        bottom_right=thresholds[3],
    )


def test_missing_corner_excludes_interpolation_cell():
    thresholds = [
        make_threshold(0, 1024, 0, 1024, 1.0),
        make_threshold(0, 1024, 1024, 2048, 2.0),
        make_threshold(1024, 2048, 0, 1024, 3.0),
    ]

    cells = build_complete_interpolation_cells(
        thresholds
    )

    assert cells == []


def test_two_by_three_grid_produces_two_cells():
    thresholds = [
        make_threshold(0, 1024, 0, 1024, 1.0),
        make_threshold(0, 1024, 1024, 2048, 2.0),
        make_threshold(0, 1024, 2048, 3072, 3.0),
        make_threshold(1024, 2048, 0, 1024, 4.0),
        make_threshold(1024, 2048, 1024, 2048, 5.0),
        make_threshold(1024, 2048, 2048, 3072, 6.0),
    ]

    cells = build_complete_interpolation_cells(
        thresholds
    )

    assert len(cells) == 2


def test_empty_threshold_list_produces_no_cells():
    assert build_complete_interpolation_cells([]) == []
def test_bilinear_threshold_returns_corner_values():
    thresholds = [
        SubtileThreshold(0, 1024, 0, 1024, 0.0),
        SubtileThreshold(0, 1024, 1024, 2048, 10.0),
        SubtileThreshold(1024, 2048, 0, 1024, 20.0),
        SubtileThreshold(1024, 2048, 1024, 2048, 30.0),
    ]

    cells = build_complete_interpolation_cells(thresholds)

    assert len(cells) == 1

    cell = cells[0]

    assert bilinear_threshold(
        cell,
        511.5,
        511.5,
    ) == 0.0

    assert bilinear_threshold(
        cell,
        511.5,
        1535.5,
    ) == 10.0

    assert bilinear_threshold(
        cell,
        1535.5,
        511.5,
    ) == 20.0

    assert bilinear_threshold(
        cell,
        1535.5,
        1535.5,
    ) == 30.0


def test_bilinear_threshold_midpoint_is_average_of_corners():
    thresholds = [
        SubtileThreshold(0, 1024, 0, 1024, 0.0),
        SubtileThreshold(0, 1024, 1024, 2048, 10.0),
        SubtileThreshold(1024, 2048, 0, 1024, 20.0),
        SubtileThreshold(1024, 2048, 1024, 2048, 30.0),
    ]

    cell = build_complete_interpolation_cells(thresholds)[0]

    result = bilinear_threshold(
        cell,
        1023.5,
        1023.5,
    )

    assert result == 15.0


def test_bilinear_threshold_rejects_point_outside_cell():
    thresholds = [
        SubtileThreshold(0, 1024, 0, 1024, 0.0),
        SubtileThreshold(0, 1024, 1024, 2048, 10.0),
        SubtileThreshold(1024, 2048, 0, 1024, 20.0),
        SubtileThreshold(1024, 2048, 1024, 2048, 30.0),
    ]

    cell = build_complete_interpolation_cells(thresholds)[0]

    try:
        bilinear_threshold(
            cell,
            2000.0,
            1000.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for point outside interpolation cell"
        )
def test_bilinear_threshold_field_returns_expected_values():
    thresholds = [
        SubtileThreshold(0, 1024, 0, 1024, 0.0),
        SubtileThreshold(0, 1024, 1024, 2048, 10.0),
        SubtileThreshold(1024, 2048, 0, 1024, 20.0),
        SubtileThreshold(1024, 2048, 1024, 2048, 30.0),
    ]

    cell = build_complete_interpolation_cells(thresholds)[0]

    rows = np.array([
        511.5,
        511.5,
        1535.5,
        1535.5,
        1023.5,
    ])

    cols = np.array([
        511.5,
        1535.5,
        511.5,
        1535.5,
        1023.5,
    ])

    result = bilinear_threshold_field(
        cell,
        rows,
        cols,
    )

    expected = np.array([
        0.0,
        10.0,
        20.0,
        30.0,
        15.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_bilinear_threshold_field_preserves_input_shape():
    thresholds = [
        SubtileThreshold(0, 1024, 0, 1024, 0.0),
        SubtileThreshold(0, 1024, 1024, 2048, 10.0),
        SubtileThreshold(1024, 2048, 0, 1024, 20.0),
        SubtileThreshold(1024, 2048, 1024, 2048, 30.0),
    ]

    cell = build_complete_interpolation_cells(thresholds)[0]

    rows = np.array([
        [511.5, 511.5],
        [1535.5, 1535.5],
    ])

    cols = np.array([
        [511.5, 1535.5],
        [511.5, 1535.5],
    ])

    result = bilinear_threshold_field(
        cell,
        rows,
        cols,
    )

    assert result.shape == rows.shape


def test_bilinear_threshold_field_rejects_mismatched_shapes():
    thresholds = [
        SubtileThreshold(0, 1024, 0, 1024, 0.0),
        SubtileThreshold(0, 1024, 1024, 2048, 10.0),
        SubtileThreshold(1024, 2048, 0, 1024, 20.0),
        SubtileThreshold(1024, 2048, 1024, 2048, 30.0),
    ]

    cell = build_complete_interpolation_cells(thresholds)[0]

    rows = np.array([511.5, 1023.5])
    cols = np.array([511.5])

    try:
        bilinear_threshold_field(
            cell,
            rows,
            cols,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for mismatched coordinate shapes"
        )
def test_initialize_threshold_field_is_undefined_everywhere():
    field, defined_mask = initialize_threshold_field(
        (4, 6)
    )

    assert field.shape == (4, 6)
    assert defined_mask.shape == (4, 6)

    assert np.all(np.isnan(field))
    assert not np.any(defined_mask)


def test_initialize_threshold_field_rejects_invalid_shape():
    try:
        initialize_threshold_field(
            (0, 10)
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for non-positive image dimension"
        )
def test_populate_threshold_field_fills_complete_cell():
    thresholds = [
        SubtileThreshold(0, 1024, 0, 1024, 0.0),
        SubtileThreshold(0, 1024, 1024, 2048, 10.0),
        SubtileThreshold(1024, 2048, 0, 1024, 20.0),
        SubtileThreshold(1024, 2048, 1024, 2048, 30.0),
    ]

    cells = build_complete_interpolation_cells(
        thresholds
    )

    field, defined_mask = populate_threshold_field(
        (2048, 2048),
        cells,
    )

    assert defined_mask[512, 512]
    assert defined_mask[1023, 1023]
    assert defined_mask[1535, 1535]

    assert np.isfinite(field[512, 512])
    assert np.isfinite(field[1023, 1023])
    assert np.isfinite(field[1535, 1535])

def test_populate_threshold_field_leaves_uncovered_pixels_undefined():
    thresholds = [
        SubtileThreshold(0, 1024, 0, 1024, 0.0),
        SubtileThreshold(0, 1024, 1024, 2048, 10.0),
        SubtileThreshold(1024, 2048, 0, 1024, 20.0),
        SubtileThreshold(1024, 2048, 1024, 2048, 30.0),
    ]

    cells = build_complete_interpolation_cells(
        thresholds
    )

    field, defined_mask = populate_threshold_field(
        (2048, 2048),
        cells,
    )

    assert not defined_mask[0, 0]
    assert np.isnan(field[0, 0])

    assert not defined_mask[2047, 2047]
    assert np.isnan(field[2047, 2047])


def test_populate_threshold_field_empty_cells_leaves_field_undefined():
    field, defined_mask = populate_threshold_field(
        (10, 10),
        [],
    )

    assert np.all(np.isnan(field))
    assert not np.any(defined_mask)