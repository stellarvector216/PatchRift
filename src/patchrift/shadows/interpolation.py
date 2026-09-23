from dataclasses import dataclass
import numpy as np
from patchrift.shadows.thresholds import SubtileThreshold


@dataclass(frozen=True)
class InterpolationCell:
    """
    A bilinear interpolation cell whose four corner threshold
    samples are available.

    The corners are ordered:

        top_left     top_right
             +---------+
             |         |
             +---------+
        bottom_left  bottom_right
    """

    top_left: SubtileThreshold
    top_right: SubtileThreshold
    bottom_left: SubtileThreshold
    bottom_right: SubtileThreshold


def build_complete_interpolation_cells(
    thresholds: list[SubtileThreshold],
) -> list[InterpolationCell]:
    """
    Build bilinear interpolation cells from qualified subtile
    threshold samples.

    A cell is created only when all four neighbouring subtile
    threshold samples are available.

    This is an implementation convention because Version 3 does
    not specify a fallback for missing interpolation neighbours.
    """

    threshold_map = {
        (
            threshold.row_start,
            threshold.col_start,
        ): threshold
        for threshold in thresholds
    }

    cells: list[InterpolationCell] = []

    for threshold in thresholds:
        row = threshold.row_start
        col = threshold.col_start

        right = threshold_map.get(
            (row, col + (threshold.col_end - threshold.col_start))
        )

        bottom = threshold_map.get(
            (row + (threshold.row_end - threshold.row_start), col)
        )

        bottom_right = threshold_map.get(
            (
                row + (threshold.row_end - threshold.row_start),
                col + (threshold.col_end - threshold.col_start),
            )
        )

        if (
            right is None
            or bottom is None
            or bottom_right is None
        ):
            continue

        cells.append(
            InterpolationCell(
                top_left=threshold,
                top_right=right,
                bottom_left=bottom,
                bottom_right=bottom_right,
            )
        )

    return cells
def bilinear_threshold(
    cell: InterpolationCell,
    row: float,
    col: float,
) -> float:
    """
    Evaluate the bilinearly interpolated threshold at one image
    coordinate.

    The four threshold samples are located at the centres of the
    four subtiles.

    Image coordinates use:
        row -> downward
        col -> rightward

    Therefore:
        u = horizontal/column interpolation coordinate
        v = vertical/row interpolation coordinate

    Version 3 does not explicitly define the threshold-sample
    anchor or boundary behaviour. This function follows the
    implementation convention that each subtile threshold is
    sampled at the subtile centre.
    """

    top_left_row, top_left_col = (
        cell.top_left.row_start + cell.top_left.row_end - 1
    ) / 2.0, (
        cell.top_left.col_start + cell.top_left.col_end - 1
    ) / 2.0

    top_right_row, top_right_col = (
        cell.top_right.row_start + cell.top_right.row_end - 1
    ) / 2.0, (
        cell.top_right.col_start + cell.top_right.col_end - 1
    ) / 2.0

    bottom_left_row, bottom_left_col = (
        cell.bottom_left.row_start + cell.bottom_left.row_end - 1
    ) / 2.0, (
        cell.bottom_left.col_start + cell.bottom_left.col_end - 1
    ) / 2.0

    bottom_right_row, bottom_right_col = (
        cell.bottom_right.row_start + cell.bottom_right.row_end - 1
    ) / 2.0, (
        cell.bottom_right.col_start + cell.bottom_right.col_end - 1
    ) / 2.0

    if not (
        top_left_row <= row <= bottom_left_row
        and top_left_col <= col <= top_right_col
    ):
        raise ValueError(
            "row and col must lie within the interpolation cell"
        )

    row_span = bottom_left_row - top_left_row
    col_span = top_right_col - top_left_col

    if row_span <= 0 or col_span <= 0:
        raise ValueError(
            "interpolation cell must have positive spatial extent"
        )

    u = (col - top_left_col) / col_span
    v = (row - top_left_row) / row_span

    tau00 = cell.top_left.threshold
    tau10 = cell.top_right.threshold
    tau01 = cell.bottom_left.threshold
    tau11 = cell.bottom_right.threshold

    return float(
        (1.0 - u) * (1.0 - v) * tau00
        + u * (1.0 - v) * tau10
        + (1.0 - u) * v * tau01
        + u * v * tau11
    )
def bilinear_threshold_field(
    cell: InterpolationCell,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the bilinearly interpolated threshold at multiple
    image coordinates.

    rows and cols must have the same shape.

    The interpolation uses the four subtile-centre threshold
    samples stored in the cell.
    """

    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)

    if rows.shape != cols.shape:
        raise ValueError(
            "rows and cols must have the same shape"
        )

    top_left_row = (
        cell.top_left.row_start
        + cell.top_left.row_end
        - 1
    ) / 2.0

    top_left_col = (
        cell.top_left.col_start
        + cell.top_left.col_end
        - 1
    ) / 2.0

    top_right_col = (
        cell.top_right.col_start
        + cell.top_right.col_end
        - 1
    ) / 2.0

    bottom_left_row = (
        cell.bottom_left.row_start
        + cell.bottom_left.row_end
        - 1
    ) / 2.0

    row_span = bottom_left_row - top_left_row
    col_span = top_right_col - top_left_col

    if row_span <= 0 or col_span <= 0:
        raise ValueError(
            "interpolation cell must have positive spatial extent"
        )

    if (
        np.any(rows < top_left_row)
        or np.any(rows > bottom_left_row)
        or np.any(cols < top_left_col)
        or np.any(cols > top_right_col)
    ):
        raise ValueError(
            "all coordinates must lie within the interpolation cell"
        )

    u = (cols - top_left_col) / col_span
    v = (rows - top_left_row) / row_span

    tau00 = cell.top_left.threshold
    tau10 = cell.top_right.threshold
    tau01 = cell.bottom_left.threshold
    tau11 = cell.bottom_right.threshold

    return (
        (1.0 - u) * (1.0 - v) * tau00
        + u * (1.0 - v) * tau10
        + (1.0 - u) * v * tau01
        + u * v * tau11
    )
def initialize_threshold_field(
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an empty image-sized threshold field.

    Pixels begin with no defined interpolated threshold.

    Returns
    -------
    threshold_field:
        Float array containing NaN wherever no threshold has
        yet been defined.

    defined_mask:
        Boolean array. True indicates that the corresponding
        threshold-field pixel has a defined value.

    Notes
    -----
    Version 3 does not specify extrapolation or fallback behaviour
    outside complete bilinear interpolation cells. Such pixels
    therefore remain undefined rather than being assigned an
    invented threshold.
    """

    if len(image_shape) != 2:
        raise ValueError(
            "image_shape must contain exactly two dimensions"
        )

    rows, cols = image_shape

    if rows <= 0 or cols <= 0:
        raise ValueError(
            "image dimensions must be positive"
        )

    threshold_field = np.full(
        image_shape,
        np.nan,
        dtype=float,
    )

    defined_mask = np.zeros(
        image_shape,
        dtype=bool,
    )

    return threshold_field, defined_mask
def populate_threshold_field(
    image_shape: tuple[int, int],
    cells: list[InterpolationCell],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Populate an image-sized threshold field from complete
    bilinear interpolation cells.

    Pixels covered by a complete interpolation cell receive the
    bilinearly interpolated threshold.

    Pixels not covered by any complete interpolation cell remain
    undefined (NaN) and are marked False in defined_mask.

    Version 3 does not specify extrapolation or fallback behaviour
    for regions without four valid interpolation samples.
    """

    threshold_field, defined_mask = initialize_threshold_field(
        image_shape
    )

    for cell in cells:
        top_left_row = (
            cell.top_left.row_start
            + cell.top_left.row_end
            - 1
        ) / 2.0

        top_left_col = (
            cell.top_left.col_start
            + cell.top_left.col_end
            - 1
        ) / 2.0

        bottom_left_row = (
            cell.bottom_left.row_start
            + cell.bottom_left.row_end
            - 1
        ) / 2.0

        top_right_col = (
            cell.top_right.col_start
            + cell.top_right.col_end
            - 1
        ) / 2.0

        row_start = int(np.ceil(top_left_row))
        row_end = int(np.floor(bottom_left_row))

        col_start = int(np.ceil(top_left_col))
        col_end = int(np.floor(top_right_col))

        if row_start > row_end or col_start > col_end:
            continue

        row_start = max(row_start, 0)
        row_end = min(row_end, image_shape[0] - 1)

        col_start = max(col_start, 0)
        col_end = min(col_end, image_shape[1] - 1)

        if row_start > row_end or col_start > col_end:
            continue

        rows = np.arange(
            row_start,
            row_end + 1,
            dtype=float,
        )

        cols = np.arange(
            col_start,
            col_end + 1,
            dtype=float,
        )

        row_grid, col_grid = np.meshgrid(
            rows,
            cols,
            indexing="ij",
        )

        values = bilinear_threshold_field(
            cell,
            row_grid,
            col_grid,
        )

        existing = defined_mask[
            row_start:row_end + 1,
            col_start:col_end + 1,
        ]

        if np.any(existing):
            existing_values = threshold_field[
                row_start:row_end + 1,
                col_start:col_end + 1,
            ][existing]

            new_values = values[existing]

            if not np.allclose(
                existing_values,
                new_values,
            ):
                raise ValueError(
                    "overlapping interpolation cells "
                    "produce inconsistent thresholds"
                )

        threshold_field[
            row_start:row_end + 1,
            col_start:col_end + 1,
        ] = values

        defined_mask[
            row_start:row_end + 1,
            col_start:col_end + 1,
        ] = True

    return threshold_field, defined_mask