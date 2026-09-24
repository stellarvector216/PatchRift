from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Subtile:
    """
    One Version 3 shadow-segmentation subtile.

    row_start and row_end define the half-open row interval:
        row_start <= row < row_end

    col_start and col_end define the half-open column interval:
        col_start <= col < col_end
    """

    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns) covered by this subtile."""
        return (
            self.row_end - self.row_start,
            self.col_end - self.col_start,
        )
    @property
    def center(self) -> tuple[float, float]:
        """
        Return the subtile centre as (row, column).

        The centre is used as the spatial sample location for the
        threshold associated with this subtile.

        Addendum 1 A1 mandates the subtile-centroid anchor.
        """
        return (
            (self.row_start + self.row_end - 1) / 2.0,
            (self.col_start + self.col_end - 1) / 2.0,
        )

def generate_subtiles(
    image: np.ndarray,
    tile_size: int = 1024,
) -> list[Subtile]:
    """
    Generate the Version 3 shadow-segmentation subtiles.

    The Version 3 specification defines a 1024×1024 subtile grid.
    Edge tiles may be smaller when the image dimensions are not
    exact multiples of 1024.

    Parameters
    ----------
    image:
        2D image whose spatial dimensions define the tiling.

    tile_size:
        Subtile size. Version 3 specifies 1024.

    Returns
    -------
    list[Subtile]
        All subtiles covering the image.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    rows, columns = image.shape

    subtiles: list[Subtile] = []

    for row_start in range(0, rows, tile_size):
        row_end = min(
            row_start + tile_size,
            rows,
        )

        for col_start in range(0, columns, tile_size):
            col_end = min(
                col_start + tile_size,
                columns,
            )

            subtiles.append(
                Subtile(
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                )
            )
    return subtiles
