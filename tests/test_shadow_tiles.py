import numpy as np
import pytest

from patchrift.shadows.tiles import (
    Subtile,
    generate_subtiles,
)


def test_generate_single_1024_tile():
    image = np.zeros(
        (1024, 1024),
        dtype=np.float32,
    )

    subtiles = generate_subtiles(image)

    assert subtiles == [
        Subtile(
            row_start=0,
            row_end=1024,
            col_start=0,
            col_end=1024,
        )
    ]


def test_generate_four_tiles_for_2048_square():
    image = np.zeros(
        (2048, 2048),
        dtype=np.float32,
    )

    subtiles = generate_subtiles(image)

    assert len(subtiles) == 4

    assert subtiles[0].shape == (1024, 1024)
    assert subtiles[1].shape == (1024, 1024)
    assert subtiles[2].shape == (1024, 1024)
    assert subtiles[3].shape == (1024, 1024)


def test_edge_tiles_are_smaller_when_needed():
    image = np.zeros(
        (1500, 2500),
        dtype=np.float32,
    )

    subtiles = generate_subtiles(image)

    assert len(subtiles) == 6

    shapes = [subtile.shape for subtile in subtiles]

    assert shapes == [
        (1024, 1024),
        (1024, 1024),
        (1024, 452),
        (476, 1024),
        (476, 1024),
        (476, 452),
    ]


def test_subtiles_cover_image_without_gaps():
    image = np.zeros(
        (2500, 3000),
        dtype=np.float32,
    )

    subtiles = generate_subtiles(image)

    coverage = np.zeros(
        image.shape,
        dtype=np.int32,
    )

    for subtile in subtiles:
        coverage[
            subtile.row_start:subtile.row_end,
            subtile.col_start:subtile.col_end,
        ] += 1

    assert np.all(coverage == 1)


def test_rejects_non_2d_image():
    image = np.zeros(
        (10, 10, 3),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        generate_subtiles(image)


def test_rejects_non_positive_tile_size():
    image = np.zeros(
        (100, 100),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        generate_subtiles(
            image,
            tile_size=0,
        )
def test_subtile_center_for_full_tile():
    subtile = Subtile(
        row_start=0,
        row_end=1024,
        col_start=0,
        col_end=1024,
    )

    assert subtile.center == (511.5, 511.5)


def test_subtile_center_for_offset_tile():
    subtile = Subtile(
        row_start=1024,
        row_end=2048,
        col_start=2048,
        col_end=3072,
    )

    assert subtile.center == (1535.5, 2559.5)


def test_edge_subtile_center_uses_actual_extent():
    subtile = Subtile(
        row_start=1024,
        row_end=1500,
        col_start=2048,
        col_end=2500,
    )

    assert subtile.center == (1261.5, 2273.5)