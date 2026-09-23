from datetime import datetime

import numpy as np

from patchrift.io import ImageProduct, ingest_array
from patchrift.io.product import ProductMetadata


def make_metadata() -> ProductMetadata:
    return ProductMetadata(
        timestamp=datetime(2026, 9, 21, 12, 0, 0),
        footprint_corners=np.array([
            [85.0, 20.0],
            [85.1, 20.0],
            [85.1, 20.1],
            [85.0, 20.1],
        ]),
        footprint_pixel_coords=np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ]),
        gsd=10.0,
        emission_angle=5.0,
    )

def test_ingest_converts_to_float32_and_builds_mask():
    raw = np.array(
        [
            [1, 2, -999],
            [4, -999, 6],
        ],
        dtype=np.int16,
    )

    product = ingest_array(
        raw,
        fill_value=-999,
        metadata=make_metadata(),
    )

    assert isinstance(product, ImageProduct)
    assert product.image.dtype == np.float32
    assert product.valid_mask.dtype == np.bool_

    expected_mask = np.array(
        [
            [True, True, False],
            [True, False, True],
        ]
    )

    np.testing.assert_array_equal(
        product.valid_mask,
        expected_mask,
    )


def test_ingest_preserves_image_values():
    raw = np.array(
        [
            [1, 2, -999],
            [4, -999, 6],
        ],
        dtype=np.int16,
    )

    product = ingest_array(
        raw,
        fill_value=-999,
        metadata=make_metadata(),
    )

    expected = raw.astype(np.float32)

    np.testing.assert_array_equal(
        product.image,
        expected,
    )


def test_metadata_is_stored():
    metadata = make_metadata()

    raw = np.ones((2, 2), dtype=np.int16)

    product = ingest_array(
        raw,
        fill_value=-999,
        metadata=metadata,
    )

    assert product.metadata is metadata
    assert product.metadata.gsd == 10.0
    assert product.metadata.emission_angle == 5.0
def test_ingest_normalizes_mirrored_product():
    raw = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ], dtype=np.int16)

    metadata = ProductMetadata(
        timestamp=datetime(2026, 9, 21, 12, 0, 0),
        footprint_corners=np.array([
            [85.0, 20.0],
            [85.1, 20.0],
            [85.1, 20.1],
            [85.0, 20.1],
        ]),
        footprint_pixel_coords=np.array([
            [3.0, 0.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [3.0, 2.0],
        ]),
        gsd=10.0,
        emission_angle=5.0,
    )

    product = ingest_array(
        raw,
        fill_value=-999,
        metadata=metadata,
    )

    expected_image = np.fliplr(raw.astype(np.float32))

    expected_coords = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 2.0],
        [0.0, 2.0],
    ])

    np.testing.assert_array_equal(
        product.image,
        expected_image,
    )

    np.testing.assert_array_equal(
        product.metadata.footprint_pixel_coords,
        expected_coords,
    )

    assert product.was_reflected is True