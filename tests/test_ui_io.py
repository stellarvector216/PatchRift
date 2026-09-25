from io import BytesIO

import numpy as np
import pytest

from patchrift.ui.io import parse_footprint_corners, parse_utc_timestamp, read_scalar_image


def test_ui_reads_scalar_npy_without_changing_shape():
    source = np.arange(20, dtype=np.float32).reshape(4, 5)
    payload = BytesIO()
    np.save(payload, source)

    actual = read_scalar_image("lunar.npy", payload.getvalue())

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, source)


def test_ui_rejects_hyperspectral_npy_with_actionable_message():
    payload = BytesIO()
    np.save(payload, np.zeros((3, 4, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="scalar band"):
        read_scalar_image("cube.npy", payload.getvalue())


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_ui_reads_grayscale_png_without_changing_pixel_values(dtype):
    pytest.importorskip("PIL")
    from PIL import Image

    source = np.arange(20, dtype=dtype).reshape(4, 5)
    payload = BytesIO()
    Image.fromarray(source).save(payload, format="PNG")

    actual = read_scalar_image("lunar.png", payload.getvalue())

    assert actual.shape == source.shape
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, source.astype(np.float32))


def test_ui_rejects_color_png_instead_of_silently_grayscaling():
    pytest.importorskip("PIL")
    from PIL import Image

    source = np.zeros((4, 5, 3), dtype=np.uint8)
    payload = BytesIO()
    Image.fromarray(source).save(payload, format="PNG")

    with pytest.raises(ValueError, match="RGB/RGBA images are not converted"):
        read_scalar_image("color.png", payload.getvalue())


def test_footprint_parser_keeps_lon_lat_and_array_corner_order():
    corners = parse_footprint_corners("10, 20\n11, 20\n11, 21\n10, 21")
    np.testing.assert_array_equal(corners, [[10, 20], [11, 20], [11, 21], [10, 21]])


@pytest.mark.parametrize("text", ["1,2\n3,4\n5,6", "1,2\n3,4\n5,6\n7,no"])
def test_footprint_parser_rejects_incomplete_or_non_numeric_corners(text):
    with pytest.raises(ValueError):
        parse_footprint_corners(text)


def test_timestamp_parser_interprets_naive_values_as_utc():
    timestamp = parse_utc_timestamp("2025-01-15T12:30:00")
    assert timestamp.utcoffset().total_seconds() == 0
