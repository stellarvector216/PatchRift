"""Input helpers kept separate from the optional Streamlit application."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np


def read_scalar_image(filename: str, payload: bytes) -> np.ndarray:
    """Read a scalar NumPy array, grayscale PNG, or single-band TIFF."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy":
        image = np.load(BytesIO(payload), allow_pickle=False)
    elif suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:  # pragma: no cover - install extra for UI use
            raise RuntimeError("TIFF uploads require the 'tifffile' UI dependency") from exc
        image = tifffile.imread(BytesIO(payload))
    elif suffix == ".png":
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Streamlit UI extra includes Pillow
            raise RuntimeError("PNG uploads require Pillow; install the 'ui' extra") from exc
        with Image.open(BytesIO(payload)) as png:
            if png.mode not in {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}:
                raise ValueError(
                    f"PNG mode {png.mode!r} is not a grayscale scalar image. "
                    "Upload a grayscale PNG; RGB/RGBA images are not converted automatically."
                )
            image = np.asarray(png)
    else:
        raise ValueError("Upload a .npy, grayscale .png, or single-band .tif/.tiff image")

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            "The prototype accepts one scalar band per image. Reduce hyperspectral "
            "IIRS cubes through the Stage-I spectral reduction before uploading."
        )
    if image.size == 0 or not np.any(np.isfinite(image)):
        raise ValueError("Image must contain at least one finite pixel")
    return image.astype(np.float32, copy=False)


def parse_footprint_corners(text: str) -> np.ndarray:
    """Parse image-array corners in (0,0), (W-1,0), (W-1,H-1), (0,H-1) order."""
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 2:
            raise ValueError("Each footprint row must contain longitude, latitude")
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except ValueError as exc:
            raise ValueError("Footprint coordinates must be numeric degrees") from exc
    corners = np.asarray(rows, dtype=float)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("Enter exactly four finite lon,lat coordinate rows")
    if np.any(np.abs(corners[:, 1]) > 90.0):
        raise ValueError("Footprint latitudes must be within -90 to 90 degrees")
    return corners


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive timestamps are interpreted as UTC."""
    text = value.strip()
    if not text:
        raise ValueError("Acquisition timestamp is required")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("Use an ISO-8601 timestamp, for example 2025-01-15T12:30:00Z") from exc
    if timestamp.tzinfo is None:
        from datetime import timezone

        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp
