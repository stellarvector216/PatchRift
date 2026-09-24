from datetime import datetime

import numpy as np

from patchrift.io import ImageProduct
from patchrift.io.product import ProductMetadata
from patchrift.pipeline.stage_i import prepare_image_product


def test_stage_i_keeps_the_ingest_validity_mask():
    metadata = ProductMetadata(
        timestamp=datetime(2026, 1, 1),
        footprint_corners=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        footprint_pixel_coords=np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float),
        gsd=1.0,
        emission_angle=0.0,
    )
    product = ImageProduct(
        image=np.array([[1.0, -999.0], [1.0, 1.0]], dtype=np.float32),
        valid_mask=np.array([[True, False], [True, True]]),
        metadata=metadata,
    )

    result = prepare_image_product(product)

    np.testing.assert_array_equal(result.product.valid_mask, product.valid_mask)
    assert result.prepared_image.shape == product.image.shape
