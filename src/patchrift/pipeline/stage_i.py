"""Stage I - radiometric preparation (§4)."""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from patchrift.io import IIRSProduct, ImageProduct, ProductMetadata, ingest_array
from patchrift.preprocessing.destripe import destripe
from patchrift.preprocessing.iirs import reduce_iirs_to_panchromatic, tmc_spectral_overlap_weights


@dataclass(frozen=True)
class StageIResult:
    """The scalar, prepared image consumed by Stage II."""

    product: ImageProduct
    prepared_image: np.ndarray
    planarity_flag: bool

    def __post_init__(self) -> None:
        if self.prepared_image.shape != self.product.image.shape:
            raise ValueError("prepared_image must match product image shape")


def prepare_image_product(product: ImageProduct) -> StageIResult:
    """Apply §4.3 to an ingested scalar OHRC or TMC product."""
    prepared = destripe(product.image, product.valid_mask)
    return StageIResult(
        product=product,
        prepared_image=prepared,
        planarity_flag=bool(
            product.metadata.emission_angle > np.deg2rad(15.0)
        ),
    )

def prepare_iirs_product(
    product: IIRSProduct,
    fill_value: float | int | tuple[float | int, ...] | None,
    metadata: ProductMetadata,
    spectral_weights: np.ndarray | None = None,
    spectral_response: Callable[[float], float] | tuple[np.ndarray, np.ndarray] | None = None,
    response_domain: tuple[float, float] | None = None,
) -> StageIResult:
    """Apply §4.2 followed by the normal scalar-image Stage I path.

    ``spectral_weights`` must be supplied from the externally sourced
    TMC-response overlap when that response is available.  Omitting it
    intentionally selects the specification's SNR fallback.
    """
    if spectral_weights is not None and spectral_response is not None:
        raise ValueError("supply spectral_weights or spectral_response, not both")
    weights = spectral_weights
    if spectral_response is not None:
        weights = tmc_spectral_overlap_weights(
            product, spectral_response, response_domain=response_domain
        )
        if not np.any(weights > 0.0):
            raise ValueError("TMC response has no overlap with any IIRS band")
    scalar = reduce_iirs_to_panchromatic(product, weights=weights)
    ingested = ingest_array(scalar, fill_value=fill_value, metadata=metadata)
    return prepare_image_product(ingested)
