"""Adjacent-resolution transformation composition for the §1 cascade."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from patchrift.pipeline.stages_iii_v import PairCorrespondenceResult


@dataclass(frozen=True)
class CascadeCorrespondenceResult:
    """Two hop-level results plus their end-to-end transform and error."""

    source_to_intermediate: PairCorrespondenceResult
    intermediate_to_target: PairCorrespondenceResult
    composed_homography: np.ndarray | None
    hop1_residual_meters: float | None
    hop2_residual_meters: float | None
    end_to_end_residual_meters: float | None
    first_hop_scale_error: float | None
    second_hop_extent_meters: float
    aligned: bool


def match_two_hop_cascade(
    hop1_inputs: dict,
    hop2_inputs: dict,
    *,
    hop1_gsd_target: float,
    hop2_gsd_target: float,
    second_hop_extent_meters: float,
    first_hop_scale_error: float | None = None,
) -> CascadeCorrespondenceResult:
    """Run and preserve both adjacent-resolution matches, then compose.

    ``hop1_inputs`` and ``hop2_inputs`` are keyword-argument mappings for
    :func:`match_tile_pair`, typically OHRC→TMC and TMC→IIRS respectively.
    """
    from patchrift.pipeline.stages_iii_v import match_tile_pair

    hop1 = match_tile_pair(**hop1_inputs)
    hop2 = match_tile_pair(**hop2_inputs)
    return compose_cascade_correspondences(
        hop1,
        hop2,
        hop1_gsd_target=hop1_gsd_target,
        hop2_gsd_target=hop2_gsd_target,
        second_hop_extent_meters=second_hop_extent_meters,
        first_hop_scale_error=first_hop_scale_error,
    )


def compose_cascade_correspondences(
    source_to_intermediate: PairCorrespondenceResult,
    intermediate_to_target: PairCorrespondenceResult,
    *,
    hop1_gsd_target: float,
    hop2_gsd_target: float,
    second_hop_extent_meters: float,
    first_hop_scale_error: float | None = None,
) -> CascadeCorrespondenceResult:
    """Compose adjacent-hop homographies according to V3 Eqs. (45)–(46).

    Hop homographies operate in pixel coordinates at each hop's target GSD.
    The intermediate coordinate is converted between those two pixel grids
    before matrix composition. Both original hop results remain attached.
    """
    values = (hop1_gsd_target, hop2_gsd_target, second_hop_extent_meters)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("GSDs and second-hop extent must be finite")
    if hop1_gsd_target <= 0.0 or hop2_gsd_target <= 0.0 or second_hop_extent_meters < 0.0:
        raise ValueError("GSDs must be positive and extent non-negative")

    first = source_to_intermediate.matching
    second = intermediate_to_target.matching
    eps1, eps2 = first.rms_residual_meters, second.rms_residual_meters
    H1, H2 = first.homography, second.homography
    if (
        not first.aligned or not second.aligned
        or H1 is None or H2 is None
        or eps1 is None or eps2 is None
        or not np.isfinite(eps1) or not np.isfinite(eps2)
    ):
        return CascadeCorrespondenceResult(
            source_to_intermediate, intermediate_to_target, None,
            eps1, eps2, None, first_hop_scale_error,
            float(second_hop_extent_meters), False,
        )

    H1, H2 = np.asarray(H1, dtype=float), np.asarray(H2, dtype=float)
    if H1.shape != (3, 3) or H2.shape != (3, 3) or not np.all(np.isfinite(H1)) or not np.all(np.isfinite(H2)):
        raise ValueError("hop homographies must be finite 3x3 matrices")
    if first_hop_scale_error is None:
        first_hop_scale_error = _residual_scale_error(H1)
    if not np.isfinite(first_hop_scale_error) or first_hop_scale_error < 0.0:
        raise ValueError("first_hop_scale_error must be finite and non-negative")

    mid_grid_conversion = np.diag(
        [hop1_gsd_target / hop2_gsd_target, hop1_gsd_target / hop2_gsd_target, 1.0]
    )
    composed = H2 @ mid_grid_conversion @ H1
    if abs(composed[2, 2]) < 1e-12:
        return CascadeCorrespondenceResult(
            source_to_intermediate, intermediate_to_target, None,
            float(eps1), float(eps2), None, float(first_hop_scale_error),
            float(second_hop_extent_meters), False,
        )
    composed = composed / composed[2, 2]

    # Eq. (46): independent hop residuals plus scale error propagated over
    # the spatial extent of the second-hop scene.
    end_to_end = float(np.sqrt(
        eps1**2 + eps2**2 + (first_hop_scale_error * second_hop_extent_meters) ** 2
    ))
    return CascadeCorrespondenceResult(
        source_to_intermediate, intermediate_to_target, composed,
        float(eps1), float(eps2), end_to_end,
        float(first_hop_scale_error), float(second_hop_extent_meters), True,
    )


def _residual_scale_error(homography):
    """Estimate fractional local scale error at the canonical origin."""
    H = np.asarray(homography, dtype=float)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError("homography must be a finite 3x3 matrix")
    h22 = H[2, 2]
    if abs(h22) < 1e-12:
        raise ValueError("homography is singular at the canonical origin")
    x = y = 0.0
    w = H[2, 0] * x + H[2, 1] * y + h22
    if abs(w) < 1e-12:
        raise ValueError("homography is singular at the canonical origin")
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    jacobian = np.array([
        [(H[0, 0] - u * H[2, 0]) / w, (H[0, 1] - u * H[2, 1]) / w],
        [(H[1, 0] - v * H[2, 0]) / w, (H[1, 1] - v * H[2, 1]) / w],
    ])
    scale = float(np.sqrt(abs(np.linalg.det(jacobian))))
    if not np.isfinite(scale):
        raise ValueError("homography scale estimate is non-finite")
    return abs(scale - 1.0)
