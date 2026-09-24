"""Descriptor matching and geometric verification for Stage V (§8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class StageVResult:
    candidate_pairs: np.ndarray
    pairs: np.ndarray
    orientation_shifts: np.ndarray
    orientation_shift_histogram: np.ndarray
    distances: np.ndarray
    inlier_mask: np.ndarray
    translation: np.ndarray | None
    homography: np.ndarray | None
    hough_peak_fraction: float
    hough_bin_size: float
    inlier_count: int
    rms_residual_pixels: float | None
    rms_residual_meters: float | None
    delta_alpha: float
    sigma_delta_alpha: float
    aligned: bool
    fallback_used: bool


def calculate_search_window(sigma_delta_alpha, N_o=12):
    if not np.isfinite(sigma_delta_alpha) or sigma_delta_alpha < 0 or N_o <= 0:
        raise ValueError("sigma_delta_alpha must be nonnegative and N_o positive")
    bin_width = np.pi / N_o
    limit = int(np.ceil((2.0 * sigma_delta_alpha) / bin_width))
    return np.arange(-limit, limit + 1)


def roll_descriptor(descriptor, shift, J=17, N_o=12):
    descriptor = np.asarray(descriptor)
    if descriptor.size != J * N_o:
        raise ValueError(f"descriptor must have {J * N_o} values")
    return np.roll(descriptor.reshape(J, N_o), int(shift), axis=1).ravel()


def compute_weighted_distance(D_A, D_B, v_A, v_B, J=17, N_o=12, epsilon=1e-8):
    a, b = np.asarray(D_A, dtype=float), np.asarray(D_B, dtype=float)
    va, vb = np.asarray(v_A, dtype=float), np.asarray(v_B, dtype=float)
    if a.size != J * N_o or b.size != J * N_o or va.size != J or vb.size != J:
        raise ValueError("descriptor or reliability dimensions do not match J and N_o")
    if not np.isfinite(epsilon) or epsilon <= 0 or np.any(va < 0) or np.any(vb < 0):
        raise ValueError("epsilon must be positive and reliability weights nonnegative")
    if not all(np.all(np.isfinite(values)) for values in (a, b, va, vb)):
        raise ValueError("descriptors and reliability weights must be finite")
    a, b = a.reshape(J, N_o), b.reshape(J, N_o)
    prod = va.reshape(J) * vb.reshape(J)
    return float(np.sum(prod * np.sum((a - b) ** 2, axis=1)) / (np.sum(prod) + epsilon))


def map_to_canonical_frame(x, y, M_X):
    canonical = np.linalg.solve(np.asarray(M_X, dtype=float), [x, y])
    return float(canonical[0]), float(canonical[1])


def calculate_hough_bin_size(h_max, e_A, e_B, g_star):
    if not all(np.isfinite(value) for value in (h_max, e_A, e_B, g_star)) or g_star <= 0 or h_max < 0:
        raise ValueError("g_star must be positive and h_max nonnegative")
    # All angles, including metadata emission angles, are radians internally (§2.2).
    tau_relief = (h_max * abs(np.tan(e_A) - np.tan(e_B))) / g_star
    return max(3.0, float(tau_relief))


def project_homography(H, x_A):
    points = np.asarray(x_A, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (np.asarray(H, dtype=float) @ homogeneous.T).T
    denom = projected[:, 2:3]
    return np.divide(projected[:, :2], denom, out=np.full_like(projected[:, :2], np.inf), where=np.abs(denom) > 1e-12)


def eq71_residuals(h_elements, x_A, x_B):
    H = np.append(h_elements, 1.0).reshape(3, 3)
    return (np.asarray(x_B) - project_homography(H, np.asarray(x_A))).ravel()


def fit_huber_homography(x_A_inliers, x_B_inliers, H_init=None):
    x_A, x_B = np.asarray(x_A_inliers, dtype=float), np.asarray(x_B_inliers, dtype=float)
    if x_A.shape != x_B.shape or x_A.ndim != 2 or x_A.shape[1] != 2 or len(x_A) < 4:
        raise ValueError("homography refinement requires at least four paired 2D points")
    _validate_homography_point_cloud(x_A, "source")
    _validate_homography_point_cloud(x_B, "destination")
    if H_init is None:
        H_init = _fit_dlt(x_A, x_B)
    H_init = np.asarray(H_init, dtype=float)
    if H_init.shape != (3, 3) or not np.isfinite(H_init).all() or abs(H_init[2, 2]) < 1e-12:
        raise ValueError("H_init must be a finite nonsingular 3x3 matrix with H[2,2] nonzero")
    if abs(np.linalg.det(H_init)) < 1e-12:
        raise ValueError("H_init must be nonsingular")
    flat = (H_init / H_init[2, 2]).ravel()[:8]
    result = least_squares(eq71_residuals, flat, args=(x_A, x_B), loss="huber", f_scale=1.0)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Huber homography refinement failed: {result.message}")
    return np.append(result.x, 1.0).reshape(3, 3)


def _validate_homography_point_cloud(points, label):
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} homography points must be finite")
    if len(np.unique(points, axis=0)) < 4:
        raise ValueError(f"{label} homography points must contain four distinct locations")
    centered = points - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 2 or singular[0] <= 0.0 or singular[1] / singular[0] < 1e-8:
        raise ValueError(f"{label} homography points are collinear or ill-conditioned")


def compute_output_transformation(x_A, y_A, t_x, t_y, g_A, g_B, delta_alpha, M_B):
    if g_A <= 0 or g_B <= 0:
        raise ValueError("GSD values must be positive")
    rotation = np.array([[np.cos(delta_alpha), -np.sin(delta_alpha)], [np.sin(delta_alpha), np.cos(delta_alpha)]])
    result = (g_A / g_B) * (rotation @ np.array([x_A, y_A], dtype=float)) + np.asarray(M_B) @ np.array([t_x, t_y])
    return float(result[0]), float(result[1])


def self_diagnose_alignment(
    num_inliers,
    N_min=10,
    *,
    hough_peak_fraction=None,
    minimum_hough_peak_fraction=None,
):
    """Apply §8.4's count and configured Hough-mass diagnosis gates."""
    if int(num_inliers) < int(N_min):
        return False
    if minimum_hough_peak_fraction is not None:
        if not 0.0 <= minimum_hough_peak_fraction <= 1.0:
            raise ValueError("minimum_hough_peak_fraction must lie in [0, 1]")
        if hough_peak_fraction is None or not np.isfinite(hough_peak_fraction):
            return False
        if hough_peak_fraction < minimum_hough_peak_fraction:
            return False
    return True


def calculate_fallback_search_window(N_o=12):
    if N_o <= 0:
        raise ValueError("N_o must be positive")
    return np.arange(N_o)


class UnweightedANNIndex:
    """ANN-style L2 shortlist index over unweighted descriptor vectors.

    The cKDTree uses only descriptor coordinates. Reliability is deliberately
    absent here and enters only in the subsequent pairwise Eq. (67) rescore.
    ``eps`` controls the approximate query tolerance and defaults to 0.1.
    """

    def __init__(self, descriptors, *, eps=0.1):
        vectors = np.asarray(descriptors, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.all(np.isfinite(vectors)):
            raise ValueError("ANN descriptors must be finite 2D vectors with nonzero width")
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("ANN eps must be finite and nonnegative")
        self.descriptors = vectors
        self.eps = float(eps)
        self._tree = cKDTree(vectors) if len(vectors) else None

    def query(self, descriptors, top_m):
        queries = np.asarray(descriptors, dtype=float)
        if queries.ndim != 2 or queries.shape[1] != self.descriptors.shape[1] or not np.all(np.isfinite(queries)):
            raise ValueError("ANN queries must be finite and match the indexed descriptor width")
        if top_m <= 0:
            raise ValueError("top_m must be positive")
        if self._tree is None:
            return np.empty((len(queries), 0), dtype=int)
        k = min(int(top_m), len(self.descriptors))
        _, indexes = self._tree.query(queries, k=k, eps=self.eps, workers=1)
        if k == 1:
            indexes = np.asarray(indexes).reshape((-1, 1))
        return np.asarray(indexes, dtype=int)


def build_unweighted_ann_shortlist(descriptors_a, descriptors_b, *, top_m=20, ann_eps=0.1):
    """Return the bidirectional top-M candidate union from unweighted L2 ANN."""
    a, b = np.asarray(descriptors_a, dtype=float), np.asarray(descriptors_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("ANN descriptor matrices must have matching 2D shapes")
    if top_m <= 0:
        raise ValueError("top_m must be positive")
    if not len(a) or not len(b):
        return np.empty((0, 2), dtype=int)
    a_to_b = UnweightedANNIndex(b, eps=ann_eps).query(a, top_m)
    b_to_a = UnweightedANNIndex(a, eps=ann_eps).query(b, top_m)
    retrieved_by_a = np.zeros((len(a), len(b)), dtype=bool)
    retrieved_by_b = np.zeros((len(a), len(b)), dtype=bool)
    for i, candidates in enumerate(a_to_b):
        retrieved_by_a[i, candidates] = True
    for j, candidates in enumerate(b_to_a):
        retrieved_by_b[candidates, j] = True
    return np.argwhere(retrieved_by_a | retrieved_by_b)


def match_feature_sets(
    features_a,
    features_b,
    *,
    delta_alpha,
    sigma_delta_alpha,
    N_o=12,
    ratio=0.8,
    hough_bin=3.0,
    minimum_inliers=10,
    top_m=20,
    ann_eps=0.1,
    minimum_hough_peak_fraction: float | None = None,
    gsd_target=1.0,
):
    """Run §8 orientation search, mutual ratio matching, and translation Hough.

    Returned point coordinates are canonical-frame coordinates from Stage IV.
    Each descriptor is compared after cyclic orientation shifts only; spatial
    cells remain fixed as required by §8.1.
    """
    if not 0.0 < ratio < 1.0 or hough_bin <= 0:
        raise ValueError("ratio must lie in (0,1) and hough_bin must be positive")
    if np.isnan(sigma_delta_alpha) or sigma_delta_alpha < 0.0 or not np.isfinite(delta_alpha):
        raise ValueError("rotation and its uncertainty must be finite, with nonnegative uncertainty")
    if N_o <= 0 or minimum_inliers < 1:
        raise ValueError("N_o and minimum_inliers must be positive")
    if not np.isfinite(gsd_target) or gsd_target <= 0.0:
        raise ValueError("gsd_target must be finite and positive")
    if top_m <= 0:
        raise ValueError("top_m must be positive")
    if minimum_hough_peak_fraction is not None and (
        not np.isfinite(minimum_hough_peak_fraction)
        or not 0.0 <= minimum_hough_peak_fraction <= 1.0
    ):
        raise ValueError("minimum_hough_peak_fraction must lie in [0, 1]")
    fallback_used = np.isinf(sigma_delta_alpha)
    active_delta_alpha = 0.0 if fallback_used else float(delta_alpha)
    shifts = (
        calculate_fallback_search_window(N_o)
        if fallback_used
        else calculate_search_window(sigma_delta_alpha, N_o)
    )
    candidate_pairs = build_unweighted_ann_shortlist(
        features_a.descriptors, features_b.descriptors, top_m=top_m, ann_eps=ann_eps
    )
    candidate_mask = _candidate_mask(len(features_a.descriptors), len(features_b.descriptors), candidate_pairs)
    candidates, shift_map = _distance_matrix(features_a, features_b, shifts, N_o, candidate_mask)
    pairs, shift_indices, distances = _mutual_ratio_matches(candidates, shift_map, ratio)
    chosen_shifts = shifts[shift_indices] if len(shift_indices) else np.empty(0, dtype=int)
    if len(pairs) == 0:
        if fallback_used:
            return _empty_stage_v(active_delta_alpha, sigma_delta_alpha, True, N_o)
        fallback_used = True
        active_delta_alpha = 0.0
        shifts = calculate_fallback_search_window(N_o)
        candidates, shift_map = _distance_matrix(features_a, features_b, shifts, N_o, candidate_mask)
        pairs, shift_indices, distances = _mutual_ratio_matches(candidates, shift_map, ratio)
        chosen_shifts = shifts[shift_indices] if len(shift_indices) else np.empty(0, dtype=int)
        if not len(pairs):
            return _empty_stage_v(active_delta_alpha, sigma_delta_alpha, True, N_o)

    positions_a = np.asarray(features_a.canonical_positions, dtype=float)[pairs[:, 0]]
    positions_b = np.asarray(features_b.canonical_positions, dtype=float)[pairs[:, 1]]
    # §8.1's cyclic shift changes descriptor orientation bins only. §8.2's
    # canonical-frame coordinates vote directly for the shared translation.
    translations = positions_b - positions_a
    inliers, translation, peak_fraction = _translation_consensus(
        translations, hough_bin, distances
    )

    if int(inliers.sum()) < minimum_inliers or (
        minimum_hough_peak_fraction is not None and peak_fraction < minimum_hough_peak_fraction
    ):
        fallback_used = True
        active_delta_alpha = 0.0
        fallback = calculate_fallback_search_window(N_o)
        candidate_pairs = build_unweighted_ann_shortlist(
            features_a.descriptors, features_b.descriptors, top_m=top_m, ann_eps=ann_eps
        )
        candidate_mask = _candidate_mask(len(features_a.descriptors), len(features_b.descriptors), candidate_pairs)
        all_candidates, shift_map = _distance_matrix(features_a, features_b, fallback, N_o, candidate_mask)
        pairs, shift_indices, distances = _mutual_ratio_matches(all_candidates, shift_map, ratio)
        chosen_shifts = fallback[shift_indices] if len(shift_indices) else np.empty(0, dtype=int)
        if not len(pairs):
            return _empty_stage_v(active_delta_alpha, sigma_delta_alpha, True, N_o)
        positions_a = np.asarray(features_a.canonical_positions, dtype=float)[pairs[:, 0]]
        positions_b = np.asarray(features_b.canonical_positions, dtype=float)[pairs[:, 1]]
        translations = positions_b - positions_a
        inliers, translation, peak_fraction = _translation_consensus(
            translations, hough_bin, distances
        )

    aligned = self_diagnose_alignment(
        int(inliers.sum()), minimum_inliers,
        hough_peak_fraction=peak_fraction,
        minimum_hough_peak_fraction=minimum_hough_peak_fraction,
    )
    homography = None
    rms_pixels = None
    rms_meters = None
    if aligned and int(inliers.sum()) >= 4:
        initial = np.array(
            [[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]], [0.0, 0.0, 1.0]]
        )
        try:
            homography = fit_huber_homography(positions_a[inliers], positions_b[inliers], initial)
            residuals = project_homography(homography, positions_a[inliers]) - positions_b[inliers]
            rms_pixels = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
            rms_meters = rms_pixels * float(gsd_target)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Translation consensus is not a valid projective fit when the
            # inlier geometry is degenerate. Fail closed for pair alignment.
            homography = None
            aligned = False
    return StageVResult(
        candidate_pairs=candidate_pairs,
        pairs=pairs,
        orientation_shifts=chosen_shifts,
        orientation_shift_histogram=np.bincount(
            np.mod(chosen_shifts, N_o), minlength=N_o
        ).astype(np.int64),
        distances=distances,
        inlier_mask=inliers,
        translation=translation,
        homography=homography,
        hough_peak_fraction=peak_fraction,
        hough_bin_size=float(hough_bin),
        inlier_count=int(inliers.sum()),
        rms_residual_pixels=rms_pixels,
        rms_residual_meters=rms_meters,
        delta_alpha=float(active_delta_alpha),
        sigma_delta_alpha=float(sigma_delta_alpha),
        aligned=aligned,
        fallback_used=fallback_used,
    )


def _translation_consensus(translations, bin_size, distances):
    """Peak-bin plus its 8 neighbours, then a match-score weighted centroid."""
    bins = np.floor(np.asarray(translations, dtype=float) / bin_size).astype(np.int64)
    unique, counts = np.unique(bins, axis=0, return_counts=True)
    peak = unique[int(np.argmax(counts))]
    inliers = np.all(np.abs(bins - peak[None, :]) <= 1, axis=1)
    # §8.2 defines the peak neighbourhood (peak bin and immediate neighbours)
    # as the inlier vote mass used by the §8.4 self-diagnosis.
    peak_fraction = float(np.sum(inliers) / len(bins)) if len(bins) else 0.0
    weights = 1.0 / (np.asarray(distances, dtype=float)[inliers] + 1e-8)
    weights /= np.sum(weights)
    translation = np.sum(np.asarray(translations, dtype=float)[inliers] * weights[:, None], axis=0)
    return inliers, translation, peak_fraction


def _distance_matrix(a, b, shifts, N_o, candidate_mask=None):
    desc_a, desc_b = np.asarray(a.descriptors), np.asarray(b.descriptors)
    rel_a, rel_b = np.asarray(a.reliability), np.asarray(b.reliability)
    if desc_a.ndim != 2 or desc_b.ndim != 2 or desc_a.shape[1] != desc_b.shape[1]:
        raise ValueError("feature descriptor arrays must have matching 2D shapes")
    if not np.all(np.isfinite(desc_a)) or not np.all(np.isfinite(desc_b)):
        raise ValueError("feature descriptors must be finite")
    J = desc_a.shape[1] // N_o
    if J * N_o != desc_a.shape[1] or rel_a.shape != (len(desc_a), J) or rel_b.shape != (len(desc_b), J):
        raise ValueError("descriptor/reliability dimensions do not match N_o")
    if not np.all(np.isfinite(rel_a)) or not np.all(np.isfinite(rel_b)) or np.any(rel_a < 0.0) or np.any(rel_b < 0.0):
        raise ValueError("reliability weights must be finite and nonnegative")
    for feature, expected in ((a, len(desc_a)), (b, len(desc_b))):
        positions = np.asarray(feature.canonical_positions)
        if positions.shape != (expected, 2) or not np.all(np.isfinite(positions)):
            raise ValueError("canonical_positions must be a finite (feature_count, 2) array")
    a_cells = desc_a.reshape(len(desc_a), J, N_o)
    b_cells = desc_b.reshape(len(desc_b), J, N_o)
    if candidate_mask is None:
        candidate_mask = np.ones((len(desc_a), len(desc_b)), dtype=bool)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != (len(desc_a), len(desc_b)):
            raise ValueError("candidate_mask must match the feature-pair matrix")
    best = np.full((len(desc_a), len(desc_b)), np.inf)
    best_shift = np.zeros((len(desc_a), len(desc_b)), dtype=int)
    for i in range(len(desc_a)):
        js = np.flatnonzero(candidate_mask[i])
        if not len(js):
            continue
        for si, shift in enumerate(shifts):
            shifted = np.roll(b_cells[js], int(shift), axis=2)
            numerator = np.zeros(len(js), dtype=float)
            denominator = np.zeros(len(js), dtype=float)
            for cell in range(J):
                weight = rel_a[i, cell] * rel_b[js, cell]
                sq_diff = np.sum((a_cells[i, cell][None, :] - shifted[:, cell, :]) ** 2, axis=1)
                numerator += weight * sq_diff
                denominator += weight
            distance = np.full(len(js), np.inf)
            supported = denominator > 1e-8
            distance[supported] = numerator[supported] / (denominator[supported] + 1e-8)
            improve = distance < best[i, js]
            best[i, js[improve]] = distance[improve]
            best_shift[i, js[improve]] = si
    return best, best_shift


def _candidate_mask(count_a, count_b, pairs):
    mask = np.zeros((count_a, count_b), dtype=bool)
    pairs = np.asarray(pairs, dtype=int).reshape((-1, 2))
    if len(pairs):
        mask[pairs[:, 0], pairs[:, 1]] = True
    return mask


def _mutual_ratio_matches(distances, shift_indices, ratio):
    if distances.shape[0] == 0 or distances.shape[1] == 0:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int), np.empty(0)
    collapsed = distances
    nearest_b = np.argmin(collapsed, axis=1)
    nearest_a = np.argmin(collapsed, axis=0)
    selected = []
    for i, j in enumerate(nearest_b):
        if nearest_a[j] != i:
            continue
        row = np.sort(collapsed[i])
        if not np.isfinite(row[0]):
            continue
        if len(row) > 1 and np.sqrt(row[0] / row[1]) >= ratio:
            continue
        si = int(shift_indices[i, j])
        selected.append((i, j, si, float(collapsed[i, j])))
    return (
        np.asarray([(i, j) for i, j, _, _ in selected], dtype=int).reshape((-1, 2)),
        np.asarray([s for _, _, s, _ in selected], dtype=int),
        np.asarray([d for _, _, _, d in selected], dtype=float),
    )


def _fit_dlt(a, b):
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.extend(([-x, -y, -1, 0, 0, 0, u*x, u*y, u], [0, 0, 0, -x, -y, -1, v*x, v*y, v]))
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("homography fit is degenerate")
    return H / H[2, 2]


def _empty_stage_v(delta_alpha, sigma_delta_alpha, fallback, N_o=12):
    empty_pairs = np.empty((0, 2), dtype=int)
    return StageVResult(
        candidate_pairs=empty_pairs,
        pairs=empty_pairs.copy(),
        orientation_shifts=np.empty(0, dtype=int),
        orientation_shift_histogram=np.zeros(N_o, dtype=np.int64),
        distances=np.empty(0),
        inlier_mask=np.empty(0, dtype=bool),
        translation=None,
        homography=None,
        hough_peak_fraction=0.0,
        hough_bin_size=float("nan"),
        inlier_count=0,
        rms_residual_pixels=None,
        rms_residual_meters=None,
        delta_alpha=float(delta_alpha),
        sigma_delta_alpha=float(sigma_delta_alpha),
        aligned=False,
        fallback_used=fallback,
    )
