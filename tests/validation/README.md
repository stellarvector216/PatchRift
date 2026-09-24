# V3 formal validation crosswalk: T1–T36

The IDs below follow the main V3 document's §10 validation plan. Addenda are
applied according to their precedence and included where they modify a case.
Stage III–V cases T22–T36 have dedicated executable tests in
`test_v3_t22_t36.py`; earlier cases map to the tests listed below.

| ID | Main-document validation case | Executable coverage |
|---|---|---|
| T1 | Shadow-side verification | `tests/test_shadow_vetting.py` centroid azimuth direction cases |
| T2 | Image handedness | `tests/test_handedness.py` |
| T3 | Gradient sign | `tests/test_scharr.py::test_gradient_orientation_points_into_shadow` |
| T4 | Contour cancellation | `tests/test_scharr.py::test_scharr_azimuth_resultant_cancels_opposite_directions` |
| T5 | Canonicalization sign | `tests/test_rotation.py::test_canonical_frame_rotation_sign_round_trips_native_displacement` |
| T6 | Light-centroid recovery | `tests/test_shadow_vetting.py::test_light_centroid_weighted_position` |
| T7 | Precision calibration with erosion/dilation | `tests/test_shadow_vetting.py` positional and angular uncertainty cases |
| T8 | Sector shape law | `tests/test_shadow_vetting.py` sector-ratio table and inversion cases |
| T9 | Axis-symmetry weighting | `tests/test_shadow_vetting.py` symmetry-weight formula and boundary cases |
| T10 | Border exclusion | `tests/test_shadow_vetting.py::test_vet_components_by_border` |
| T11 | Empty ensemble fallback | `tests/test_stage_ii.py::test_stage_ii_uses_scharr_only_when_crater_ensemble_is_empty` |
| T12 | Moderate outlier | `tests/test_fusion.py` robust fusion cluster/outlier cases |
| T13 | Strong outlier | `tests/test_fusion.py::test_robust_circular_fusion_rejects_large_outlier` |
| T14 | Scale floor | `tests/test_fusion.py::test_irls_scale_respects_scale_floor` |
| T15 | Fresh weighting | `tests/test_fusion.py` IRLS weight update and robust fusion cases |
| T16 | Bimodal fallback | `tests/test_fusion.py` zero-weight and low-confidence cases |
| T17 | Seed robustness | `tests/test_fusion.py` robust circular fusion convergence/outlier cases |
| T18 | Standard-error calibration | `tests/test_fusion.py` effective sample size and azimuth uncertainty cases |
| T19 | Azimuth bias against elevation | `tests/test_stage_i_ii_integrations.py` geometry and footprint diagnostic cases |
| T20 | Slope diagnostic | `tests/test_stage_ii.py::test_crater_slope_diagnostic_uses_tile_solar_elevation` |
| T21 | Azimuth against footprint | `tests/test_stage_i_ii_integrations.py` footprint angle/disagreement integration case |
| T22 | Patch-size bound | `tests/validation/test_v3_t22_t36.py::test_t22_patch_size_respects_both_sampling_bounds` |
| T23 | Filter-bank coverage, DC, quadrature and bandwidth | `tests/validation/test_v3_t22_t36.py::test_t23_filter_bank_has_dc_rejection_coverage_and_quadrature_response` |
| T24 | Rayleigh noise floor | `tests/validation/test_v3_t22_t36.py::test_t24_rayleigh_noise_floor_keeps_phase_congruency_finite_and_bounded` |
| T25 | Same physical patch ground extent | `tests/validation/test_v3_t22_t36.py::test_t25_native_and_target_grid_preserve_physical_spacing` and Stage III grid tests |
| T26 | Equal-alpha interpolation symmetry; affine reproduction regression | `tests/validation/test_v3_t22_t36.py::test_t26_equal_canonicalization_has_equal_per_scale_amplitude_ratios` and `test_lanczos3_interpolation_reproduces_affine_ramp` |
| T27 | IIRS reduction order, band inclusion and single-band behavior | `tests/validation/test_v3_t22_t36.py::test_t27_iirs_band_order_is_irrelevant_to_reduction` and `tests/test_iirs.py` |
| T28 | Canonical coordinate round-trip and Eq. 72 | `tests/validation/test_v3_t22_t36.py::test_t28_canonical_coordinate_transform_matches_scale_rotation_translation` and `tests/test_stage_iii_geometry.py` |
| T29 | Detector repeatability and Mmin versus raw response | `tests/validation/test_v3_t22_t36.py::test_t29_detector_repeatability_and_subpixel_relocalization_are_deterministic` |
| T30 | Reliability precision/recall weighting | `tests/validation/test_v3_t22_t36.py::test_t30_reliability_weights_affect_rescore_after_unweighted_retrieval` |
| T31 | Dense rotations | `tests/validation/test_v3_t22_t36.py::test_t31_dense_rotation_search_covers_the_full_orientation_ring` |
| T32 | One-point Hough versus four-point RANSAC | `tests/validation/test_v3_t22_t36.py::test_t32_hough_consensus_rejects_translation_outliers` |
| T33 | Solar-separation envelope | `tests/validation/test_v3_t22_t36.py::test_t33_solar_rotation_uncertainty_expands_orientation_search_window` |
| T34 | Parallax threshold | `tests/validation/test_v3_t22_t36.py::test_t34_parallax_sets_hough_threshold_with_three_pixel_floor` |
| T35 | Corrupt α by 30° and C4 fallback | `tests/validation/test_v3_t22_t36.py::test_t35_corrupt_rotation_uses_full_ring_c4_fallback` |
| T36 | Report inliers, RMS metres, σΔα and τ | `tests/validation/test_v3_t22_t36.py::test_t36_stage_v_result_exposes_reporting_fields` |

Run the complete suite with `pytest`. The tests use deterministic synthetic
inputs or injectable geometry; mission imagery and SPICE kernels are not
required for this validation subset.
