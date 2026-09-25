# V3 algorithm stage-to-code map

The V3 document defines five numbered stages in execution order: **§4 Stage I**,
**§5 Stage II**, **§6 Stage III**, **§7 Stage IV**, and **§8 Stage V**. The code
does not use five separate top-level stage folders; implementation is grouped
under `src/patchrift/` by technical domain. This map identifies the stage-owner
files and the files shared by multiple stages.

## Stage I — Radiometric preparation (§4)

**What it does:** reads the product, builds the no-data mask, normalizes image
handedness, reduces IIRS to a scalar band where needed, plans tiles, and removes
column striping and impulse noise.

**Main files:**

- `src/patchrift/pipeline/stage_i.py` — entry points
  `prepare_image_product()` and `prepare_iirs_product()`.
- `src/patchrift/io/product.py` — `ProductMetadata` and `ImageProduct` models.
- `src/patchrift/io/ingest.py` — array ingest, valid-pixel mask, and call to
  handedness normalization.
- `src/patchrift/geometry/handedness.py` — footprint-to-pixel fit and
  one-time reflection for mirrored frames (§2.3).
- `src/patchrift/io/iirs.py` — raw IIRS cube and band-metadata model.
- `src/patchrift/preprocessing/iirs.py` — IIRS band reduction and TMC-response
  overlap weighting (§4.2).
- `src/patchrift/preprocessing/destripe.py` — column destriping and
  impulse-noise median filter (§4.3).
- `src/patchrift/geometry/tiling.py` — `PixelTile`, footprint mapping, and
  `bisect_tiles_for_solar_azimuth()` (§4.1 / Addendum B1).

**Boundary detail:** adaptive tile planning is specified in Stage I §4.1, but
the runtime call is in `src/patchrift/pipeline/workflow.py`; that function has
the Stage-I product and the Stage-II solar-azimuth callbacks it needs. See the
Stage-II list for this orchestration file.

## Stage II — Solar geometry (§5)

**What it does:** computes ephemeris solar geometry, segments and vets shadows,
estimates per-crater azimuths, cross-checks them with Scharr gradients, fuses
measurements, and reports consistency diagnostics plus sensor-frame rotation
and uncertainty.

**Main files:**

- `src/patchrift/pipeline/workflow.py` — image-level workflow: adaptive tile
  plan, tile-centre geometry and derivatives, and per-tile Stage-II calls.
- `src/patchrift/pipeline/stage_ii.py` — `segment_shadows()`,
  `process_solar_tile()`, result models, and pair-level
  `pair_solar_geometry()`.
- `src/patchrift/geometry/spice.py` — SPICE kernel loading, solar vector,
  azimuth/elevation, finite-difference derivatives, and default derivative
  steps.
- `src/patchrift/geometry/solar.py` — solar/config models and usable-elevation
  gate.
- `src/patchrift/geometry/diagnostics.py` — footprint angle, median/MAD, slope,
  shadow-length elevation, and discrepancy diagnostics.
- `src/patchrift/geometry/rotation.py` — pixel north angle and pairwise
  inter-image rotation/uncertainty helpers.
- `src/patchrift/shadows/segmentation.py` — local Otsu and shadow threshold.
- `src/patchrift/shadows/thresholds.py`, `interpolation.py`, `tiles.py`,
  `variance.py` — subtile thresholds, valid interpolation cells, and
  low-variance exclusions.
- `src/patchrift/shadows/mask.py` and `morphology.py` — shadow masks,
  morphology, connected components, rings, and interiors.
- `src/patchrift/shadows/vetting.py` — component gates, crater geometry,
  centroid azimuth measurements, uncertainty, and component weights.
- `src/patchrift/shadows/scharr.py` — boundary-gradient azimuth cross-check and
  fallback.
- `src/patchrift/shadows/fusion.py` — robust circular fusion and fused
  uncertainty.

The `shadows/` package contains Stage-II subroutines; it is not an additional
pipeline stage.

## Stage III — Geometric conditioning (§6)

**What it does:** normalizes the image pair to the coarser working GSD, carries
masks through resampling, and removes the broad directional illumination trend
while preserving descriptor-scale terrain structure.

**Main files:**

- `src/patchrift/cascade/stage_iii.py` — `condition_image()`, scale
  normalization, mask resampling, and directional photometric conditioning.
- `src/patchrift/pipeline/stages_iii_v.py` —
  `extract_tile_features()` connects Stage-II tile results to Stage-III
  conditioning and then hands the result to Stage IV.

Here, `cascade/stage_iii.py` is the per-image Stage-III conditioning code. The
separate `pipeline/cascade.py` composes two-resolution correspondences; see
Stage V.

## Stage IV — Feature extraction (§7)

**What it does:** creates Log-Gabor responses, computes phase congruency,
detects and refines keypoints, extracts canonical patches, builds the Maximum
Index Map/log-polar descriptor, and computes reliability weights.

**Main files:**

- `src/patchrift/features/stage_iv.py` — filter bank, phase congruency, moment
  analysis, detector, subpixel refinement, canonical Lanczos sampling,
  descriptor/reliability calculation, and dense-keypoint formulation switch.
- `src/patchrift/features/__init__.py` — public feature API exports.
- `src/patchrift/utils/complexity.py` — §9 complexity estimate and the
  `KP² > N/2` patch-versus-full-image decision used by Stage IV.
- `src/patchrift/pipeline/stages_iii_v.py` — invokes feature detection and
  description for each selected tile.

## Stage V — Correspondence and verification (§8)

**What it does:** builds an unweighted descriptor shortlist, performs the
rotation-aware weighted match scoring, verifies translation by Hough voting,
refines inliers with a homography, reports the transform, and diagnoses or
falls back when alignment confidence is insufficient.

**Main files:**

- `src/patchrift/matching/stage_v.py` — ANN shortlist, descriptor distances,
  cyclic orientation handling, ratio test, Hough consensus, homography,
  transformation output, and C4 fallback logic.
- `src/patchrift/matching/__init__.py` — public Stage-V exports.
- `src/patchrift/pipeline/stages_iii_v.py` — `match_tile_pair()` connects
  Stage-II pair geometry, Stage-III conditioning, Stage-IV features, Stage-V
  matching, and the full C4 retry path.

## Shared entry points and non-stage code

- `src/patchrift/pipeline/__init__.py` exports the main orchestration APIs for
  Stage I, Stage II, tile matching, and two-hop cascade matching.
- `src/patchrift/geometry/rotation.py` is shared by Stage II angle estimation
  and Stage-V conditioned orientation matching.
- `src/patchrift/pipeline/cascade.py` is cross-stage glue for the resolution
  cascade in §6.1: it consumes two adjacent-resolution Stage-III–V match
  results (for example OHRC→TMC and TMC→IIRS), then composes their transforms
  and reports per-hop/end-to-end residuals.
- `src/patchrift/ui/` and `docs/ui-prototype.md` are the prototype interface and
  usage instructions; they are not mathematical pipeline stages.
- `tests/validation/README.md` maps formal V3 validation cases T1–T36 to their
  tests; supporting test modules are under `tests/`.

## Suggested reading split

| Team role | Start with | Then follow |
|---|---|---|
| Stage I | `pipeline/stage_i.py` | `io/`, `preprocessing/`, `geometry/handedness.py`, `geometry/tiling.py` |
| Stage II | `pipeline/stage_ii.py` | `pipeline/workflow.py`, `geometry/spice.py`, `geometry/diagnostics.py`, `shadows/` |
| Stage III | `cascade/stage_iii.py` | `pipeline/stages_iii_v.py::extract_tile_features()` |
| Stage IV | `features/stage_iv.py` | `features/__init__.py`, Stage-IV calls in `pipeline/stages_iii_v.py` |
| Stage V | `matching/stage_v.py` | `pipeline/stages_iii_v.py::match_tile_pair()`, then `pipeline/cascade.py` for the two-hop output |

For algorithm decisions, follow the agreed document precedence: main V3 is
binding except where an addendum explicitly changes a clause. Addendum 2 takes
precedence over Addendum 1 wherever it explicitly changes the main document or
the earlier addendum.
