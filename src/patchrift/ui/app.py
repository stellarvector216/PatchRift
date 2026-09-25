"""Streamlit prototype UI. Install with ``pip install -e '.[ui]'``."""

from __future__ import annotations

import csv
import hashlib
import json
from io import StringIO
from pathlib import Path

import numpy as np
import streamlit as st

from patchrift.geometry.solar import SpiceKernelConfig
from patchrift.geometry.spice import load_spice_kernels
from patchrift.io import ProductMetadata, ingest_array
from patchrift.pipeline import (
    StageIIConfig,
    match_tile_pair,
    prepare_image_product,
    process_stage_ii_image,
)
from patchrift.ui.io import parse_footprint_corners, parse_utc_timestamp, read_scalar_image
from patchrift.ui.tile_matching import match_tile_pairs_incrementally, plan_tile_pairs
from patchrift.ui.tile_summary import tile_summary_rows
from patchrift.ui.solar_mode import (
    COMPARE_MODE,
    METADATA_MODE,
    SPICE_MODE,
    degrees_to_geometry,
    metadata_geometry_kwargs,
    uses_spice,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
KERNELS = PROJECT_ROOT / "kernels" / "spice"
KERNEL_CONFIG = SpiceKernelConfig(
    lsk_path=str(KERNELS / "naif0012.tls"),
    pck_path=str(KERNELS / "moon_pa_de440_200625.bpc"),
    spk_path=str(KERNELS / "de440.bsp"),
    fk_path=str(KERNELS / "moon_de440_250416.tf"),
)

st.set_page_config(
    page_title="PatchRift · Lunar Correspondence",
    page_icon="🌘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      [data-testid="stHeader"] { background: rgba(245,247,251,.92); }
      .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
      .hero { padding: 1.4rem 1.7rem; border-radius: 18px;
              background: linear-gradient(115deg,#14243a,#203b5c); color: #f4f7fb;
              margin-bottom: 1.35rem; }
      .hero h1 { margin: .1rem 0 .35rem 0; font-size: 2rem; }
      .hero p { margin: 0; color: #cad7e6; }
      .eyebrow { color: #93c5fd !important; font-size: .76rem; font-weight: 700;
                 letter-spacing: .11em; text-transform: uppercase; }
      .panel { padding: 1rem 1.1rem; border: 1px solid #e2e8f0; border-radius: 14px;
               background: white; }
      div[data-testid="stMetric"] { background: white; padding: 1rem; border-radius: 12px;
                                     border: 1px solid #e2e8f0; }
      .stButton > button[kind="primary"] { background: #1769aa; border-color: #1769aa; }
      footer { visibility: hidden; }
    </style>
    <div class="hero">
      <p class="eyebrow">SIH 2026 · Chandrayaan-2</p>
      <h1>PatchRift Correspondence Workbench</h1>
      <p>Prepare a sensor pair, inspect solar geometry, and estimate image correspondence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _product_form(label: str, key: str, geometry_mode: str) -> dict:
    st.subheader(label)
    st.caption(
        "Enter values from this image product's label, metadata file, or geolocation sidecar. "
        "Do not use approximate map coordinates for the footprint."
    )
    upload = st.file_uploader(
        "Image data · single band (.png, .npy, .tif/.tiff)",
        type=["png", "npy", "tif", "tiff"],
        key=f"upload_{key}",
        help="Upload a two-dimensional rows × columns numeric array or grayscale PNG. Use the calibrated scalar OHRC/TMC image, or an IIRS image after the specified spectral reduction. Do not upload a colour screenshot or a raw multi-band IIRS cube.",
    )
    st.caption(
        "Accepted: grayscale PNG (8/16-bit), 2D NumPy array, or single-band TIFF. "
        "RGB/RGBA PNGs are rejected rather than converted. For raw IIRS cubes, first run "
        "the Stage-I band reduction to produce one scalar image."
    )
    sensor = st.selectbox(
        "Sensor label (shown in results)",
        ["OHRC", "TMC", "IIRS (reduced scalar)"],
        key=f"sensor_{key}",
        help="This selector labels the input in the UI. It does not perform sensor-specific spectral reduction.",
    )
    timestamp = st.text_input(
        "Acquisition timestamp · UTC",
        placeholder="2025-01-15T12:30:00Z",
        key=f"time_{key}",
        help="The UTC capture time from the image product metadata. SPICE uses it to calculate where the Sun was during acquisition.",
    )
    st.caption("Format: `YYYY-MM-DDTHH:MM:SSZ`, for example `2025-01-15T12:30:00Z`.")
    solar_azimuth_deg = solar_elevation_deg = None
    if geometry_mode != SPICE_MODE:
        st.markdown("**Scene-wide solar angles from metadata**")
        st.caption(
            "Enter the product's scene-wide solar azimuth and elevation in degrees. "
            "This mode treats both values as constant across all tiles. Confirm the "
            "metadata's azimuth convention matches the V3 solar-azimuth convention."
        )
        angle_col_az, angle_col_el = st.columns(2)
        solar_azimuth_deg = angle_col_az.number_input(
            "Solar azimuth · degrees", min_value=0.0, max_value=360.0,
            value=None, format="%.4f", key=f"solar_azimuth_{key}",
            help="Scene-wide solar azimuth from product metadata, clockwise from the product's documented reference direction.",
        )
        solar_elevation_deg = angle_col_el.number_input(
            "Solar elevation · degrees", min_value=-90.0, max_value=90.0,
            value=None, format="%.4f", key=f"solar_elevation_{key}",
            help="Scene-wide elevation of the Sun above the local horizon, from product metadata.",
        )
    gsd = st.number_input(
        "Ground sample distance · GSD (metres per pixel)", min_value=0.000001,
        value=None, format="%.6f", key=f"gsd_{key}",
        help="The nominal ground distance represented by one image pixel. Read this from product metadata; do not enter image width or altitude.",
    )
    st.caption("Example: if one pixel represents 25 metres on the lunar surface, enter `25`.")
    emission_deg = st.number_input(
        "Emission angle · sensor viewing angle (degrees)",
        value=None, format="%.4f", key=f"emission_{key}",
        help="The angle between the local surface normal and the line of sight from the ground to the spacecraft, from product metadata. This is not the solar elevation angle.",
    )
    st.caption("Used to flag when the planar-geometry assumption may be limited. Enter the product's emission angle, not the Sun angle.")
    corners = st.text_area(
        "Image footprint · four image-array corner coordinates",
        placeholder="\n".join(["lon, lat"] * 4),
        height=105,
        key=f"corners_{key}",
        help="Enter decimal-degree longitude,latitude pairs from the product geolocation. Lines map in order to array pixels (column,row): (0,0), (W−1,0), (W−1,H−1), then (0,H−1). The image is not assumed north-up, so these are array corners, not geographic north/south corners.",
    )
    st.caption(
        "Enter one `longitude, latitude` pair per line, in this exact pixel order: "
        "top-left array pixel `(0,0)`; top-right `(W−1,0)`; bottom-right "
        "`(W−1,H−1)`; bottom-left `(0,H−1)`. Coordinates are decimal degrees. "
        "Use actual geolocation values from the product, not an estimated bounding box."
    )
    with st.expander("Additional metadata (relief, fill value, uncertainty)", expanded=False):
        hmax = st.number_input(
            "Regional relief bound · hmax (metres)", min_value=0.0, value=300.0,
            format="%.1f", key=f"hmax_{key}",
            help="Maximum expected local terrain relief used to set the geometric matching tolerance. Use a regional terrain bound if available; otherwise keep the V3 conservative default of 300 m.",
        )
        st.caption("This is a local terrain-height bound, not spacecraft altitude. Leave the V3 default of 300 m if no regional relief bound is available.")
        fill_text = st.text_input(
            "No-data / fill sentinel (optional)", placeholder="For example: -9999; blank if none", key=f"fill_{key}",
            help="The exact numeric value used in the file to mark pixels with no observation. Non-finite pixels are excluded automatically.",
        )
        has_geo_sigma = st.checkbox(
            "I have geolocation uncertainty values in the product metadata", value=False, key=f"has_sigma_{key}"
        )
        latitude_sigma = longitude_sigma = None
        if has_geo_sigma:
            st.caption("Enter each coordinate's 1σ position uncertainty in arcseconds. These values feed the Stage-II geometric uncertainty calculation.")
            sigma_col1, sigma_col2 = st.columns(2)
            latitude_sigma = sigma_col1.number_input(
                "Latitude 1σ uncertainty (arcsec)", min_value=0.0, value=None,
                key=f"sigma_lat_{key}",
            )
            longitude_sigma = sigma_col2.number_input(
                "Longitude 1σ uncertainty (arcsec)", min_value=0.0, value=None,
                key=f"sigma_lon_{key}",
            )
    return {
        "upload": upload,
        "sensor": sensor,
        "timestamp": timestamp,
        "solar_azimuth_rad": None if solar_azimuth_deg is None else float(np.deg2rad(solar_azimuth_deg)),
        "solar_elevation_rad": None if solar_elevation_deg is None else float(np.deg2rad(solar_elevation_deg)),
        "gsd": None if gsd is None else float(gsd),
        "emission_rad": None if emission_deg is None else float(np.deg2rad(emission_deg)),
        "corners_text": corners,
        "hmax": float(hmax),
        "fill_text": fill_text,
        "has_geo_sigma": has_geo_sigma,
        "latitude_sigma_arcsec": latitude_sigma,
        "longitude_sigma_arcsec": longitude_sigma,
    }


def _make_product(form: dict):
    if form["upload"] is None:
        raise ValueError("Upload an image for both inputs")
    if form["gsd"] is None or form["emission_rad"] is None:
        raise ValueError("Enter GSD and emission angle for both images")
    image = read_scalar_image(form["upload"].name, form["upload"].getvalue())
    corners = parse_footprint_corners(form["corners_text"])
    timestamp = parse_utc_timestamp(form["timestamp"])
    rows, cols = image.shape
    pixel_corners = np.array(
        [[0.0, 0.0], [cols - 1.0, 0.0], [cols - 1.0, rows - 1.0], [0.0, rows - 1.0]]
    )
    sigma_lat = form["latitude_sigma_arcsec"]
    sigma_lon = form["longitude_sigma_arcsec"]
    if (sigma_lat is None) != (sigma_lon is None):
        raise ValueError("Enter both latitude and longitude uncertainties")
    if sigma_lat is None and form["has_geo_sigma"]:
        raise ValueError("Enter both geolocation uncertainties or turn that option off")
    metadata = ProductMetadata(
        timestamp=timestamp,
        footprint_corners=corners,
        footprint_pixel_coords=pixel_corners,
        gsd=form["gsd"],
        emission_angle=form["emission_rad"],
        hmax=form["hmax"],
        latitude_uncertainty=(None if sigma_lat is None else np.deg2rad(sigma_lat / 3600.0)),
        longitude_uncertainty=(None if sigma_lon is None else np.deg2rad(sigma_lon / 3600.0)),
    )
    fill_text = form["fill_text"].strip()
    try:
        fill_value = None if not fill_text else float(fill_text)
    except ValueError as exc:
        raise ValueError("Fill/no-data value must be numeric") from exc
    product = ingest_array(image, fill_value, metadata)
    return image, prepare_image_product(product), form["sensor"]


def _preview(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros((32, 32), dtype=np.uint8)
    low, high = np.percentile(image[finite], [2, 98])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((np.nan_to_num(image, nan=low) - low) / (high - low), 0.0, 1.0)
    stride = max(1, int(np.ceil(max(image.shape) / 1200)))
    return (scaled[::stride, ::stride] * 255).astype(np.uint8)


def _pair_results_csv(records: list[dict]) -> str:
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "tile_a", "tile_b", "index_a", "index_b", "x_a", "y_a", "x_b", "y_b",
        "descriptor_distance", "inlier",
    ])
    for record in records:
        for row in record.get("correspondence_rows", ()):
            writer.writerow([
                record["tile_a"] + 1, record["tile_b"] + 1, *row,
            ])
    return out.getvalue()


def _pair_results_json(records: list[dict], mode: str) -> str:
    pairs = []
    for record in records:
        matching = record.get("matching")
        pairs.append({
            "tile_a": record["tile_a"] + 1,
            "tile_b": record["tile_b"] + 1,
            "error": record.get("error"),
            "aligned": None if matching is None else bool(matching.aligned),
            "inlier_count": None if matching is None else int(matching.inlier_count),
            "rms_residual_meters": None if matching is None or matching.rms_residual_meters is None or not np.isfinite(matching.rms_residual_meters) else float(matching.rms_residual_meters),
            "delta_alpha_radians": None if matching is None or matching.delta_alpha is None or not np.isfinite(matching.delta_alpha) else float(matching.delta_alpha),
            "sigma_delta_alpha_radians": None if matching is None or matching.sigma_delta_alpha is None or not np.isfinite(matching.sigma_delta_alpha) else float(matching.sigma_delta_alpha),
            "translation": None if matching is None or matching.translation is None else matching.translation.tolist(),
            "homography": None if matching is None or matching.homography is None else matching.homography.tolist(),
        })
    return json.dumps({"tile_pairing_mode": mode, "pairs": pairs}, indent=2, allow_nan=False)


def _run_signature(form_a: dict, form_b: dict, q_value: float, tile_budget: int, geometry_mode: str) -> str:
    def serializable(form):
        return {
            key: value for key, value in form.items()
            if key != "upload"
        } | {
            "upload_name": None if form["upload"] is None else form["upload"].name,
            "upload_id": None if form["upload"] is None else getattr(form["upload"], "file_id", None),
            "upload_size": None if form["upload"] is None else form["upload"].size,
        }

    content = json.dumps(
        {"a": serializable(form_a), "b": serializable(form_b), "q": q_value, "tile_budget": tile_budget, "geometry_mode": geometry_mode},
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


with st.sidebar:
    st.markdown("### Run settings")
    geometry_mode = st.radio(
        "Solar geometry source",
        [SPICE_MODE, METADATA_MODE, COMPARE_MODE],
        index=0,
        help="Metadata-only mode uses the scene-wide azimuth and elevation you enter for every tile and does not load or call SPICE. Comparison mode runs SPICE and reports metadata-minus-SPICE angle differences.",
    )
    if geometry_mode == METADATA_MODE:
        st.info("SPICE is bypassed. Scene-wide solar angles are treated as constant across the image; geographic variation and its derivative are unavailable. Stage V handles the unknown angular contribution conservatively and may use its full-ring fallback.")
    elif geometry_mode == COMPARE_MODE:
        st.caption("SPICE computes the Stage-II geometry; the entered scene-wide metadata angles are shown beside it for comparison.")
    q = st.slider(
        "Shadow-threshold bias q", 0.05, 0.10, 0.075, 0.005,
        help="V3 §4.3 biases the local Otsu threshold toward the darker shadow class. The permitted range is 0.05–0.10; 0.075 is the starting value.",
    )
    st.caption("Leave at 0.075 unless you are deliberately tuning the documented shadow-threshold bias.")
    max_pixels_per_tile = st.number_input(
        "Tile memory cap · maximum pixels", min_value=10000,
        value=1_000_000, step=10000,
        help="Upper bound on the number of image pixels processed in one Stage-II tile. Lower this if memory is limited; solar-azimuth variation may subdivide tiles further.",
    )
    st.caption("This is a pixel count, not bytes or image dimensions.")
    lambda_max = st.number_input(
        "Largest descriptor filter wavelength (pixels)", min_value=1.0,
        value=24.0, step=1.0,
        help="Largest centre wavelength in the Stage-IV Log-Gabor filter bank, measured in working-image pixels. V3 defaults to 3 px minimum × 2³ across four scales = 24 px.",
    )
    minimum_inliers = st.number_input(
        "Minimum verified matches · inliers", min_value=4, value=10, step=1,
        help="Minimum number of descriptor correspondences that must agree with the translation consensus before the tile pair is reported as aligned.",
    )
    if geometry_mode == SPICE_MODE:
        st.caption("Solar azimuth/elevation are calculated from acquisition time and footprint using the bundled SPICE kernels.")

input_col_a, input_col_b = st.columns(2, gap="large")
with input_col_a:
    form_a = _product_form("Image A", "a", geometry_mode)
with input_col_b:
    form_b = _product_form("Image B", "b", geometry_mode)

run_col, note_col = st.columns([1, 3], vertical_alignment="center")
with run_col:
    analyze = st.button("Prepare images & estimate solar geometry", type="primary", use_container_width=True)
with note_col:
    required = "image file, UTC acquisition time, GSD, emission angle, and four image-array corner geolocations"
    if geometry_mode != SPICE_MODE:
        required += ", plus scene-wide solar azimuth and elevation"
    st.caption(f"Required for each image: {required}. Optional: fill value and geolocation uncertainty.")

current_signature = _run_signature(form_a, form_b, float(q), int(max_pixels_per_tile), geometry_mode)
if st.session_state.get("patchrift_run_signature") != current_signature:
    st.session_state.pop("patchrift_run", None)
    st.session_state.pop("patchrift_pair_results", None)
    st.session_state.pop("patchrift_match_signature", None)

if analyze:
    try:
        with st.spinner("Reading images and running Stages I–II for both products…"):
            image_a, stage_i_a, sensor_a = _make_product(form_a)
            image_b, stage_i_b, sensor_b = _make_product(form_b)
            if uses_spice(geometry_mode) and not st.session_state.get("patchrift_spice_loaded", False):
                load_spice_kernels(KERNEL_CONFIG)
                st.session_state["patchrift_spice_loaded"] = True
            config = StageIIConfig(q=float(q))
            def process_product(stage_i, form):
                if geometry_mode != SPICE_MODE and (
                    form["solar_azimuth_rad"] is None or form["solar_elevation_rad"] is None
                ):
                    raise ValueError("Enter scene-wide solar azimuth and elevation for both images")
                if geometry_mode != SPICE_MODE:
                    degrees_to_geometry(
                        np.rad2deg(form["solar_azimuth_rad"]),
                        np.rad2deg(form["solar_elevation_rad"]),
                    )
                return process_stage_ii_image(
                    stage_i, config, max_pixels_per_tile=int(max_pixels_per_tile),
                    **metadata_geometry_kwargs(
                        geometry_mode, form["solar_azimuth_rad"], form["solar_elevation_rad"]
                    ),
                )

            stage_ii_a = process_product(stage_i_a, form_a)
            stage_ii_b = process_product(stage_i_b, form_b)
            st.session_state["patchrift_run"] = {
                "image_a": image_a, "image_b": image_b,
                "stage_i_a": stage_i_a, "stage_i_b": stage_i_b,
                "stage_ii_a": stage_ii_a, "stage_ii_b": stage_ii_b,
                "sensor_a": sensor_a, "sensor_b": sensor_b,
                "geometry_mode": geometry_mode,
                "metadata_geometry_a": (form_a["solar_azimuth_rad"], form_a["solar_elevation_rad"]),
                "metadata_geometry_b": (form_b["solar_azimuth_rad"], form_b["solar_elevation_rad"]),
            }
            st.session_state["patchrift_run_signature"] = current_signature
            st.session_state.pop("patchrift_pair_results", None)
    except Exception as exc:
        st.error(f"Could not process this pair: {exc}")

run_data = st.session_state.get("patchrift_run")
if run_data:
    st.divider()
    st.markdown("## Stage I–II overview")
    image_left, image_right = st.columns(2, gap="large")
    with image_left:
        st.markdown(f"**Image A · {run_data['sensor_a']}**")
        st.image(_preview(run_data["image_a"]), caption=f"{run_data['image_a'].shape[1]} × {run_data['image_a'].shape[0]} px", use_container_width=True, clamp=True)
        geometry_a = run_data["metadata_geometry_a"] if run_data["geometry_mode"] == COMPARE_MODE else None
        st.dataframe(tile_summary_rows(run_data["stage_ii_a"], geometry_a), hide_index=True, use_container_width=True)
    with image_right:
        st.markdown(f"**Image B · {run_data['sensor_b']}**")
        st.image(_preview(run_data["image_b"]), caption=f"{run_data['image_b'].shape[1]} × {run_data['image_b'].shape[0]} px", use_container_width=True, clamp=True)
        geometry_b = run_data["metadata_geometry_b"] if run_data["geometry_mode"] == COMPARE_MODE else None
        st.dataframe(tile_summary_rows(run_data["stage_ii_b"], geometry_b), hide_index=True, use_container_width=True)

    stage_ii_a, stage_ii_b = run_data["stage_ii_a"], run_data["stage_ii_b"]
    if stage_ii_a.planarity_flag or stage_ii_b.planarity_flag:
        st.warning("At least one image has emission angle above approximately 15°. V3 flags the planar-geometry limitation; inspect the result before interpreting a match.")

    st.markdown("## Stage III–V correspondence")
    st.caption("Tile matching is planned automatically from the two image footprints. Each candidate pair runs on its own; its result is saved and displayed as soon as it finishes, then the next pair starts automatically.")
    pairing_label = st.radio(
        "Tile-pair search",
        ["Geographic overlap only (recommended)", "All Image A tiles × all Image B tiles"],
        horizontal=True,
        key="tile_pairing_mode",
        help="Overlap mode avoids work where the metadata footprints do not intersect. Exhaustive mode tests every possible tile pair and can take substantially longer.",
    )
    pairing_mode = "all" if pairing_label.startswith("All ") else "overlap"
    try:
        candidate_pairs = plan_tile_pairs(stage_ii_a, stage_ii_b, pairing_mode)
    except Exception as exc:
        candidate_pairs = []
        st.error(f"Could not plan tile pairs from the footprint metadata: {exc}")
    if pairing_mode == "all":
        st.warning(f"Exhaustive mode will test {len(candidate_pairs):,} tile pairs. Runtime grows with both tile counts.")
    elif not candidate_pairs:
        st.warning("The supplied footprints show no overlapping tile pairs. Check the corner coordinates or use exhaustive mode if the footprints are approximate.")
    else:
        st.info(f"Automatic geographic-overlap planning found {len(candidate_pairs):,} candidate tile pair(s).")
    with st.expander("How automatic tile pairing chooses pairs"):
        st.markdown(
            """
            1. Stage II independently subdivides each image to satisfy the solar-azimuth
               variation limit and the configured pixels-per-tile memory cap.
            2. Each adaptive tile's four array-corner locations are estimated from the
               product's image-corner geolocation metadata.
            3. In **Geographic overlap** mode, every Image A tile is compared with every
               Image B tile. The corner polygons are projected to a local
               equirectangular plane (with longitude wrap handled at the date line),
               and convex-polygon clipping computes their intersection area. Pairs with
               positive area above the numerical tolerance are candidates. Touching
               edges alone do not count as overlap.
            4. In **All tiles × all tiles** mode, the candidate list is the full Cartesian
               product: `(A0,B0), (A0,B1), …, (A1,B0), …`. Geographic overlap is not
               used to filter this list.
            5. Candidate pairing only decides which tile pairs to try. Stages III–V
               still extract features and verify correspondence independently for each
               pair; tile indices are never assumed to correspond across images.
            """
        )

    match_signature = (
        current_signature, pairing_mode, tuple(candidate_pairs),
        float(lambda_max), int(minimum_inliers),
    )
    if st.session_state.get("patchrift_match_signature") != match_signature:
        st.session_state.pop("patchrift_pair_results", None)
    st.session_state["patchrift_match_signature"] = match_signature
    records = st.session_state.get("patchrift_pair_results", [])
    processed_pairs = {(record["tile_a"], record["tile_b"]) for record in records}
    remaining_pairs = [pair for pair in candidate_pairs if pair not in processed_pairs]
    if records and remaining_pairs:
        st.caption(f"{len(records):,} tile pair(s) already have saved results. New results appear as each remaining pair finishes; processing continues automatically.")
    elif not records and candidate_pairs:
        st.caption("Each completed tile pair will appear below immediately. Processing then continues to the next pair automatically.")

    if remaining_pairs and st.button(
        "Continue remaining tile pairs" if records else "Match planned tile pairs",
        type="primary",
    ):
        progress = st.progress(
            len(records) / len(candidate_pairs),
            text=f"Starting at pair {len(records) + 1:,} of {len(candidate_pairs):,}…",
        )
        live_results = st.container()

        def match_one(tile_a, tile_b):
            return match_tile_pair(
                stage_ii_a.stage_i, stage_ii_a.tiles[tile_a], stage_ii_a.tile_results[tile_a],
                stage_ii_b.stage_i, stage_ii_b.tiles[tile_b], stage_ii_b.tile_results[tile_b],
                gsd_a=stage_ii_a.stage_i.product.metadata.gsd,
                gsd_b=stage_ii_b.stage_i.product.metadata.gsd,
                gsd_target=max(stage_ii_a.stage_i.product.metadata.gsd, stage_ii_b.stage_i.product.metadata.gsd),
                lambda_max_a=float(lambda_max), lambda_max_b=float(lambda_max),
                matching_options={"minimum_inliers": int(minimum_inliers)},
            )

        def save_and_render(raw_record):
            result = raw_record["result"]
            record = {
                "tile_a": raw_record["tile_a"], "tile_b": raw_record["tile_b"],
                "matching": None, "correspondence_rows": [], "error": raw_record["error"],
            }
            if result is not None:
                try:
                    matching = result.matching
                    points_a = result.features_a.features.canonical_positions
                    points_b = result.features_b.features.canonical_positions
                    record["matching"] = matching
                    record["correspondence_rows"] = [
                        [int(index_a), int(index_b), *points_a[index_a].tolist(),
                         *points_b[index_b].tolist(), float(matching.distances[position]),
                         bool(matching.inlier_mask[position])]
                        for position, (index_a, index_b) in enumerate(matching.pairs)
                    ]
                except Exception as exc:
                    record["error"] = f"Could not save match output: {exc}"
            records.append(record)
            st.session_state["patchrift_pair_results"] = list(records)
            matching = record["matching"]
            with live_results:
                with st.container(border=True):
                    st.markdown(f"**Completed · Image A tile {record['tile_a'] + 1} ↔ Image B tile {record['tile_b'] + 1}**")
                    if matching is None:
                        st.error(record["error"])
                    else:
                        if matching.aligned:
                            st.success(f"Verified alignment · {matching.inlier_count} inliers")
                        else:
                            st.info(f"No verified alignment · {matching.inlier_count} inliers")
                        residual = matching.rms_residual_meters
                        st.caption("RMS residual: unavailable" if residual is None else f"RMS residual: {residual:.2f} m")

        progress_position = len(records)
        for _record in match_tile_pairs_incrementally(
            remaining_pairs, match_one, save_and_render
        ):
            progress_position += 1
            progress.progress(
                progress_position / len(candidate_pairs),
                text=f"Completed {progress_position:,} of {len(candidate_pairs):,} tile pairs…",
            )

    records = st.session_state.get("patchrift_pair_results", [])
    if records:
        summary = []
        for record in records:
            matching = record["matching"]
            summary.append({
                "Image A tile": record["tile_a"] + 1,
                "Image B tile": record["tile_b"] + 1,
                "Aligned": None if matching is None else bool(matching.aligned),
                "Inliers": None if matching is None else matching.inlier_count,
                "RMS residual (m)": None if matching is None else matching.rms_residual_meters,
                "Status": record["error"] or ("verified" if matching and matching.aligned else "no verified alignment"),
            })
        st.dataframe(summary, hide_index=True, use_container_width=True)
        aligned_records = [record for record in records if record["matching"] is not None and record["matching"].aligned]
        if aligned_records:
            st.success(f"Verified alignment in {len(aligned_records)} of {len(records)} processed tile pair(s).")
        else:
            st.warning("No verified alignment was found among the processed tile pairs. Review the Stage-I/II quality and product metadata.")
        for record in records:
            matching = record["matching"]
            with st.expander(f"Tile A {record['tile_a'] + 1} ↔ Tile B {record['tile_b'] + 1}", expanded=bool(matching and matching.aligned)):
                if matching is None:
                    st.error(record["error"])
                    continue
                cols = st.columns(4)
                cols[0].metric("Inliers", matching.inlier_count)
                cols[1].metric("RMS residual", "—" if matching.rms_residual_meters is None else f"{matching.rms_residual_meters:.2f} m")
                cols[2].metric("Δα uncertainty", "—" if matching.sigma_delta_alpha is None or not np.isfinite(matching.sigma_delta_alpha) else f"{np.rad2deg(matching.sigma_delta_alpha):.3f}°")
                cols[3].metric("Hough threshold τ", "—" if matching.hough_bin_size is None else f"{matching.hough_bin_size:.2f} px")
                if matching.fallback_used:
                    st.info("The V3 C4 unrotated/full-ring fallback was used for this tile pair.")
                st.code(json.dumps({
                    "translation_px": None if matching.translation is None else matching.translation.tolist(),
                    "homography": None if matching.homography is None else matching.homography.tolist(),
                }, indent=2), language="json")
        download_csv, download_json = st.columns(2)
        download_csv.download_button(
            "Download all correspondences (CSV)", data=_pair_results_csv(records),
            file_name="patchrift_correspondences.csv", mime="text/csv",
        )
        download_json.download_button(
            "Download all pair results (JSON)", data=_pair_results_json(records, pairing_mode),
            file_name="patchrift_results.json", mime="application/json",
        )

    st.caption("Prototype note: uploads are single-band scalar arrays. Raw IIRS cubes require the documented Stage-I spectral reduction before upload.")
