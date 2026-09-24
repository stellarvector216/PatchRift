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


def _product_form(label: str, key: str) -> dict:
    st.subheader(label)
    st.caption(
        "Enter values from this image product's label, metadata file, or geolocation sidecar. "
        "Do not use approximate map coordinates for the footprint."
    )
    upload = st.file_uploader(
        "Image data · single band (.npy or .tif/.tiff)",
        type=["npy", "tif", "tiff"],
        key=f"upload_{key}",
        help="Upload a two-dimensional rows × columns numeric array. Use the calibrated scalar OHRC/TMC image, or an IIRS image after the specified spectral reduction. Do not upload a colour screenshot or a raw multi-band IIRS cube.",
    )
    st.caption(
        "Accepted: 2D NumPy array or single-band TIFF. For raw IIRS cubes, first run the "
        "Stage-I band reduction to produce one scalar image."
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


def _tile_summary(result) -> list[dict]:
    rows = []
    for index, (tile, solar) in enumerate(zip(result.tiles, result.tile_results), start=1):
        rows.append({
            "Tile": index,
            "Rows": f"{tile.row_start}:{tile.row_end}",
            "Columns": f"{tile.col_start}:{tile.col_end}",
            "Solar elevation (°)": round(float(np.rad2deg(solar.solar_geometry.elevation)), 3),
            "Shadow azimuth (°)": (None if solar.pixel_azimuth is None else round(float(np.rad2deg(solar.pixel_azimuth)), 3)),
            "North angle (°)": (None if solar.north_angle is None else round(float(np.rad2deg(solar.north_angle)), 3)),
            "Planarity warning": result.planarity_flag,
            "Low confidence": solar.low_confidence,
            "C4 fallback": solar.c4,
        })
    return rows


def _correspondence_csv(pair_result) -> str:
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["index_a", "index_b", "x_a", "y_a", "x_b", "y_b", "descriptor_distance", "inlier"])
    a = pair_result.features_a.features.canonical_positions
    b = pair_result.features_b.features.canonical_positions
    matching = pair_result.matching
    for position, (idx_a, idx_b) in enumerate(matching.pairs):
        writer.writerow([
            int(idx_a), int(idx_b), *a[idx_a].tolist(), *b[idx_b].tolist(),
            float(matching.distances[position]), bool(matching.inlier_mask[position]),
        ])
    return out.getvalue()


def _matching_json(pair_result) -> str:
    result = pair_result.matching
    def optional_number(value):
        return None if value is None or not np.isfinite(value) else float(value)

    return json.dumps({
        "aligned": result.aligned,
        "fallback_used": result.fallback_used,
        "inlier_count": result.inlier_count,
        "rms_residual_pixels": optional_number(result.rms_residual_pixels),
        "rms_residual_meters": optional_number(result.rms_residual_meters),
        "delta_alpha_radians": optional_number(result.delta_alpha),
        "sigma_delta_alpha_radians": optional_number(result.sigma_delta_alpha),
        "hough_bin_size_pixels": optional_number(result.hough_bin_size),
        "translation": None if result.translation is None else result.translation.tolist(),
        "homography": None if result.homography is None else result.homography.tolist(),
    }, indent=2, allow_nan=False)


def _run_signature(form_a: dict, form_b: dict, q_value: float, tile_budget: int) -> str:
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
        {"a": serializable(form_a), "b": serializable(form_b), "q": q_value, "tile_budget": tile_budget},
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


with st.sidebar:
    st.markdown("### Run settings")
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
    st.caption("Solar azimuth/elevation are calculated from acquisition time and footprint using the bundled SPICE kernels.")

input_col_a, input_col_b = st.columns(2, gap="large")
with input_col_a:
    form_a = _product_form("Image A", "a")
with input_col_b:
    form_b = _product_form("Image B", "b")

run_col, note_col = st.columns([1, 3], vertical_alignment="center")
with run_col:
    analyze = st.button("Prepare images & estimate solar geometry", type="primary", use_container_width=True)
with note_col:
    st.caption("Required for each image: image file, UTC acquisition time, GSD, emission angle, and four image-array corner geolocations. Optional: fill value and geolocation uncertainty.")

current_signature = _run_signature(form_a, form_b, float(q), int(max_pixels_per_tile))
if st.session_state.get("patchrift_run_signature") != current_signature:
    st.session_state.pop("patchrift_run", None)
    st.session_state.pop("patchrift_pair_result", None)
    st.session_state.pop("patchrift_match_signature", None)

if analyze:
    try:
        with st.spinner("Reading images and running Stages I–II for both products…"):
            image_a, stage_i_a, sensor_a = _make_product(form_a)
            image_b, stage_i_b, sensor_b = _make_product(form_b)
            if not st.session_state.get("patchrift_spice_loaded", False):
                load_spice_kernels(KERNEL_CONFIG)
                st.session_state["patchrift_spice_loaded"] = True
            config = StageIIConfig(q=float(q))
            stage_ii_a = process_stage_ii_image(
                stage_i_a, config, max_pixels_per_tile=int(max_pixels_per_tile)
            )
            stage_ii_b = process_stage_ii_image(
                stage_i_b, config, max_pixels_per_tile=int(max_pixels_per_tile)
            )
            st.session_state["patchrift_run"] = {
                "image_a": image_a, "image_b": image_b,
                "stage_i_a": stage_i_a, "stage_i_b": stage_i_b,
                "stage_ii_a": stage_ii_a, "stage_ii_b": stage_ii_b,
                "sensor_a": sensor_a, "sensor_b": sensor_b,
            }
            st.session_state["patchrift_run_signature"] = current_signature
            st.session_state.pop("patchrift_pair_result", None)
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
        st.dataframe(_tile_summary(run_data["stage_ii_a"]), hide_index=True, use_container_width=True)
    with image_right:
        st.markdown(f"**Image B · {run_data['sensor_b']}**")
        st.image(_preview(run_data["image_b"]), caption=f"{run_data['image_b'].shape[1]} × {run_data['image_b'].shape[0]} px", use_container_width=True, clamp=True)
        st.dataframe(_tile_summary(run_data["stage_ii_b"]), hide_index=True, use_container_width=True)

    stage_ii_a, stage_ii_b = run_data["stage_ii_a"], run_data["stage_ii_b"]
    if stage_ii_a.planarity_flag or stage_ii_b.planarity_flag:
        st.warning("At least one image has emission angle above approximately 15°. V3 flags the planar-geometry limitation; inspect the result before interpreting a match.")

    st.markdown("## Stage III–V correspondence")
    st.caption("A tile is a rectangular subsection of an image created to meet memory and solar-angle limits. Select the same lunar region in Image A and Image B; use the row/column ranges above and the footprint metadata to choose corresponding tiles.")
    tile_col_a, tile_col_b = st.columns(2)
    selected_a = tile_col_a.selectbox(
        "Image A tile", range(len(stage_ii_a.tiles)), format_func=lambda ix: f"Tile {ix + 1} · {stage_ii_a.tiles[ix].shape[1]} × {stage_ii_a.tiles[ix].shape[0]} px",
        key="match_tile_a",
    )
    selected_b = tile_col_b.selectbox(
        "Image B tile", range(len(stage_ii_b.tiles)), format_func=lambda ix: f"Tile {ix + 1} · {stage_ii_b.tiles[ix].shape[1]} × {stage_ii_b.tiles[ix].shape[0]} px",
        key="match_tile_b",
    )
    match_signature = (
        current_signature, int(selected_a), int(selected_b),
        float(lambda_max), int(minimum_inliers),
    )
    if st.session_state.get("patchrift_match_signature") != match_signature:
        st.session_state.pop("patchrift_pair_result", None)
    if st.button("Match selected tiles", type="primary"):
        try:
            with st.spinner("Conditioning tiles, building descriptors, and verifying correspondences…"):
                result = match_tile_pair(
                    stage_ii_a.stage_i, stage_ii_a.tiles[selected_a], stage_ii_a.tile_results[selected_a],
                    stage_ii_b.stage_i, stage_ii_b.tiles[selected_b], stage_ii_b.tile_results[selected_b],
                    gsd_a=stage_ii_a.stage_i.product.metadata.gsd,
                    gsd_b=stage_ii_b.stage_i.product.metadata.gsd,
                    gsd_target=max(stage_ii_a.stage_i.product.metadata.gsd, stage_ii_b.stage_i.product.metadata.gsd),
                    lambda_max_a=float(lambda_max), lambda_max_b=float(lambda_max),
                    matching_options={"minimum_inliers": int(minimum_inliers)},
                )
                st.session_state["patchrift_pair_result"] = result
                st.session_state["patchrift_match_signature"] = match_signature
        except Exception as exc:
            st.error(f"Tile matching did not complete: {exc}")

    pair_result = st.session_state.get("patchrift_pair_result")
    if pair_result:
        matching = pair_result.matching
        if not matching.aligned:
            st.warning("No verified alignment for this tile pair. Try another geographically corresponding tile pair or review product metadata.")
        else:
            st.success("Geometric alignment verified.")
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Inliers", matching.inlier_count)
        metric2.metric("RMS residual", "—" if matching.rms_residual_meters is None else f"{matching.rms_residual_meters:.2f} m")
        metric3.metric("Δα uncertainty", f"{np.rad2deg(matching.sigma_delta_alpha):.3f}°")
        metric4.metric("Hough threshold τ", f"{matching.hough_bin_size:.2f} px")
        if matching.fallback_used:
            st.info("The V3 C4 unrotated/full-ring fallback was used for this tile pair.")
        transform = {
            "translation_px": None if matching.translation is None else matching.translation.tolist(),
            "homography": None if matching.homography is None else matching.homography.tolist(),
        }
        st.markdown("**Estimated transform**")
        st.code(json.dumps(transform, indent=2), language="json")
        st.download_button(
            "Download correspondence CSV", data=_correspondence_csv(pair_result),
            file_name="patchrift_correspondences.csv", mime="text/csv",
        )
        st.download_button(
            "Download run summary", data=_matching_json(pair_result),
            file_name="patchrift_result.json", mime="application/json",
        )

    st.caption("Prototype note: uploads are single-band scalar arrays. Raw IIRS cubes require the documented Stage-I spectral reduction before upload.")
