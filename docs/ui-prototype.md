# PatchRift UI prototype

The optional Streamlit workbench runs the existing pipeline without changing
the algorithm implementation. It accepts a pair of scalar image arrays,
collects acquisition and footprint metadata, runs Stages I–II for each image,
then lets the operator select one adaptive tile from each side for Stage III–V
matching.

## Start

From the `Programs` project directory:

```sh
python -m pip install -e '.[ui]'
streamlit run src/patchrift/ui/app.py
```

The app uses the four SPICE kernels under `kernels/spice/`.
The project Streamlit configuration allows individual uploads up to 1500 MB;
restart the app after changing this setting. If the UI is later hosted behind a
proxy or on a managed Streamlit service, that deployment may enforce its own
request-size limit as well.

## Inputs

- Upload a single-band `.npy` array or `.tif`/`.tiff` image for each side. Arrays
  must be two-dimensional (`rows × columns`).
- Enter an ISO-8601 acquisition timestamp, GSD in metres per pixel, and emission
  angle in degrees.
- Enter four geolocations as `longitude, latitude` in degrees. Their order is
  the image-array corner order `(0,0)`, `(W−1,0)`, `(W−1,H−1)`, `(0,H−1)`; it
  does not assume the image is north-up.
- Optionally enter the fill value and 1σ latitude/longitude uncertainty. The
  latter is entered in arcseconds and converted to radians for the pipeline.
- The relief bound defaults to the V3 fallback of 300 m.

Uploads must be scalar products. Reduce a raw IIRS hyperspectral cube through
the documented Stage-I spectral reduction before uploading it as an IIRS
reduced scalar image.

## Outputs

The first run displays the input previews, adaptive tile plans, per-tile solar
geometry, confidence/fallback flags, and the planarity warning. After selecting
geographically corresponding tiles, the matching step shows alignment status,
inliers, residual, angular uncertainty, the canonical-frame transform, and
download links for a correspondence CSV and JSON run summary.

The tile pairing is deliberately operator-selected in this prototype: it avoids
guessing a correspondence from footprints when products have different tiling
or imperfect georeferencing. Process additional tile pairs individually when
the scene spans more than one adaptive tile.
