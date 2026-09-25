# PatchRift UI prototype

The optional Streamlit workbench runs the existing pipeline without changing
the algorithm implementation. It accepts a pair of scalar image arrays,
collects acquisition and footprint metadata, runs Stages I–II for each image,
then automatically plans adaptive tile pairs for Stage III–V matching.

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

- Upload a grayscale `.png`, single-band `.tif`/`.tiff`, or 2D `.npy` array for
  each side. PNG supports 8-bit and 16-bit grayscale. RGB/RGBA PNGs are rejected
  instead of being silently converted, so pixel values remain meaningful.
  Arrays must be two-dimensional (`rows × columns`).
- Enter an ISO-8601 acquisition timestamp, GSD in metres per pixel, and emission
  angle in degrees.
- Choose one solar-geometry source:
  - **SPICE ephemeris:** use timestamp and footprint with the bundled kernels.
  - **Use metadata geometry · skip SPICE:** enter scene-wide solar azimuth and
  elevation in degrees. These values are treated as constant across the
  image, so spatial azimuth derivatives and their geolocation contribution
  to `sigma_geo` are unavailable. The UI does not report them as zero; Stage V
  handles the unknown angular contribution conservatively and may use its
  full-ring fallback.
  - **Metadata plus SPICE comparison:** run SPICE for Stage II and show the
    entered scene-wide metadata angles alongside the per-tile SPICE values,
    including wrapped azimuth and elevation differences.
  Metadata angles should follow the product's solar azimuth convention
  (clockwise from selenographic north) and should be truly scene-wide values.
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
geometry, confidence/fallback flags, and the planarity warning. The matching
step defaults to geographic-overlap planning and matches every intersecting
tile pair automatically. An exhaustive option tests every Image A tile against
every Image B tile. Each pair is processed independently and the UI reports
alignment, inliers, residual, angular uncertainty, and its canonical-frame
transform. Download links export all pair correspondences as CSV and all pair
summaries as JSON. A result is saved and shown as soon as its tile pair
finishes, then the next pair starts automatically. If a run is interrupted,
completed results remain saved and the remaining pairs can be resumed.
Exhaustive search can take substantially longer.

## Automatic tile-pair selection

Stage II independently divides each image into adaptive tiles to respect its
solar-azimuth variation bound and configured maximum pixel count. For the
recommended geographic-overlap mode, the workbench estimates each tile's four
geographic corners from the product's image-corner geolocation metadata. It
compares every A-tile polygon with every B-tile polygon in a local
equirectangular projection, wraps longitude across the date line, and clips the
convex polygons to measure intersection area. Pairs with positive intersection
area above the numerical tolerance become candidates; edge-touching alone is
not enough. In exhaustive mode, all index combinations are used, regardless of
footprint: `(A0,B0), (A0,B1), …, (A1,B0), …`.

The candidate planner does not declare a match. Stages III–V still extract and
verify visual correspondence for each candidate pair. Tile numbers are not
assumed to correspond between products. Use exhaustive mode when footprint
metadata is approximate or to search beyond geographic overlap.

The Stage-II overview reports both **Solar azimuth**, the Sun direction from
SPICE or metadata, and **Shadow azimuth**, the image-frame direction estimated
from shadow/gradient evidence. Shadow azimuth may be unavailable when Stage II
uses C4 or cannot estimate a valid shadow direction; its status is shown in the
table. That does not mean the solar azimuth is unavailable.
