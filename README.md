# IPDE — Image Precision Data Extractor

IPDE extracts depth images, HDR gain maps, portrait/semantic mattes, Apple
Spatial Photo stereo views, and other HEIF auxiliary images without
display-oriented processing. The project has two parts:

- a Python command-line extractor built on `pillow-heif >= 1.5.0`;
- a native Qt 6 desktop GUI that runs the extractor with structured JSON I/O.

## Precision model

IPDE preserves the values produced by the HEIF decoder. Raw outputs never receive
gamma correction, tone mapping, color enhancement, range stretching, or
normalization. A separately named metric-depth EXR is an explicit, documented
calibration and never replaces the raw depth PNG/NPY.

The GUI separates **Source precision** from **Export storage**. Generated RAFT,
classical, and calibrated float32 products are not images stored in the HEIC.
On macOS, **Apple native disparity/depth** also preserves ImageIO's float16 or
float32 buffer in an EXR of the same precision, including its accuracy metadata.
This is a decoded representation: an 8-bit encoded disparity plane can become
Apple float16 values without gaining more than its 256 original levels. The
encoded codes remain available separately. Apple's `relative` accuracy flag is
reported; such a map is not a guarantee of absolute scene distances.

There is an important distinction between *decoded-sample preservation* and the
original scene:

- HEVC auxiliary images may have been encoded lossily by the camera. No decoder
  can reconstruct information that the original encoding discarded.
- A depth plane may store uniform depth, inverse depth, disparity, or a nonlinear
  representation. The numeric plane alone is not necessarily distance in meters.
  IPDE exposes depth-representation metadata in the inspection report and optional JSON manifest.
- `pillow-heif` can expand 10/12-bit codes into a 16-bit display range. IPDE
  explicitly disables that behavior with `hdr_to_16bit=False`; a decoded 12-bit
  value of `1` remains `1`, not `16`.
- `pillow-heif 1.5` inventories high-bit auxiliary items but rejects decoding
  them. For those items only, IPDE calls the same libheif bundled in the wheel,
  requests 10/12-bit output in unshifted `uint16` storage, verifies every code is
  within the source range, and records this fallback in the manifest. No 8-bit
  intermediary is created.

Every extractable plane gets two outputs by default:

1. A lossless exchange file: 8/16-bit PNG for unsigned integer planes, or
   losslessly compressed OpenEXR for `float16`/`float32` planes.
2. A NumPy `.npy` file containing the exact dtype, shape, byte order, and decoded
   sample bits.

An 8-bit, single-channel depth plane whose metadata declares
`uniform_disparity` and supplies valid `d_min`/`d_max` bounds also gets two
verified 32-bit float products. The calibration is performed entirely in
`float32`, in this exact order:

```text
normalized = float32(raw) / float32(255.0)
disparity = normalized * (float32(d_max) - float32(d_min)) + float32(d_min)
depth_meters = float32(1.0) / disparity
```

- `<name>_depth_disparity.exr` stores calibrated disparity in `1/m`. It remains
  linear with the source codes and larger values mean nearer geometry. This is
  usually the appropriate derived input when a proximity-style displacement or
  normal map should retain the source depth map's direction and uniform steps.
- `<name>_depth_meters.exr` stores physical distance in meters. Because distance
  is the reciprocal of disparity, nearer geometry is numerically smaller. The
  reciprocal also makes the original 8-bit quantization steps larger in the far
  field, so normals can reveal more contouring there.

Float32 calibration does **not** restore precision. A source with 256 possible
codes still has at most 256 distinct calibrated values; float32 merely stores
those values and their physical units accurately. IPDE records the source code
count, code-step sizes, value direction, and this limitation in the manifest and
EXR attributes. Other source dtypes, missing/invalid bounds, and unsupported
representations are not guessed: raw outputs remain available and the manifest
reports why calibrated products were skipped.

## Apple Spatial Photos and RAFT-Stereo

Apple Spatial Video uses stereo MV-HEVC, but an Apple Spatial Photo is a stereo
HEIC: two same-sized top-level images in an ImageIO `StereoPair` group. On macOS,
IPDE reads that group and its camera model from ImageIO while using
`pillow-heif` to decode the stereo and encoded auxiliary images. ImageIO receives the same immutable file
snapshot as pillow-heif, so the group metadata and decoded samples cannot come
from different versions of a file.

For every validated Spatial Photo, normal extraction writes:

- `<name>_spatial_left.png` and `<name>_spatial_right.png`, which preserve the
  full decoded `uint8` RGB arrays;
- exact `.npy` companions unless `--no-npy` is selected; and
- the left/right indices, camera intrinsics, extrinsic positions/rotations,
  baseline, orientation, stereo aggressors, and disparity adjustment in the
  manifest.

If the photo has a separate monoscopic display image, `<name>_display.png`
preserves its own full resolution and framing. Resizing it does not register it
to either stereo view. Stereo outputs align to the **left stereo view**, not
the display image or embedded depth grid. Decoded stereo dimensions must match
the associated camera calibration; IPDE refuses substitutions or implicit resizing.

The raw arrays are not rotated for display. They stay in the stored coordinate
system to remain registered with the camera intrinsics; the EXIF orientation is
recorded in the manifest.

The GUI lists RAFT and classical stereo as separate selectable outputs. The CLI
option `--stereo-comparison` exports both full-resolution near-is-high float32
pixel-disparity maps:

- `<name>_spatial_stereo_matching_height.exr` uses OpenCV StereoSGBM, a classical
  semi-global block matcher. It uses RGB inference copies with a mild Gaussian
  filter (sigma 1 pixel, 7×7 kernel) to accommodate differences in camera detail,
  noise, and sharpening. Raw extracted views are untouched. Disable this with
  **Tolerate camera detail differences** in the GUI or `--stereo-noise-sigma 0`.
  No resizing, normalization, gamma correction, or hole filling is applied.
  OpenCV's 1/16-pixel fixed-point disparities are preserved in float32;
  pixels rejected by the matcher are explicit `NaN` values. An independent reverse
  match must agree within one pixel at both bracketing coordinates. Small disparity
  components (200 pixels or fewer, with a two-pixel neighbor tolerance) are rejected
  after consistency and photometric checking. At either the native or shared
  detail scale, a 9×9 grayscale patch must have correlation
  at least 0.8 and mean squared **horizontal** gradient at least 1 in both
  views (Sobel derivative scaled by 1/8, in code values per pixel). Rectified
  stereo searches in one dimension; vertical edges constrain that search and
  must not be rejected merely for lacking a 2-D corner. Flat surfaces and
  horizontal edges can otherwise agree on a false near-zero
  disparity in both directions, producing enormous false distances. These checks
  reject unsupported values as NaN; accepted samples are never smoothed or filled.
  Equal computational margins on both inputs avoid OpenCV's automatic exclusion
  of a full search-width strip at the image edges. The margins are removed from
  the output, and only correspondences inside the original images can pass
  validation. Genuine occlusions and unavailable overlap remain unsupported.
- `<name>_spatial_raft_stereo_height.exr` uses the official
  [Princeton RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) model. It is
  checked against an independent mirrored reverse inference. The forward and
  reverse correspondences must agree within one pixel at both bracketing
  coordinates. This check produces a **separate support mask**; it does not erase
  forward estimates from depth/disparity/displacement. Previous versions replaced
  failures with NaN, cutting outlines into otherwise dense predictions. Programs
  that display NaN as black, or displace it to zero, made those outlines look like
  trenches. The new dense outputs preserve the predictions, including unverified
  estimates at occlusions and image boundaries. Select **RAFT supported depth**
  for the conservative, masked result. The signed flow diagnostic always retains
  the untouched forward result. Reverse inference roughly doubles
  inference time. Learned convex upsampling from the model's internal coarse
  grid can still smooth fine detail: native-sized output is not evidence of
  independent measurements at every pixel.

Exact `.npy` companions are included when **Write exact .npy companions** is
selected. `--stereo-matching` and `--raft-stereo` select either height map
individually. IPDE uses RAFT's memory-efficient `alt` correlation implementation,
does not resize or tile either view, and defaults to the Middlebury checkpoint,
which the upstream project recommends for in-the-wild images. Automatic device
selection prefers CUDA, then Apple Metal (MPS), then CPU.

Before either matcher runs, IPDE checks the actual images for small vertical
misregistration. Spatial metadata alone does not prove that corresponding features
lie on the same row. Well-distributed SIFT matches support a deterministic RANSAC
fit of `y_left = a*x_right + b*y_right + c`. Only the right inference image is
resampled vertically; its horizontal coordinates and the left reference image
remain unchanged. Fits without sufficient coverage, consensus, or subpixel
residual accuracy are not applied. Raw extracted views remain bit-exact. The
matrix, support counts, residuals, and interpolation policy are embedded in each
inference EXR, including when the JSON manifest is disabled.

Correspondences outside the right image or across padded registration borders
are marked unsupported. Classical matches and RAFT's explicitly selected supported
depth product exclude them as NaN. RAFT's dense estimates retain them, with the
support policy recorded in the EXR. This validation cannot identify every
inference error or recover geometry the cameras did not observe.

The GUI offers **Color Matching** as an opt-in inference preprocessing step. The selected
Hero view (left by default) remains unchanged. IPDE constructs three monotone
256-entry lookup tables from its RGB channel CDFs and applies them to the other
view before either matcher runs. Raw extracted left/right PNG and NPY arrays are
never modified. The manifest records the complete LUTs, Hero side, input/output
array hashes, channel statistics, interpolation and rounding policy, and confirms
that no ICC conversion, gamma correction, tone mapping, or normalization occurred.

This is per-channel histogram matching, not a literal camera-profile conversion:
it aligns marginal RGB code-value distributions but cannot guarantee pixelwise
color equality because the cameras see slightly different content. CLI users opt
in with `--color-matching` and select the unchanged Hero with
`--color-hero {left,right}`. Color-matched outputs include `_color_matched` in
their filename so processed and unprocessed inference cannot be confused.

The normal RAFT option writes only the height map. `--raft-diagnostics` adds two
specialist diagnostic EXRs. All are read back bit-for-bit before commit:

```text
signed_flow = RAFT(left, right)                   # x_right - x_left, pixels
height = (cx_right - cx_left) - signed_flow    # near is generally high
height[nonfinite_or_negative] = NaN           # impossible disparity, not abs()
depth_meters = float32(focal_px * baseline_m) / height
support = positive_disparity & in_view & reverse_consistent
supported_depth = where(support, depth_meters, NaN)  # separate, opt-in product
```

- `<name>_spatial_raft_stereo_height.exr` is the raw pixel-disparity map:
  nonnegative disparity, generally larger for nearer geometry. This quantity
  is proportional to inverse distance, not linear physical relief.
- `<name>_spatial_raft_stereo_signed_flow.exr` is the unmodified horizontal model
  correspondence. For the supplied photo its values are negative, so an image
  viewer that clamps negative values to black will show a black image even though
  the stored data is valid.
- `<name>_spatial_raft_stereo_depth_meters.exr` converts the height map using the
  stored focal length and physical baseline. Smaller values mean nearer geometry,
  the opposite direction from a displacement height map.

These are data EXRs, not display-ready grayscale pictures. A viewer that assumes
`0..1` may clip a pixel-disparity height map to white or signed flow to black.
Set the viewer/importer to Raw or Non-Color and adjust only its display exposure
or node mapping; the EXR values themselves should remain unchanged. Directly
converting a disparity EXR to integer PNG without an explicit display range often
turns every positive value into white and every `NaN` into black. That PNG is a
clipped validity mask, not a faithful rendering of the height values.

For software that expects a `0..1` displacement range, select **RAFT dense estimate —
linear depth 0–1 displacement** (or its classical counterpart) in the GUI. CLI users can
select it with `--select raft-displacement --no-npy`, or add it to legacy exports
with `--displacement-maps`. IPDE then writes
separately named full-resolution float32 derivatives:

- `<name>_spatial_stereo_matching_displacement_0_to_1.exr`
- `<name>_spatial_raft_stereo_displacement_0_to_1.exr`

Each displacement map now converts disparity to camera-axis depth before mapping
its finite positive distance range:

```text
Z = float32(focal_px * baseline_m) / disparity_pixels
height_0_to_1 = (far_m - Z) / (far_m - near_m)
```

This is linear in distance. For example, surfaces at 1, 2, and 3 meters map to
1, 0.5, and 0. Scaling disparity directly would put the middle surface at 0.25,
which distorts physical relief. The pixel-disparity products remain available
separately, with their inverse-depth units clearly labeled in the GUI.

No percentile tails are clipped. Each map uses independent bounds; constant
finite maps become zero. Nonpositive disparities and nonfinite depths become NaN,
and cannot determine the mapping bounds. EXR attributes retain the near/far
bounds in meters and the displacement scale (`far_m - near_m`), so the height
can be converted back to distance. Raw extracted data is never normalized by
requesting this explicit derivative. A full perspective reconstruction also
requires camera intrinsics; a 2D displacement texture alone is not a point cloud.

For inspection, select **RAFT depth preview** (`--select raft-preview`) or
**Classical depth preview** (`--select stereo-preview`). These explicitly named
`_depth_preview.png` files map the same full linear-depth range to 16-bit gray,
with nearer geometry white. Missing values are transparent rather than black.
PNG metadata records the depth bounds, mapping, and quantization. Previews are
for viewing only; use float EXR for displacement. No gamma or tone mapping is
applied to either the preview or the scientific data.

`--select raft-support` exports a float EXR with 1 for supported correspondence
and 0 for unsupported/occluded estimates. This mask is not depth or a confidence
probability. `--select raft-supported-depth` exports metric depth with unsupported
pixels as NaN. The equivalent `stereo-support` and `stereo-supported-depth`
products expose the sparse classical result. Preview alpha indicates whether an
estimate exists, not whether it passes support checks. All products remain
individually selectable; requesting one does not silently write the others.

These optional derivatives incur float32 rounding. They cannot fix every bad
stereo estimate: untextured, blurred, or occluded regions may remain missing or
incorrect. Independent ranges must not be compared numerically across matchers;
compare metric depth or raw disparity instead.

Apple encodes disparity adjustment as a signed integer in `[-10000, 10000]`,
mapping to `[-1, +1]` times image width. It controls the presentation zero-parallax
plane; it is not an optical calibration term. IPDE preserves both the encoded and
fractional values but deliberately does not add the presentation shift to
geometric depth. For example, the supplied `IMG_6942.HEIC` stores `240`, or
`+2.4%` of the stereo-view width.

Both StereoSGBM and RAFT-Stereo are estimates. StereoSGBM deliberately leaves
unreliable regions unmatched; RAFT can infer plausible structure in blank or
occluded regions, but neither can guarantee a flawless or mathematically exact
scene reconstruction. IPDE's precision guarantee means the algorithm outputs and
documented derivations survive EXR/NPY storage unchanged; it does not turn an
inference into a source measurement.

For normal/displacement tools, import these maps as raw or non-color data. Do not
judge numerical equivalence by converting the meter EXR to a color-managed PNG:
an sRGB transfer function and display normalization change the values being fed
to the normal calculation.

With `--manifest`, an optional `<source>_aux_manifest.json` records source/output SHA-256 hashes, dimensions,
dtype, source bit depth, decoder settings, depth semantics, auxiliary URNs, and
raw per-asset metadata blocks. PNG and EXR files are read back and compared before
they are committed. Outputs are written through same-directory temporary files
and existing files are never replaced unless `--overwrite` is requested.

## Requirements

- macOS with Python 3.14 at
  `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14`
- Qt 6.11.1 at `/opt/Qt/6.11.1/macos`
- CMake 3.24+

Create the local environment:

```sh
make setup
```

This also installs an `ipde` console command in `.venv/bin`.

The dependencies are NumPy, [pillow-heif](https://pillow-heif.readthedocs.io/),
headless OpenCV for StereoSGBM, the official
[OpenEXR Python module](https://openexr.com/en/latest/python.html), PyTorch,
SciPy, and opt_einsum. OpenEXR is installed so floating-point data can never be
silently downgraded.

RAFT inference also needs an upstream RAFT-Stereo checkout and checkpoint. IPDE
auto-detects the supplied sibling paths `../RAFT-Stereo`,
`../models/raftstereo-middlebury.pth`, and `../models.zip`.
For another layout, pass `--raft-root` and `--raft-model`, or set
`IPDE_RAFT_STEREO_DIR` and `IPDE_RAFT_MODEL`. A ZIP model source defaults to the
`raftstereo-middlebury.pth` member; `--raft-model-member` selects another supplied
checkpoint.

## Command line

Inspect a file without writing anything:

```sh
.venv/bin/python ipde_extract.py --inspect portrait.heic
```

Extract beside the source image:

```sh
.venv/bin/python ipde_extract.py portrait.heic
```

Extract several files to a chosen directory:

```sh
.venv/bin/python ipde_extract.py --output-dir /path/to/output photo1.heic photo2.heic
```

Export matching classical and RAFT-Stereo height maps with Apple Metal:

```sh
.venv/bin/python ipde_extract.py --stereo-comparison --displacement-maps --color-matching \
  --color-hero left --raft-device mps spatial.heic
```

Useful options:

```text
--inspect       Inventory planes and available product IDs without writing files
--select ID     Export only this product; repeat to select more (see --inspect --json)
--json          Emit one machine-readable JSON object per input file
--overwrite     Atomically replace colliding output files
--no-npy        Omit exact-array companions (PNG/EXR remain verified and lossless)
--manifest      Write a provenance JSON manifest (off by default)
--no-metric-depth  Omit calibrated float32 distance in meters
--no-physical-disparity  Omit calibrated float32 disparity in inverse meters
--stereo-comparison  Export StereoSGBM and RAFT-Stereo height maps together
--displacement-maps  Also export per-map float32 0..1 displacement linear in depth
--stereo-matching  Export only the classical full-resolution height map
--stereo-max-disparity PIXELS  Override the classical disparity search range
--stereo-noise-sigma PIXELS  Shared-detail scale, default 1; 0 disables, maximum 3
--color-matching  Match the non-Hero view's RGB histograms before inference
--color-hero {left,right}  Select the unchanged Hero view (default: left)
--raft-stereo   Export only the full-resolution RAFT-Stereo height map
--raft-diagnostics  Also export RAFT signed flow and metric distance
--raft-device {auto,cpu,mps,cuda}  Select inference backend
--raft-iterations N  Select 1..256 recurrent updates (default: 32)
--raft-root PATH  Select a RAFT-Stereo checkout
--raft-model PATH  Select a .pth checkpoint or models.zip
```

The process returns nonzero if any input fails. In JSON mode, errors are emitted
as JSON objects as well, which is what the GUI consumes.

## Qt GUI

Build and launch against the supplied Qt installation:

```sh
make gui
```

Files may be added with the picker or drag-and-drop. IPDE inventories them first,
listing each available output with dimensions and precision. Select one row and
click **Export this map**, or check multiple rows and click **Export checked**.
Only those products are written; RAFT-only exports do not run classical matching
or write stereo RGB views, gain maps, diagnostics, or other unselected products.
**Write exact .npy companions** is off by default. PNG/EXR exports are still
lossless and verified bit-for-bit. **Write JSON manifest** is also off by default.
When enabled, its selection-specific name allows separate exports to the same
folder. Turning it off does not delete or overwrite previously exported manifests.
Numerical derivations remain embedded in inference EXRs without a sidecar.

Depth source samples, calibrated disparity, and metric distance are separate
choices. Spatial photos additionally offer RAFT and classical pixel-disparity
maps (inverse depth), optional linear-depth 0–1 displacement maps, and RAFT flow/distance diagnostics.
Raw pixel-disparity EXRs can look white in a viewer restricted to 0–1. Choose the
explicit 0–1 product for that workflow; unmatched classical regions remain NaN
and may display black. No smoothing or invented hole filling is applied.

Use **RAFT model: Choose…** to select a `.pth`, `.pt`, or `.zip` checkpoint, and
**RAFT source folder: Choose…** if automatic source lookup fails. The source
folder must contain `core/raft_stereo.py`. For a ZIP, the optional member field
selects the checkpoint (default `raftstereo-middlebury.pth`); nested archive
folders are supported. These paths persist between launches. Explicit invalid
paths report an error instead of silently selecting a different model. Use the
original upstream checkpoint filename to identify its model architecture.

Color Matching is opt-in. It changes only inference inputs. The output folder
may be left blank to write beside each source, and extraction runs one source at
a time to bound peak memory use.

CLI examples for exporting a single product:

```sh
.venv/bin/python ipde_extract.py --inspect --json portrait.heic
.venv/bin/python ipde_extract.py --select raw:0 --no-npy portrait.heic
.venv/bin/python ipde_extract.py --select raft-displacement --no-npy \
  --raft-model /path/to/raftstereo-middlebury.pth spatial.heic
```

Use the actual raw ID from the inventory; `raw:0` is only an example. With no
`--select`, the CLI preserves its legacy full-extraction behavior.

Build without launching:

```sh
make build
```

The app bundle is `build/IPDE.app`. The Python extractor and package are copied
into the bundle resources, while the configured Python runtime remains external.

## Validation

```sh
make test
make smoke
```

The test suite covers exact 8/16-bit PNG round trips, exact NPY round trips,
floating-point EXR round trips when OpenEXR is installed, metadata preservation,
collision refusal, auxiliary enumeration, exact float32 metric-depth operation
ordering, source-quantization accounting, Apple stereo-group validation,
presentation-versus-geometric disparity semantics, full-resolution StereoSGBM
with explicit unmatched pixels, exact per-channel histogram LUT construction and
raw-view isolation, exact RAFT float32 derivation, comparison-only versus
diagnostic RAFT output integration, linear-depth displacement spacing, vertical registration, visibility and reverse-match validation,
selective exports, manifest opt-in, explicit-path failures, and nested checkpoint ZIPs,
resource resolution, and CLI behavior.

Earlier versions were exercised on several full-resolution spatial photos, but
those files are not part of the repository. For current validation, run the test
suite and use your own spatial photo to assess correspondence quality. Synthetic
known-shift tests distinguish the disparity sign and calibration math from
viewer clipping; float32 EXR round trips verify stored data independently of a
viewer. A precision-preserving export does not guarantee accurate inferred
geometry in untextured or occluded regions.

## Output naming

For a typical Apple portrait named `IMG_0001.HEIC`, outputs look like:

```text
IMG_0001_depth.png
IMG_0001_depth.npy
IMG_0001_depth_disparity.exr
IMG_0001_depth_meters.exr
IMG_0001_hdr_gain_map.png
IMG_0001_hdr_gain_map.npy
IMG_0001_hdr_gain_map_metadata0.xmp
IMG_0001_spatial_left.png
IMG_0001_spatial_left.npy
IMG_0001_spatial_right.png
IMG_0001_spatial_right.npy
IMG_0001_spatial_stereo_matching_color_matched_height.exr
IMG_0001_spatial_stereo_matching_color_matched_height.npy
IMG_0001_spatial_stereo_matching_color_matched_displacement_0_to_1.exr
IMG_0001_spatial_stereo_matching_color_matched_displacement_0_to_1.npy
IMG_0001_spatial_raft_stereo_color_matched_height.exr
IMG_0001_spatial_raft_stereo_color_matched_height.npy
IMG_0001_spatial_raft_stereo_color_matched_displacement_0_to_1.exr
IMG_0001_spatial_raft_stereo_color_matched_displacement_0_to_1.npy
IMG_0001_aux_manifest.json  # only with --manifest
```

With `--raft-diagnostics`, the additional names are
`IMG_0001_spatial_raft_stereo_color_matched_signed_flow.{exr,npy}` and
`IMG_0001_spatial_raft_stereo_color_matched_depth_meters.{exr,npy}` when Color
Matching is active. Without `--color-matching`, the `_color_matched` component is
omitted.

Indices are added when a container has multiple top-level images or repeated
auxiliary types. Auxiliary metadata sidecars preserve their original bytes; the
optional manifest also contains a base64 copy and SHA-256 digest.
