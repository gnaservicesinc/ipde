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

There is an important distinction between *decoded-sample preservation* and the
original scene:

- HEVC auxiliary images may have been encoded lossily by the camera. No decoder
  can reconstruct information that the original encoding discarded.
- A depth plane may store uniform depth, inverse depth, disparity, or a nonlinear
  representation. The numeric plane alone is not necessarily distance in meters.
  IPDE preserves the depth-representation metadata in the JSON manifest.
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
IPDE reads that group and its camera model from ImageIO while continuing to use
`pillow-heif` as the only pixel decoder. ImageIO receives the same immutable file
snapshot as pillow-heif, so the group metadata and decoded samples cannot come
from different versions of a file.

For every validated Spatial Photo, normal extraction writes:

- `<name>_spatial_left.png` and `<name>_spatial_right.png`, which preserve the
  full decoded `uint8` RGB arrays;
- exact `.npy` companions unless `--no-npy` is selected; and
- the left/right indices, camera intrinsics, extrinsic positions/rotations,
  baseline, orientation, stereo aggressors, and disparity adjustment in the
  manifest.

The raw arrays are not rotated for display. They stay in the stored coordinate
system to remain registered with the camera intrinsics; the EXIF orientation is
recorded in the manifest.

Selecting **Compare Stereo Matching + RAFT-Stereo height maps** in the GUI, or
passing `--stereo-comparison`, creates two directly comparable, full-resolution,
near-is-high float32 height maps:

- `<name>_spatial_stereo_matching_height.exr` uses OpenCV StereoSGBM, a classical
  semi-global block matcher. It consumes the decoded RGB codes directly without
  grayscale conversion, resizing, normalization, gamma correction, or hole
  filling. OpenCV's 1/16-pixel fixed-point disparities are preserved in float32;
  pixels rejected by the matcher are explicit `NaN` values.
- `<name>_spatial_raft_stereo_height.exr` uses the official
  [Princeton RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) model. It is
  normally dense, including in blank or occluded regions where classical matching
  has no reliable correspondence.

Exact `.npy` companions are included when **Write exact .npy companions** is
selected. `--stereo-matching` and `--raft-stereo` select either height map
individually. IPDE uses RAFT's memory-efficient `alt` correlation implementation,
does not resize or tile either view, and defaults to the Middlebury checkpoint,
which the upstream project recommends for in-the-wild images. Automatic device
selection prefers CUDA, then Apple Metal (MPS), then CPU.

The GUI enables **Color Matching** by default for this comparison. The selected
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
height = abs(signed_flow + (cx_right - cx_left)) # near is generally high
depth_meters = float32(focal_px * baseline_m) / height
```

- `<name>_spatial_raft_stereo_height.exr` is the comparison/displacement map:
  nonnegative pixel disparity, generally larger for nearer geometry.
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

For software that insists on a `0..1` displacement range, enable **0–1
displacement maps** in the GUI or pass `--displacement-maps`. IPDE then writes
separately named full-resolution float32 derivatives:

- `<name>_spatial_stereo_matching_displacement_0_to_1.exr`
- `<name>_spatial_raft_stereo_displacement_0_to_1.exr`

IPDE pools the finite pixel-disparity samples from every selected matcher, finds
one shared 1st–99th percentile range, and applies
`clip((height - lower) / (upper - lower), 0, 1)` to both maps. Sharing the exact
recorded bounds makes the StereoSGBM and RAFT results visually comparable rather
than independently auto-scaling them. Near remains high. StereoSGBM `NaN` pixels
remain `NaN` because they are genuinely unmatched; RAFT is normally dense. The
original `_height.exr` and `.npy` pixel-disparity products are retained unchanged,
so this convenient normalization never replaces the scientific algorithm output.

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

A `<source>_aux_manifest.json` records source/output SHA-256 hashes, dimensions,
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
auto-detects the supplied sibling paths `../RAFT-Stereo` and `../models.zip`.
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
--inspect       Inventory and decode auxiliary planes without writing files
--json          Emit one machine-readable JSON object per input file
--overwrite     Atomically replace colliding output files
--no-npy        Omit exact-array companions (PNG/EXR remain verified and lossless)
--no-metric-depth  Omit calibrated float32 distance in meters
--no-physical-disparity  Omit calibrated float32 disparity in inverse meters
--stereo-comparison  Export StereoSGBM and RAFT-Stereo height maps together
--displacement-maps  Also export shared-range float32 0..1 displacement maps
--stereo-matching  Export only the classical full-resolution height map
--stereo-max-disparity PIXELS  Override the classical disparity search range
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
showing every depth, auxiliary, alpha, and spatial-view plane with its dimensions,
dtype, and source bit depth. Spatial Photos are labeled with their left/right
indices, baseline, and disparity adjustment. Extraction runs one source at a time
to keep peak memory use bounded. Metric-depth EXR calibration is enabled by
default and can be toggled independently from calibrated disparity. Spatial
comparison is opt-in and writes one classical Stereo Matching height map plus
one RAFT-Stereo height map. Color Matching is enabled for that comparison by
default, with explicit Left Hero and Right Hero choices. Explicit 0–1
displacement derivatives are also enabled by default; unchecking that option
leaves only the scientific pixel-disparity maps. Automatic, Apple Metal, and CPU
RAFT device choices are available. The output folder may be left blank to write
beside each source.

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
diagnostic RAFT output integration, shared-range float32 displacement mapping,
resource resolution, and CLI behavior.

The supplied `IMG_6942.HEIC` was also exercised end-to-end at its full
2688x2016 stereo resolution with the supplied Middlebury checkpoint, 32 recurrent
updates, and Apple Metal. The comparison height EXRs passed exact read-back
verification; optional RAFT diagnostics use the same verified output path.

`IMG_6979.HEIC` was validated through the complete left-Hero Color Matching path.
The right-view RGB means changed from approximately `(110.72, 137.72, 142.26)` to
`(136.77, 133.45, 131.36)`, closely matching the left Hero's
`(136.73, 133.47, 131.41)`, while the raw exported views remained bit-identical.
StereoSGBM finite coverage increased modestly from 63.31% to 63.95%; this confirms
that its visible color cast was real but was not the main source of the attached
black/white PNG artifact. The color-matched StereoSGBM and RAFT EXRs both passed
exact read-back verification.

`IMG_6998.HEIC` reproduced the apparent two-color failure in a `0..1`-clipping
viewer even though its scientific StereoSGBM map contained 3,352 distinct finite
pixel-disparity values. With the shared recorded range of approximately
`31.411194..65.9375` pixels, the new displacement EXRs contain 554 distinct finite
StereoSGBM levels and 3,729,527 RAFT levels. Both span `0..1` and passed exact EXR
read-back verification; the RAFT map has no nonfinite pixels.

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
IMG_0001_aux_manifest.json
```

With `--raft-diagnostics`, the additional names are
`IMG_0001_spatial_raft_stereo_color_matched_signed_flow.{exr,npy}` and
`IMG_0001_spatial_raft_stereo_color_matched_depth_meters.{exr,npy}` when Color
Matching is active. Without `--color-matching`, the `_color_matched` component is
omitted.

Indices are added when a container has multiple top-level images or repeated
auxiliary types. Auxiliary metadata sidecars preserve their original bytes; the
manifest also contains a base64 copy and SHA-256 digest.
