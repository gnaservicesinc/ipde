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

The GUI lists RAFT and classical stereo as separate selectable outputs. The CLI
option `--stereo-comparison` exports both full-resolution near-is-high float32
pixel-disparity maps:

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
height[height < 0] = NaN                       # invalid correspondence, not abs()
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

For software that expects a `0..1` displacement range, select **RAFT height —
0–1 displacement** (or its classical counterpart) in the GUI. CLI users can
select it with `--select raft-displacement --no-npy`, or add it to legacy exports
with `--displacement-maps`. IPDE then writes
separately named full-resolution float32 derivatives:

- `<name>_spatial_stereo_matching_displacement_0_to_1.exr`
- `<name>_spatial_raft_stereo_displacement_0_to_1.exr`

Each displacement map uses its own finite minimum and maximum and applies
`(height - minimum) / (maximum - minimum)` in float32. No percentile tails are
clipped by default, and a classical result cannot squash the contrast of the
RAFT result. Constant finite maps become zero; NaN remains NaN. The exact range
and formula are recorded in the EXR and manifest. These optional derivatives
change units and incur float32 rounding; the raw pixel-disparity product remains
available separately and is never changed by requesting a displacement map.
Independent scales mean normalized maps must not be compared numerically across
matchers; compare the raw pixel-disparity maps for that purpose.

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
--no-metric-depth  Omit calibrated float32 distance in meters
--no-physical-disparity  Omit calibrated float32 disparity in inverse meters
--stereo-comparison  Export StereoSGBM and RAFT-Stereo height maps together
--displacement-maps  Also export per-map full-range float32 0..1 displacement maps
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
listing each available output with dimensions and precision. Select one row and
click **Export this map**, or check multiple rows and click **Export checked**.
Only those products are written; RAFT-only exports do not run classical matching
or write stereo RGB views, gain maps, diagnostics, or other unselected products.
**Write exact .npy companions** is off by default. PNG/EXR exports are still
lossless and verified bit-for-bit. A provenance manifest is always included;
its selection-specific name allows separate exports to the same folder.

Depth source samples, calibrated disparity, and metric distance are separate
choices. Spatial photos additionally offer RAFT and classical pixel-disparity
height maps, optional 0–1 displacement maps, and RAFT flow/distance diagnostics.
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
diagnostic RAFT output integration, per-map float32 displacement mapping,
selective exports, explicit-path failures, and nested checkpoint ZIPs,
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
