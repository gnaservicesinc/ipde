# Stereo math and output audit: IMG_0835

Source: `IMG_0835.HEIC`, 3,705,032 bytes, SHA-256
`a5e2368b7f8e2d97dc3cd866f84ee02f314282d76d1c7e60624ec63f71d2d879`.
The original file and supplied exports were read without modification.

## Geometry checked from the source

A fresh pillow-heif 1.7.0 decode exactly matched the previously extracted stereo
arrays. ImageIO identifies image 2 as left and image 1 as right, both 2688×2016,
orientation 1. Intrinsics have `fx = fy = 1948.3553034067154` pixels and equal
principal points `(1344, 1008)`. The baseline is 0.019272 meters. Both camera
rotations are identity. Thus `f * B = 37.54870340725422` pixel-meters before the
documented float32 conversion.

For these rectified cameras:

```text
xL = f X/Z + cxL
xR = f (X-B)/Z + cxR
flow = xR-xL
d = (cxR-cxL)-flow = f B/Z
Z = f B/d
```

`Z` is camera-axis distance, not radial distance and not image brightness.
The encoded 240 disparity adjustment is a presentation shift of 2.4% of image
width. Applying it to raw pixel correspondence would change the geometry
incorrectly. See Apple's [spatial metadata explanation](https://developer.apple.com/videos/play/wwdc2024/10166/)
and the upstream [RAFT-Stereo implementation and depth convention](https://github.com/princeton-vl/RAFT-Stereo).
The local Middlebury checkpoint configuration agrees with upstream; inputs remain
native RGB code values cast to float32, and RAFT performs its own internal input
transform. No additional gamma or input normalization is applied by IPDE.

The actual pair has small vertical misregistration despite its idealized
metadata. The existing vertical-only correction reduces median feature residual
from 2.107 to 0.206 pixels. Horizontal coordinates, focal length, baseline, and
the left reference grid are unchanged. This empirical correction is useful, but
does not establish perfectly calibrated physical measurement.

## Findings and changes

1. **The black signed-flow PNG was completely clipped.** All PNG pixels are zero;
   the float EXR contains values from -75.630493 to -29.411419 pixels, median
   -55.744667. Those numbers are not clustered near zero. Signed flow is a
   correspondence diagnostic, not a near-white displacement texture.
2. **Depth outlines were NaNs introduced by support filtering.** The previous
   metric EXR has 549,129 NaNs (10.1334%), and no zero-valued depths. That exactly
   accounts for the black fraction in the supplied PNG. A reverse-match failure
   can indicate occlusion or disagreement; it is not a measured zero-depth trench.
   Dense RAFT depth/disparity now preserve the forward predictions. A separate
   binary support mask and explicitly selected supported-depth EXR retain the
   conservative result. No hole filling or new model prediction was introduced.
3. **The classical PNG was a clipped mask.** Every finite disparity in the
   supplied EXR (35.25–76.5625 pixels) became white; every NaN became black.
   The file did not show the actual variation in disparity.
4. **The classical texture gate used the wrong dimensionality.** A minimum
   structure-tensor eigenvalue requires a 2-D corner. Rectified stereo estimates
   only horizontal displacement: vertical texture is informative, while purely
   horizontal texture is not. The filter now checks horizontal gradient energy.
   Correlation, reverse consistency, visibility and small-component rejection
   remain. Accepted SGBM samples retain their exact 1/16-pixel values.
5. **Viewing and geometry are explicit separate products.** Depth-preview PNGs
   use a recorded full-range linear depth mapping with alpha for missing values.
   They are quantized viewing aids. Float32 EXR/NPY outputs retain scientific
   values; normalized displacement remains a separately selected derivative.
   Preview alpha indicates a finite estimate, not correspondence support.

## Validation on the supplied photo

Two fresh full-resolution RAFT forward/reverse runs used Metal, 32 iterations,
the Middlebury checkpoint SHA-256
`d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819`, and upstream
source commit `6e93ed2169bd858dbb43033988563f3b0bb49506`.

- New raw signed flow is **bit-for-bit equal** to the supplied EXR.
- New supported-depth EXR is **bit-for-bit equal** to the previous masked depth.
- New dense depth is 100% finite, range **0.496476–1.276671 meters**. Restoring
  predictions does not make the unsupported 10.1334% verified geometry.
- Dense displacement uses the full depth range, near = 1, far = 0. Its preview
  exactly equals rounding that float derivative to uint16, with no gamma change.
- Classical supported coverage is 25.8966%, versus 25.1973% previously for the
  same non-color-matched pair. Color-matched coverage is 25.7978%. Large gaps
  remain on low-texture surfaces; this is not a reliable dense reconstruction.
- Every generated EXR/PNG passed the writer's mandatory sample-exact readback.

An independent SIFT comparison used 12,000 requested features, a 0.6 descriptor
ratio, positive horizontal disparity, and less than 0.5-pixel vertical residual
after registration. At 3,900 feature correspondences, RAFT absolute disparity
errors against feature locations were 0.175 / 0.479 / 1.052 pixels at the
50th / 90th / 99th percentiles. At 3,401 supported feature samples, classical
errors were 0.158 / 0.427 / 0.885 pixels. These are correspondence consistency
checks, **not surveyed depth ground truth**, and do not measure accuracy on
textureless or occluded surfaces.

Regression coverage includes independent pinhole projection with unequal
principal points, equal-distance displacement steps, horizontal observability
without corners through the actual SGBM matcher, retained RAFT predictions under
reverse disagreement, separate support/metric exports, preview mapping and
transparency, and collision preflight before inference. All 65 tests passed.
Qt 6.11.2 was found installed (the supplied 6.11.1 path is absent); the Release
build and native Cocoa smoke test passed. Native screenshot inspection was
unavailable because the computer-use native pipe failed to start.

## Reproduce the exports

```sh
env PYTHONPATH=src .venv/bin/python -m ipde /path/to/IMG_0835.HEIC \
  --output-dir out/reviewed \
  --select raft-depth --select raft-displacement --select raft-preview \
  --select raft-support --select raft-supported-depth --select raft-flow \
  --select stereo-height --select stereo-supported-depth --select stereo-preview \
  --select stereo-support --select stereo-displacement \
  --raft-root /opt/ipde/RAFT-Stereo \
  --raft-model /opt/ipde/models/raftstereo-middlebury.pth \
  --raft-device mps --no-npy --manifest
```

Use `--color-matching` in a separate export for the optional histogram-matched
comparison. Generated files for this audit are in
`out/IMG_0835-reviewed-20260921`; numerical reports and independent feature
coordinates are in `out/stereo-audit-20260921`.
