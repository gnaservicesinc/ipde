# Stereo detail and border audit: IMG_0845 and IMG_0835

This extends the [earlier RAFT/math audit](stereo-math-audit-2026-09-21.md).
The current changes affect classical StereoSGBM matching. They do not change
RAFT input processing, calibrated disparity-to-depth conversion, or raw extraction.

## Findings from the supplied sources

Fresh decoding of `IMG_0845.HEIC` exactly matches both user-supplied spatial PNGs.
The app is using the correct stereo pair at 2688×2016. Both decoded views have
full rectangular RGB grids; they do not contain the broad transparency bars
seen in the inferred depth. The original files remain unchanged.

| Photo | Left HEIF image | Right HEIF image | Left physical lens | Right physical lens |
| --- | ---: | ---: | --- | --- |
| IMG_0845 | 1 | 2 | 6.765 mm f/1.78 | 2.22 mm f/2.2 |
| IMG_0835 | 2 | 1 | 2.22 mm f/2.2 | 6.765 mm f/1.78 |

These lens values come from each image's EXIF LensModel/FocalLength, not from an
assumption about the left view. The stored stereo intrinsics already describe
the processed, matched fields of view. Do not reapply the physical focal-length
ratio to these arrays. Nor can the cleaner physical camera always be identified
by the left/right label: its assignment switches between these captures.

For IMG_0845, the calibrated focal length is 1978.7499668598175 pixels, baseline
0.019272 meters, and principal-point difference zero. Accepted disparity remains
in pixels; camera-axis depth is `float32(f) * float32(B) / disparity`. The supported
range is 0.243476–0.421667 meters. These are inferred calibrated distances, not
independently measured physical ground truth.

Apple's stored pair still has a measured median vertical residual of 0.855 pixels
on IMG_0845. The existing vertical-only fit reduces it to 0.187 pixels (90th
percentile 0.483). It introduces **zero invalid border pixels** for this photo.
Thus neither source selection nor vertical registration explains its large bars.
Horizontal parallax is retained because it is the signal used to infer depth.

The broad bars come from OpenCV's search domain: its
[StereoSGBM implementation](https://raw.githubusercontent.com/opencv/opencv/4.x/modules/calib3d/src/stereosgbm.cpp)
starts computing at `maxDisparity`. With the default 336-pixel search, it skips
the first 336 columns regardless of their actual disparity. The independent
reverse check adds a corresponding loss near the opposite edge. This behavior
must be accommodated by the caller.

The cameras also have unequal high-frequency detail. Across 4,428 corresponding
31×31 feature patches in IMG_0845, median high-frequency variance is about
100.7 on the left and 75.5 on the right. This is evidence of detail differences;
it cannot separate true texture, sharpening and sensor noise. Copying sharp RGB
texture into depth would turn albedo variation into invented surface relief.

## Changes

- Add equal computational margins to both SGBM inputs in each direction and
  remove them from the result. This preserves relative x coordinates and pixel
  units. Original-image visibility, independent reverse agreement, photometric
  support and small-component rejection still decide which estimates survive.
  Padded correspondences are not exported as observed geometry.
- Match RGB inference copies with a symmetric 7×7 Gaussian, sigma 1 pixel.
  Photometric checking can find support at native or shared detail scale. Each
  scale still requires correlation ≥0.8 and observable horizontal structure in
  both views; the global acceptance thresholds have not been lowered.
- Compute texture evidence in the right source image before sampling it at the
  correspondence. A disparity discontinuity cannot manufacture this evidence.
- Preserve raw arrays and the left reference grid. Accepted disparities retain
  SGBM's 1/16-pixel quantization; there is no depth smoothing or hole filling.
- Expose **Tolerate camera detail differences** (enabled by default) in Qt.
  `--stereo-noise-sigma 0` disables the detail processing, while values through
  3 pixels allow explicit CLI control. EXR metadata records filtering, padding,
  rounding, source/input hashes and support counts.

## Real-photo checks

Baseline is the code immediately before these border/detail changes, including
the earlier corrected horizontal texture gate. Both comparisons use the same
raw views, registration, search range, native output grid and calibration.

| Measurement | IMG_0845 before → after | IMG_0835 before → after |
| --- | --- | --- |
| Supported pixels | 55.5932% → 74.9873% | 25.8966% → 34.8742% |
| First/last nonempty column, zero-based | 336–2490 → 101–2683 | 336–2405 → 54–2683 |
| Supported independent feature samples | 4,411 → 4,427 | 3,401 → 3,880 |
| Common-feature disparity error, median | 0.1613 → 0.1603 px | 0.1577 → 0.1584 px |
| Common-feature disparity error, p90 | 0.4017 → 0.3864 px | 0.4266 → 0.4070 px |
| Common-feature disparity error, p99 | 0.6634 → 0.6439 px | 0.8845 → 0.8312 px |

The independent comparison requests 12,000 SIFT features, uses a 0.6 descriptor
ratio and <0.5-pixel vertical residual, then compares horizontal correspondences
against bilinearly sampled output disparity. Common-feature rows above use the
same locations before and after. The newly supported 479 cabinet features have
median/p90/p99 errors of 0.184/0.444/0.992 pixels. This checks correspondence
consistency in textured areas; it does not validate every recovered pixel.

Visual inspection confirms that the artificial search-width bars are gone and
the towel surface is more continuous. Genuine missing overlap remains on the
left, with occlusions and ambiguous areas elsewhere. The cabinet still has only
34.9% supported coverage: classical matching remains a sparse, conservative
result there, not a complete displacement surface. Missing pixels are NaN in
EXR and transparent in preview PNG; a black viewer background is not zero depth.
Previews are explicitly scaled for viewing; raw and metric EXRs are not.

All 70 tests passed, including six invalid-scale subtests. New numerical
regressions cover search-width independence, unavailable border rejection,
unequal blur/noise at a known shift, unrelated-image rejection, unchanged raw
inputs, and CLI-to-matcher option forwarding. On the synthetic flat plane with
an 8-pixel shift, accepted errors stay below 0.2 pixels even with unequal detail.

Both real photos were exported again through the CLI. All ten EXR/PNG outputs
passed exact writer readback, file hashes, and independent depth/displacement/
preview derivation checks. Fresh source decoding still matches cached raw input;
the towel views additionally match the user's PNGs sample-for-sample. Qt 6.11.2
build and native Cocoa `--smoke-test` passed; bundled Python files match source.
This is not a native interactive screenshot/click-through verification.

## Artifacts and reproduction

- Towel outputs: `out/IMG_0845-reviewed-20260921/`
- Cabinet outputs: `out/IMG_0835-border-detail-20260921/`
- Exact comparison statistics: `out/stereo-audit-0845/comparison.json`
- Output and source checks: `out/stereo-audit-0845/verification.json`
- Read-only verifier: `out/stereo-audit-0845/verify.py`

`IMG_0845.HEIC` SHA-256:
`28b6457cbf35d3b16f6ff2e108a349e0ca769a3f64c34733f60a8f3895a537c2`.
`IMG_0835.HEIC` SHA-256:
`a5e2368b7f8e2d97dc3cd866f84ee02f314282d76d1c7e60624ec63f71d2d879`.

Run from the repository, choosing a fresh output directory:

```sh
env PYTHONPATH=src .venv/bin/python -m ipde /path/to/IMG_0845.HEIC \
  --output-dir /path/to/new-output \
  --select stereo-height --select stereo-supported-depth \
  --select stereo-preview --select stereo-support --select stereo-displacement \
  --stereo-noise-sigma 1 --no-npy --manifest
env PYTHONPATH=src .venv/bin/python -m pytest -q
env PYTHONPATH=src .venv/bin/python out/stereo-audit-0845/verify.py
build/IPDE.app/Contents/MacOS/IPDE --smoke-test
```
