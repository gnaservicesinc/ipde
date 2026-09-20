# IMG_0835 validation

The updated app is `/opt/ipde/ipde/build/IPDE.app`. This folder contains fresh exports; the original HEIC and supplied GIMP PNGs were not modified.

| Image or representation | Dimensions | Precision |
| --- | --- | --- |
| Display image | 5712 × 4284 | 8-bit RGB encoded samples |
| Left stereo view (HEIF image 2, GIMP ch) | 2688 × 2016 | 8-bit RGB encoded samples |
| Right stereo view (HEIF image 1, GIMP ch2) | 2688 × 2016 | 8-bit RGB encoded samples |
| HDR gain map | 2856 × 2142 | 8-bit encoded samples |
| Encoded disparity codes | 768 × 576 | 8-bit integer, 256 distinct codes |
| Apple native disparity | 768 × 576 | float16, 256 distinct values |
| Generated stereo products | 2688 × 2016 | float32 estimates |

Apple's native float16 disparity is bit-exactly equal to the calibrated encoded codes rounded to float16 for this photo. Both representations are now available. Apple labels this depth as relative accuracy, so it is not an absolute-distance ground truth. All raw arrays remain unnormalized.

The display image has its own framing. Stereo results use the left stereo view's coordinates. Feature matches to the supplied GIMP exports give median position errors of 0.0086 pixels (left, 2876 matches) and 0.0072 pixels (right, 2883 matches). RGB code differences average about 1.5–1.6 on a 0–255 scale; the two decoders' RGB values are not claimed bit-identical.

Classical stereo now rejects photometrically unsupported matches, including ambiguous flat surfaces and single edges, before depth inversion. For the color-matched case, the 10th–90th percentiles of valid displacement improve from 0.9957–0.9983 to 0.5183–0.7588 without percentile clipping. About 25.2% of pixels pass the conservative classical checks; unmatched pixels remain NaN. Every retained unprocessed classical disparity is bit-identical to its previous value.

RAFT uses the full 2688 × 2016 RGB views, with the existing measured vertical correction on its right inference input. Independent mirrored reverse inference now rejects inconsistent geometry. About 89.9% of pixels remain valid. Signed-flow diagnostics retain the unfiltered forward output. The model still uses a fourfold internal feature reduction and learned upsampling; this change does not claim to restore missing fine detail or eliminate all model smoothing. Reverse verification roughly doubles inference time.

Validation: 59 regression tests passed; Qt 6.11.2 build and launch smoke passed; embedded Python matches source; 16 output files passed independent manifest SHA-256 checks and each export passed its pixel round-trip check. Native UI inspection was unavailable because the computer-use bridge failed to start.
