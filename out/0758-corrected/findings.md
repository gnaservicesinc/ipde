# IMG_0758 depth investigation

The source was decoded with pillow-heif without changing its samples. The extracted
left and right arrays match the supplied PNG arrays exactly. No source files or
previous exports were overwritten; corrected exports are in this directory.

## Findings and changes

- **Manifest:** JSON sidecars are now opt-in in both the Qt GUI and CLI
  (`--manifest`). Verification and embedded EXR derivations still run without one.
- **Vertical registration:** the stereo metadata describes rectified cameras, but
  image features had a median 2.62-pixel vertical discrepancy. A constrained
  right-image vertical affine correction reduced the inlier median to 0.24 pixels
  (90th percentile 0.68 pixels). It changes inference inputs only. Horizontal
  coordinates and the raw exports are preserved. Fits require distributed feature
  support, consensus, and bounded residual/correction size.
- **Displacement math:** pixel disparity is proportional to inverse depth. Scaling
  it directly to 0–1 does not preserve physical depth spacing. The explicit
  displacement product now computes `Z = focal_pixels * baseline_meters / disparity`
  and then `(far - Z) / (far - near)`. Distances 1, 2, 3 meters therefore become
  heights 1, 0.5, 0. Raw disparity remains a separate product. The EXR carries
  the meter bounds and displacement scale; no percentile clipping is applied.
- **Invalid correspondences:** off-image/padded-border predictions are NaN in
  geometry products. Classical matching also requires an independent reverse
  match and rejects small disconnected components after that check. Rejected
  values do not control the displacement range. Retained fixed-point estimates
  remain unchanged; there is no smoothing or hole filling.

## Check against Apple

A separate native AVDepthData read of the original HEIC gave 0.11863–0.29561 m.
The existing pillow-heif calibration gave 0.11863–0.29561 m, with a maximum
per-pixel difference of approximately 0.0001074 m. The original Apple conversion
was not the source of the large visual discrepancy. Native AVDepthData reports
relative accuracy for this file, so it is a reference estimate, not exact scene
measurements. Its primary-camera map is also not registered to the stereo left
view, so cross-camera per-pixel error comparisons would be misleading.

The supplied RAFT PNG contains only 65535 values: it has actually been clipped
white. Its EXR contains 111.44–314.64-pixel disparities. The supplied Apple PNG,
by contrast, contains a broad uint16 range, despite appearing white in the chat
preview. The comparison chart renders the numerical arrays explicitly.

## Corrected sample

- RAFT: 90.73% finite geometry after visibility checks; depth 0.12929–0.33277 m.
- Classical: 27.32% finite geometry after alignment, reverse checking, and
  component rejection; retained disparities 122.8125–335 pixels.
- Classical displacement now spans its finite range: 25th/50th/75th percentiles
  approximately 0.660/0.768/0.863, rather than being compressed above 0.998 by
  isolated false far-distance matches.
- At jointly valid pixels, 95.2% of retained classical estimates agree with the
  corrected RAFT estimate within 3 pixels. This is agreement between estimators,
  not a ground-truth accuracy score.

Classical stereo remains sparse on this blurred/unevenly exposed pair. RAFT is
dense over the shared field of view but still inferred. Neither algorithm can
recover unobserved geometry or promise an exact reconstruction. The corrected
0–1 map is linear in camera-axis depth; a perspective point cloud additionally
requires back-projection through the camera intrinsics.

All outputs were exported at their original decoded resolution as losslessly
compressed float32 EXR and read back bit-for-bit. The updated application built
and passed the GUI smoke test; all 50 automated tests passed. Tests include
known-depth spacing, a known horizontal/vertical stereo shift, visibility,
contradictory reverse matches, isolated far-disparity outliers, and manifest opt-in.

## References

- [Apple: Writing spatial photos](https://developer.apple.com/documentation/ImageIO/writing-spatial-photos)
- [Apple: Capturing photos with depth](https://developer.apple.com/documentation/avfoundation/capturing-photos-with-depth)
- [RAFT-Stereo: disparity-to-depth conversion](https://github.com/princeton-vl/RAFT-Stereo#converting-disparity-to-depth)
- [OpenCV StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)
