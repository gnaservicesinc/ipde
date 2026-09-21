"""Geometric regressions independent of display brightness and model weights."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from ipde.spatial import (
    DisplacementMappingError, StereoMatchingOptions, correspondence_validity,
    linear_depth_displacement, register_stereo_rows, run_stereo_matching,
    reverse_correspondence_support, stereo_photometric_support,
)


class DepthGeometryTests(unittest.TestCase):
    def test_reverse_inconsistent_estimates_are_not_exported_as_geometry(self):
        rgb = np.zeros((40, 96, 3), np.uint8)
        forward = np.full((40, 96), 64, np.int16)  # four pixels, fixed point
        reverse = forward.copy()
        reverse[10:30, 36:56] = 192  # contradictory twelve-pixel match
        with (patch("cv2.StereoSGBM.create") as create,
              patch("ipde.spatial.stereo_photometric_support", return_value=(np.ones((40, 96), bool), {}))):
            create.return_value.compute.side_effect = [forward, reverse[:, ::-1]]
            result = run_stereo_matching(rgb, rgb, {
                "left_camera": {"width": 96, "height": 40},
                "rectified_stereo_ready": True, "principal_point_delta_x_pixels": 0,
            }, StereoMatchingOptions(maximum_disparity=16))
        self.assertTrue(np.isnan(result.height_disparity_pixels[10:30, 40:60]).all())
        valid = result.height_disparity_pixels[np.isfinite(result.height_disparity_pixels)]
        self.assertGreater(valid.size, 200)
        np.testing.assert_array_equal(valid, np.full(valid.shape, 4, np.float32))

    def test_tiny_far_disparity_island_does_not_collapse_displacement_range(self):
        rgb = np.zeros((40, 96, 3), np.uint8)
        forward = np.full((40, 96), 64, np.int16)
        reverse = forward.copy()
        forward[16:24, 48:56] = 16  # isolated one-pixel disparity
        reverse[16:24, 47:55] = 16
        calibration = {"left_camera": {"width": 96, "height": 40},
                       "rectified_stereo_ready": True, "principal_point_delta_x_pixels": 0,
                       "focal_length_pixels_for_depth": 100, "baseline_meters": .1}
        with (patch("cv2.StereoSGBM.create") as create,
              patch("ipde.spatial.stereo_photometric_support", return_value=(np.ones((40, 96), bool), {}))):
            create.return_value.compute.side_effect = [forward, reverse[:, ::-1]]
            result = run_stereo_matching(rgb, rgb, calibration, StereoMatchingOptions(maximum_disparity=16))
        self.assertTrue(np.isnan(result.height_disparity_pixels[16:24, 48:56]).all())
        mapped, details = linear_depth_displacement(result.height_disparity_pixels, calibration)
        self.assertTrue(details["constant_map"])
        self.assertEqual(np.nanmax(mapped), 0)

    def test_equal_distance_steps_produce_equal_displacement_steps(self):
        # f*B = 6: disparities 6,3,2 are distances 1,2,3 meters.
        # Scaling disparity directly would incorrectly put the middle at 0.25.
        disparity = np.array([[6, 3, 2]], dtype=np.float32)
        original = disparity.copy()
        mapped, details = linear_depth_displacement(disparity, {
            "focal_length_pixels_for_depth": 60, "baseline_meters": .1,
            "principal_point_delta_x_pixels": 9,  # already corrected; never add twice
        })
        np.testing.assert_array_equal(mapped, [[1, .5, 0]])
        np.testing.assert_array_equal(disparity, original)
        self.assertEqual(details["displacement_scale_meters_float32"], 2)

    def test_unbounded_or_impossible_depth_does_not_set_displacement_range(self):
        disparity = np.array([[6, 3, 2, 0, -1, np.nan, np.inf]], dtype=np.float32)
        mapped, _ = linear_depth_displacement(disparity, {
            "focal_length_pixels_for_depth": 60, "baseline_meters": .1,
        })
        np.testing.assert_equal(mapped, [[1, .5, 0, np.nan, np.nan, np.nan, np.nan]])
        with self.assertRaisesRegex(DisplacementMappingError, "no finite positive"):
            linear_depth_displacement(np.zeros((2, 2), np.float32), {
                "focal_length_pixels_for_depth": 60, "baseline_meters": .1,
            })

    def test_constant_depth_has_zero_relief(self):
        mapped, details = linear_depth_displacement(np.full((2, 3), 5, np.float32), {
            "focal_length_pixels_for_depth": 60, "baseline_meters": .1,
        })
        np.testing.assert_array_equal(mapped, np.zeros((2, 3), np.float32))
        self.assertTrue(details["constant_map"])

    def test_correspondences_cannot_use_off_image_or_padded_samples(self):
        flow = np.array([[-1, -.5, 1, np.nan], [0, -.5, 0, 1]], np.float32)
        valid = np.ones((2, 4), bool)
        valid[1, 0] = False
        np.testing.assert_array_equal(correspondence_validity(flow, valid),
                                      [[False, True, True, False], [False, False, True, False]])

    def test_alignment_recovers_vertical_shift_without_erasing_horizontal_disparity(self):
        rng = np.random.default_rng(42)
        left = cv2.GaussianBlur(rng.integers(0, 256, (256, 384, 3), np.uint8), (3, 3), .6)
        right = np.zeros_like(left)
        right[3:, :-24] = left[:-3, 24:]
        original = right.copy()
        aligned, valid, details = register_stereo_rows(left, right)
        self.assertTrue(details["applied"])
        np.testing.assert_array_equal(right, original)
        np.testing.assert_array_equal(details["right_to_aligned_affine"][0], [1, 0, 0])
        self.assertLess(details["median_vertical_error_after_pixels"], .1)
        np.testing.assert_array_equal(aligned[10:-10, 10:-30], left[10:-10, 34:-6])
        self.assertFalse(valid[-1].any())
        calibration = {"left_camera": {"width": 384, "height": 256},
                       "rectified_stereo_ready": True, "principal_point_delta_x_pixels": 0.0}
        result = run_stereo_matching(left, right, calibration, StereoMatchingOptions(maximum_disparity=48))
        core = result.height_disparity_pixels[10:-10, 60:-60]
        self.assertGreater(np.isfinite(core).mean(), .95)
        self.assertLess(np.nanmedian(abs(core - 24)), .1)

    def test_textureless_inputs_do_not_invent_a_registration(self):
        left = np.zeros((64, 96, 3), np.uint8)
        right = np.full_like(left, 10)
        aligned, valid, details = register_stereo_rows(left, right)
        self.assertFalse(details["applied"])
        self.assertIs(aligned, right)
        self.assertTrue(valid.all())

    def test_flat_bidirectionally_consistent_regions_are_not_geometry(self):
        rgb = np.full((64, 96, 3), 80, np.uint8)
        fake = np.full((64, 96), 16, np.int16)
        with patch("cv2.StereoSGBM.create") as create:
            create.return_value.compute.side_effect = [fake, fake.copy()]
            result = run_stereo_matching(rgb, rgb, {
                "left_camera": {"width": 96, "height": 64},
                "rectified_stereo_ready": True, "principal_point_delta_x_pixels": 0,
            }, StereoMatchingOptions(maximum_disparity=16))
        self.assertTrue(np.isnan(result.height_disparity_pixels).all())

    def test_patch_support_accepts_exposure_change_but_rejects_false_disparity(self):
        rng = np.random.default_rng(18)
        left = rng.integers(30, 160, (64, 96, 3), np.uint8)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:] + 30
        disparity = np.full((64, 96), 8, np.float32)
        accepted, _ = stereo_photometric_support(left, right, disparity)
        self.assertTrue(accepted[8:-8, 20:-12].all())
        disparity[20:44, 30:70] = .0625
        accepted, _ = stereo_photometric_support(left, right, disparity)
        self.assertFalse(accepted[24:40, 34:66].any())

    def test_reverse_flow_checks_sign_subpixel_neighbors_and_occlusion(self):
        forward = np.full((2, 10), -2.5, np.float32)
        reverse = np.full((2, 10), 2.5, np.float32)
        reverse[1, 3] = 8
        accepted = reverse_correspondence_support(forward, reverse)
        self.assertFalse(accepted[:, :3].any())
        self.assertTrue(accepted[0, 3:].all())
        self.assertFalse(accepted[1, 5:7].any())

    def test_rectified_stereo_needs_horizontal_information_not_corners(self):
        # Texture constant along y has no 2-D corners, but its x shift is observable.
        rng = np.random.default_rng(16)
        stripe = rng.integers(30, 190, (1, 160, 1), np.uint8)
        left = np.tile(stripe, (64, 1, 3))
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        disparity = np.full((64, 160), 8, np.float32)
        supported, _ = stereo_photometric_support(left, right, disparity)
        self.assertTrue(supported[8:-8, 20:-12].all())
        result = run_stereo_matching(left, right, {
            "left_camera": {"width": 160, "height": 64},
            "rectified_stereo_ready": True, "principal_point_delta_x_pixels": 0,
        }, StereoMatchingOptions(maximum_disparity=16))
        core = result.height_disparity_pixels[8:-8, 24:-24]
        self.assertGreater(np.isfinite(core).mean(), .95)
        self.assertLess(np.nanmax(abs(core - 8)), .2)
        # The transpose varies only vertically: horizontal disparity is unobservable.
        horizontal_stripes = np.ascontiguousarray(left.transpose(1, 0, 2))
        unsupported, _ = stereo_photometric_support(
            horizontal_stripes, horizontal_stripes, np.zeros(horizontal_stripes.shape[:2], np.float32))
        self.assertFalse(unsupported.any())

    def test_projected_two_plane_scene_preserves_depth_steps(self):
        # An independent pinhole projection with known f, B, and cx difference:
        # xL = f*X/Z + cxL; xR = f*(X-B)/Z + cxR.
        from ipde.spatial import derive_raft_height_and_depth
        f, baseline, cx_l, cx_r = 800., .06, 300., 307.
        z = np.array([[.8, 1.2, 2.4]], np.float32)
        x = np.array([[.1, -.2, .3]], np.float32)
        xl = f * x / z + cx_l
        xr = f * (x - baseline) / z + cx_r
        disparity, depth = derive_raft_height_and_depth(xr - xl, {
            "focal_length_pixels_for_depth": f, "baseline_meters": baseline,
            "principal_point_delta_x_pixels": cx_r-cx_l,
        })
        np.testing.assert_allclose(depth, z, rtol=2e-6)
        np.testing.assert_allclose(disparity, f*baseline/z, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
