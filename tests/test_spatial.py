from __future__ import annotations

import tempfile
import unittest
import zipfile
from unittest.mock import patch
from pathlib import Path

import numpy as np

from ipde.formats import arrays_bit_equal
from ipde.spatial import (
    _checkpoint_bytes,
    ColorMatchingError,
    DisplacementMappingError,
    RaftStereoError,
    RaftStereoOptions,
    SpatialPhotoError,
    StereoMatchingError,
    StereoMatchingOptions,
    analyze_spatial_photo,
    derive_raft_height_and_depth,
    histogram_match_stereo_pair,
    normalize_height_maps_for_displacement,
    resolve_raft_resources,
    run_stereo_matching,
)


def camera_image(position_x: float, *, width: int = 640, height: int = 480) -> dict:
    return {
        "PixelWidth": width,
        "PixelHeight": height,
        "Orientation": 1,
        "{HEIF}": {
            "CameraExtrinsics": {
                "CoordinateSystemID": 0,
                "Position": [position_x, 0.0, 0.0],
                "Rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            "CameraModel": {
                "ModelType": "SimplifiedPinhole",
                "Intrinsics": [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
            },
        },
    }


def spatial_metadata() -> dict:
    return {
        "groups": [
            {
                "GroupType": "StereoPair",
                "GroupIndex": 0,
                "GroupImageIndexLeft": 1,
                "GroupImageIndexRight": 2,
                "GroupImageIndexMonoscopic": 0,
                "GroupImageDisparityAdjustment": 250,
            }
        ],
        "images": [{}, camera_image(0.0), camera_image(0.064)],
    }


class SpatialPhotoTests(unittest.TestCase):
    def test_spatial_group_preserves_calibration_and_presentation_adjustment(self) -> None:
        result = analyze_spatial_photo(spatial_metadata())
        assert result is not None
        self.assertEqual(result["left_image_index"], 1)
        self.assertEqual(result["right_image_index"], 2)
        self.assertEqual(result["baseline_meters"], 0.064)
        self.assertEqual(result["encoded_disparity_adjustment"], 250)
        self.assertEqual(result["disparity_adjustment_fraction_of_width"], 0.025)
        self.assertEqual(result["disparity_adjustment_pixels"], 16.0)
        self.assertTrue(result["rectified_stereo_ready"])
        self.assertTrue(result["raft_stereo_ready"])
        self.assertIn("Presentation-only", result["disparity_adjustment_semantics"])

    def test_spatial_group_rejects_missing_disparity_adjustment(self) -> None:
        metadata = spatial_metadata()
        del metadata["groups"][0]["GroupImageDisparityAdjustment"]
        with self.assertRaisesRegex(SpatialPhotoError, "disparity adjustment"):
            analyze_spatial_photo(metadata)

    def test_float32_height_and_metric_depth_operation_order(self) -> None:
        flow = np.array([[-10.0, -20.0, 0.0]], dtype=np.float32)
        calibration = {
            "principal_point_delta_x_pixels": 2.0,
            "focal_length_pixels_for_depth": 100.0,
            "baseline_meters": 0.1,
        }
        height, depth = derive_raft_height_and_depth(flow, calibration)
        expected_height = np.array([[12.0, 22.0, 2.0]], dtype=np.float32)
        expected_depth = np.float32(np.float32(100.0) * np.float32(0.1)) / expected_height
        self.assertTrue(arrays_bit_equal(height, expected_height))
        self.assertTrue(arrays_bit_equal(depth, expected_depth))
        self.assertEqual(height.dtype, np.dtype("float32"))
        self.assertEqual(depth.dtype, np.dtype("float32"))

    def test_raft_resource_resolution_accepts_explicit_portable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "RAFT-Stereo"
            (root / "core").mkdir(parents=True)
            (root / "core" / "raft_stereo.py").write_text("# fixture\n", encoding="utf-8")
            model = Path(directory) / "model.pth"
            model.write_bytes(b"checkpoint")
            resolved_root, resolved_model, member = resolve_raft_resources(
                RaftStereoOptions(root=root, model=model)
            )
            self.assertEqual(resolved_root, root.resolve())
            self.assertEqual(resolved_model, model.resolve())
            self.assertIsNone(member)

    def test_raft_does_not_fold_wrong_sign_flow_into_near_geometry(self) -> None:
        flow = np.array([[-5, 0, 3, np.nan, np.inf]], dtype=np.float32)
        original = flow.copy()
        height, depth = derive_raft_height_and_depth(flow, {
            "principal_point_delta_x_pixels": 0, "focal_length_pixels_for_depth": 100,
            "baseline_meters": 0.1,
        })
        np.testing.assert_equal(height, [[5, 0, np.nan, np.nan, np.nan]])
        np.testing.assert_equal(depth, [[2, np.inf, np.nan, np.nan, np.nan]])
        self.assertTrue(arrays_bit_equal(flow, original))

    def test_selected_missing_model_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RaftStereoError, "Selected.*model"):
                resolve_raft_resources(RaftStereoOptions(model=Path(directory) / "missing.pth"))
            with self.assertRaisesRegex(RaftStereoError, "Selected.*source"):
                resolve_raft_resources(RaftStereoOptions(root=Path(directory)))

    def test_zip_checkpoint_accepts_nested_member_but_not_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("models/raftstereo-middlebury.pth", b"model")
            self.assertEqual(_checkpoint_bytes(path, "raftstereo-middlebury.pth"),
                             (b"model", "raftstereo-middlebury.pth"))
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("other/raftstereo-middlebury.pth", b"other")
            with self.assertRaisesRegex(RaftStereoError, "ambiguous"):
                _checkpoint_bytes(path, "raftstereo-middlebury.pth")

    def test_displacement_default_keeps_extremes_and_constant_maps(self) -> None:
        values = np.linspace(30, 40, 10000, dtype=np.float32).reshape(100, 100)
        values[0, 0] = 0
        values[-1, -1] = 200
        original = values.copy()
        mapped, details = normalize_height_maps_for_displacement({"raft": values})
        self.assertEqual(np.count_nonzero(mapped["raft"] == 0), 1)
        self.assertEqual(np.count_nonzero(mapped["raft"] == 1), 1)
        self.assertEqual(details["lower_percentile"], 0)
        self.assertEqual(details["upper_percentile"], 100)
        self.assertTrue(arrays_bit_equal(values, original))
        constant = np.array([[15, 15, np.nan]], dtype=np.float32)
        mapped, details = normalize_height_maps_for_displacement({"raft": constant})
        np.testing.assert_equal(mapped["raft"], [[0, 0, np.nan]])
        self.assertTrue(details["constant_map"])

    def test_raft_derivation_rejects_implicit_dtype_conversion(self) -> None:
        with self.assertRaisesRegex(RaftStereoError, "float32"):
            derive_raft_height_and_depth(
                np.array([[1.0]], dtype=np.float64),
                {
                    "principal_point_delta_x_pixels": 0.0,
                    "focal_length_pixels_for_depth": 100.0,
                    "baseline_meters": 0.1,
                },
            )

    def test_stereo_matching_is_full_resolution_float32_and_marks_unmatched_pixels(self) -> None:
        generator = np.random.default_rng(7)
        left = generator.integers(0, 256, size=(32, 64, 3), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-4] = left[:, 4:]
        spatial = analyze_spatial_photo(
            {
                "groups": [
                    {
                        "GroupType": "StereoPair",
                        "GroupIndex": 0,
                        "GroupImageIndexLeft": 1,
                        "GroupImageIndexRight": 2,
                        "GroupImageIndexMonoscopic": 0,
                        "GroupImageDisparityAdjustment": 0,
                    }
                ],
                "images": [
                    {},
                    camera_image(0.0, width=64, height=32),
                    camera_image(0.064, width=64, height=32),
                ],
            }
        )
        assert spatial is not None
        result = run_stereo_matching(
            left,
            right,
            spatial,
            StereoMatchingOptions(maximum_disparity=16, block_size=5),
        )
        height = result.height_disparity_pixels
        self.assertEqual(height.shape, (32, 64))
        self.assertEqual(height.dtype, np.dtype("float32"))
        self.assertTrue(height.flags.c_contiguous)
        self.assertGreater(np.count_nonzero(np.isfinite(height)), 0)
        self.assertGreater(np.count_nonzero(np.isnan(height)), 0)
        finite_sixteenths = height[np.isfinite(height)] * np.float32(16.0)
        self.assertTrue(np.all(finite_sixteenths == np.round(finite_sixteenths)))
        self.assertTrue(result.details["full_decoded_resolution"])
        self.assertFalse(result.details["resized_or_tiled"])
        self.assertEqual(result.details["invalid_output_representation"], "float32 NaN")

    def test_stereo_matching_refuses_high_bit_input_instead_of_reducing_it(self) -> None:
        spatial = analyze_spatial_photo(spatial_metadata())
        assert spatial is not None
        left = np.zeros((480, 640, 3), dtype=np.uint16)
        with self.assertRaisesRegex(StereoMatchingError, "will not reduce"):
            run_stereo_matching(left, left.copy(), spatial)

    def test_histogram_color_matching_preserves_hero_and_records_exact_luts(self) -> None:
        left = np.array(
            [
                [[10, 50, 90], [20, 60, 100]],
                [[30, 70, 110], [40, 80, 120]],
            ],
            dtype=np.uint8,
        )
        right = np.array(
            [
                [[0, 130, 200], [1, 140, 210]],
                [[2, 150, 220], [3, 160, 230]],
            ],
            dtype=np.uint8,
        )
        original_left = left.copy()
        original_right = right.copy()
        result = histogram_match_stereo_pair(left, right, hero_side="left")

        self.assertTrue(arrays_bit_equal(result.left, original_left))
        self.assertTrue(arrays_bit_equal(result.right, original_left))
        self.assertTrue(arrays_bit_equal(left, original_left))
        self.assertTrue(arrays_bit_equal(right, original_right))
        self.assertEqual(result.details["hero_side"], "left")
        self.assertEqual(result.details["transformed_side"], "right")
        self.assertFalse(result.details["raw_assets_modified"])
        self.assertEqual(len(result.details["lookup_tables"]), 3)
        self.assertTrue(all(len(lut) == 256 for lut in result.details["lookup_tables"]))
        self.assertEqual(result.details["lookup_tables"][0][0:4], [10, 20, 30, 40])

        reverse = histogram_match_stereo_pair(left, right, hero_side="right")
        self.assertTrue(arrays_bit_equal(reverse.right, original_right))
        self.assertTrue(arrays_bit_equal(reverse.left, original_right))
        self.assertEqual(reverse.details["hero_side"], "right")
        self.assertEqual(reverse.details["transformed_side"], "left")

    def test_histogram_color_matching_refuses_implicit_bit_depth_reduction(self) -> None:
        value = np.zeros((2, 2, 3), dtype=np.uint16)
        with self.assertRaisesRegex(ColorMatchingError, "will not reduce"):
            histogram_match_stereo_pair(value, value, hero_side="left")

    def test_displacement_mapping_uses_one_explicit_float32_range(self) -> None:
        classical = np.array(
            [[np.nan, 0.0, 2.0], [4.0, 6.0, 8.0]], dtype=np.float32
        )
        raft = np.array([[1.0, 3.0, 5.0], [7.0, 9.0, 10.0]], dtype=np.float32)

        mapped, details = normalize_height_maps_for_displacement(
            {"stereo_matching": classical, "raft_stereo": raft},
            lower_percentile=0.0,
            upper_percentile=100.0,
        )

        expected_classical = np.clip(
            (classical - np.float32(0.0)) / np.float32(10.0),
            np.float32(0.0),
            np.float32(1.0),
        ).astype(np.float32)
        expected_raft = np.clip(
            (raft - np.float32(0.0)) / np.float32(10.0),
            np.float32(0.0),
            np.float32(1.0),
        ).astype(np.float32)
        self.assertTrue(arrays_bit_equal(mapped["stereo_matching"], expected_classical))
        self.assertTrue(arrays_bit_equal(mapped["raft_stereo"], expected_raft))
        self.assertTrue(np.isnan(mapped["stereo_matching"][0, 0]))
        self.assertEqual(details["lower_bound_pixels_float32"], 0.0)
        self.assertEqual(details["upper_bound_pixels_float32"], 10.0)
        self.assertEqual(
            details["shared_bounds_across_maps"],
            ["stereo_matching", "raft_stereo"],
        )
        self.assertFalse(details["scientific_pixel_disparity_replaced"])

    def test_displacement_mapping_rejects_implicit_dtype_conversion(self) -> None:
        with self.assertRaisesRegex(DisplacementMappingError, "float32"):
            normalize_height_maps_for_displacement(
                {"height": np.ones((2, 2), dtype=np.float64)}
            )


if __name__ == "__main__":
    unittest.main()
