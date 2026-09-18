from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ipde.extractor import (
    ExtractOptions,
    ExtractionError,
    MetricDepthError,
    extract_file,
    inspect_file,
    reconstruct_metric_depth,
    reconstruct_physical_disparity,
    semantic_name,
)
from ipde.formats import arrays_bit_equal, read_exr_exact, read_npy_exact, read_png_exact
from ipde.libheif_aux import DecodedAuxiliary
from ipde.spatial import (
    HistogramMatchedStereoPair,
    RaftStereoResult,
    StereoMatchingResult,
)


class FakeImage:
    def __init__(
        self,
        array: np.ndarray,
        mode: str,
        bit_depth: int,
        *,
        metadata: list[dict] | None = None,
        info: dict | None = None,
        has_alpha: bool = False,
        premultiplied_alpha: bool = False,
    ) -> None:
        self.array = np.ascontiguousarray(array)
        self.mode = mode
        self.size = (int(array.shape[1]), int(array.shape[0]))
        self.info = info or {}
        self.has_alpha = has_alpha
        self.premultiplied_alpha = premultiplied_alpha
        self._c_image = SimpleNamespace(
            bit_depth=bit_depth,
            metadata=metadata or [],
            chroma=0,
            colorspace=2,
            color_profile={},
            camera_intrinsic_matrix=None,
            camera_extrinsic_matrix_rot=None,
        )

    @property
    def __array_interface__(self) -> dict:
        return self.array.__array_interface__


class FakeParent(FakeImage):
    def __init__(self, *args, auxiliary: dict[int, FakeImage] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.auxiliary = auxiliary or {}

    def get_aux_image(self, aux_id: int) -> FakeImage:
        value = self.auxiliary[aux_id]
        if isinstance(value, BaseException):
            raise value
        return value


class FakeFile:
    def __init__(self, images: list[FakeParent], primary_index: int = 0) -> None:
        self.images = images
        self.primary_index = primary_index
        self.mimetype = "image/heic"

    def __iter__(self):
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> FakeParent:
        return self.images[index]


def fixture_file() -> FakeFile:
    depth = FakeImage(
        np.array([[0, 1, 4095], [12, 2048, 1023]], dtype=np.uint16),
        "I;12",
        12,
        info={
            "metadata": {
                "d_min": 1.5,
                "d_max": 4.5,
                "representation_type": 1,
                "disparity_reference_view": 0,
            }
        },
    )
    xmp = b"<x:xmpmeta>gain metadata</x:xmpmeta>\n"
    gain = FakeImage(
        np.array([[0, 7], [100, 255]], dtype=np.uint8),
        "L",
        8,
        metadata=[{"type": "mime", "content_type": "application/rdf+xml", "data": xmp}],
    )
    parent = FakeParent(
        np.zeros((4, 6, 3), dtype=np.uint8),
        "RGB",
        8,
        info={
            "primary": True,
            "bit_depth": 8,
            "depth_images": [depth],
            "aux": {"urn:com:apple:photo:2020:aux:hdrgainmap": [52]},
            "thumbnails": [],
        },
        auxiliary={52: gain},
    )
    return FakeFile([parent])


class ExtractorTests(unittest.TestCase):
    def test_spatial_photo_extracts_comparable_height_maps_and_optional_diagnostics(self) -> None:
        primary = FakeParent(
            np.zeros((2, 3, 3), dtype=np.uint8),
            "RGB",
            8,
            info={"primary": True, "bit_depth": 8, "depth_images": [], "aux": {}, "thumbnails": []},
        )
        left_array = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        right_array = np.flip(left_array, axis=1).copy()
        left = FakeParent(
            left_array,
            "RGB",
            8,
            info={"primary": False, "bit_depth": 8, "depth_images": [], "aux": {}, "thumbnails": []},
        )
        right = FakeParent(
            right_array,
            "RGB",
            8,
            info={"primary": False, "bit_depth": 8, "depth_images": [], "aux": {}, "thumbnails": []},
        )
        camera_model = {
            "ModelType": "SimplifiedPinhole",
            "Intrinsics": [100.0, 0.0, 1.5, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0],
        }
        imageio_metadata = {
            "groups": [
                {
                    "GroupType": "StereoPair",
                    "GroupIndex": 0,
                    "GroupImageIndexLeft": 1,
                    "GroupImageIndexRight": 2,
                    "GroupImageIndexMonoscopic": 0,
                    "GroupImageDisparityAdjustment": 100,
                }
            ],
            "images": [
                {},
                {
                    "PixelWidth": 3,
                    "PixelHeight": 2,
                    "Orientation": 1,
                    "{HEIF}": {
                        "CameraExtrinsics": {
                            "CoordinateSystemID": 0,
                            "Position": [0.0, 0.0, 0.0],
                            "Rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        },
                        "CameraModel": camera_model,
                    },
                },
                {
                    "PixelWidth": 3,
                    "PixelHeight": 2,
                    "Orientation": 1,
                    "{HEIF}": {
                        "CameraExtrinsics": {
                            "CoordinateSystemID": 0,
                            "Position": [0.05, 0.0, 0.0],
                            "Rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        },
                        "CameraModel": camera_model,
                    },
                },
            ],
        }
        signed_flow = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
        height = np.abs(signed_flow)
        depth = np.float32(5.0) / height
        raft_result = RaftStereoResult(
            signed_flow,
            height,
            depth,
            {"engine": "RAFT-Stereo fixture", "precision_scope": "test"},
        )
        stereo_height = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=np.float32)
        stereo_result = StereoMatchingResult(
            stereo_height,
            {"engine": "StereoSGBM fixture", "precision_scope": "test"},
        )
        matched_right = np.bitwise_xor(right_array, np.uint8(1))
        matched_pair = HistogramMatchedStereoPair(
            left_array.copy(),
            matched_right,
            {
                "applied": True,
                "hero_side": "left",
                "transformed_side": "right",
                "raw_assets_modified": False,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "spatial.heic"
            source.write_bytes(b"fake-spatial-heif")
            comparison_dir = Path(directory) / "comparison"
            diagnostics_dir = Path(directory) / "diagnostics"
            with (
                patch(
                    "ipde.extractor.pillow_heif.open_heif",
                    return_value=FakeFile([primary, left, right]),
                ),
                patch(
                    "ipde.extractor.read_apple_imageio_metadata",
                    return_value=imageio_metadata,
                ),
                patch(
                    "ipde.extractor.run_stereo_matching",
                    return_value=stereo_result,
                ) as stereo_runner,
                patch(
                    "ipde.extractor.histogram_match_stereo_pair",
                    return_value=matched_pair,
                ) as color_runner,
                patch("ipde.extractor.run_raft_stereo", return_value=raft_result) as runner,
            ):
                report = extract_file(
                    source,
                    ExtractOptions(
                        output_dir=comparison_dir,
                        write_npy=False,
                        write_metric_depth=False,
                        write_physical_disparity=False,
                        write_stereo_matching=True,
                        write_raft_stereo=True,
                        histogram_color_matching=True,
                        color_matching_hero="left",
                        write_displacement_maps=True,
                    ),
                )
                diagnostic_report = extract_file(
                    source,
                    ExtractOptions(
                        output_dir=diagnostics_dir,
                        write_npy=False,
                        write_metric_depth=False,
                        write_physical_disparity=False,
                        write_raft_stereo=True,
                        write_raft_diagnostics=True,
                    ),
                )

            self.assertEqual(runner.call_count, 2)
            stereo_runner.assert_called_once()
            color_runner.assert_called_once()
            self.assertTrue(arrays_bit_equal(stereo_runner.call_args.args[0], left_array))
            self.assertTrue(arrays_bit_equal(stereo_runner.call_args.args[1], matched_right))
            self.assertTrue(
                arrays_bit_equal(read_png_exact(comparison_dir / "spatial_spatial_left.png"), left_array)
            )
            self.assertTrue(
                arrays_bit_equal(read_png_exact(comparison_dir / "spatial_spatial_right.png"), right_array)
            )
            self.assertTrue(
                arrays_bit_equal(
                    read_exr_exact(
                        comparison_dir
                        / "spatial_spatial_stereo_matching_color_matched_height.exr",
                        stereo_height.shape,
                    ),
                    stereo_height,
                )
            )
            self.assertTrue(
                arrays_bit_equal(
                    read_exr_exact(
                        comparison_dir
                        / "spatial_spatial_raft_stereo_color_matched_height.exr",
                        height.shape,
                    ),
                    height,
                )
            )
            stereo_depth = np.float32(5.0) / stereo_height
            expected_stereo_displacement = (np.float32(5.0) - stereo_depth) / (np.float32(5.0) - np.float32(5.0 / 6.0))
            expected_raft_displacement = (np.float32(5.0) - depth) / (np.float32(5.0) - np.float32(5.0 / 6.0))
            self.assertTrue(
                arrays_bit_equal(
                    read_exr_exact(
                        comparison_dir
                        / "spatial_spatial_stereo_matching_color_matched_displacement_0_to_1.exr",
                        stereo_height.shape,
                    ),
                    expected_stereo_displacement,
                )
            )
            self.assertTrue(
                arrays_bit_equal(
                    read_exr_exact(
                        comparison_dir
                        / "spatial_spatial_raft_stereo_color_matched_displacement_0_to_1.exr",
                        height.shape,
                    ),
                    expected_raft_displacement,
                )
            )
            self.assertFalse(
                (
                    comparison_dir
                    / "spatial_spatial_raft_stereo_color_matched_signed_flow.exr"
                ).exists()
            )
            self.assertFalse(
                (
                    comparison_dir
                    / "spatial_spatial_raft_stereo_color_matched_depth_meters.exr"
                ).exists()
            )
            self.assertTrue(
                arrays_bit_equal(
                    read_exr_exact(
                        diagnostics_dir / "spatial_spatial_raft_stereo_signed_flow.exr",
                        signed_flow.shape,
                    ),
                    signed_flow,
                )
            )
            self.assertTrue(
                arrays_bit_equal(
                    read_exr_exact(
                        diagnostics_dir / "spatial_spatial_raft_stereo_depth_meters.exr",
                        depth.shape,
                    ),
                    depth,
                )
            )
            self.assertTrue(report["source"]["spatial_photo"]["is_spatial_photo"])
            roles = [output["role"] for output in report["assets"][0]["outputs"]]
            self.assertIn("derived_stereo_matching_height_map", roles)
            self.assertIn("derived_raft_stereo_height_map", roles)
            self.assertIn("derived_stereo_matching_displacement_0_to_1", roles)
            self.assertIn("derived_raft_stereo_displacement_0_to_1", roles)
            matching_output = next(
                output
                for output in report["assets"][0]["outputs"]
                if output["role"] == "derived_stereo_matching_height_map"
            )
            self.assertTrue(matching_output["derivation"]["input_color_matching"]["applied"])
            self.assertEqual(
                matching_output["derivation"]["input_color_matching"]["hero_side"],
                "left",
            )
            displacement_output = next(
                output
                for output in report["assets"][0]["outputs"]
                if output["role"] == "derived_raft_stereo_displacement_0_to_1"
            )
            mapping = displacement_output["derivation"]["displacement_mapping"]
            self.assertEqual(mapping["far_depth_meters_float32"], 5.0)
            self.assertAlmostEqual(mapping["near_depth_meters_float32"], 5.0 / 6.0)
            self.assertFalse(mapping["scientific_pixel_disparity_replaced"])
            diagnostic_roles = [
                output["role"] for output in diagnostic_report["assets"][0]["outputs"]
            ]
            self.assertIn("derived_raft_stereo_signed_flow", diagnostic_roles)
            self.assertIn("derived_raft_stereo_metric_depth", diagnostic_roles)

    def test_metric_depth_reconstruction_uses_exact_float32_operation_order(self) -> None:
        raw = np.array([[0, 1, 127, 255], [254, 128, 64, 32]], dtype=np.uint8)
        metadata = {"d_min": 1.5, "d_max": 4.5, "representation_type": 1}

        normalized = raw.astype(np.float32) / np.float32(255.0)
        disparity = reconstruct_physical_disparity(raw, metadata)
        expected_disparity = normalized * (np.float32(4.5) - np.float32(1.5)) + np.float32(1.5)
        actual = reconstruct_metric_depth(raw, metadata)
        expected = np.float32(1.0) / disparity

        self.assertTrue(arrays_bit_equal(disparity, expected_disparity))
        self.assertEqual(actual.dtype, np.dtype("float32"))
        self.assertTrue(actual.flags.c_contiguous)
        self.assertTrue(arrays_bit_equal(actual, expected))
        self.assertEqual(np.unique(actual).size, np.unique(raw).size)
        self.assertEqual(actual[0, 0], np.float32(1.0) / np.float32(1.5))
        self.assertEqual(actual[0, 3], np.float32(1.0) / np.float32(4.5))

    def test_metric_depth_reconstruction_rejects_mislabelled_inputs(self) -> None:
        raw = np.array([[0, 255]], dtype=np.uint8)
        with self.assertRaisesRegex(MetricDepthError, "uniform_disparity"):
            reconstruct_metric_depth(
                raw,
                {"d_min": 1.0, "d_max": 3.0, "representation_type": 2},
            )
        with self.assertRaisesRegex(MetricDepthError, "uint8"):
            reconstruct_metric_depth(
                raw.astype(np.uint16),
                {"d_min": 1.0, "d_max": 3.0, "representation_type": 1},
            )
        with self.assertRaisesRegex(MetricDepthError, "d_min and d_max"):
            reconstruct_metric_depth(raw, {"representation_type": 1})

    @unittest.skipUnless(importlib.util.find_spec("OpenEXR"), "OpenEXR is not installed")
    def test_metric_depth_is_exported_as_verified_float32_exr(self) -> None:
        raw = np.array([[0, 1, 127, 255], [5, 25, 125, 250]], dtype=np.uint8)
        metadata = {"d_min": 1.5, "d_max": 4.5, "representation_type": 1}
        depth = FakeImage(raw, "L", 8, info={"metadata": metadata})
        parent = FakeParent(
            np.zeros((2, 4, 3), dtype=np.uint8),
            "RGB",
            8,
            info={
                "primary": True,
                "bit_depth": 8,
                "depth_images": [depth],
                "aux": {},
                "thumbnails": [],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "metric.heic"
            source.write_bytes(b"fake-heif")
            with patch("ipde.extractor.pillow_heif.open_heif", return_value=FakeFile([parent])):
                report = extract_file(source, ExtractOptions(write_npy=False))

            exr_path = Path(directory) / "metric_depth_meters.exr"
            disparity_path = Path(directory) / "metric_depth_disparity.exr"
            self.assertTrue(exr_path.is_file())
            self.assertTrue(disparity_path.is_file())
            expected = reconstruct_metric_depth(raw, metadata)
            expected_disparity = reconstruct_physical_disparity(raw, metadata)
            self.assertTrue(arrays_bit_equal(read_exr_exact(exr_path, raw.shape), expected))
            self.assertTrue(
                arrays_bit_equal(read_exr_exact(disparity_path, raw.shape), expected_disparity)
            )

            metric_output = next(
                item
                for item in report["assets"][0]["outputs"]
                if item["role"] == "derived_metric_depth"
            )
            disparity_output = next(
                item
                for item in report["assets"][0]["outputs"]
                if item["role"] == "derived_physical_disparity"
            )
            self.assertTrue(metric_output["verified"])
            self.assertEqual(metric_output["derivation"]["output_dtype"], "float32")
            self.assertEqual(metric_output["derivation"]["depth_units"], "m")
            self.assertIn("nearer geometry is numerically smaller", metric_output["derivation"]["value_direction"])
            self.assertFalse(
                metric_output["derivation"]["source_quantization"]["restores_additional_precision"]
            )
            self.assertEqual(
                metric_output["derivation"]["source_quantization"]["source_codes_present"],
                np.unique(raw).size,
            )
            self.assertEqual(
                metric_output["derivation"]["distinct_values_present"],
                np.unique(raw).size,
            )
            self.assertEqual(metric_output["derivation"]["d_min"], 1.5)
            self.assertEqual(metric_output["derivation"]["d_max"], 4.5)
            self.assertTrue(disparity_output["verified"])
            self.assertEqual(disparity_output["derivation"]["units"], "1/m")
            self.assertEqual(
                disparity_output["derivation"]["value_direction"],
                "larger values indicate nearer geometry",
            )

    def test_semantic_urn_names(self) -> None:
        self.assertEqual(semantic_name("urn:com:apple:photo:2020:aux:hdrgainmap"), "hdr_gain_map")
        self.assertEqual(semantic_name("urn:test:PortraitMatte"), "portrait_matte")
        self.assertEqual(
            semantic_name("tag:apple.com,2023:photo:aux:linearthumbnail"), "linear_thumbnail"
        )
        self.assertEqual(semantic_name("urn:test:custom-map"), "custom_map")

    def test_high_bit_auxiliary_uses_unshifted_libheif_fallback(self) -> None:
        array = np.array([[[0, 1, 1023], [512, 7, 9]]], dtype=np.uint16)
        parent = FakeParent(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "RGB",
            8,
            info={
                "primary": True,
                "bit_depth": 8,
                "depth_images": [],
                "aux": {"tag:apple.com,2023:photo:aux:linearthumbnail": [52]},
                "thumbnails": [],
            },
            auxiliary={
                52: NotImplementedError(
                    "Only 8-bit AUX images are currently supported. Got 10-bit image."
                )
            },
        )
        fallback = DecodedAuxiliary(
            array=array,
            mode="RGB;10",
            source_bit_depth=10,
            metadata_blocks=[],
            details={"backend": "test fallback", "decoded_storage": "unshifted"},
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "high-bit.heic"
            source.write_bytes(b"fake-heif")
            with (
                patch("ipde.extractor.pillow_heif.open_heif", return_value=FakeFile([parent])),
                patch("ipde.extractor.decode_high_bit_auxiliary", return_value=fallback) as decoder,
            ):
                report = extract_file(source)
            decoder.assert_called_once_with(b"fake-heif", 0, 52)
            asset = report["assets"][0]
            self.assertEqual(asset["semantic_name"], "linear_thumbnail")
            self.assertEqual(asset["source_bit_depth"], 10)
            self.assertEqual(asset["maximum"], 1023)
            self.assertIn("high_bit_fallback", asset["metadata"])
            result = read_png_exact(Path(directory) / "high-bit_linear_thumbnail.png")
            self.assertEqual(int(result[0, 0, 1]), 1)
            self.assertEqual(int(result[0, 0, 2]), 1023)

    def test_inspection_uses_native_code_mode_and_preserves_depth_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "photo.heic"
            source.write_bytes(b"fake-heif")
            with patch("ipde.extractor.pillow_heif.open_heif", return_value=fixture_file()) as opener:
                report = inspect_file(source)
            kwargs = opener.call_args.kwargs
            self.assertEqual(opener.call_args.args[0], b"fake-heif")
            self.assertFalse(kwargs["convert_hdr_to_8bit"])
            self.assertFalse(kwargs["hdr_to_16bit"])
            self.assertTrue(kwargs["remove_stride"])
            self.assertEqual(report["asset_count"], 2)
            self.assertEqual(report["assets"][0]["source_bit_depth"], 12)
            self.assertEqual(report["assets"][0]["maximum"], 4095)
            self.assertEqual(report["assets"][0]["metadata"]["representation_name"], "uniform_disparity")

    def test_extraction_writes_and_verifies_png_npy_metadata_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photo.edit.heic"
            output = root / "output"
            source.write_bytes(b"fake-heif")
            with patch("ipde.extractor.pillow_heif.open_heif", return_value=fixture_file()):
                report = extract_file(source, ExtractOptions(output_dir=output, write_manifest=True))

            depth_png = output / "photo.edit_depth.png"
            depth_npy = output / "photo.edit_depth.npy"
            gain_png = output / "photo.edit_hdr_gain_map.png"
            xmp = output / "photo.edit_hdr_gain_map_metadata0.xmp"
            manifest = output / "photo.edit_aux_manifest.json"
            for path in (depth_png, depth_npy, gain_png, xmp, manifest):
                self.assertTrue(path.is_file(), path)
            self.assertEqual(int(read_png_exact(depth_png)[0, 1]), 1)
            self.assertEqual(int(read_npy_exact(depth_npy)[0, 2]), 4095)
            self.assertEqual(xmp.read_bytes(), b"<x:xmpmeta>gain metadata</x:xmpmeta>\n")
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], "ipde-extraction-manifest-v1")
            self.assertEqual(loaded["asset_count"], 2)
            self.assertEqual(Path(report["manifest_path"]), manifest.resolve())
            self.assertTrue(all(output["verified"] for asset in loaded["assets"] for output in asset["outputs"]))
            self.assertFalse((output / "photo.edit_depth_meters.exr").exists())
            self.assertTrue(any("requires uint8" in warning for warning in loaded["warnings"]))

    def test_collision_refuses_before_replacing_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photo.heic"
            source.write_bytes(b"fake-heif")
            existing = root / "photo_depth.png"
            existing.write_bytes(b"keep-me")
            with patch("ipde.extractor.pillow_heif.open_heif", return_value=fixture_file()):
                with self.assertRaisesRegex(ExtractionError, "output already exists"):
                    extract_file(source)
            self.assertEqual(existing.read_bytes(), b"keep-me")
            self.assertFalse((root / "photo_depth.npy").exists())

    def test_alpha_is_extracted_as_an_independent_plane(self) -> None:
        alpha_values = np.array([[0, 1], [1023, 500]], dtype=np.uint16)
        rgba = np.zeros((2, 2, 4), dtype=np.uint16)
        rgba[:, :, 3] = alpha_values
        parent = FakeParent(
            rgba,
            "RGBA;10",
            10,
            has_alpha=True,
            premultiplied_alpha=True,
            info={"primary": True, "bit_depth": 10, "depth_images": [], "aux": {}, "thumbnails": []},
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "alpha.heic"
            source.write_bytes(b"fake-heif")
            with patch("ipde.extractor.pillow_heif.open_heif", return_value=FakeFile([parent])):
                report = extract_file(source)
            self.assertEqual(report["asset_count"], 1)
            self.assertEqual(report["assets"][0]["kind"], "alpha")
            self.assertTrue(report["assets"][0]["premultiplied_alpha"])
            self.assertTrue(np.array_equal(read_png_exact(Path(directory) / "alpha_alpha.png"), alpha_values))


if __name__ == "__main__":
    unittest.main()
