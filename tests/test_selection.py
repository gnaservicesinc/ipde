"""Selective export must not write or compute unrequested products."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipde.cli import main
from ipde.extractor import Asset, Discovery, ExtractOptions, ExtractionError, extract_file, inspect_file
from ipde.formats import read_exr_exact, read_png_exact
from ipde.spatial import RaftStereoResult, StereoMatchingResult


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "photo.heic"
        self.source.write_bytes(b"fixture")
        self.depth = np.arange(6, dtype=np.uint8).reshape(2, 3)
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        self.discovery = Discovery(self.source, 7, "hash", "image/heic", 0, [], [
            Asset("depth", 0, 0, self.depth, "L", 8, "depth", metadata={
                "representation_type": 1, "d_min": 1.0, "d_max": 4.0,
                "depth_representation": "uniform_disparity",
            }),
            Asset("auxiliary", 0, 0, self.depth, "L", 8, "hdr_gain_map"),
            Asset("spatial_view", 1, 0, rgb, "RGB", 8, "spatial_left"),
            Asset("spatial_view", 2, 0, rgb, "RGB", 8, "spatial_right"),
        ], spatial_photo={"left_image_index": 1, "right_image_index": 2,
                          "rectified_stereo_ready": True,
                          "left_camera": {"width": 3, "height": 2}})
        self.patcher = patch("ipde.extractor.discover_file", return_value=self.discovery)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.height = np.array([[20, 21, 22], [23, 24, 25]], dtype=np.float32)
        self.raft = RaftStereoResult(-self.height, self.height, 1 / self.height, {})

    def export(self, *selection, **kwargs):
        return extract_file(self.source, ExtractOptions(
            output_dir=self.directory / "out", selected_products=selection, write_npy=False, **kwargs))

    def files(self):
        return {p.name for p in (self.directory / "out").iterdir() if p.suffix != ".json"}

    def test_raw_depth_only_is_bit_exact_and_runs_no_matcher(self):
        with patch("ipde.extractor.run_raft_stereo") as raft, patch("ipde.extractor.run_stereo_matching") as stereo:
            report = self.export("raw:0")
        raft.assert_not_called()
        stereo.assert_not_called()
        self.assertEqual(self.files(), {"photo_depth.png"})
        np.testing.assert_array_equal(read_png_exact(self.directory / "out/photo_depth.png"), self.depth)
        self.assertEqual(report["selected_products"], ["raw:0"])

    def test_raft_only_omits_views_companions_and_classical_matching(self):
        with patch("ipde.extractor.run_raft_stereo", return_value=self.raft) as raft, patch("ipde.extractor.run_stereo_matching") as stereo:
            self.export("raft-displacement")
        raft.assert_called_once()
        stereo.assert_not_called()
        self.assertEqual(self.files(), {"photo_spatial_raft_stereo_displacement_0_to_1.exr"})
        result = read_exr_exact(self.directory / "out/photo_spatial_raft_stereo_displacement_0_to_1.exr", (2, 3))
        np.testing.assert_array_equal(result, np.array([[0, .2, .4], [.6, .8, 1]], dtype=np.float32))

    def test_raw_raft_height_stays_in_pixels(self):
        with patch("ipde.extractor.run_raft_stereo", return_value=self.raft):
            self.export("raft-height")
        self.assertEqual(self.files(), {"photo_spatial_raft_stereo_height.exr"})
        result = read_exr_exact(self.directory / "out/photo_spatial_raft_stereo_height.exr", (2, 3))
        np.testing.assert_array_equal(result, self.height)

    def test_calibrated_depth_only_omits_raw_and_other_calibrations(self):
        self.export("meters:0")
        self.assertEqual(self.files(), {"photo_depth_meters.exr"})
        result = read_exr_exact(self.directory / "out/photo_depth_meters.exr", (2, 3))
        expected = np.float32(1) / (self.depth.astype(np.float32) / np.float32(255) * np.float32(3) + np.float32(1))
        np.testing.assert_array_equal(result, expected)

    def test_raw_height_does_not_change_and_unselected_collision_does_not_block(self):
        out = self.directory / "out"
        out.mkdir()
        (out / "photo_spatial_raft_stereo_height.exr").write_bytes(b"existing")
        with patch("ipde.extractor.run_raft_stereo", return_value=self.raft):
            self.export("raft-displacement")
        self.assertEqual((out / "photo_spatial_raft_stereo_height.exr").read_bytes(), b"existing")
        np.testing.assert_array_equal(self.raft.height_disparity_pixels, self.height)

    def test_comparison_does_not_change_raft_displacement_range(self):
        classical = StereoMatchingResult(self.height * 100, {})
        with patch("ipde.extractor.run_raft_stereo", return_value=self.raft), patch("ipde.extractor.run_stereo_matching", return_value=classical):
            self.export("raft-displacement", "stereo-displacement")
        result = read_exr_exact(self.directory / "out/photo_spatial_raft_stereo_displacement_0_to_1.exr", (2, 3))
        self.assertEqual(float(result.max()), 1)
        self.assertEqual(float(result.min()), 0)
        self.assertGreater(len(np.unique(result)), 2)

    def test_independent_exports_do_not_collide_on_manifest(self):
        first = self.export("raw:0")
        second = self.export("raw:1")
        self.assertNotEqual(first["manifest_path"], second["manifest_path"])
        self.assertEqual(self.files(), {"photo_depth.png", "photo_hdr_gain_map.png"})

    def test_invalid_selection_is_rejected_before_any_output(self):
        with patch("ipde.extractor.run_raft_stereo") as raft:
            with self.assertRaisesRegex(ExtractionError, "unavailable"):
                self.export("raw:999")
        raft.assert_not_called()
        self.assertFalse((self.directory / "out").exists())

    def test_cli_select_drives_inference_without_legacy_flags(self):
        with patch("ipde.cli.extract_file", return_value={}) as extract, patch("builtins.print"):
            self.assertEqual(main([str(self.source), "--json", "--select", "raft-height", "--no-npy", "--raft-model", "/tmp/chosen.pth"]), 0)
        config = extract.call_args.args[1]
        self.assertEqual(config.selected_products, ("raft-height",))
        self.assertFalse(config.write_npy)
        self.assertEqual(config.raft_model, Path("/tmp/chosen.pth"))

    def test_inventory_exposes_raw_and_generated_choices(self):
        products = {p["id"] for p in inspect_file(self.source)["available_products"]}
        self.assertTrue({"raw:0", "raw:1", "raft-height", "raft-displacement", "stereo-height"} <= products)
