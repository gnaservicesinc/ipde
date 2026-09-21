"""Exercise tensor preparation and reverse validation without downloading weights."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np

from ipde.spatial import RaftStereoOptions, run_raft_stereo


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch is not installed')
class RaftRunnerTests(unittest.TestCase):
    def test_native_rgb_tensors_and_reverse_validation_preserve_raw_flow(self):
        import torch
        calls = []

        class Model:
            def __init__(self, configuration):
                pass
            def load_state_dict(self, state, strict):
                pass
            def to(self, device):
                return self
            def eval(self):
                return self
            def __call__(self, left, right, **kwargs):
                calls.append((left.clone(), right.clone()))
                # Reverse disagrees everywhere, despite a finite forward result.
                flow = torch.full((1, 1, left.shape[2], left.shape[3]),
                                  -8.0 if len(calls) == 1 else -20.0)
                return flow, flow

        class Padder:
            def __init__(self, shape, divis_by):
                pass
            def pad(self, *arrays):
                return [torch.nn.functional.pad(a, (8, 8, 4, 4), mode='replicate') for a in arrays]
            def unpad(self, array):
                return array[:, :, 4:-4, 8:-8]

        raft = ModuleType('core.raft_stereo')
        raft.RAFTStereo = Model
        utils = ModuleType('core.utils.utils')
        utils.InputPadder = Padder
        left = np.arange(40 * 96 * 3, dtype=np.uint8).reshape(40, 96, 3)
        right = left[:, ::-1].copy()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (patch.dict(sys.modules, {'core.raft_stereo': raft, 'core.utils.utils': utils}),
                  patch('ipde.spatial.resolve_raft_resources', return_value=(root, root / 'model.pth', None)),
                  patch('ipde.spatial._checkpoint_bytes', return_value=(b'fixture', 'model.pth')),
                  patch('torch.load', return_value={}),
                  patch('ipde.spatial.register_stereo_rows', return_value=(right, np.ones((40, 96), bool), {}))):
                result = run_raft_stereo(left, right, {
                    'left_camera': {'width': 96, 'height': 40}, 'raft_stereo_ready': True,
                    'principal_point_delta_x_pixels': 0, 'focal_length_pixels_for_depth': 100,
                    'baseline_meters': .1,
                }, RaftStereoOptions(device='cpu'))
        self.assertEqual(len(calls), 2)
        actual_left = calls[0][0][0, :, 4:-4, 8:-8].permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(actual_left, left.astype(np.float32))
        self.assertTrue(torch.equal(calls[1][0], torch.flip(calls[0][1], [3])))
        self.assertTrue(torch.equal(calls[1][1], torch.flip(calls[0][0], [3])))
        np.testing.assert_array_equal(result.signed_flow_pixels, np.full((40, 96), -8, np.float32))
        np.testing.assert_array_equal(result.height_disparity_pixels, np.full((40, 96), 8, np.float32))
        np.testing.assert_array_equal(result.depth_meters, np.full((40, 96), 1.25, np.float32))
        self.assertFalse(result.support_mask.any())
        self.assertFalse(result.details['depth_and_disparity_filtered_by_support'])
        self.assertEqual(result.details['model_internal_downsample_factor'], 4)
