import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ipde.apple_imageio import ImageIOMetadataError, decode_native_depth_buffer
from ipde.extractor import ExtractOptions, ExtractionError, extract_file, inspect_file
from ipde.formats import arrays_bit_equal, read_exr_exact
from test_extractor import FakeFile, FakeImage, FakeParent
from test_spatial import camera_image


class NativeDepthTests(unittest.TestCase):
    def test_row_padding_is_removed_without_changing_half_float_bits(self):
        # Includes negative zero, a subnormal and a noncanonical NaN payload.
        bits = np.array([[0, 0x8000, 0x0001], [0x3c00, 0x7e31, 0x7c00]], np.uint16)
        packed = b''.join(row.tobytes() + b'padding!' for row in bits)
        result = decode_native_depth_buffer(packed, {
            'Width': 3, 'Height': 2, 'BytesPerRow': 14,
            'PixelFormat': int.from_bytes(b'hdis', 'big'),
        })
        self.assertTrue(arrays_bit_equal(result, bits.view(np.float16)))
        self.assertTrue(result.flags.c_contiguous)

    def test_float32_is_not_demoted_and_truncation_is_rejected(self):
        array = np.array([[1.0000001, -0.0]], np.float32)
        description = {'Width': 2, 'Height': 1, 'BytesPerRow': 8,
                       'PixelFormat': int.from_bytes(b'fdep', 'big')}
        self.assertTrue(arrays_bit_equal(decode_native_depth_buffer(array.tobytes(), description), array))
        with self.assertRaises(ImageIOMetadataError):
            decode_native_depth_buffer(array.tobytes()[:-1], description)
        with self.assertRaises(ImageIOMetadataError):
            decode_native_depth_buffer(array.tobytes(), {**description, 'BytesPerRow': 2})

    def test_inventory_separates_encoded_precision_native_storage_and_generated_products(self):
        raw = FakeImage(np.array([[0, 255]], np.uint8), 'L', 8,
                        info={'metadata': {'representation_type': 1, 'd_min': 1, 'd_max': 2}})
        parent = FakeParent(np.zeros((4, 6, 3), np.uint16), 'RGB;16', 16,
                            info={'bit_depth': 10, 'primary': True, 'depth_images': [raw]})
        native = np.array([[1, 2]], np.float16)
        apple = {'native_depth_images': [{'parent_image_index': 0, 'semantic': 'disparity',
                    'array': native, 'description': {}, 'accuracy': 'relative', 'xmp': b''}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.heic'
            path.write_bytes(b'fixture')
            with (patch('ipde.extractor.pillow_heif.open_heif', return_value=FakeFile([parent])),
                  patch('ipde.extractor.read_apple_imageio_metadata', return_value=apple)):
                report = extract_file(path, ExtractOptions(selected_products=('raw:1',), write_npy=False))
        products = {p['id']: p for p in report['available_products']}
        self.assertEqual(products['raw:0']['source_precision'], '8-bit encoded')
        self.assertEqual(products['raw:1']['precision'], '16-bit float EXR')
        self.assertEqual(products['raw:1']['origin'], 'Apple decoded')
        self.assertEqual(products['meters:0']['origin'], 'Calculated')
        self.assertEqual(report['source']['top_level_images'][0]['source_bit_depth'], 10)
        self.assertEqual(report['assets'][1]['source_bit_depth'], 8)
        self.assertEqual(report['assets'][1]['decoded_storage_bit_depth'], 16)
        self.assertIn('relative accuracy', ' '.join(report['warnings']))

    def test_native_depth_export_round_trips_as_half_not_float32(self):
        native = np.array([[.1234, 1.25]], np.float16)
        parent = FakeParent(np.zeros((2, 3, 3), np.uint8), 'RGB', 8)
        apple = {'native_depth_images': [{'parent_image_index': 0, 'semantic': 'depth',
                    'array': native, 'description': {}, 'accuracy': 'absolute', 'xmp': b''}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.heic'
            path.write_bytes(b'fixture')
            with (patch('ipde.extractor.pillow_heif.open_heif', return_value=FakeFile([parent])),
                  patch('ipde.extractor.read_apple_imageio_metadata', return_value=apple)):
                extract_file(path, ExtractOptions(write_npy=False))
            self.assertTrue(arrays_bit_equal(read_exr_exact(Path(folder) / 'fixture_apple_depth.exr', native.shape), native))

    def test_stereo_roles_and_display_dimensions_are_independent(self):
        primary = FakeParent(np.zeros((8, 12, 3), np.uint8), 'RGB', 8, info={'primary': True})
        right = FakeParent(np.full((2, 3, 3), 10, np.uint8), 'RGB', 8)
        left = FakeParent(np.full((2, 3, 3), 20, np.uint8), 'RGB', 8)
        metadata = {'groups': [{'GroupType': 'StereoPair', 'GroupImageIndexLeft': 2,
                    'GroupImageIndexRight': 1, 'GroupImageIndexMonoscopic': 0, 'GroupImageDisparityAdjustment': 0}],
                    'images': [{}, camera_image(0, width=3, height=2), camera_image(-.02, width=3, height=2)]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.heic'
            path.write_bytes(b'fixture')
            with (patch('ipde.extractor.pillow_heif.open_heif', return_value=FakeFile([primary, right, left])),
                  patch('ipde.extractor.read_apple_imageio_metadata', return_value=metadata)):
                report = inspect_file(path)
                assets = {a['semantic_name']: a for a in report['assets']}
                self.assertEqual(assets['spatial_left']['minimum'], 20)
                self.assertEqual(assets['spatial_right']['minimum'], 10)
                self.assertEqual(assets['display']['shape'], [8, 12, 3])
                self.assertEqual(assets['spatial_left']['shape'], [2, 3, 3])
                metadata['images'][2]['PixelWidth'] = 4
                metadata['images'][1]['PixelWidth'] = 4
                with self.assertRaisesRegex(ExtractionError, 'refusing.*resize'):
                    inspect_file(path)
