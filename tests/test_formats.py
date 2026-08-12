from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ipde.formats import (
    arrays_bit_equal,
    read_exr_exact,
    read_npy_exact,
    read_png_exact,
    verify_exr,
    verify_npy,
    verify_png,
    write_exr,
    write_npy,
    write_png,
)


class LosslessFormatTests(unittest.TestCase):
    def test_png_uint8_grayscale_round_trip(self) -> None:
        array = np.arange(35, dtype=np.uint8).reshape(5, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.png"
            write_png(path, array)
            verify_png(path, array)
            self.assertTrue(arrays_bit_equal(array, read_png_exact(path)))
            self.assertEqual(path.read_bytes()[24], 8)  # IHDR bit-depth field.

    def test_png_uint16_rgb_preserves_low_range_codes(self) -> None:
        array = np.array(
            [
                [[0, 1, 2], [1023, 2048, 4095]],
                [[17, 33, 65], [65535, 32768, 9]],
            ],
            dtype=np.uint16,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.png"
            write_png(path, array)
            verify_png(path, array)
            result = read_png_exact(path)
            self.assertTrue(arrays_bit_equal(array, result))
            self.assertEqual(int(result[0, 0, 1]), 1)
            self.assertEqual(path.read_bytes()[24], 16)

    def test_npy_preserves_dtype_endianness_shape_and_nan_bits(self) -> None:
        bit_patterns = np.array([0x3F800000, 0x7FC01234, 0x80000000], dtype=np.uint32)
        array = bit_patterns.view(np.float32).reshape(1, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npy"
            write_npy(path, array)
            verify_npy(path, array)
            self.assertTrue(arrays_bit_equal(array, read_npy_exact(path)))

    @unittest.skipUnless(importlib.util.find_spec("OpenEXR"), "OpenEXR is not installed")
    def test_exr_float16_round_trip_is_bit_exact(self) -> None:
        array = np.array([[0.0, 1.0, -2.0], [0.5, 10.0, 65504.0]], dtype=np.float16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.exr"
            write_exr(path, array)
            verify_exr(path, array)
            self.assertTrue(arrays_bit_equal(array, read_exr_exact(path, array.shape)))


if __name__ == "__main__":
    unittest.main()

