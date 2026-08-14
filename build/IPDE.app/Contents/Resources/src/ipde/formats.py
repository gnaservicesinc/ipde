"""Lossless array writers with mandatory round-trip verification."""

from __future__ import annotations

import binascii
import hashlib
import struct
import zlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}
_PNG_CHANNEL_COUNTS = {0: 1, 4: 2, 2: 3, 6: 4}


class FormatError(RuntimeError):
    """Raised when a lossless format cannot represent or verify an array."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def _normalized_png_array(array: np.ndarray) -> tuple[np.ndarray, int]:
    value = np.asarray(array)
    if value.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise FormatError(f"PNG requires uint8 or uint16, not {value.dtype}")
    if value.ndim == 2:
        channels = 1
    elif value.ndim == 3 and value.shape[2] in _PNG_COLOR_TYPES:
        channels = int(value.shape[2])
        if channels == 1:
            value = value[:, :, 0]
    else:
        raise FormatError(f"PNG requires HxW or HxWx(1..4), not shape {value.shape}")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise FormatError("PNG cannot store an empty array")
    return np.ascontiguousarray(value), channels


def write_png(path: Path, array: np.ndarray) -> None:
    """Write unsigned samples directly to PNG without Pillow conversions."""
    value, channels = _normalized_png_array(array)
    height, width = value.shape[:2]
    bit_depth = value.dtype.itemsize * 8
    if bit_depth == 16:
        # PNG is network byte order. astype swaps bytes only; numeric codes do not change.
        encoded = value.astype(">u2", copy=False).tobytes(order="C")
        row_bytes = width * channels * 2
    else:
        encoded = value.tobytes(order="C")
        row_bytes = width * channels
    scanlines = b"".join(b"\x00" + encoded[offset : offset + row_bytes] for offset in range(0, len(encoded), row_bytes))
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, _PNG_COLOR_TYPES[channels], 0, 0, 0)
    payload = (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def read_png_exact(path: Path) -> np.ndarray:
    """Read the deliberately simple PNG subset emitted by :func:`write_png`."""
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise FormatError("invalid PNG signature")
    position = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    compressed = bytearray()
    saw_end = False
    while position < len(data):
        if position + 12 > len(data):
            raise FormatError("truncated PNG chunk")
        length = struct.unpack_from(">I", data, position)[0]
        position += 4
        kind = data[position : position + 4]
        position += 4
        end = position + length
        if end + 4 > len(data):
            raise FormatError("truncated PNG payload")
        payload = data[position:end]
        expected_crc = struct.unpack_from(">I", data, end)[0]
        if (binascii.crc32(kind + payload) & 0xFFFFFFFF) != expected_crc:
            raise FormatError(f"PNG CRC mismatch in {kind!r}")
        position = end + 4
        if kind == b"IHDR":
            if len(payload) != 13:
                raise FormatError("invalid PNG IHDR")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            saw_end = True
            break
    if ihdr is None or not saw_end:
        raise FormatError("incomplete PNG")
    width, height, bit_depth, color_type, compression, filtering, interlace = ihdr
    if bit_depth not in (8, 16) or color_type not in _PNG_CHANNEL_COUNTS:
        raise FormatError("unsupported PNG layout")
    if (compression, filtering, interlace) != (0, 0, 0):
        raise FormatError("unsupported PNG encoding flags")
    channels = _PNG_CHANNEL_COUNTS[color_type]
    row_bytes = width * channels * (bit_depth // 8)
    raw = zlib.decompress(bytes(compressed))
    expected_size = height * (row_bytes + 1)
    if len(raw) != expected_size:
        raise FormatError("PNG scanline size mismatch")
    rows = []
    for row in range(height):
        offset = row * (row_bytes + 1)
        if raw[offset] != 0:
            raise FormatError("unexpected PNG filter")
        rows.append(raw[offset + 1 : offset + row_bytes + 1])
    pixels = b"".join(rows)
    dtype = np.dtype("uint8") if bit_depth == 8 else np.dtype(">u2")
    result = np.frombuffer(pixels, dtype=dtype).reshape(height, width, channels)
    if bit_depth == 16:
        result = result.astype(np.uint16)
    else:
        result = result.copy()
    return result[:, :, 0] if channels == 1 else result


def write_npy(path: Path, array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)


def read_npy_exact(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.load(stream, allow_pickle=False)


def _exr_channels(array: np.ndarray) -> dict[str, np.ndarray]:
    value = np.ascontiguousarray(array)
    if value.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise FormatError(f"OpenEXR requires float16 or float32, not {value.dtype}")
    if value.ndim == 2:
        return {"Y": value}
    if value.ndim != 3:
        raise FormatError(f"OpenEXR requires HxW or HxWxC, not shape {value.shape}")
    channels = value.shape[2]
    if channels == 1:
        return {"Y": value[:, :, 0]}
    if channels == 2:
        return {"Y": np.ascontiguousarray(value[:, :, 0]), "A": np.ascontiguousarray(value[:, :, 1])}
    if channels == 3:
        return {"RGB": value}
    if channels == 4:
        return {"RGBA": value}
    raise FormatError(f"OpenEXR supports 1..4 channels here, not {channels}")


def write_exr(
    path: Path,
    array: np.ndarray,
    *,
    storage_description: str = "Decoded sample values; no normalization or transfer function applied",
    attributes: Mapping[str, str] | None = None,
) -> None:
    try:
        import OpenEXR  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FormatError("floating-point auxiliary data requires OpenEXR>=3.4.4") from exc
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
        "ipdeStorage": storage_description,
    }
    if attributes:
        header.update(attributes)
    with OpenEXR.File(header, _exr_channels(array)) as output:
        output.write(str(path))


def read_exr_exact(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        import OpenEXR  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FormatError("cannot verify EXR without OpenEXR") from exc
    with OpenEXR.File(str(path)) as source:
        channels = source.channels()
        if len(expected_shape) == 2 or expected_shape[-1] == 1:
            value = channels["Y"].pixels
            return value if len(expected_shape) == 2 else value[:, :, None]
        if expected_shape[-1] == 2:
            return np.stack((channels["Y"].pixels, channels["A"].pixels), axis=2)
        key = "RGB" if expected_shape[-1] == 3 else "RGBA"
        return channels[key].pixels


def arrays_bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare dtype, shape, and every stored bit (including NaN payloads)."""
    a = np.ascontiguousarray(left)
    b = np.ascontiguousarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def verify_png(path: Path, expected: np.ndarray) -> None:
    actual = read_png_exact(path)
    if not arrays_bit_equal(np.asarray(expected, dtype=actual.dtype), actual):
        raise FormatError(f"PNG round-trip mismatch: {path.name}")


def verify_npy(path: Path, expected: np.ndarray) -> None:
    if not arrays_bit_equal(expected, read_npy_exact(path)):
        raise FormatError(f"NPY round-trip mismatch: {path.name}")


def verify_exr(path: Path, expected: np.ndarray) -> None:
    actual = read_exr_exact(path, tuple(expected.shape))
    if not arrays_bit_equal(expected, actual):
        raise FormatError(f"EXR round-trip mismatch: {path.name}")
