"""High-bit auxiliary decoding fallback for pillow-heif 1.5.

pillow-heif 1.5 inventories 10/12-bit auxiliary items but its C binding rejects
them before decoding. This module calls the libheif bundled with pillow-heif for
only that rejected handle. It requests unshifted 16-bit storage and never routes
through an 8-bit conversion.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pillow_heif


class HighBitAuxiliaryError(RuntimeError):
    """Raised when libheif cannot losslessly decode a high-bit auxiliary item."""


class _HeifError(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_int),
        ("subcode", ctypes.c_int),
        ("message", ctypes.c_char_p),
    ]


class _NclxProfile(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint8),
        ("color_primaries", ctypes.c_int),
        ("transfer_characteristics", ctypes.c_int),
        ("matrix_coefficients", ctypes.c_int),
        ("full_range_flag", ctypes.c_uint8),
        ("color_primary_red_x", ctypes.c_float),
        ("color_primary_red_y", ctypes.c_float),
        ("color_primary_green_x", ctypes.c_float),
        ("color_primary_green_y", ctypes.c_float),
        ("color_primary_blue_x", ctypes.c_float),
        ("color_primary_blue_y", ctypes.c_float),
        ("color_primary_white_x", ctypes.c_float),
        ("color_primary_white_y", ctypes.c_float),
    ]


@dataclass(frozen=True)
class DecodedAuxiliary:
    array: np.ndarray
    mode: str
    source_bit_depth: int
    metadata_blocks: list[dict[str, Any]]
    details: dict[str, Any]


def _function(library: ctypes.CDLL, name: str, arguments: list[Any], result: Any) -> Any:
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result
    return function


def _bundled_library_path() -> Path | str:
    package = Path(pillow_heif.__file__).resolve().parent
    candidates: list[Path] = []
    for directory_name in (".dylibs", ".libs"):
        directory = package / directory_name
        if directory.is_dir():
            candidates.extend(directory.glob("libheif*.dylib"))
            candidates.extend(directory.glob("libheif*.so*"))
    if candidates:
        return sorted(candidates, key=lambda path: len(path.name), reverse=True)[0]
    system = ctypes.util.find_library("heif")
    if system:
        return system
    raise HighBitAuxiliaryError("pillow-heif's bundled libheif library could not be located")


@lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    try:
        library = ctypes.CDLL(str(_bundled_library_path()))
    except OSError as exc:
        raise HighBitAuxiliaryError(f"could not load pillow-heif's bundled libheif: {exc}") from exc

    void_pointer = ctypes.c_void_p
    error = _HeifError
    item_id = ctypes.c_uint32

    _function(library, "heif_context_alloc", [], void_pointer)
    _function(library, "heif_context_free", [void_pointer], None)
    _function(
        library,
        "heif_context_read_from_memory",
        [void_pointer, void_pointer, ctypes.c_size_t, void_pointer],
        error,
    )
    _function(library, "heif_context_get_number_of_top_level_images", [void_pointer], ctypes.c_int)
    _function(
        library,
        "heif_context_get_list_of_top_level_image_IDs",
        [void_pointer, ctypes.POINTER(item_id), ctypes.c_int],
        ctypes.c_int,
    )
    _function(
        library,
        "heif_context_get_image_handle",
        [void_pointer, item_id, ctypes.POINTER(void_pointer)],
        error,
    )
    _function(
        library,
        "heif_image_handle_get_auxiliary_image_handle",
        [void_pointer, item_id, ctypes.POINTER(void_pointer)],
        error,
    )
    _function(library, "heif_image_handle_release", [void_pointer], None)
    _function(library, "heif_image_handle_get_width", [void_pointer], ctypes.c_int)
    _function(library, "heif_image_handle_get_height", [void_pointer], ctypes.c_int)
    _function(library, "heif_image_handle_get_luma_bits_per_pixel", [void_pointer], ctypes.c_int)
    _function(
        library,
        "heif_image_handle_get_preferred_decoding_colorspace",
        [void_pointer, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
        error,
    )
    _function(
        library,
        "heif_decode_image",
        [void_pointer, ctypes.POINTER(void_pointer), ctypes.c_int, ctypes.c_int, void_pointer],
        error,
    )
    _function(library, "heif_image_release", [void_pointer], None)
    _function(library, "heif_image_get_primary_width", [void_pointer], ctypes.c_int)
    _function(library, "heif_image_get_primary_height", [void_pointer], ctypes.c_int)
    _function(
        library,
        "heif_image_get_plane_readonly2",
        [void_pointer, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)],
        ctypes.POINTER(ctypes.c_uint8),
    )
    _function(library, "heif_image_get_bits_per_pixel_range", [void_pointer, ctypes.c_int], ctypes.c_int)

    _function(
        library,
        "heif_image_handle_get_number_of_metadata_blocks",
        [void_pointer, ctypes.c_char_p],
        ctypes.c_int,
    )
    _function(
        library,
        "heif_image_handle_get_list_of_metadata_block_IDs",
        [void_pointer, ctypes.c_char_p, ctypes.POINTER(item_id), ctypes.c_int],
        ctypes.c_int,
    )
    _function(
        library,
        "heif_image_handle_get_metadata_type",
        [void_pointer, item_id],
        ctypes.c_char_p,
    )
    _function(
        library,
        "heif_image_handle_get_metadata_content_type",
        [void_pointer, item_id],
        ctypes.c_char_p,
    )
    _function(
        library,
        "heif_image_handle_get_metadata_size",
        [void_pointer, item_id],
        ctypes.c_size_t,
    )
    _function(
        library,
        "heif_image_handle_get_metadata",
        [void_pointer, item_id, void_pointer],
        error,
    )

    _function(
        library,
        "heif_image_handle_get_nclx_color_profile",
        [void_pointer, ctypes.POINTER(ctypes.POINTER(_NclxProfile))],
        error,
    )
    _function(library, "heif_nclx_color_profile_free", [ctypes.POINTER(_NclxProfile)], None)
    _function(library, "heif_image_handle_get_raw_color_profile_size", [void_pointer], ctypes.c_size_t)
    _function(
        library,
        "heif_image_handle_get_raw_color_profile",
        [void_pointer, void_pointer],
        error,
    )
    return library


def _message(error: _HeifError) -> str:
    return error.message.decode("utf-8", "replace") if error.message else "unknown libheif error"


def _check(error: _HeifError, operation: str) -> None:
    if error.code != 0:
        raise HighBitAuxiliaryError(
            f"{operation} failed: {_message(error)} (code={error.code}, subcode={error.subcode})"
        )


def _metadata_blocks(library: ctypes.CDLL, handle: ctypes.c_void_p) -> list[dict[str, Any]]:
    count = library.heif_image_handle_get_number_of_metadata_blocks(handle, None)
    if count <= 0:
        return []
    ids = (ctypes.c_uint32 * count)()
    actual = library.heif_image_handle_get_list_of_metadata_block_IDs(handle, None, ids, count)
    blocks: list[dict[str, Any]] = []
    for index in range(actual):
        metadata_id = ids[index]
        type_bytes = library.heif_image_handle_get_metadata_type(handle, metadata_id) or b""
        content_bytes = library.heif_image_handle_get_metadata_content_type(handle, metadata_id) or b""
        size = int(library.heif_image_handle_get_metadata_size(handle, metadata_id))
        if size < 0 or size > 256 * 1024 * 1024:
            raise HighBitAuxiliaryError(f"auxiliary metadata block has unreasonable size {size}")
        buffer = ctypes.create_string_buffer(size)
        if size:
            _check(
                library.heif_image_handle_get_metadata(handle, metadata_id, buffer),
                f"reading auxiliary metadata block {metadata_id}",
            )
            data = buffer.raw[:size]
        else:
            data = b""
        blocks.append(
            {
                "type": type_bytes.decode("utf-8", "replace"),
                "content_type": content_bytes.decode("utf-8", "replace"),
                "data": data,
            }
        )
    return blocks


def _color_profile(library: ctypes.CDLL, handle: ctypes.c_void_p) -> dict[str, Any]:
    result: dict[str, Any] = {}
    profile_pointer = ctypes.POINTER(_NclxProfile)()
    error = library.heif_image_handle_get_nclx_color_profile(handle, ctypes.byref(profile_pointer))
    if error.code == 0 and profile_pointer:
        try:
            profile = profile_pointer.contents
            result["nclx"] = {
                "color_primaries": profile.color_primaries,
                "transfer_characteristics": profile.transfer_characteristics,
                "matrix_coefficients": profile.matrix_coefficients,
                "full_range_flag": bool(profile.full_range_flag),
            }
        finally:
            library.heif_nclx_color_profile_free(profile_pointer)

    raw_size = int(library.heif_image_handle_get_raw_color_profile_size(handle))
    if raw_size:
        if raw_size > 256 * 1024 * 1024:
            raise HighBitAuxiliaryError(f"auxiliary ICC profile has unreasonable size {raw_size}")
        raw_buffer = ctypes.create_string_buffer(raw_size)
        _check(
            library.heif_image_handle_get_raw_color_profile(handle, raw_buffer),
            "reading auxiliary ICC profile",
        )
        result["icc"] = raw_buffer.raw[:raw_size]
    return result


def decode_high_bit_auxiliary(
    source_bytes: bytes,
    parent_image_index: int,
    auxiliary_id: int,
) -> DecodedAuxiliary:
    """Decode one rejected 10/12-bit auxiliary item into unshifted uint16 codes."""
    library = _library()
    context = library.heif_context_alloc()
    if not context:
        raise HighBitAuxiliaryError("libheif could not allocate a decoding context")
    parent_handle = ctypes.c_void_p()
    auxiliary_handle = ctypes.c_void_p()
    decoded_image = ctypes.c_void_p()
    try:
        source_buffer = ctypes.create_string_buffer(source_bytes)
        _check(
            library.heif_context_read_from_memory(context, source_buffer, len(source_bytes), None),
            "reading HEIF snapshot",
        )
        image_count = int(library.heif_context_get_number_of_top_level_images(context))
        if parent_image_index < 0 or parent_image_index >= image_count:
            raise HighBitAuxiliaryError(
                f"parent image index {parent_image_index} is outside the {image_count}-image container"
            )
        image_ids = (ctypes.c_uint32 * image_count)()
        actual_count = int(
            library.heif_context_get_list_of_top_level_image_IDs(context, image_ids, image_count)
        )
        if parent_image_index >= actual_count:
            raise HighBitAuxiliaryError("libheif returned an incomplete top-level image list")
        _check(
            library.heif_context_get_image_handle(
                context, image_ids[parent_image_index], ctypes.byref(parent_handle)
            ),
            f"opening parent image {parent_image_index}",
        )
        _check(
            library.heif_image_handle_get_auxiliary_image_handle(
                parent_handle, auxiliary_id, ctypes.byref(auxiliary_handle)
            ),
            f"opening auxiliary image {auxiliary_id}",
        )

        bit_depth = int(library.heif_image_handle_get_luma_bits_per_pixel(auxiliary_handle))
        if bit_depth not in (10, 12):
            raise HighBitAuxiliaryError(
                f"fallback is restricted to 10/12-bit auxiliary images; item {auxiliary_id} is {bit_depth}-bit"
            )
        preferred_colorspace = ctypes.c_int()
        preferred_chroma = ctypes.c_int()
        _check(
            library.heif_image_handle_get_preferred_decoding_colorspace(
                auxiliary_handle,
                ctypes.byref(preferred_colorspace),
                ctypes.byref(preferred_chroma),
            ),
            f"querying auxiliary image {auxiliary_id} colorspace",
        )
        metadata_blocks = _metadata_blocks(library, auxiliary_handle)
        color_profile = _color_profile(library, auxiliary_handle)

        if preferred_colorspace.value == 2:  # heif_colorspace_monochrome
            output_colorspace = 2
            output_chroma = 0
            output_channel = 0
            channels = 1
            mode = f"I;{bit_depth}"
        elif preferred_colorspace.value in (0, 1):  # YCbCr or RGB
            output_colorspace = 1  # RGB
            output_chroma = 14  # interleaved RRGGBB little-endian
            output_channel = 10  # interleaved
            channels = 3
            mode = f"RGB;{bit_depth}"
        else:
            raise HighBitAuxiliaryError(
                f"unsupported preferred colorspace {preferred_colorspace.value} for auxiliary image {auxiliary_id}"
            )

        _check(
            library.heif_decode_image(
                auxiliary_handle,
                ctypes.byref(decoded_image),
                output_colorspace,
                output_chroma,
                None,
            ),
            f"decoding {bit_depth}-bit auxiliary image {auxiliary_id}",
        )
        width = int(library.heif_image_get_primary_width(decoded_image))
        height = int(library.heif_image_get_primary_height(decoded_image))
        if width <= 0 or height <= 0:
            raise HighBitAuxiliaryError(f"decoded auxiliary image has invalid size {width}x{height}")
        decoded_range = int(library.heif_image_get_bits_per_pixel_range(decoded_image, output_channel))
        if decoded_range != bit_depth:
            raise HighBitAuxiliaryError(
                f"decoded range changed from {bit_depth} to {decoded_range} bits for auxiliary image {auxiliary_id}"
            )
        stride = ctypes.c_size_t()
        plane = library.heif_image_get_plane_readonly2(
            decoded_image, output_channel, ctypes.byref(stride)
        )
        if not plane:
            raise HighBitAuxiliaryError(f"libheif returned no pixel plane for auxiliary image {auxiliary_id}")
        row_bytes = width * channels * 2
        if stride.value < row_bytes:
            raise HighBitAuxiliaryError(
                f"decoded stride {stride.value} is smaller than row data {row_bytes}"
            )
        plane_address = ctypes.addressof(plane.contents)
        packed = bytearray(row_bytes * height)
        for row in range(height):
            start = row * row_bytes
            packed[start : start + row_bytes] = ctypes.string_at(
                plane_address + row * stride.value, row_bytes
            )
        shape = (height, width) if channels == 1 else (height, width, channels)
        array = np.frombuffer(packed, dtype="<u2").reshape(shape).copy()
        maximum_code = (1 << bit_depth) - 1
        if int(array.max()) > maximum_code:
            raise HighBitAuxiliaryError(
                f"decoded sample exceeds the {bit_depth}-bit range; refusing a shifted conversion"
            )
        return DecodedAuxiliary(
            array=array,
            mode=mode,
            source_bit_depth=bit_depth,
            metadata_blocks=metadata_blocks,
            details={
                "backend": "pillow-heif bundled libheif ctypes fallback",
                "reason": "pillow-heif 1.5 rejects auxiliary images above 8 bits",
                "auxiliary_id": auxiliary_id,
                "parent_image_index": parent_image_index,
                "preferred_colorspace": preferred_colorspace.value,
                "preferred_chroma": preferred_chroma.value,
                "decoded_colorspace": "monochrome" if channels == 1 else "RGB",
                "decoded_storage": "little-endian uint16, unshifted source-range codes",
                "source_bit_depth": bit_depth,
                "decoded_bit_depth_range": decoded_range,
                "color_profile": color_profile,
            },
        )
    finally:
        if decoded_image:
            library.heif_image_release(decoded_image)
        if auxiliary_handle:
            library.heif_image_handle_release(auxiliary_handle)
        if parent_handle:
            library.heif_image_handle_release(parent_handle)
        library.heif_context_free(context)
