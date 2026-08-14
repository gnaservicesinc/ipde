"""Read Apple HEIF stereo-group metadata without decoding image samples.

ImageIO is the authoritative API for Apple's spatial-photo group and camera
metadata.  pillow-heif remains the sole pixel decoder; this module only reads
container properties from the same immutable byte snapshot.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any


class ImageIOMetadataError(RuntimeError):
    """Raised when macOS ImageIO cannot inspect an otherwise valid container."""


_CF_STRING_ENCODING_UTF8 = 0x08000100
_CF_NUMBER_SINT64 = 4
_CF_NUMBER_FLOAT64 = 6


class _ImageIOBridge:
    def __init__(self) -> None:
        try:
            self.cf = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            self.imageio = ctypes.CDLL(
                "/System/Library/Frameworks/ImageIO.framework/ImageIO"
            )
        except OSError as exc:
            raise ImageIOMetadataError(f"could not load macOS ImageIO: {exc}") from exc

        pointer = ctypes.c_void_p
        index = ctypes.c_long
        type_id = ctypes.c_ulong

        self.cf.CFRelease.argtypes = [pointer]
        self.cf.CFGetTypeID.argtypes = [pointer]
        self.cf.CFGetTypeID.restype = type_id
        for name in (
            "CFStringGetTypeID",
            "CFNumberGetTypeID",
            "CFBooleanGetTypeID",
            "CFArrayGetTypeID",
            "CFDictionaryGetTypeID",
            "CFDataGetTypeID",
        ):
            function = getattr(self.cf, name)
            function.restype = type_id

        self.cf.CFStringGetLength.argtypes = [pointer]
        self.cf.CFStringGetLength.restype = index
        self.cf.CFStringGetMaximumSizeForEncoding.argtypes = [index, ctypes.c_uint32]
        self.cf.CFStringGetMaximumSizeForEncoding.restype = index
        self.cf.CFStringGetCString.argtypes = [
            pointer,
            ctypes.c_char_p,
            index,
            ctypes.c_uint32,
        ]
        self.cf.CFStringGetCString.restype = ctypes.c_bool

        self.cf.CFNumberIsFloatType.argtypes = [pointer]
        self.cf.CFNumberIsFloatType.restype = ctypes.c_bool
        self.cf.CFNumberGetValue.argtypes = [pointer, ctypes.c_int, pointer]
        self.cf.CFNumberGetValue.restype = ctypes.c_bool
        self.cf.CFBooleanGetValue.argtypes = [pointer]
        self.cf.CFBooleanGetValue.restype = ctypes.c_bool

        self.cf.CFArrayGetCount.argtypes = [pointer]
        self.cf.CFArrayGetCount.restype = index
        self.cf.CFArrayGetValueAtIndex.argtypes = [pointer, index]
        self.cf.CFArrayGetValueAtIndex.restype = pointer
        self.cf.CFDictionaryGetCount.argtypes = [pointer]
        self.cf.CFDictionaryGetCount.restype = index
        self.cf.CFDictionaryGetKeysAndValues.argtypes = [
            pointer,
            ctypes.POINTER(pointer),
            ctypes.POINTER(pointer),
        ]
        self.cf.CFDataGetLength.argtypes = [pointer]
        self.cf.CFDataGetLength.restype = index
        self.cf.CFDataGetBytePtr.argtypes = [pointer]
        self.cf.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_ubyte)
        self.cf.CFCopyDescription.argtypes = [pointer]
        self.cf.CFCopyDescription.restype = pointer
        self.cf.CFDataCreate.argtypes = [
            pointer,
            ctypes.POINTER(ctypes.c_ubyte),
            index,
        ]
        self.cf.CFDataCreate.restype = pointer

        self.imageio.CGImageSourceCreateWithData.argtypes = [pointer, pointer]
        self.imageio.CGImageSourceCreateWithData.restype = pointer
        self.imageio.CGImageSourceCopyProperties.argtypes = [pointer, pointer]
        self.imageio.CGImageSourceCopyProperties.restype = pointer
        self.imageio.CGImageSourceCopyPropertiesAtIndex.argtypes = [
            pointer,
            index,
            pointer,
        ]
        self.imageio.CGImageSourceCopyPropertiesAtIndex.restype = pointer
        self.imageio.CGImageSourceGetCount.argtypes = [pointer]
        self.imageio.CGImageSourceGetCount.restype = index

        self.type_ids = {
            self.cf.CFStringGetTypeID(): "string",
            self.cf.CFNumberGetTypeID(): "number",
            self.cf.CFBooleanGetTypeID(): "boolean",
            self.cf.CFArrayGetTypeID(): "array",
            self.cf.CFDictionaryGetTypeID(): "dictionary",
            self.cf.CFDataGetTypeID(): "data",
        }

    def _string(self, value: int) -> str:
        length = int(self.cf.CFStringGetLength(value))
        capacity = int(
            self.cf.CFStringGetMaximumSizeForEncoding(
                length, _CF_STRING_ENCODING_UTF8
            )
        ) + 1
        if capacity <= 0 or capacity > 16 * 1024 * 1024:
            raise ImageIOMetadataError("ImageIO returned an invalid metadata string length")
        buffer = ctypes.create_string_buffer(capacity)
        if not self.cf.CFStringGetCString(
            value, buffer, capacity, _CF_STRING_ENCODING_UTF8
        ):
            raise ImageIOMetadataError("ImageIO returned a metadata string that is not UTF-8")
        return buffer.value.decode("utf-8")

    def _convert(self, value: int, depth: int = 0) -> Any:
        if not value:
            return None
        if depth > 32:
            raise ImageIOMetadataError("ImageIO metadata nesting exceeds the safety limit")
        kind = self.type_ids.get(self.cf.CFGetTypeID(value))
        if kind == "string":
            return self._string(value)
        if kind == "boolean":
            return bool(self.cf.CFBooleanGetValue(value))
        if kind == "number":
            if self.cf.CFNumberIsFloatType(value):
                result = ctypes.c_double()
                if not self.cf.CFNumberGetValue(
                    value, _CF_NUMBER_FLOAT64, ctypes.byref(result)
                ):
                    raise ImageIOMetadataError("could not read a floating-point ImageIO value")
                return result.value
            result = ctypes.c_longlong()
            if not self.cf.CFNumberGetValue(
                value, _CF_NUMBER_SINT64, ctypes.byref(result)
            ):
                raise ImageIOMetadataError("could not read an integer ImageIO value")
            return result.value
        if kind == "array":
            count = int(self.cf.CFArrayGetCount(value))
            if count < 0 or count > 100_000:
                raise ImageIOMetadataError("ImageIO returned an invalid metadata array size")
            return [
                self._convert(self.cf.CFArrayGetValueAtIndex(value, item), depth + 1)
                for item in range(count)
            ]
        if kind == "dictionary":
            count = int(self.cf.CFDictionaryGetCount(value))
            if count < 0 or count > 100_000:
                raise ImageIOMetadataError("ImageIO returned an invalid metadata dictionary size")
            pointer = ctypes.c_void_p
            keys = (pointer * count)()
            values = (pointer * count)()
            self.cf.CFDictionaryGetKeysAndValues(value, keys, values)
            return {
                str(self._convert(keys[item], depth + 1)): self._convert(
                    values[item], depth + 1
                )
                for item in range(count)
            }
        if kind == "data":
            length = int(self.cf.CFDataGetLength(value))
            if length < 0:
                raise ImageIOMetadataError("ImageIO returned an invalid metadata byte count")
            pointer = self.cf.CFDataGetBytePtr(value)
            return bytes(pointer[:length]) if length else bytes()

        description = self.cf.CFCopyDescription(value)
        if not description:
            return "<unknown CoreFoundation value>"
        try:
            return self._string(description)
        finally:
            self.cf.CFRelease(description)

    def _dictionary_subset(self, dictionary: int, wanted: set[str]) -> dict[str, Any]:
        if not dictionary:
            return {}
        count = int(self.cf.CFDictionaryGetCount(dictionary))
        if count < 0 or count > 100_000:
            raise ImageIOMetadataError("ImageIO returned an invalid property dictionary size")
        pointer = ctypes.c_void_p
        keys = (pointer * count)()
        values = (pointer * count)()
        self.cf.CFDictionaryGetKeysAndValues(dictionary, keys, values)
        result: dict[str, Any] = {}
        for item in range(count):
            if self.type_ids.get(self.cf.CFGetTypeID(keys[item])) != "string":
                continue
            key = self._string(keys[item])
            if key in wanted:
                result[key] = self._convert(values[item])
        return result

    def read(self, source_bytes: bytes) -> dict[str, Any]:
        if not source_bytes:
            raise ImageIOMetadataError("cannot inspect an empty HEIF container")
        byte_buffer = (ctypes.c_ubyte * len(source_bytes)).from_buffer_copy(source_bytes)
        data = self.cf.CFDataCreate(None, byte_buffer, len(source_bytes))
        if not data:
            raise ImageIOMetadataError("could not create an immutable ImageIO data source")
        source = None
        global_properties = None
        try:
            source = self.imageio.CGImageSourceCreateWithData(data, None)
            if not source:
                raise ImageIOMetadataError("ImageIO did not recognize the HEIF container")
            global_properties = self.imageio.CGImageSourceCopyProperties(source, None)
            global_subset = self._dictionary_subset(
                global_properties,
                {"{Groups}", "PrimaryImage"},
            )
            images: list[dict[str, Any]] = []
            image_count = int(self.imageio.CGImageSourceGetCount(source))
            if image_count < 0 or image_count > 100_000:
                raise ImageIOMetadataError("ImageIO returned an invalid image count")
            for image_index in range(image_count):
                properties = self.imageio.CGImageSourceCopyPropertiesAtIndex(
                    source, image_index, None
                )
                if not properties:
                    images.append({})
                    continue
                try:
                    images.append(
                        self._dictionary_subset(
                            properties,
                            {"{HEIF}", "PixelWidth", "PixelHeight", "Orientation"},
                        )
                    )
                finally:
                    self.cf.CFRelease(properties)
            return {
                "groups": global_subset.get("{Groups}", []),
                "primary_image_index": global_subset.get("PrimaryImage"),
                "images": images,
            }
        finally:
            if global_properties:
                self.cf.CFRelease(global_properties)
            if source:
                self.cf.CFRelease(source)
            self.cf.CFRelease(data)


def read_apple_imageio_metadata(source_bytes: bytes) -> dict[str, Any] | None:
    """Return spatial-group metadata on macOS, or ``None`` off Apple platforms."""
    if sys.platform != "darwin":
        return None
    return _ImageIOBridge().read(source_bytes)
