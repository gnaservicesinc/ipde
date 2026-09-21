"""Discovery, metadata capture, and transactional extraction for HEIF assets."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pillow_heif

from .apple_imageio import ImageIOMetadataError, read_apple_imageio_metadata
from .formats import (
    FormatError,
    sha256_array,
    sha256_file,
    verify_exr,
    verify_npy,
    verify_png,
    write_exr,
    write_npy,
    write_png,
)
from .libheif_aux import HighBitAuxiliaryError, decode_high_bit_auxiliary
from .spatial import (
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
    linear_depth_displacement,
    run_raft_stereo,
    run_stereo_matching,
)


MINIMUM_PILLOW_HEIF = (1, 5, 0)
DEPTH_REPRESENTATIONS = {
    0: "uniform_inverse_z",
    1: "uniform_disparity",
    2: "uniform_z",
    3: "non_uniform_disparity",
}


class ExtractionError(RuntimeError):
    """A precise, user-facing extraction failure."""


@dataclass(frozen=True)
class ExtractOptions:
    selected_products: tuple[str, ...] | None = None
    output_dir: Path | None = None
    write_npy: bool = True
    write_manifest: bool = False
    write_metric_depth: bool = True
    write_physical_disparity: bool = True
    write_stereo_matching: bool = False
    write_raft_stereo: bool = False
    write_raft_diagnostics: bool = False
    histogram_color_matching: bool = False
    color_matching_hero: str = "left"
    write_displacement_maps: bool = False
    stereo_maximum_disparity: int | None = None
    stereo_noise_sigma_pixels: float = 1.0
    raft_root: Path | None = None
    raft_model: Path | None = None
    raft_model_member: str | None = None
    raft_device: str = "auto"
    raft_iterations: int = 32
    overwrite: bool = False


@dataclass
class Asset:
    kind: str
    parent_image_index: int
    ordinal: int
    array: np.ndarray
    mode: str
    source_bit_depth: int
    semantic_name: str
    aux_type: str | None = None
    aux_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_blocks: list[dict[str, Any]] = field(default_factory=list)
    premultiplied_alpha: bool | None = None


@dataclass
class Discovery:
    source: Path
    source_size: int
    source_sha256: str
    mimetype: str
    primary_index: int
    top_level_images: list[dict[str, Any]]
    assets: list[Asset]
    spatial_photo: dict[str, Any] | None = None
    spatial_metadata_warning: str | None = None


@dataclass
class PendingOutput:
    final_path: Path
    role: str
    asset_index: int | None
    write: Callable[[Path], None]
    verify: Callable[[Path], None]
    details: dict[str, Any] = field(default_factory=dict)
    temporary_path: Path | None = None


class MetricDepthError(ValueError):
    """Raised when a depth plane cannot be reconstructed as physical meters."""


def reconstruct_physical_disparity(raw_depth: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    """Convert an 8-bit uniform-disparity plane to calibrated float32 inverse meters.

    This is a calibrated representation conversion, not precision restoration:
    the result retains exactly the source plane's quantization levels.
    """
    raw = np.asarray(raw_depth)
    if raw.dtype != np.dtype("uint8"):
        raise MetricDepthError(f"metric reconstruction requires uint8 source samples, not {raw.dtype}")
    if raw.ndim != 2 or raw.size == 0:
        raise MetricDepthError(f"metric reconstruction requires a nonempty single-channel HxW plane, not {raw.shape}")

    representation = metadata.get("representation_type")
    try:
        representation_value = int(representation)
    except (TypeError, ValueError) as exc:
        raise MetricDepthError("depth metadata has no valid representation_type") from exc
    if representation_value != 1:
        name = DEPTH_REPRESENTATIONS.get(representation_value, "unknown")
        raise MetricDepthError(
            f"the requested reciprocal mapping is valid only for uniform_disparity (1), not {name} ({representation_value})"
        )

    try:
        d_min_value = float(metadata["d_min"])
        d_max_value = float(metadata["d_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MetricDepthError("uniform-disparity metadata requires numeric d_min and d_max") from exc
    if not math.isfinite(d_min_value) or not math.isfinite(d_max_value):
        raise MetricDepthError("d_min and d_max must be finite")
    if d_min_value < 0.0 or d_max_value <= 0.0 or d_max_value < d_min_value:
        raise MetricDepthError(
            f"invalid physical disparity bounds d_min={d_min_value!r}, d_max={d_max_value!r}"
        )

    d_min = np.float32(d_min_value)
    d_max = np.float32(d_max_value)
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        raise MetricDepthError("d_min or d_max cannot be represented as finite float32")

    float_samples = raw.astype(np.float32)
    normalized = float_samples / np.float32(255.0)
    disparity = normalized * (d_max - d_min) + d_min
    return np.ascontiguousarray(disparity, dtype=np.float32)


def reconstruct_metric_depth(raw_depth: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    """Convert an 8-bit uniform-disparity plane to float32 physical depth in meters.

    Every operation is vectorized and explicitly performed in float32. Physical
    depth is the reciprocal of disparity, so nearer geometry has smaller values.
    The conversion cannot recreate precision absent from the uint8 source.
    """
    disparity = reconstruct_physical_disparity(raw_depth, metadata)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_meters = np.float32(1.0) / disparity
    return np.ascontiguousarray(depth_meters, dtype=np.float32)


def _version_tuple(text: str) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in re.findall(r"\d+", text)[:3])
    return (parsed + (0, 0, 0))[:3]


def _check_runtime() -> str:
    current = getattr(pillow_heif, "__version__", "0")
    if _version_tuple(current) < MINIMUM_PILLOW_HEIF:
        raise ExtractionError(f"pillow-heif >= 1.5.0 is required; found {current}")
    return current


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return {"name": value.name, "value": value.value}
    if isinstance(value, bytes | bytearray | memoryview):
        raw = bytes(value)
        return {
            "encoding": "base64",
            "length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _metadata_blocks(image: Any) -> list[dict[str, Any]]:
    c_image = getattr(image, "_c_image", None)
    blocks = getattr(c_image, "metadata", []) if c_image is not None else []
    result: list[dict[str, Any]] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        raw = block.get("data", b"")
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raw = bytes()
        result.append(
            {
                "type": str(block.get("type", "")),
                "content_type": str(block.get("content_type", "")),
                "data": bytes(raw),
            }
        )
    return result


def _source_bit_depth(image: Any, fallback: int = 0) -> int:
    c_image = getattr(image, "_c_image", None)
    # Public bit_depth is encoded precision, whereas mode/dtype describe storage.
    for value in ((getattr(image, "info", {}) or {}).get("bit_depth"),
                  getattr(c_image, "bit_depth", None), fallback):
        try:
            if int(value) > 0:
                return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r";(10|12|16)", str(getattr(image, "mode", "")))
    return int(match.group(1)) if match else 8


def _camel_to_snake(text: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first)


def semantic_name(aux_type: str) -> str:
    token = re.split(r"[:/#]", aux_type.rstrip(":/#"))[-1] or "auxiliary"
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", _camel_to_snake(token)).strip("_").lower()
    aliases = {
        "hdrgainmap": "hdr_gain_map",
        "gainmap": "gain_map",
        "linearthumbnail": "linear_thumbnail",
        "styledeltamap": "style_delta_map",
        "portraitmatte": "portrait_matte",
        "portraiteffectsmatte": "portrait_effects_matte",
        "semanticsegmentationmatte": "semantic_segmentation_matte",
        "semanticskymatte": "semantic_sky_matte",
        "semanticskinmatte": "semantic_skin_matte",
        "semantichairmatte": "semantic_hair_matte",
        "semanticteethmatte": "semantic_teeth_matte",
        "semanticglassesmatte": "semantic_glasses_matte",
        "disparitymap": "disparity_map",
    }
    return aliases.get(normalized, normalized or "auxiliary")


def _array_copy(image: Any) -> np.ndarray:
    try:
        decoded = np.asarray(image)
        value = np.array(decoded, copy=True, order="C", subok=False)
    except Exception as exc:
        raise ExtractionError(f"could not decode {image!r}: {exc}") from exc
    if value.dtype.hasobject:
        raise ExtractionError("object arrays are not valid decoded image data")
    if value.ndim not in (2, 3) or value.size == 0:
        raise ExtractionError(f"decoded image has unsupported shape {value.shape}")
    return value


def _depth_metadata(image: Any) -> dict[str, Any]:
    info = getattr(image, "info", {}) or {}
    metadata = dict(info.get("metadata", {}) or {})
    representation = metadata.get("representation_type")
    if isinstance(representation, int):
        metadata["representation_name"] = DEPTH_REPRESENTATIONS.get(representation, "unknown")
    return _jsonable(metadata)


def _image_context(image: Any) -> dict[str, Any]:
    c_image = getattr(image, "_c_image", None)
    info = getattr(image, "info", {}) or {}
    return _jsonable(
        {
            "mode": getattr(image, "mode", ""),
            "size": list(getattr(image, "size", (0, 0))),
            "source_bit_depth": _source_bit_depth(image, int(info.get("bit_depth", 0) or 0)),
            "primary": bool(info.get("primary", False)),
            "has_alpha": bool(getattr(image, "has_alpha", False)),
            "premultiplied_alpha": bool(getattr(image, "premultiplied_alpha", False)),
            "thumbnail_sizes": info.get("thumbnails", []),
            "depth_image_count": len(info.get("depth_images", [])),
            "auxiliary_images": info.get("aux", {}),
            "tiling": info.get("tiling"),
            "pixel_aspect_ratio": info.get("pixel_aspect_ratio"),
            "chroma": getattr(c_image, "chroma", info.get("chroma")),
            "colorspace": getattr(c_image, "colorspace", None),
            "color_profile": getattr(c_image, "color_profile", None),
            "camera_intrinsic_matrix": getattr(c_image, "camera_intrinsic_matrix", None),
            "camera_extrinsic_matrix_rotation": getattr(c_image, "camera_extrinsic_matrix_rot", None),
            "metadata_blocks": _metadata_blocks(image),
        }
    )


def discover_file(source: Path) -> Discovery:
    _check_runtime()
    path = source.expanduser().resolve()
    if not path.is_file():
        raise ExtractionError(f"input is not a regular file: {path}")
    try:
        # pillow-heif reads the complete file internally as well. Keeping this immutable
        # snapshot ensures the recorded hash describes the bytes that were decoded even
        # if another process replaces the source path during a long extraction.
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"could not read input {path}: {exc}") from exc
    if not source_bytes:
        raise ExtractionError(f"input is empty: {path}")
    try:
        heif = pillow_heif.open_heif(
            source_bytes,
            convert_hdr_to_8bit=False,
            hdr_to_16bit=False,
            bgr_mode=False,
            remove_stride=True,
        )
    except Exception as exc:
        raise ExtractionError(f"could not open HEIF container {path.name}: {exc}") from exc

    spatial_metadata_warning: str | None = None
    try:
        imageio_metadata = read_apple_imageio_metadata(source_bytes)
    except ImageIOMetadataError as exc:
        imageio_metadata = None
        spatial_metadata_warning = f"Apple spatial metadata was unavailable: {exc}"
    try:
        spatial_photo = analyze_spatial_photo(imageio_metadata)
    except SpatialPhotoError as exc:
        raise ExtractionError(f"invalid Apple spatial-photo metadata: {exc}") from exc

    assets: list[Asset] = []
    image_contexts: list[dict[str, Any]] = []
    for image_index, image in enumerate(heif):
        image_contexts.append(_image_context(image))
        for depth_index, depth in enumerate(image.info.get("depth_images", []) or []):
            blocks = _metadata_blocks(depth)
            assets.append(
                Asset(
                    kind="depth",
                    parent_image_index=image_index,
                    ordinal=depth_index,
                    array=_array_copy(depth),
                    mode=depth.mode,
                    source_bit_depth=_source_bit_depth(depth),
                    semantic_name="depth",
                    metadata=_depth_metadata(depth),
                    metadata_blocks=blocks,
                )
            )

        aux = image.info.get("aux", {}) or {}
        for aux_type in sorted(aux):
            for aux_index, aux_id in enumerate(aux[aux_type]):
                try:
                    auxiliary = image.get_aux_image(aux_id)
                    blocks = _metadata_blocks(auxiliary)
                    auxiliary_array = _array_copy(auxiliary)
                    auxiliary_mode = auxiliary.mode
                    auxiliary_bit_depth = _source_bit_depth(auxiliary)
                    auxiliary_metadata: dict[str, Any] = {"metadata_blocks": _jsonable(blocks)}
                except NotImplementedError as exc:
                    if "Only 8-bit AUX images" not in str(exc):
                        raise ExtractionError(
                            f"could not retrieve auxiliary image {aux_id} ({aux_type}): {exc}"
                        ) from exc
                    try:
                        fallback = decode_high_bit_auxiliary(source_bytes, image_index, int(aux_id))
                    except HighBitAuxiliaryError as fallback_error:
                        raise ExtractionError(
                            f"could not decode high-bit auxiliary image {aux_id} ({aux_type}): "
                            f"{fallback_error}"
                        ) from fallback_error
                    blocks = fallback.metadata_blocks
                    auxiliary_array = fallback.array
                    auxiliary_mode = fallback.mode
                    auxiliary_bit_depth = fallback.source_bit_depth
                    auxiliary_metadata = {
                        "metadata_blocks": _jsonable(blocks),
                        "high_bit_fallback": _jsonable(fallback.details),
                    }
                except Exception as exc:
                    raise ExtractionError(f"could not retrieve auxiliary image {aux_id} ({aux_type}): {exc}") from exc
                assets.append(
                    Asset(
                        kind="auxiliary",
                        parent_image_index=image_index,
                        ordinal=aux_index,
                        array=auxiliary_array,
                        mode=auxiliary_mode,
                        source_bit_depth=auxiliary_bit_depth,
                        semantic_name=semantic_name(aux_type),
                        aux_type=str(aux_type),
                        aux_id=int(aux_id),
                        metadata=auxiliary_metadata,
                        metadata_blocks=blocks,
                    )
                )

        if bool(getattr(image, "has_alpha", False)):
            decoded = _array_copy(image)
            if decoded.ndim != 3 or decoded.shape[2] not in (2, 4):
                raise ExtractionError(f"image {image_index} reports alpha but decoded shape is {decoded.shape}")
            assets.append(
                Asset(
                    kind="alpha",
                    parent_image_index=image_index,
                    ordinal=0,
                    array=np.ascontiguousarray(decoded[:, :, -1]),
                    mode=f"alpha-from-{image.mode}",
                    source_bit_depth=_source_bit_depth(image, int(image.info.get("bit_depth", 0) or 0)),
                    semantic_name="alpha",
                    metadata={},
                    premultiplied_alpha=bool(getattr(image, "premultiplied_alpha", False)),
                )
            )

    if spatial_photo is not None:
        left_index = int(spatial_photo["left_image_index"])
        right_index = int(spatial_photo["right_image_index"])
        if max(left_index, right_index) >= len(heif):
            raise ExtractionError(
                "Apple spatial-photo group does not match pillow-heif's top-level image inventory"
            )
        roles = (
            ("left", left_index, spatial_photo["left_camera"]),
            ("right", right_index, spatial_photo["right_camera"]),
        )
        for role, image_index, camera in roles:
            image = heif[image_index]
            decoded = _array_copy(image)
            if decoded.shape[:2] != (camera["height"], camera["width"]):
                raise ExtractionError(
                    f"{role} stereo image {image_index} decoded at {decoded.shape[1]}x{decoded.shape[0]}, "
                    f"but its camera calibration describes {camera['width']}x{camera['height']}; "
                    "refusing a display-image substitution or implicit resize"
                )
            if 0 <= image_index < len(image_contexts):
                image_contexts[image_index]["spatial_role"] = role
            assets.append(
                Asset(
                    kind="spatial_view",
                    parent_image_index=image_index,
                    ordinal=0,
                    array=decoded,
                    mode=image.mode,
                    source_bit_depth=_source_bit_depth(
                        image, int(image.info.get("bit_depth", 0) or 0)
                    ),
                    semantic_name=f"spatial_{role}",
                    metadata={
                        "spatial_role": role,
                        "group_index": spatial_photo["group_index"],
                        "camera": camera,
                        "decoded_orientation_policy": spatial_photo[
                            "decoded_orientation_policy"
                        ],
                    },
                )
            )

    # Append representations so existing raw product indices remain stable.
    for native in (imageio_metadata or {}).get("native_depth_images", []):
        parent = native["parent_image_index"]
        encoded = [a for a in assets if a.kind == "depth" and a.parent_image_index == parent]
        encoded_bits = encoded[0].source_bit_depth if len(encoded) == 1 else 0
        for asset in encoded:
            asset.metadata["apple_depth_accuracy"] = native["accuracy"]
            asset.metadata["apple_native_dtype"] = native["array"].dtype.name
        assets.append(Asset(
            kind="native_depth", parent_image_index=parent, ordinal=0,
            array=native["array"], mode=native["array"].dtype.name,
            source_bit_depth=encoded_bits, semantic_name=f"apple_{native['semantic']}",
            metadata={"decoder": "macOS ImageIO native depth buffer", "representation": native["semantic"],
                      "accuracy": native["accuracy"], "description": native["description"],
                      "encoded_source_bit_depth": encoded_bits or None,
                      "native_storage_bit_depth": native["array"].dtype.itemsize * 8,
                      "precision_note": "Apple's decoded floating-point representation; not additional captured precision"},
            metadata_blocks=[{"type": "mime", "content_type": "application/rdf+xml", "data": native["xmp"]}]
            if native["xmp"] else [],
        ))
    if spatial_photo is not None:
        mono = spatial_photo.get("monoscopic_image_index")
        if isinstance(mono, int) and mono not in (left_index, right_index) and 0 <= mono < len(heif):
            image = heif[mono]
            assets.append(Asset(
                kind="display_view", parent_image_index=mono, ordinal=0,
                array=_array_copy(image), mode=image.mode, source_bit_depth=_source_bit_depth(image),
                semantic_name="display", metadata={"spatial_role": "monoscopic_display",
                    "coordinate_system": "separate display image; not the stereo or depth reference grid",
                    "stereo_input": False},
            ))

    return Discovery(
        source=path,
        source_size=len(source_bytes),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        mimetype=str(heif.mimetype),
        primary_index=int(heif.primary_index),
        top_level_images=image_contexts,
        assets=assets,
        spatial_photo=spatial_photo,
        spatial_metadata_warning=spatial_metadata_warning,
    )


def _stat_value(value: np.generic | float | int) -> int | float | str:
    scalar = value.item() if isinstance(value, np.generic) else value
    return _jsonable(scalar)


def _asset_record(asset: Asset) -> dict[str, Any]:
    array = asset.array
    if np.issubdtype(array.dtype, np.floating):
        finite = np.isfinite(array)
        if finite.any():
            minimum: int | float | str = _stat_value(array[finite].min())
            maximum: int | float | str = _stat_value(array[finite].max())
        else:
            minimum = maximum = "NaN"
        nonfinite_count = int(array.size - np.count_nonzero(finite))
    else:
        minimum = _stat_value(array.min())
        maximum = _stat_value(array.max())
        nonfinite_count = 0
    record = {
        "kind": asset.kind,
        "semantic_name": asset.semantic_name,
        "parent_image_index": asset.parent_image_index,
        "ordinal": asset.ordinal,
        "aux_type": asset.aux_type,
        "aux_id": asset.aux_id,
        "mode": asset.mode,
        "source_bit_depth": asset.source_bit_depth,
        "decoded_storage_bit_depth": array.dtype.itemsize * 8,
        "dtype": array.dtype.str,
        "dtype_name": array.dtype.name,
        "shape": list(array.shape),
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "channels": int(array.shape[2]) if array.ndim == 3 else 1,
        "minimum": minimum,
        "maximum": maximum,
        "nonfinite_count": nonfinite_count,
        "decoded_data_sha256": sha256_array(array),
        "metadata": _jsonable(asset.metadata),
        "premultiplied_alpha": asset.premultiplied_alpha,
        "outputs": [],
    }
    return record


def _base_names(discovery: Discovery) -> list[str]:
    counts: dict[str, int] = {}
    for asset in discovery.assets:
        counts[asset.semantic_name] = counts.get(asset.semantic_name, 0) + 1
    occurrences: dict[str, int] = {}
    result = []
    stem = discovery.source.stem
    for asset in discovery.assets:
        name = asset.semantic_name
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        if asset.kind == "spatial_view" and counts[name] == 1:
            suffix = name
        elif counts[name] == 1 and asset.parent_image_index == discovery.primary_index:
            suffix = name
        else:
            suffix = f"image{asset.parent_image_index}_{name}{occurrence}"
        result.append(f"{stem}_{suffix}")
    return result


def _metadata_extension(block: dict[str, Any]) -> str:
    block_type = str(block.get("type", "")).lower()
    content_type = str(block.get("content_type", "")).lower()
    if block_type == "exif":
        return ".exif"
    if "rdf+xml" in content_type or "xml" in content_type:
        return ".xmp"
    return ".bin"


def _exchange_output(path_base: Path, asset: Asset, asset_index: int) -> PendingOutput:
    array = asset.array
    if array.dtype in (np.dtype("uint8"), np.dtype("uint16")) and (
        array.ndim == 2 or (array.ndim == 3 and array.shape[2] in (1, 2, 3, 4))
    ):
        path = Path(f"{path_base}.png")
        return PendingOutput(
            final_path=path,
            role="lossless_exchange",
            asset_index=asset_index,
            write=lambda temp, a=array: write_png(temp, a),
            verify=lambda temp, a=array: verify_png(temp, a),
        )
    if array.dtype in (np.dtype("float16"), np.dtype("float32")) and (
        array.ndim == 2 or (array.ndim == 3 and array.shape[2] in (1, 2, 3, 4))
    ):
        path = Path(f"{path_base}.exr")
        attributes = {}
        if asset.kind == "native_depth":
            attributes = {"ipdeDecoder": "macOS ImageIO native depth buffer",
                          "ipdeDepthAccuracy": asset.metadata.get("accuracy") or "unspecified",
                          "ipdeEncodedSourceBits": str(asset.source_bit_depth or "unknown"),
                          "ipdeSemantic": str(asset.metadata.get("representation", "depth"))}
        return PendingOutput(
            final_path=path,
            role="lossless_exchange",
            asset_index=asset_index,
            write=lambda temp, a=array, attrs=attributes: write_exr(temp, a, attributes=attrs),
            verify=lambda temp, a=array: verify_exr(temp, a),
        )
    raise FormatError(
        f"{asset.semantic_name} has dtype/shape {array.dtype}/{array.shape}, which PNG and OpenEXR cannot preserve"
    )


def _finite_array_summary(array: np.ndarray, unit_suffix: str) -> dict[str, Any]:
    finite = np.isfinite(array)
    if finite.any():
        minimum: int | float | str = _stat_value(array[finite].min())
        maximum: int | float | str = _stat_value(array[finite].max())
    else:
        minimum = maximum = "NaN"
    return {
        f"minimum_finite_{unit_suffix}": minimum,
        f"maximum_finite_{unit_suffix}": maximum,
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
        "distinct_values_present": int(np.unique(array).size),
    }


def _source_quantization_details(asset: Asset) -> dict[str, Any]:
    return {
        "source_dtype": asset.array.dtype.name,
        "source_bit_depth": asset.source_bit_depth,
        "possible_source_code_count": 256,
        "source_codes_present": int(np.unique(asset.array).size),
        "restores_additional_precision": False,
        "note": (
            "Float32 stores calibrated values, but each output sample still derives from one uint8 code; "
            "no missing scene measurements are reconstructed."
        ),
    }


def _physical_disparity_output(
    path_base: Path,
    asset: Asset,
    asset_index: int,
    disparity: np.ndarray,
) -> PendingOutput:
    d_min = float(asset.metadata["d_min"])
    d_max = float(asset.metadata["d_max"])
    path = Path(f"{path_base}_disparity.exr")
    derivation = {
        "name": "uniform_disparity_to_physical_disparity",
        "output_dtype": disparity.dtype.name,
        "normalization_divisor": 255.0,
        "d_min": d_min,
        "d_max": d_max,
        "units": "1/m",
        "value_direction": "larger values indicate nearer geometry",
        "formula": (
            "normalized = float32(raw) / float32(255.0); "
            "disparity = normalized * (float32(d_max) - float32(d_min)) + float32(d_min)"
        ),
        "nominal_code_step_inverse_meters": _stat_value(
            (np.float32(d_max) - np.float32(d_min)) / np.float32(255.0)
        ),
        "shape": list(disparity.shape),
        "derived_data_sha256": sha256_array(disparity),
        "source_quantization": _source_quantization_details(asset),
        **_finite_array_summary(disparity, "inverse_meters"),
    }
    return PendingOutput(
        final_path=path,
        role="derived_physical_disparity",
        asset_index=asset_index,
        write=lambda temp, a=disparity: write_exr(
            temp,
            a,
            storage_description="Derived float32 calibrated disparity; each Y sample is inverse meters",
            attributes={
                "ipdeUnits": "inverse meters",
                "ipdeSemantic": "disparity/proximity; near values are larger",
                "ipdeTransform": "disparity=raw/255*(d_max-d_min)+d_min",
                "ipdePrecision": "calibrated from uint8; no additional source precision",
            },
        ),
        verify=lambda temp, a=disparity: verify_exr(temp, a),
        details={"derivation": derivation},
    )


def _metric_depth_output(
    path_base: Path,
    asset: Asset,
    asset_index: int,
    depth_meters: np.ndarray,
) -> PendingOutput:
    d_min = float(asset.metadata["d_min"])
    d_max = float(asset.metadata["d_max"])
    path = Path(f"{path_base}_meters.exr")
    codebook = reconstruct_metric_depth(np.arange(256, dtype=np.uint8).reshape(1, 256), asset.metadata)[0]
    code_steps = np.abs(np.diff(codebook))
    derivation = {
        "name": "uniform_disparity_to_metric_depth",
        "output_dtype": depth_meters.dtype.name,
        "normalization_divisor": 255.0,
        "d_min": d_min,
        "d_max": d_max,
        "disparity_units": "1/m",
        "depth_units": "m",
        "value_direction": "larger values indicate farther geometry; nearer geometry is numerically smaller",
        "formula": (
            "normalized = float32(raw) / float32(255.0); "
            "disparity = normalized * (float32(d_max) - float32(d_min)) + float32(d_min); "
            "depth_meters = float32(1.0) / disparity"
        ),
        "shape": list(depth_meters.shape),
        "minimum_one_code_step_meters": _stat_value(code_steps.min()),
        "maximum_one_code_step_meters": _stat_value(code_steps.max()),
        "derived_data_sha256": sha256_array(depth_meters),
        "source_quantization": _source_quantization_details(asset),
        **_finite_array_summary(depth_meters, "meters"),
    }
    return PendingOutput(
        final_path=path,
        role="derived_metric_depth",
        asset_index=asset_index,
        write=lambda temp, a=depth_meters: write_exr(
            temp,
            a,
            storage_description="Derived float32 physical depth; each Y sample is distance in meters",
            attributes={
                "ipdeUnits": "meters",
                "ipdeSemantic": "physical distance; near values are smaller",
                "ipdeTransform": "depth=1/(raw/255*(d_max-d_min)+d_min)",
                "ipdePrecision": "calibrated from uint8; no additional source precision",
            },
        ),
        verify=lambda temp, a=depth_meters: verify_exr(temp, a),
        details={"derivation": derivation},
    )


def _inference_output_details(
    array: np.ndarray,
    inference: Mapping[str, Any],
    *,
    name: str,
    units: str,
    value_direction: str,
) -> dict[str, Any]:
    finite = np.isfinite(array)
    return {
        "derivation": {
            **dict(inference),
            "name": name,
            "units": units,
            "value_direction": value_direction,
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "derived_data_sha256": sha256_array(array),
            "minimum_finite": _stat_value(array[finite].min()) if finite.any() else "NaN",
            "maximum_finite": _stat_value(array[finite].max()) if finite.any() else "NaN",
            "nonfinite_count": int(array.size - np.count_nonzero(finite)),
        }
    }


def _raft_exr_output(
    path: Path,
    role: str,
    asset_index: int,
    array: np.ndarray,
    details: dict[str, Any],
    *,
    units: str,
    semantic: str,
    transform: str,
) -> PendingOutput:
    color_matching = details.get("derivation", {}).get("input_color_matching", {})
    color_matching_description = (
        f"per-channel uint8 CDF; Hero {color_matching.get('hero_side', 'unspecified')}"
        if color_matching.get("applied")
        else "disabled"
    )
    return PendingOutput(
        final_path=path,
        role=role,
        asset_index=asset_index,
        write=lambda temp, a=array: write_exr(
            temp,
            a,
            storage_description=(
                "Full-resolution RAFT-Stereo float32 inference; no display normalization, "
                "gamma correction, or tone mapping applied"
            ),
            attributes={
                "ipdeUnits": units,
                "ipdeSemantic": semantic,
                "ipdeTransform": transform,
                "ipdePrecision": "AI-inferred float32 estimate; not measured source depth",
                "ipdeColorMatching": color_matching_description,
                "ipdeDerivation": json.dumps(details["derivation"], sort_keys=True, allow_nan=False),
            },
        ),
        verify=lambda temp, a=array: verify_exr(temp, a),
        details=details,
    )


def _raft_pending_outputs(
    output_dir: Path,
    discovery: Discovery,
    asset_index: int,
    result: Any,
    *,
    write_npy_companions: bool,
    include_diagnostics: bool,
    color_matched: bool,
) -> list[PendingOutput]:
    color_suffix = "_color_matched" if color_matched else ""
    stem = output_dir / f"{discovery.source.stem}_spatial_raft_stereo{color_suffix}"
    specifications = [
        (
            "height",
            result.height_disparity_pixels,
            "derived_raft_stereo_height_map",
            "pixels",
            "nonnegative stereo disparity height map; near values are generally larger",
            "(cx_right-cx_left)-signed_flow; negative/nonfinite is NaN",
            "larger values generally indicate nearer geometry",
        ),
    ]
    if include_diagnostics:
        specifications.extend(
            (
                (
                    "signed_flow",
                    result.signed_flow_pixels,
                    "derived_raft_stereo_signed_flow",
                    "pixels",
                    "signed horizontal x_right - x_left correspondence displacement",
                    "model output without numeric transformation",
                    "signed correspondence displacement; sign depends on stored stereo coordinates",
                ),
                (
                    "depth_meters",
                    result.depth_meters,
                    "derived_raft_stereo_metric_depth",
                    "meters",
                    "metric camera distance; near values are smaller",
                    "float32(focal_length_pixels*baseline_meters)/height_disparity_pixels",
                    "smaller values indicate nearer geometry",
                ),
            )
        )
    pending: list[PendingOutput] = []
    for suffix, array, role, units, semantic, transform, direction in specifications:
        path_base = Path(f"{stem}_{suffix}")
        details = _inference_output_details(
            array,
            result.details,
            name=role,
            units=units,
            value_direction=direction,
        )
        pending.append(
            _raft_exr_output(
                Path(f"{path_base}.exr"),
                role,
                asset_index,
                array,
                details,
                units=units,
                semantic=semantic,
                transform=transform,
            )
        )
        if write_npy_companions:
            pending.append(
                PendingOutput(
                    final_path=Path(f"{path_base}.npy"),
                    role=f"{role}_exact_array",
                    asset_index=asset_index,
                    write=lambda temp, a=array: write_npy(temp, a),
                    verify=lambda temp, a=array: verify_npy(temp, a),
                    details=details,
                )
            )
    return pending


def _stereo_matching_pending_outputs(
    output_dir: Path,
    discovery: Discovery,
    asset_index: int,
    result: Any,
    *,
    write_npy_companions: bool,
    color_matched: bool,
) -> list[PendingOutput]:
    color_suffix = "_color_matched" if color_matched else ""
    path_base = (
        output_dir
        / f"{discovery.source.stem}_spatial_stereo_matching{color_suffix}_height"
    )
    array = result.height_disparity_pixels
    role = "derived_stereo_matching_height_map"
    details = _inference_output_details(
        array,
        result.details,
        name=role,
        units="pixels",
        value_direction="larger values generally indicate nearer geometry; NaN means unmatched",
    )
    color_matching = details.get("derivation", {}).get("input_color_matching", {})
    color_matching_description = (
        f"per-channel uint8 CDF; Hero {color_matching.get('hero_side', 'unspecified')}"
        if color_matching.get("applied")
        else "disabled"
    )
    pending = [
        PendingOutput(
            final_path=Path(f"{path_base}.exr"),
            role=role,
            asset_index=asset_index,
            write=lambda temp, a=array: write_exr(
                temp,
                a,
                storage_description=(
                    "Full-resolution OpenCV StereoSGBM float32 height estimate; no display "
                    "normalization, gamma correction, tone mapping, resizing, or hole filling applied"
                ),
                attributes={
                    "ipdeUnits": "pixels",
                    "ipdeSemantic": (
                        "classical stereo-matching disparity height; near is generally high; "
                        "NaN is unmatched"
                    ),
                    "ipdeTransform": (
                        "(StereoSGBM fixed disparity / 16) + (cx_right-cx_left); negative/unmatched is NaN"
                    ),
                    "ipdePrecision": (
                        "classical 1/16-pixel inferred estimate; not measured source depth"
                    ),
                    "ipdeColorMatching": color_matching_description,
                    "ipdeDerivation": json.dumps(details["derivation"], sort_keys=True, allow_nan=False),
                },
            ),
            verify=lambda temp, a=array: verify_exr(temp, a),
            details=details,
        )
    ]
    if write_npy_companions:
        pending.append(
            PendingOutput(
                final_path=Path(f"{path_base}.npy"),
                role=f"{role}_exact_array",
                asset_index=asset_index,
                write=lambda temp, a=array: write_npy(temp, a),
                verify=lambda temp, a=array: verify_npy(temp, a),
                details=details,
            )
        )
    return pending


def _displacement_pending_outputs(
    output_dir: Path,
    discovery: Discovery,
    asset_index: int,
    array: np.ndarray,
    *,
    engine_name: str,
    inference_details: Mapping[str, Any],
    mapping_details: Mapping[str, Any],
    color_matched: bool,
    write_npy_companions: bool,
) -> list[PendingOutput]:
    if engine_name == "stereo_matching":
        filename_engine = "stereo_matching"
        role = "derived_stereo_matching_displacement_0_to_1"
        semantic_engine = "classical StereoSGBM"
    elif engine_name == "raft_stereo":
        filename_engine = "raft_stereo"
        role = "derived_raft_stereo_displacement_0_to_1"
        semantic_engine = "RAFT-Stereo"
    else:
        raise ExtractionError(f"unsupported displacement-map engine {engine_name!r}")
    color_suffix = "_color_matched" if color_matched else ""
    path_base = (
        output_dir
        / f"{discovery.source.stem}_spatial_{filename_engine}{color_suffix}_displacement_0_to_1"
    )
    combined_details = {
        **dict(inference_details),
        "normalization": True,
        "displacement_mapping": dict(mapping_details),
    }
    details = _inference_output_details(
        array,
        combined_details,
        name=role,
        units="normalized 0..1",
        value_direction="larger values indicate nearer geometry; NaN remains unmatched",
    )
    color_matching = combined_details.get("input_color_matching", {})
    color_matching_description = (
        f"per-channel uint8 CDF; Hero {color_matching.get('hero_side', 'unspecified')}"
        if color_matching.get("applied")
        else "disabled"
    )
    bounds = mapping_details
    pending = [
        PendingOutput(
            final_path=Path(f"{path_base}.exr"),
            role=role,
            asset_index=asset_index,
            write=lambda temp, a=array: write_exr(
                temp,
                a,
                storage_description=(
                    "Explicit full-resolution 0..1 displacement derivative; per-map full-range linear "
                    "mapping from camera-axis depth in meters; raw disparity available separately"
                ),
                attributes={
                    "ipdeUnits": "normalized 0..1",
                    "ipdeSemantic": (
                        f"{semantic_engine} displacement-ready height; near is high; NaN is unmatched"
                    ),
                    "ipdeTransform": str(bounds["formula"]),
                    "ipdeDepthBoundsMeters": (
                        f"{bounds['near_depth_meters_float32']},"
                        f"{bounds['far_depth_meters_float32']}"
                    ),
                    "ipdeDisplacementScaleMeters": str(bounds["displacement_scale_meters_float32"]),
                    "ipdePrecision": (
                        "explicit normalized derivative; scientific pixel-disparity output selectable separately"
                    ),
                    "ipdeColorMatching": color_matching_description,
                    "ipdeDerivation": json.dumps(details["derivation"], sort_keys=True, allow_nan=False),
                },
            ),
            verify=lambda temp, a=array: verify_exr(temp, a),
            details=details,
        )
    ]
    if write_npy_companions:
        pending.append(
            PendingOutput(
                final_path=Path(f"{path_base}.npy"),
                role=f"{role}_exact_array",
                asset_index=asset_index,
                write=lambda temp, a=array: write_npy(temp, a),
                verify=lambda temp, a=array: verify_npy(temp, a),
                details=details,
            )
        )
    return pending


def _stereo_review_outputs(
    output_dir: Path, discovery: Discovery, asset_index: int,
    engine: str, disparity: np.ndarray, support: np.ndarray,
    inference: Mapping[str, Any], selected: set[str] | None, *,
    color_matched: bool, write_npy_companions: bool,
) -> list[PendingOutput]:
    """Explicit review products; never silently normalize a scientific export."""
    if selected is None:
        return []
    filename_engine = "raft_stereo" if engine == "raft" else "stereo_matching"
    color_suffix = "_color_matched" if color_matched else ""
    stem = output_dir / f"{discovery.source.stem}_spatial_{filename_engine}{color_suffix}"
    pending: list[PendingOutput] = []
    for product in (f"{engine}-support", f"{engine}-preview", f"{engine}-supported-depth"):
        if product not in selected:
            continue
        inference_details = dict(inference)
        if product.endswith("-support"):
            array = support.astype(np.float32)
            suffix, units = "support", "binary support (0 or 1)"
            semantic = "1 = supported correspondence; 0 = unknown, not zero depth or a confidence probability"
            transform = "binary correspondence support; no modification of depth samples"
        elif product.endswith("-supported-depth"):
            _, depth = derive_raft_height_and_depth(-disparity, {
                **discovery.spatial_photo, "principal_point_delta_x_pixels": 0.0,
            })
            array = np.where(support, depth, np.float32(np.nan))
            suffix, units = "supported_depth_meters", "meters"
            semantic = "camera-axis depth restricted to supported correspondences; NaN is unknown"
            transform = "Z = float32(focal_px * baseline_m) / disparity; unsupported = NaN"
            inference_details["depth_and_disparity_filtered_by_support"] = True
        else:
            try:
                mapped, mapping = linear_depth_displacement(disparity, discovery.spatial_photo)
            except DisplacementMappingError:
                # An entirely unmatched classical result has a useful empty preview.
                mapped = np.full(disparity.shape, np.nan, np.float32)
                mapping = {"empty_map": True, "formula": "no finite positive depth"}
            finite = np.isfinite(mapped)
            gray = np.rint(np.where(finite, mapped, 0).astype(np.float64) * 65535).astype(np.uint16)
            array = np.stack((gray, np.where(finite, 65535, 0).astype(np.uint16)), axis=-1)
            suffix, units = "depth_preview", "16-bit display codes, not scientific data"
            semantic = "linear camera-axis depth preview; near white, far black; unknown pixels transparent"
            transform = "round(65535 * (far_m - Z) / (far_m - near_m)); alpha = finite depth"
            inference_details.update({
                "preview_only": True, "normalization": True, "displacement_mapping": mapping,
                "quantization": "nearest uint16 code; use float32 EXR for displacement",
                "gamma_correction": False, "tone_mapping": False,
                "alpha_semantics": "finite estimate, not correspondence support; see separate support map",
            })
        role = _SPATIAL_PRODUCTS[product][1]
        details = _inference_output_details(array, inference_details, name=role, units=units, value_direction=semantic)
        attrs = {"ipdeUnits": units, "ipdeSemantic": semantic, "ipdeTransform": transform,
                 "ipdeDerivation": json.dumps(details["derivation"], sort_keys=True, allow_nan=False)}
        if product.endswith("-preview"):
            pending.append(PendingOutput(
                final_path=Path(f"{stem}_{suffix}.png"), role=role, asset_index=asset_index,
                write=lambda temp, a=array, at=attrs: write_png(temp, a, attributes=at),
                verify=lambda temp, a=array: verify_png(temp, a), details=details,
            ))
        else:
            pending.append(PendingOutput(
                final_path=Path(f"{stem}_{suffix}.exr"), role=role, asset_index=asset_index,
                write=lambda temp, a=array, at=attrs: write_exr(temp, a, attributes=at),
                verify=lambda temp, a=array: verify_exr(temp, a), details=details,
            ))
            if write_npy_companions:
                pending.append(PendingOutput(
                    final_path=Path(f"{stem}_{suffix}.npy"), role=f"{role}_exact_array", asset_index=asset_index,
                    write=lambda temp, a=array: write_npy(temp, a),
                    verify=lambda temp, a=array: verify_npy(temp, a), details=details,
                ))
    return pending


# Product IDs are stable within a decoded inventory and also serve as CLI selectors.
_SPATIAL_PRODUCTS = {
    "raft-displacement": ("RAFT dense estimate — linear depth 0–1 displacement", "derived_raft_stereo_displacement_0_to_1"),
    "raft-preview": ("RAFT depth preview — view only, near white", "derived_raft_stereo_depth_preview"),
    "raft-depth": ("RAFT dense estimate — camera-axis depth in meters", "derived_raft_stereo_metric_depth"),
    "raft-support": ("RAFT support mask — white supported, black unknown", "derived_raft_stereo_support"),
    "raft-supported-depth": ("RAFT supported depth — meters, unknown is NaN", "derived_raft_stereo_supported_depth"),
    "raft-height": ("RAFT disparity diagnostic — pixels, inverse depth", "derived_raft_stereo_height_map"),
    "raft-flow": ("RAFT signed flow diagnostic — negative pixels", "derived_raft_stereo_signed_flow"),
    "stereo-displacement": ("Classical sparse matches — linear depth 0–1 displacement", "derived_stereo_matching_displacement_0_to_1"),
    "stereo-preview": ("Classical depth preview — unknown transparent", "derived_stereo_matching_depth_preview"),
    "stereo-supported-depth": ("Classical supported depth — meters, unknown is NaN", "derived_stereo_matching_supported_depth"),
    "stereo-support": ("Classical support mask — white supported, black unknown", "derived_stereo_matching_support"),
    "stereo-height": ("Classical disparity diagnostic — pixels, inverse depth", "derived_stereo_matching_height_map"),
}


def _available_products(discovery: Discovery) -> list[dict[str, Any]]:
    products = []
    for index, asset in enumerate(discovery.assets):
        common = {"asset_index": index, "width": asset.array.shape[1], "height": asset.array.shape[0],
                  "channels": asset.array.shape[2] if asset.array.ndim == 3 else 1}
        storage = f"{asset.array.dtype.itemsize * 8}-bit {'float EXR' if asset.array.dtype.kind == 'f' else 'integer PNG'}"
        source_precision = f"{asset.source_bit_depth}-bit encoded" if asset.source_bit_depth else "Encoded precision unavailable"
        description = "Original decoded code values; no gamma, normalization, or resampling."
        name = f"{asset.semantic_name} — original samples"
        if asset.kind == "native_depth":
            name = f"Apple native {asset.metadata['representation']} — decoded values"
            description = (f"Apple ImageIO {asset.array.dtype.name}; accuracy: {asset.metadata['accuracy'] or 'unspecified'}. "
                           "A decoded floating-point representation, not additional encoded precision.")
        elif asset.kind == "display_view":
            name = "Display image — separate camera grid"
            description = "The full-size monoscopic image has its own framing. Stereo results align to spatial_left, not this display image."
        elif asset.kind == "spatial_view":
            description = f"HEIF image {asset.parent_image_index}; full native stereo dimensions. Results use the left view's pixel grid."
        products.append({**common, "id": f"raw:{index}",
                         "precision": storage, "name": name, "source_precision": source_precision,
                         "origin": "Apple decoded" if asset.kind == "native_depth" else "Embedded image",
                         "description": description})
        if asset.kind == "depth":
            try:
                reconstruct_physical_disparity(asset.array, asset.metadata)
            except MetricDepthError:
                continue
            for key, label in (("disparity", "Calibrated disparity — inverse meters"),
                               ("meters", "Depth distance — meters")):
                products.append({**common, "id": f"{key}:{index}", "name": f"{asset.semantic_name}: {label}",
                                 "precision": "32-bit float EXR", "source_precision": source_precision,
                                 "origin": "Calculated", "description": "Calculated from encoded depth codes. Float32 is export storage, not import bit depth. "
                                 f"Apple depth accuracy: {asset.metadata.get('apple_depth_accuracy') or 'unspecified'}."})
    if discovery.spatial_photo and discovery.spatial_photo.get("rectified_stereo_ready"):
        camera = discovery.spatial_photo["left_camera"]
        for key, (label, _) in _SPATIAL_PRODUCTS.items():
            description = "Computed on the left-view grid; inferred, not measured source depth. "
            if key.endswith("-preview"):
                description += ("Viewable PNG with explicit linear depth mapping and transparent missing pixels. "
                                "Quantized for viewing only; use the 0–1 float EXR for displacement.")
            elif key.endswith("-support"):
                description += "Binary support evidence, not depth or a calibrated probability."
            elif key.endswith("-height") or key.endswith("-flow"):
                description += ("Pixel units are not brightness. Signed flow can be negative and disparity can exceed 1; "
                                "direct PNG conversion clips them. Choose Depth preview for viewing.")
            elif key == "raft-supported-depth" or key.startswith("stereo-"):
                description += "Unsupported pixels remain NaN. Do not turn them into zero displacement."
            else:
                description += ("Dense forward estimate, including unverified occluded regions. "
                                "Export the support mask to identify them; no hole filling or confidence masking.")
            products.append({"id": key, "name": label, "width": camera["width"],
                             "height": camera["height"],
                             "precision": "16-bit PNG + alpha (view only)" if key.endswith("-preview") else "32-bit float EXR",
                             "source_precision": "Generated estimate", "origin": "Inferred",
                             "description": description})
    return products


def _product_id(output: PendingOutput) -> str:
    role = output.role.removesuffix("_exact_array")
    for key, (_, spatial_role) in _SPATIAL_PRODUCTS.items():
        if role == spatial_role:
            return key
    prefix = {"derived_physical_disparity": "disparity", "derived_metric_depth": "meters"}.get(role, "raw")
    return f"{prefix}:{output.asset_index}"


def _build_manifest(discovery: Discovery, asset_records: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    try:
        numpy_version = version("numpy")
    except PackageNotFoundError:
        numpy_version = np.__version__
    return {
        "schema": "ipde-extraction-manifest-v1",
        "source": {
            "path": str(discovery.source),
            "filename": discovery.source.name,
            "size_bytes": discovery.source_size,
            "sha256": discovery.source_sha256,
            "mimetype": discovery.mimetype,
            "primary_image_index": discovery.primary_index,
            "top_level_images": discovery.top_level_images,
            "spatial_photo": _jsonable(discovery.spatial_photo),
        },
        "decoder": {
            "pillow_heif": _check_runtime(),
            "libheif": pillow_heif.libheif_version(),
            "numpy": numpy_version,
            "settings": {
                "convert_hdr_to_8bit": False,
                "hdr_to_16bit": False,
                "bgr_mode": False,
                "remove_stride": True,
                "gamma_correction": False,
                "tone_mapping": False,
                "normalization": False,
            },
        },
        "precision_scope": (
            "Raw output samples are bit-exact copies of pillow-heif/libheif decoded arrays. "
            "Apple native depth assets separately preserve the ImageIO float16/float32 buffers bit-for-bit; "
            "native storage precision and encoded source precision are reported independently. "
            "Derived physical disparity and metric depth outputs, when present, are explicitly "
            "documented float32 calibrations of a raw uint8 uniform-disparity plane. Float32 changes "
            "the numeric representation and units, not the source quantization or amount of captured "
            "scene information. Classical StereoSGBM and RAFT-Stereo outputs, when requested, are "
            "explicitly identified inferred float32 estimates derived from the preserved stereo views. "
            "No output can restore information lost when the source HEIF was encoded."
        ),
        "available_products": _available_products(discovery),
        "asset_count": len(asset_records),
        "assets": asset_records,
        "warnings": warnings,
    }


def _discovery_warnings(discovery: Discovery) -> list[str]:
    warnings: list[str] = []
    if discovery.spatial_metadata_warning:
        warnings.append(discovery.spatial_metadata_warning)
    if any(a.kind == "native_depth" and a.metadata.get("accuracy") == "relative" for a in discovery.assets):
        warnings.append("Apple labels this embedded depth as relative accuracy; its calibrated values are not guaranteed absolute scene distances.")
    if discovery.spatial_photo:
        for aggressor in discovery.spatial_photo.get("stereo_aggressors", []):
            if isinstance(aggressor, Mapping):
                aggressor_type = str(aggressor.get("Type", "unspecified"))
            else:
                aggressor_type = str(aggressor)
            warnings.append(f"Apple spatial-photo stereo aggressor: {aggressor_type}")
    return warnings


def _report(discovery: Discovery) -> dict[str, Any]:
    records = [_asset_record(asset) for asset in discovery.assets]
    warnings = _discovery_warnings(discovery)
    if not records:
        warnings.append(
            "No depth, non-alpha auxiliary, alpha, or spatial-view planes were exposed."
        )
    return _build_manifest(discovery, records, warnings)


def inspect_file(source: Path | str) -> dict[str, Any]:
    return _report(discover_file(Path(source)))


def _temporary_path(final_path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent)
    os.close(descriptor)
    return Path(name)


def _verify_raw_bytes(path: Path, expected: bytes) -> None:
    if path.read_bytes() != expected:
        raise FormatError(f"metadata round-trip mismatch: {path.name}")


def _fsync_path(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _commit_outputs(
    installs: list[tuple[Path, Path]],
    *,
    overwrite: bool,
    created_temps: list[Path],
) -> None:
    """Commit a group and restore the prior group if any rename fails."""
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        if overwrite:
            for _, final in installs:
                if not final.exists():
                    continue
                backup = _temporary_path(final)
                backup.unlink()
                os.replace(final, backup)
                backups[final] = backup

        for temporary, final in installs:
            if overwrite:
                os.replace(temporary, final)
            else:
                # Same-directory hard linking is atomic and refuses a racing writer;
                # os.replace() would unexpectedly overwrite it.
                os.link(temporary, final)
                temporary.unlink()
            created_temps.remove(temporary)
            committed.append(final)

        directory_fd = os.open(installs[0][1].parent, os.O_RDONLY) if installs else None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        for final in reversed(committed):
            try:
                final.unlink(missing_ok=True)
            except OSError:
                pass
        for final, backup in backups.items():
            try:
                if backup.exists():
                    os.replace(backup, final)
            except OSError:
                pass
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _check_output_paths(paths: list[Path], *, overwrite: bool) -> None:
    if len({os.path.normcase(str(path)) for path in paths}) != len(paths):
        raise ExtractionError("generated output names collide")
    collisions = [path for path in paths if path.exists()]
    if collisions and not overwrite:
        joined = ", ".join(path.name for path in collisions[:5])
        if len(collisions) > 5:
            joined += f", and {len(collisions) - 5} more"
        raise ExtractionError(f"output already exists (use --overwrite): {joined}")


def extract_file(source: Path | str, options: ExtractOptions | None = None) -> dict[str, Any]:
    config = options or ExtractOptions()
    discovery = discover_file(Path(source))
    selected = None if config.selected_products is None else set(config.selected_products)
    if selected is not None:
        available = {product["id"] for product in _available_products(discovery)}
        if not selected or selected - available:
            raise ExtractionError(f"Select available products from --inspect; unavailable selection: {sorted(selected - available)}")
        config = replace(
            config,
            write_stereo_matching=any(key.startswith("stereo-") for key in selected),
            write_raft_stereo=any(key.startswith("raft-") for key in selected),
            write_raft_diagnostics=bool(selected & {"raft-flow", "raft-depth"}),
            write_displacement_maps=bool(selected & {"raft-displacement", "stereo-displacement"}),
            write_metric_depth=any(key.startswith("meters:") for key in selected),
            write_physical_disparity=any(key.startswith("disparity:") for key in selected),
        )
    def wanted(output: PendingOutput) -> bool:
        return selected is None or _product_id(output) in selected

    output_dir = (config.output_dir.expanduser().resolve() if config.output_dir else discovery.source.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ExtractionError(f"output path is not a directory: {output_dir}")

    names = _base_names(discovery)
    records = [_asset_record(asset) for asset in discovery.assets]
    warnings = _discovery_warnings(discovery)
    if not records:
        warnings.append(
            "No depth, non-alpha auxiliary, alpha, or spatial-view planes were exposed."
        )
    pending: list[PendingOutput] = []

    for index, (asset, name) in enumerate(zip(discovery.assets, names, strict=True)):
        if selected is not None and not selected.intersection({f"raw:{index}", f"meters:{index}", f"disparity:{index}"}):
            continue
        base = output_dir / name
        try:
            pending.append(_exchange_output(base, asset, index))
        except FormatError as exc:
            if not config.write_npy:
                raise ExtractionError(str(exc)) from exc
            warnings.append(f"{exc}; exact NPY was written instead.")
        if config.write_npy:
            npy_path = Path(f"{base}.npy")
            pending.append(
                PendingOutput(
                    final_path=npy_path,
                    role="exact_array",
                    asset_index=index,
                    write=lambda temp, a=asset.array: write_npy(temp, a),
                    verify=lambda temp, a=asset.array: verify_npy(temp, a),
                )
            )
        if (config.write_metric_depth or config.write_physical_disparity) and asset.kind == "depth":
            try:
                disparity = reconstruct_physical_disparity(asset.array, asset.metadata)
            except MetricDepthError as exc:
                warnings.append(f"{name}: calibrated float32 depth products were not written: {exc}")
            else:
                if config.write_physical_disparity:
                    pending.append(_physical_disparity_output(base, asset, index, disparity))
                if config.write_metric_depth:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        depth_meters = np.ascontiguousarray(np.float32(1.0) / disparity, dtype=np.float32)
                    pending.append(_metric_depth_output(base, asset, index, depth_meters))
        for metadata_index, block in enumerate(asset.metadata_blocks):
            raw = bytes(block.get("data", b""))
            if not raw:
                continue
            metadata_path = output_dir / f"{name}_metadata{metadata_index}{_metadata_extension(block)}"
            pending.append(
                PendingOutput(
                    final_path=metadata_path,
                    role="raw_metadata",
                    asset_index=index,
                    write=lambda temp, data=raw: temp.write_bytes(data),
                    verify=lambda temp, data=raw: _verify_raw_bytes(temp, data),
                )
            )

    pending = [item for item in pending if wanted(item)]
    selection_suffix = "" if selected is None else "_" + hashlib.sha256(
        "\n".join(sorted(selected)).encode("utf-8")).hexdigest()[:12]
    manifest_path = output_dir / f"{discovery.source.stem}{selection_suffix}_aux_manifest.json"
    manifest_paths = [manifest_path] if config.write_manifest else []
    write_raft = config.write_raft_stereo or config.write_raft_diagnostics
    write_spatial_height = config.write_stereo_matching or write_raft
    if config.write_displacement_maps and not write_spatial_height:
        raise ExtractionError(
            "0..1 displacement maps require --stereo-matching, --raft-stereo, or --stereo-comparison"
        )
    if write_spatial_height and discovery.spatial_photo is None:
        warnings.append(
            "Spatial stereo height maps were requested, but this HEIF has no Apple stereo-pair group."
        )
    elif write_spatial_height:
        predicted_bases: list[Path] = []
        color_suffix = "_color_matched" if config.histogram_color_matching else ""
        if config.write_stereo_matching:
            predicted_bases.append(
                output_dir
                / f"{discovery.source.stem}_spatial_stereo_matching{color_suffix}_height"
            )
            if config.write_displacement_maps:
                predicted_bases.append(
                    output_dir
                    / f"{discovery.source.stem}_spatial_stereo_matching{color_suffix}_displacement_0_to_1"
                )
        if write_raft:
            predicted_bases.append(
                output_dir
                / f"{discovery.source.stem}_spatial_raft_stereo{color_suffix}_height"
            )
            if config.write_displacement_maps:
                predicted_bases.append(
                    output_dir
                    / f"{discovery.source.stem}_spatial_raft_stereo{color_suffix}_displacement_0_to_1"
                )
            if config.write_raft_diagnostics:
                predicted_bases.extend(
                    output_dir
                    / f"{discovery.source.stem}_spatial_raft_stereo{color_suffix}_{suffix}"
                    for suffix in ("signed_flow", "depth_meters")
                )
        if selected is not None:
            suffixes = {
                "raft-height": f"raft_stereo{color_suffix}_height",
                "raft-displacement": f"raft_stereo{color_suffix}_displacement_0_to_1",
                "raft-flow": f"raft_stereo{color_suffix}_signed_flow",
                "raft-depth": f"raft_stereo{color_suffix}_depth_meters",
                "stereo-height": f"stereo_matching{color_suffix}_height",
                "stereo-displacement": f"stereo_matching{color_suffix}_displacement_0_to_1",
            }
            for engine, filename in (("raft", "raft_stereo"), ("stereo", "stereo_matching")):
                for key, suffix in (("support", "support"), ("supported-depth", "supported_depth_meters"),
                                    ("preview", "depth_preview")):
                    suffixes[f"{engine}-{key}"] = f"{filename}{color_suffix}_{suffix}"
            predicted_bases = [output_dir / f"{discovery.source.stem}_spatial_{suffix}"
                               for key, suffix in suffixes.items() if key in selected]
        predicted_paths = [Path(f"{base}.png" if str(base).endswith("_depth_preview") else f"{base}.exr")
                           for base in predicted_bases]
        if config.write_npy:
            predicted_paths.extend(Path(f"{base}.npy") for base in predicted_bases
                                   if not str(base).endswith("_depth_preview"))
        _check_output_paths(
            [item.final_path for item in pending] + predicted_paths + manifest_paths,
            overwrite=config.overwrite,
        )
        left_image_index = int(discovery.spatial_photo["left_image_index"])
        right_image_index = int(discovery.spatial_photo["right_image_index"])
        left_asset_index = next(
            (
                index
                for index, asset in enumerate(discovery.assets)
                if asset.kind == "spatial_view"
                and asset.parent_image_index == left_image_index
            ),
            None,
        )
        right_asset_index = next(
            (
                index
                for index, asset in enumerate(discovery.assets)
                if asset.kind == "spatial_view"
                and asset.parent_image_index == right_image_index
            ),
            None,
        )
        if left_asset_index is None or right_asset_index is None:
            raise ExtractionError("spatial-photo left/right decoded assets are missing")
        left_array = discovery.assets[left_asset_index].array
        right_array = discovery.assets[right_asset_index].array
        color_matching_details: dict[str, Any] = {
            "applied": False,
            "policy": "disabled",
            "raw_assets_modified": False,
        }
        inference_left = left_array
        inference_right = right_array
        height_maps: dict[str, np.ndarray] = {}
        inference_details: dict[str, Mapping[str, Any]] = {}
        if config.histogram_color_matching:
            try:
                matched_pair = histogram_match_stereo_pair(
                    left_array,
                    right_array,
                    hero_side=config.color_matching_hero,
                )
            except ColorMatchingError as exc:
                raise ExtractionError(str(exc)) from exc
            inference_left = matched_pair.left
            inference_right = matched_pair.right
            color_matching_details = matched_pair.details
        if config.write_stereo_matching:
            try:
                stereo_result = run_stereo_matching(
                    inference_left,
                    inference_right,
                    discovery.spatial_photo,
                    StereoMatchingOptions(
                        maximum_disparity=config.stereo_maximum_disparity,
                        noise_sigma_pixels=config.stereo_noise_sigma_pixels,
                    ),
                )
            except StereoMatchingError as exc:
                raise ExtractionError(str(exc)) from exc
            stereo_result.details = {
                **stereo_result.details,
                "input_color_matching": color_matching_details,
            }
            height_maps["stereo_matching"] = stereo_result.height_disparity_pixels
            inference_details["stereo_matching"] = stereo_result.details
            pending.extend(
                _stereo_matching_pending_outputs(
                    output_dir,
                    discovery,
                    left_asset_index,
                    stereo_result,
                    write_npy_companions=config.write_npy,
                    color_matched=config.histogram_color_matching,
                )
            )
            pending.extend(_stereo_review_outputs(
                output_dir, discovery, left_asset_index, "stereo", stereo_result.height_disparity_pixels,
                np.isfinite(stereo_result.height_disparity_pixels), stereo_result.details, selected,
                color_matched=config.histogram_color_matching, write_npy_companions=config.write_npy,
            ))
        if write_raft:
            try:
                raft_result = run_raft_stereo(
                    inference_left,
                    inference_right,
                    discovery.spatial_photo,
                    RaftStereoOptions(
                        root=config.raft_root,
                        model=config.raft_model,
                        model_member=config.raft_model_member,
                        device=config.raft_device,
                        iterations=config.raft_iterations,
                    ),
                )
            except RaftStereoError as exc:
                raise ExtractionError(str(exc)) from exc
            raft_result.details = {
                **raft_result.details,
                "input_color_matching": color_matching_details,
            }
            height_maps["raft_stereo"] = raft_result.height_disparity_pixels
            inference_details["raft_stereo"] = raft_result.details
            pending.extend(
                _raft_pending_outputs(
                    output_dir,
                    discovery,
                    left_asset_index,
                    raft_result,
                    write_npy_companions=config.write_npy,
                    include_diagnostics=config.write_raft_diagnostics,
                    color_matched=config.histogram_color_matching,
                )
            )
            if selected and selected & {"raft-support", "raft-supported-depth", "raft-preview"}:
                if raft_result.support_mask is None:
                    raise ExtractionError("RAFT inference did not return correspondence support")
                pending.extend(_stereo_review_outputs(
                    output_dir, discovery, left_asset_index, "raft", raft_result.height_disparity_pixels,
                    raft_result.support_mask, raft_result.details, selected,
                    color_matched=config.histogram_color_matching, write_npy_companions=config.write_npy,
                ))
            support_fraction = raft_result.details.get("support_pixel_fraction")
            if support_fraction is not None and support_fraction < 1:
                warnings.append(
                    f"RAFT dense estimate: {100 * (1 - support_fraction):.2f}% lacks correspondence support. "
                    "These predictions are retained, not verified geometry. Select RAFT support mask "
                    "or RAFT supported depth for conservative reconstruction."
                )
        if config.write_displacement_maps:
            for engine_name, height_map in height_maps.items():
                product = "raft-displacement" if engine_name == "raft_stereo" else "stereo-displacement"
                if selected is not None and product not in selected:
                    continue
                try:
                    displacement, mapping_details = linear_depth_displacement(height_map, discovery.spatial_photo)
                except (DisplacementMappingError, RaftStereoError) as exc:
                    raise ExtractionError(str(exc)) from exc
                pending.extend(_displacement_pending_outputs(
                    output_dir, discovery, left_asset_index, displacement,
                    engine_name=engine_name, inference_details=inference_details[engine_name],
                    mapping_details=mapping_details, color_matched=config.histogram_color_matching,
                    write_npy_companions=config.write_npy,
                ))
        for engine_name, height_map in height_maps.items():
            registration = inference_details[engine_name].get("vertical_registration", {})
            if registration.get("applied"):
                warnings.append(
                    f"{engine_name}: corrected vertical alignment for inference; median feature error "
                    f"{registration['median_vertical_error_before_pixels']:.2f} to "
                    f"{registration['median_vertical_error_after_pixels']:.2f} pixels. Raw views are unchanged."
                )
            invalid = int(np.count_nonzero(~np.isfinite(height_map)))
            if invalid:
                warnings.append(f"{engine_name}: {invalid}/{height_map.size} pixels have no finite estimate; "
                                "these remain NaN (often displayed black), not invented geometry.")

    pending = [item for item in pending if wanted(item)]

    all_final_paths = [item.final_path for item in pending] + manifest_paths
    _check_output_paths(all_final_paths, overwrite=config.overwrite)

    created_temps: list[Path] = []
    try:
        for item in pending:
            temp = _temporary_path(item.final_path)
            item.temporary_path = temp
            created_temps.append(temp)
            item.write(temp)
            _fsync_path(temp)
            item.verify(temp)
            output_record = {
                "role": item.role,
                "path": str(item.final_path),
                "filename": item.final_path.name,
                "size_bytes": temp.stat().st_size,
                "sha256": sha256_file(temp),
                "verified": True,
            }
            output_record.update(_jsonable(item.details))
            if item.asset_index is not None:
                records[item.asset_index]["outputs"].append(output_record)

        manifest = _build_manifest(discovery, records, warnings)
        manifest["selected_products"] = sorted(selected) if selected is not None else None
        manifest["manifest_path"] = str(manifest_path) if config.write_manifest else None

        installs: list[tuple[Path, Path]] = []
        for item in pending:
            assert item.temporary_path is not None
            installs.append((item.temporary_path, item.final_path))
        if config.write_manifest:
            manifest_temp = _temporary_path(manifest_path)
            created_temps.append(manifest_temp)
            manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            _fsync_path(manifest_temp)
            json.loads(manifest_temp.read_text(encoding="utf-8"))
            installs.append((manifest_temp, manifest_path))
        _commit_outputs(installs, overwrite=config.overwrite, created_temps=created_temps)
        return manifest
    except Exception as exc:
        raise ExtractionError(f"could not commit verified outputs: {exc}") from exc
    finally:
        for temp in created_temps:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
