"""Discovery, metadata capture, and transactional extraction for HEIF assets."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pillow_heif

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
    output_dir: Path | None = None
    write_npy: bool = True
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


@dataclass
class PendingOutput:
    final_path: Path
    role: str
    asset_index: int | None
    write: Callable[[Path], None]
    verify: Callable[[Path], None]
    temporary_path: Path | None = None


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
    value = getattr(c_image, "bit_depth", fallback) if c_image is not None else fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r";(10|12|16)", str(getattr(image, "mode", "")))
        return int(match.group(1)) if match else int(fallback or 8)


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

    return Discovery(
        source=path,
        source_size=len(source_bytes),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        mimetype=str(heif.mimetype),
        primary_index=int(heif.primary_index),
        top_level_images=image_contexts,
        assets=assets,
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
        if counts[name] == 1 and asset.parent_image_index == discovery.primary_index:
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
        return PendingOutput(
            final_path=path,
            role="lossless_exchange",
            asset_index=asset_index,
            write=lambda temp, a=array: write_exr(temp, a),
            verify=lambda temp, a=array: verify_exr(temp, a),
        )
    raise FormatError(
        f"{asset.semantic_name} has dtype/shape {array.dtype}/{array.shape}, which PNG and OpenEXR cannot preserve"
    )


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
            "Output samples are bit-exact copies of pillow-heif/libheif decoded arrays. "
            "They cannot restore information lost when the source HEIF was encoded."
        ),
        "asset_count": len(asset_records),
        "assets": asset_records,
        "warnings": warnings,
    }


def _report(discovery: Discovery) -> dict[str, Any]:
    records = [_asset_record(asset) for asset in discovery.assets]
    warnings = []
    if not records:
        warnings.append("No depth, non-alpha auxiliary, or alpha planes were exposed by pillow-heif.")
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


def extract_file(source: Path | str, options: ExtractOptions | None = None) -> dict[str, Any]:
    config = options or ExtractOptions()
    discovery = discover_file(Path(source))
    output_dir = (config.output_dir.expanduser().resolve() if config.output_dir else discovery.source.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ExtractionError(f"output path is not a directory: {output_dir}")

    names = _base_names(discovery)
    records = [_asset_record(asset) for asset in discovery.assets]
    warnings: list[str] = []
    if not records:
        warnings.append("No depth, non-alpha auxiliary, or alpha planes were exposed by pillow-heif.")
    pending: list[PendingOutput] = []

    for index, (asset, name) in enumerate(zip(discovery.assets, names, strict=True)):
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

    manifest_path = output_dir / f"{discovery.source.stem}_aux_manifest.json"
    all_final_paths = [item.final_path for item in pending] + [manifest_path]
    if len({os.path.normcase(str(path)) for path in all_final_paths}) != len(all_final_paths):
        raise ExtractionError("generated output names collide")
    collisions = [path for path in all_final_paths if path.exists()]
    if collisions and not config.overwrite:
        joined = ", ".join(path.name for path in collisions[:5])
        if len(collisions) > 5:
            joined += f", and {len(collisions) - 5} more"
        raise ExtractionError(f"output already exists (use --overwrite): {joined}")

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
            if item.asset_index is not None:
                records[item.asset_index]["outputs"].append(output_record)

        manifest = _build_manifest(discovery, records, warnings)
        manifest["manifest_path"] = str(manifest_path)
        manifest_temp = _temporary_path(manifest_path)
        created_temps.append(manifest_temp)
        manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        _fsync_path(manifest_temp)
        json.loads(manifest_temp.read_text(encoding="utf-8"))

        installs: list[tuple[Path, Path]] = []
        for item in pending:
            assert item.temporary_path is not None
            installs.append((item.temporary_path, item.final_path))
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
