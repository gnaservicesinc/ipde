"""Spatial-photo validation and full-resolution stereo inference."""

from __future__ import annotations

import hashlib
import io
import math
import os
import sys
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


class SpatialPhotoError(ValueError):
    """Raised when spatial-photo metadata is incomplete or inconsistent."""


class RaftStereoError(RuntimeError):
    """Raised when RAFT-Stereo resources or inference are unavailable."""


class StereoMatchingError(RuntimeError):
    """Raised when classical stereo matching cannot produce a height map."""


class ColorMatchingError(RuntimeError):
    """Raised when a stereo pair cannot be histogram-matched safely."""


class DisplacementMappingError(RuntimeError):
    """Raised when disparity maps cannot be mapped to displacement-ready values."""


@dataclass(frozen=True)
class StereoMatchingOptions:
    maximum_disparity: int | None = None
    block_size: int = 5


@dataclass(frozen=True)
class RaftStereoOptions:
    root: Path | None = None
    model: Path | None = None
    model_member: str | None = None
    device: str = "auto"
    iterations: int = 32


@dataclass
class RaftStereoResult:
    signed_flow_pixels: np.ndarray
    height_disparity_pixels: np.ndarray
    depth_meters: np.ndarray
    details: dict[str, Any]


@dataclass
class StereoMatchingResult:
    height_disparity_pixels: np.ndarray
    details: dict[str, Any]


@dataclass
class HistogramMatchedStereoPair:
    left: np.ndarray
    right: np.ndarray
    details: dict[str, Any]


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SpatialPhotoError(f"{name} must contain exactly {length} numeric values")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise SpatialPhotoError(f"{name} contains a nonnumeric value") from exc
    if not all(math.isfinite(item) for item in result):
        raise SpatialPhotoError(f"{name} contains a nonfinite value")
    return result


def _camera_record(image: Mapping[str, Any], image_index: int) -> dict[str, Any]:
    heif = image.get("{HEIF}")
    if not isinstance(heif, Mapping):
        raise SpatialPhotoError(f"stereo image {image_index} has no HEIF camera metadata")
    extrinsics = heif.get("CameraExtrinsics")
    model = heif.get("CameraModel")
    if not isinstance(extrinsics, Mapping) or not isinstance(model, Mapping):
        raise SpatialPhotoError(
            f"stereo image {image_index} lacks camera extrinsics or camera-model metadata"
        )
    position = _finite_vector(extrinsics.get("Position"), 3, "camera position")
    rotation = _finite_vector(extrinsics.get("Rotation"), 9, "camera rotation")
    intrinsics = _finite_vector(model.get("Intrinsics"), 9, "camera intrinsics")
    model_type = str(model.get("ModelType", ""))
    if model_type not in {"SimplifiedPinhole", "GenericPinhole"}:
        raise SpatialPhotoError(
            f"stereo image {image_index} uses unsupported camera model {model_type!r}"
        )
    try:
        width = int(image["PixelWidth"])
        height = int(image["PixelHeight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialPhotoError(f"stereo image {image_index} has no valid dimensions") from exc
    if width <= 0 or height <= 0:
        raise SpatialPhotoError(f"stereo image {image_index} has invalid dimensions")
    orientation = image.get("Orientation")
    return {
        "image_index": image_index,
        "width": width,
        "height": height,
        "orientation": int(orientation) if isinstance(orientation, int) else orientation,
        "coordinate_system_id": extrinsics.get("CoordinateSystemID"),
        "position_meters": position,
        "rotation_row_major": rotation,
        "model_type": model_type,
        "intrinsics_row_major": intrinsics,
        "focal_length_x_pixels": intrinsics[0],
        "focal_length_y_pixels": intrinsics[4],
        "principal_point_x_pixels": intrinsics[2],
        "principal_point_y_pixels": intrinsics[5],
    }


def analyze_spatial_photo(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate the first Apple stereo-pair group and derive its calibration."""
    if not metadata:
        return None
    raw_groups = metadata.get("groups", [])
    groups = raw_groups if isinstance(raw_groups, list) else [raw_groups]
    stereo_groups = [
        group
        for group in groups
        if isinstance(group, Mapping) and str(group.get("GroupType", "")) == "StereoPair"
    ]
    if not stereo_groups:
        return None
    if len(stereo_groups) > 1:
        raise SpatialPhotoError("multiple stereo-pair groups are not currently supported")
    group = stereo_groups[0]
    try:
        left_index = int(group["GroupImageIndexLeft"])
        right_index = int(group["GroupImageIndexRight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialPhotoError("stereo group has no valid left/right image indices") from exc
    if left_index == right_index or min(left_index, right_index) < 0:
        raise SpatialPhotoError("stereo group left/right image indices are invalid")
    images = metadata.get("images")
    if not isinstance(images, list) or max(left_index, right_index) >= len(images):
        raise SpatialPhotoError("stereo group references an image outside the HEIF container")

    left = _camera_record(images[left_index], left_index)
    right = _camera_record(images[right_index], right_index)
    if (left["width"], left["height"]) != (right["width"], right["height"]):
        raise SpatialPhotoError("left and right spatial-photo images have different dimensions")

    left_position = np.asarray(left["position_meters"], dtype=np.float64)
    right_position = np.asarray(right["position_meters"], dtype=np.float64)
    baseline_vector = right_position - left_position
    baseline_meters = float(np.linalg.norm(baseline_vector))
    if not math.isfinite(baseline_meters) or baseline_meters <= 0.0:
        raise SpatialPhotoError("spatial-photo camera baseline is not positive and finite")

    encoded_adjustment_value = group.get("GroupImageDisparityAdjustment")
    try:
        encoded_adjustment = int(encoded_adjustment_value)
    except (TypeError, ValueError) as exc:
        raise SpatialPhotoError("stereo group has no valid disparity adjustment") from exc
    if not -10_000 <= encoded_adjustment <= 10_000:
        raise SpatialPhotoError(
            f"encoded disparity adjustment {encoded_adjustment} is outside [-10000, 10000]"
        )
    adjustment_fraction = encoded_adjustment / 10_000.0

    left_intrinsics = np.asarray(left["intrinsics_row_major"], dtype=np.float64)
    right_intrinsics = np.asarray(right["intrinsics_row_major"], dtype=np.float64)
    left_rotation = np.asarray(left["rotation_row_major"], dtype=np.float64)
    right_rotation = np.asarray(right["rotation_row_major"], dtype=np.float64)
    intrinsics_match = bool(np.allclose(left_intrinsics, right_intrinsics, rtol=0.0, atol=1e-9))
    rotations_match = bool(np.allclose(left_rotation, right_rotation, rtol=0.0, atol=1e-9))
    baseline_is_horizontal = bool(
        baseline_vector[0] > 0.0
        and abs(float(baseline_vector[1])) <= max(1e-9, baseline_meters * 1e-6)
        and abs(float(baseline_vector[2])) <= max(1e-9, baseline_meters * 1e-6)
    )
    orientations_match = left["orientation"] == right["orientation"]
    raft_ready = (
        intrinsics_match
        and rotations_match
        and baseline_is_horizontal
        and orientations_match
    )
    validation_notes: list[str] = []
    if not intrinsics_match:
        validation_notes.append("left and right intrinsics differ")
    if not rotations_match:
        validation_notes.append("left and right rotations differ")
    if not baseline_is_horizontal:
        validation_notes.append("right camera is not on the positive horizontal baseline")
    if not orientations_match:
        validation_notes.append("left and right orientations differ")

    monoscopic = group.get("GroupImageIndexMonoscopic")
    aggressors = group.get("GroupImageStereoAggressors", [])
    return {
        "is_spatial_photo": True,
        "metadata_source": "macOS ImageIO container properties",
        "group_index": int(group.get("GroupIndex", 0)),
        "left_image_index": left_index,
        "right_image_index": right_index,
        "monoscopic_image_index": int(monoscopic) if isinstance(monoscopic, int) else monoscopic,
        "monoscopic_image_location": group.get("GroupImageIndexMonoscopicImageLocation"),
        "encoded_disparity_adjustment": encoded_adjustment,
        "disparity_adjustment_fraction_of_width": adjustment_fraction,
        "disparity_adjustment_pixels": adjustment_fraction * left["width"],
        "disparity_adjustment_semantics": (
            "Presentation-only zero-parallax shift; retained in metadata and not applied to geometric depth."
        ),
        "stereo_aggressors": aggressors if isinstance(aggressors, list) else [aggressors],
        "left_camera": left,
        "right_camera": right,
        "baseline_vector_meters": baseline_vector.tolist(),
        "baseline_meters": baseline_meters,
        "baseline_millimeters": baseline_meters * 1000.0,
        "focal_length_pixels_for_depth": left["focal_length_x_pixels"],
        "principal_point_delta_x_pixels": (
            right["principal_point_x_pixels"] - left["principal_point_x_pixels"]
        ),
        "rectified_stereo_ready": raft_ready,
        "rectified_stereo_validation_notes": validation_notes,
        "raft_stereo_ready": raft_ready,
        "raft_stereo_validation_notes": validation_notes,
        "decoded_orientation_policy": (
            "Stereo inference uses pillow-heif decoded sample coordinates without applying EXIF orientation, "
            "so the arrays remain registered to the stored camera intrinsics."
        ),
    }


def _candidate_ancestors() -> list[Path]:
    here = Path(__file__).resolve()
    result: list[Path] = []
    for parent in here.parents:
        if parent not in result:
            result.append(parent)
        if len(result) >= 8:
            break
    return result


def resolve_raft_resources(options: RaftStereoOptions) -> tuple[Path, Path, str | None]:
    if options.root is not None and not (options.root.expanduser() / "core" / "raft_stereo.py").is_file():
        raise RaftStereoError(f"Selected RAFT-Stereo source is invalid: {options.root}")
    if options.model is not None and not options.model.expanduser().is_file():
        raise RaftStereoError(f"Selected RAFT-Stereo model does not exist: {options.model}")
    root_candidates: list[Path] = []
    if options.root is not None:
        root_candidates.append(options.root.expanduser())
    environment_root = os.environ.get("IPDE_RAFT_STEREO_DIR")
    if environment_root:
        root_candidates.append(Path(environment_root).expanduser())
    for parent in _candidate_ancestors():
        root_candidates.extend((parent / "RAFT-Stereo", parent / "raft_stereo"))
    root = next(
        (
            candidate.resolve()
            for candidate in root_candidates
            if (candidate / "core" / "raft_stereo.py").is_file()
        ),
        None,
    )
    if root is None:
        raise RaftStereoError(
            "RAFT-Stereo source was not found; use --raft-root or IPDE_RAFT_STEREO_DIR"
        )

    model_candidates: list[Path] = []
    if options.model is not None:
        model_candidates.append(options.model.expanduser())
    environment_model = os.environ.get("IPDE_RAFT_MODEL")
    if environment_model:
        model_candidates.append(Path(environment_model).expanduser())
    model_candidates.extend(
        (
            root / "models" / "raftstereo-middlebury.pth",
            root.parent / "models.zip",
        )
    )
    for parent in _candidate_ancestors():
        model_candidates.extend(
            (
                parent / "models" / "raftstereo-middlebury.pth",
                parent / "models.zip",
            )
        )
    model = next((candidate.resolve() for candidate in model_candidates if candidate.is_file()), None)
    if model is None:
        raise RaftStereoError(
            "RAFT-Stereo checkpoint was not found; use --raft-model or IPDE_RAFT_MODEL"
        )
    member = options.model_member
    if model.suffix.lower() == ".zip" and member is None:
        member = "raftstereo-middlebury.pth"
    return root, model, member


def _checkpoint_bytes(model: Path, member: str | None) -> tuple[bytes, str]:
    if model.suffix.lower() != ".zip":
        try:
            return model.read_bytes(), model.name
        except OSError as exc:
            raise RaftStereoError(f"could not read RAFT-Stereo checkpoint {model}: {exc}") from exc
    if not member:
        raise RaftStereoError("a checkpoint member is required when --raft-model names a ZIP archive")
    try:
        with zipfile.ZipFile(model) as archive:
            matches = [name for name in archive.namelist() if name == member]
            if not matches:
                matches = [name for name in archive.namelist() if Path(name).name == member]
            if len(matches) != 1:
                raise RaftStereoError(f"Checkpoint {member!r} is missing or ambiguous in {model}")
            info = archive.getinfo(matches[0])
            if info.is_dir() or info.file_size <= 0 or info.file_size > 2 * 1024 * 1024 * 1024:
                raise RaftStereoError(f"invalid checkpoint member size for {member!r}")
            return archive.read(info), member
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise RaftStereoError(
            f"could not read checkpoint member {member!r} from {model}: {exc}"
        ) from exc


def _model_configuration(checkpoint_name: str) -> SimpleNamespace:
    lowered = checkpoint_name.lower()
    values: dict[str, Any] = {
        "hidden_dims": [128, 128, 128],
        "corr_implementation": "alt",
        "shared_backbone": False,
        "corr_levels": 4,
        "corr_radius": 4,
        "n_downsample": 2,
        "context_norm": "batch",
        "slow_fast_gru": False,
        "n_gru_layers": 3,
        "mixed_precision": False,
    }
    if "iraftstereo_rvc" in lowered:
        values["context_norm"] = "instance"
    elif "realtime" in lowered:
        values.update(
            {
                "shared_backbone": True,
                "n_downsample": 3,
                "n_gru_layers": 2,
                "slow_fast_gru": True,
            }
        )
    return SimpleNamespace(**values)


def _select_device(torch: Any, requested: str) -> str:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise RaftStereoError(f"unsupported RAFT-Stereo device {requested!r}")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RaftStereoError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RaftStereoError("Apple Metal (MPS) was requested but is unavailable")
    return requested


def _validate_raft_inputs(
    left: np.ndarray, right: np.ndarray, spatial: Mapping[str, Any], iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != np.dtype("uint8") or right_array.dtype != np.dtype("uint8"):
        raise RaftStereoError("the supplied pretrained RAFT-Stereo models require uint8 RGB inputs")
    if (
        left_array.ndim != 3
        or right_array.ndim != 3
        or left_array.shape[2] != 3
        or right_array.shape[2] != 3
    ):
        raise RaftStereoError("RAFT-Stereo requires HxWx3 RGB left/right arrays")
    if left_array.shape != right_array.shape or left_array.size == 0:
        raise RaftStereoError("RAFT-Stereo left/right arrays must have identical nonempty shapes")
    left_camera = spatial.get("left_camera", {})
    try:
        expected_shape = (
            int(left_camera["height"]),
            int(left_camera["width"]),
            3,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RaftStereoError("spatial-photo metadata has no valid left-camera dimensions") from exc
    if left_array.shape != expected_shape:
        raise RaftStereoError(
            f"RAFT-Stereo requires the full decoded spatial view {expected_shape}, not {left_array.shape}"
        )
    if not bool(spatial.get("raft_stereo_ready")):
        notes = spatial.get("raft_stereo_validation_notes", [])
        raise RaftStereoError(
            "spatial-photo calibration is not rectified for RAFT-Stereo"
            + (f": {', '.join(str(item) for item in notes)}" if notes else "")
        )
    if not 1 <= iterations <= 256:
        raise RaftStereoError("RAFT-Stereo iterations must be in [1, 256]")
    return np.ascontiguousarray(left_array), np.ascontiguousarray(right_array)


def _validate_stereo_matching_inputs(
    left: np.ndarray,
    right: np.ndarray,
    spatial: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != np.dtype("uint8") or right_array.dtype != np.dtype("uint8"):
        raise StereoMatchingError(
            "OpenCV StereoSGBM requires uint8 inputs; IPDE will not reduce higher-bit spatial views"
        )
    if (
        left_array.ndim != 3
        or right_array.ndim != 3
        or left_array.shape[2] != 3
        or right_array.shape[2] != 3
    ):
        raise StereoMatchingError("StereoSGBM requires HxWx3 RGB left/right arrays")
    if left_array.shape != right_array.shape or left_array.size == 0:
        raise StereoMatchingError(
            "StereoSGBM left/right arrays must have identical nonempty shapes"
        )
    left_camera = spatial.get("left_camera", {})
    try:
        expected_shape = (
            int(left_camera["height"]),
            int(left_camera["width"]),
            3,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StereoMatchingError(
            "spatial-photo metadata has no valid left-camera dimensions"
        ) from exc
    if left_array.shape != expected_shape:
        raise StereoMatchingError(
            f"StereoSGBM requires the full decoded spatial view {expected_shape}, not {left_array.shape}"
        )
    ready = spatial.get("rectified_stereo_ready", spatial.get("raft_stereo_ready"))
    if not bool(ready):
        notes = spatial.get(
            "rectified_stereo_validation_notes",
            spatial.get("raft_stereo_validation_notes", []),
        )
        raise StereoMatchingError(
            "spatial-photo calibration is not rectified for StereoSGBM"
            + (f": {', '.join(str(item) for item in notes)}" if notes else "")
        )
    return np.ascontiguousarray(left_array), np.ascontiguousarray(right_array)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _channel_summary(array: np.ndarray) -> dict[str, list[float]]:
    flattened = array.reshape(-1, array.shape[2]).astype(np.float64, copy=False)
    return {
        "mean": [float(value) for value in flattened.mean(axis=0)],
        "standard_deviation": [float(value) for value in flattened.std(axis=0)],
    }


def _uint8_histogram_lut(source: np.ndarray, hero: np.ndarray) -> np.ndarray:
    source_histogram = np.bincount(source.ravel(), minlength=256).astype(np.int64)
    hero_histogram = np.bincount(hero.ravel(), minlength=256).astype(np.int64)
    source_values = np.flatnonzero(source_histogram)
    hero_values = np.flatnonzero(hero_histogram)
    source_quantiles = np.cumsum(
        source_histogram[source_values], dtype=np.float64
    ) / np.float64(source.size)
    hero_quantiles = np.cumsum(
        hero_histogram[hero_values], dtype=np.float64
    ) / np.float64(hero.size)
    mapped = np.interp(
        source_quantiles,
        hero_quantiles,
        hero_values.astype(np.float64),
    )
    lut = np.arange(256, dtype=np.uint8)
    lut[source_values] = np.clip(np.rint(mapped), 0.0, 255.0).astype(np.uint8)
    return lut


def histogram_match_stereo_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    hero_side: str = "left",
) -> HistogramMatchedStereoPair:
    """Match the non-Hero view's uint8 RGB channel CDFs to the Hero view."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if hero_side not in {"left", "right"}:
        raise ColorMatchingError("Color Matching Hero must be 'left' or 'right'")
    if left_array.dtype != np.dtype("uint8") or right_array.dtype != np.dtype("uint8"):
        raise ColorMatchingError(
            "per-channel histogram Color Matching requires uint8 stereo views; "
            "IPDE will not reduce higher-bit inputs"
        )
    if (
        left_array.ndim != 3
        or right_array.ndim != 3
        or left_array.shape[2] != 3
        or right_array.shape[2] != 3
        or left_array.shape != right_array.shape
        or left_array.size == 0
    ):
        raise ColorMatchingError(
            "per-channel histogram Color Matching requires identical nonempty HxWx3 RGB views"
        )

    hero = left_array if hero_side == "left" else right_array
    source = right_array if hero_side == "left" else left_array
    matched = np.empty_like(source)
    lookup_tables: list[list[int]] = []
    for channel in range(3):
        lut = _uint8_histogram_lut(source[:, :, channel], hero[:, :, channel])
        matched[:, :, channel] = lut[source[:, :, channel]]
        lookup_tables.append([int(value) for value in lut])

    if hero_side == "left":
        output_left = np.ascontiguousarray(left_array.copy())
        output_right = np.ascontiguousarray(matched)
        transformed_side = "right"
    else:
        output_left = np.ascontiguousarray(matched)
        output_right = np.ascontiguousarray(right_array.copy())
        transformed_side = "left"

    details = {
        "applied": True,
        "policy": "per-channel uint8 CDF histogram matching",
        "hero_side": hero_side,
        "transformed_side": transformed_side,
        "channel_order": "RGB",
        "lookup_table_length_per_channel": 256,
        "lookup_tables": lookup_tables,
        "interpolation": "float64 linear interpolation between populated Hero-channel CDF values",
        "quantization": "round-to-nearest-even with np.rint, then clamp to uint8 [0,255]",
        "spatial_correspondence_used_to_fit_curves": False,
        "icc_profile_conversion": False,
        "gamma_correction": False,
        "tone_mapping": False,
        "normalization": False,
        "raw_assets_modified": False,
        "left_input_sha256": _array_sha256(left_array),
        "right_input_sha256": _array_sha256(right_array),
        "left_inference_input_sha256": _array_sha256(output_left),
        "right_inference_input_sha256": _array_sha256(output_right),
        "left_input_channels": _channel_summary(left_array),
        "right_input_channels": _channel_summary(right_array),
        "left_inference_input_channels": _channel_summary(output_left),
        "right_inference_input_channels": _channel_summary(output_right),
        "semantic_scope": (
            "This matches each RGB channel's marginal code-value distribution. It is a derived "
            "radiometric preprocessing step, not an ICC transform or proof of pixelwise color equality."
        ),
    }
    return HistogramMatchedStereoPair(output_left, output_right, details)


def normalize_height_maps_for_displacement(
    height_maps: Mapping[str, np.ndarray],
    *,
    lower_percentile: float = 0.0,
    upper_percentile: float = 100.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map one or more pixel-disparity fields to a shared display/displacement range."""
    if not height_maps:
        raise DisplacementMappingError("at least one height map is required")
    if not (
        math.isfinite(lower_percentile)
        and math.isfinite(upper_percentile)
        and 0.0 <= lower_percentile < upper_percentile <= 100.0
    ):
        raise DisplacementMappingError(
            "displacement percentiles must satisfy 0 <= lower < upper <= 100"
        )

    shape: tuple[int, int] | None = None
    arrays: dict[str, np.ndarray] = {}
    finite_values: list[np.ndarray] = []
    finite_counts: dict[str, int] = {}
    for name, value in height_maps.items():
        array = np.asarray(value)
        if array.dtype != np.dtype("float32") or array.ndim != 2 or array.size == 0:
            raise DisplacementMappingError(
                f"height map {name!r} must be a nonempty float32 HxW array"
            )
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise DisplacementMappingError("all height maps must have the same shape")
        finite = np.isfinite(array)
        count = int(np.count_nonzero(finite))
        finite_counts[str(name)] = count
        if count:
            finite_values.append(array[finite])
        arrays[str(name)] = array
    if not finite_values:
        raise DisplacementMappingError("height maps contain no finite values")

    pooled = np.concatenate(finite_values)
    bounds = np.percentile(
        pooled,
        [lower_percentile, upper_percentile],
        method="linear",
    )
    lower = np.float32(bounds[0])
    upper = np.float32(bounds[1])
    del pooled
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise DisplacementMappingError(
            "displacement bounds are not finite and ordered"
        )
    with np.errstate(over="ignore"):
        scale = np.float32(upper - lower)
    if not np.isfinite(scale):
        raise DisplacementMappingError("displacement range exceeds finite float32")

    normalized: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        mapped = np.ascontiguousarray(
            np.divide(
                np.subtract(array, lower, dtype=np.float32),
                scale if scale > 0 else np.float32(1.0),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        np.clip(mapped, np.float32(0.0), np.float32(1.0), out=mapped)
        normalized[name] = mapped

    details = {
        "policy": "explicit linear pixel-disparity mapping",
        "constant_map": bool(upper == lower),
        "purpose": "display and normalized displacement input",
        "source_units": "pixels",
        "output_units": "normalized 0..1",
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "lower_bound_pixels_float32": float(lower),
        "upper_bound_pixels_float32": float(upper),
        "formula": (
            "finite samples map to zero; NaN remains NaN" if upper == lower else
            "clip((height_disparity_pixels - lower_bound) / (upper_bound - lower_bound), 0, 1)"
        ),
        "percentile_method": "NumPy linear",
        "shared_bounds_across_maps": list(arrays),
        "finite_input_counts": finite_counts,
        "nonfinite_policy": "NaN remains NaN; negative/positive infinity clips to 0/1",
        "scientific_pixel_disparity_replaced": False,
        "precision_scope": (
            "This is an explicitly normalized derivative for display/displacement convenience. "
            "The separate pixel-disparity EXR/NPY remains the calibrated algorithm output."
        ),
    }
    return normalized, details


def _stereo_search_range(width: int, requested: int | None) -> int:
    if requested is None:
        requested = max(16, math.ceil(width / 8))
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise StereoMatchingError("StereoSGBM maximum disparity must be an integer")
    if requested <= 0:
        raise StereoMatchingError("StereoSGBM maximum disparity must be positive")
    disparities = math.ceil(requested / 16) * 16
    if disparities >= width:
        raise StereoMatchingError(
            f"StereoSGBM disparity search range {disparities} must be smaller than image width {width}"
        )
    if disparities > 2048:
        raise StereoMatchingError(
            "StereoSGBM disparity search range exceeds the signed fixed-point output limit"
        )
    return disparities


def register_stereo_rows(
    left: np.ndarray, right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Correct small, evidenced vertical registration errors for inference only.

    Fit y_left = a*x_right + b*y_right + c; x_right is never changed.
    A general homography would also alter horizontal disparity and invalidate
    the supplied focal length/baseline. Keep the left reference and raw assets.
    """
    import cv2

    height, width = left.shape[:2]
    valid = np.ones((height, width), dtype=bool)
    details: dict[str, Any] = {
        "applied": False, "raw_assets_modified": False,
        "horizontal_coordinates_changed": False, "reference_view": "left",
        "method": "SIFT ratio matches; deterministic RANSAC vertical affine fit",
        "reason": "insufficient well-distributed feature support",
    }
    detector = cv2.SIFT_create(nfeatures=8000)
    key_l, desc_l = detector.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), None)
    key_r, desc_r = detector.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY), None)
    if desc_l is None or desc_r is None or len(desc_r) < 2:
        return right, valid, details
    pairs = cv2.BFMatcher().knnMatch(desc_l, desc_r, k=2)
    # Reject ambiguous descriptors and duplicate target features.
    matches = sorted((p[0] for p in pairs if len(p) == 2 and p[0].distance < .65 * p[1].distance),
                     key=lambda m: m.distance)
    unique: dict[int, Any] = {}
    for match in matches:
        unique.setdefault(match.trainIdx, match)
    if len(unique) < 16:
        return right, valid, details
    pl = np.array([key_l[m.queryIdx].pt for m in unique.values()], dtype=np.float64)
    pr = np.array([key_r[m.trainIdx].pt for m in unique.values()], dtype=np.float64)
    max_shift = max(8.0, height * .01)
    keep = (np.abs(pl[:, 1] - pr[:, 1]) < max_shift) & (np.abs(pl[:, 0] - pr[:, 0]) < width * .5)
    pl, pr = pl[keep], pr[keep]
    details["feature_match_count"] = len(pl)
    if len(pl) < 16:
        return right, valid, details
    design = np.column_stack((pr[:, 0] / width, pr[:, 1] / height, np.ones(len(pr))))
    delta = pl[:, 1] - pr[:, 1]
    best = np.zeros(len(pr), dtype=bool)
    generator = np.random.default_rng(0)
    for _ in range(512):
        indices = generator.choice(len(pr), 3, replace=False)
        try:
            fit = np.linalg.solve(design[indices], delta[indices])
        except np.linalg.LinAlgError:
            continue
        inliers = np.abs(design @ fit - delta) < 1.0
        if np.count_nonzero(inliers) > np.count_nonzero(best):
            best = inliers
    count = int(np.count_nonzero(best))
    if count < max(16, math.ceil(.6 * len(pr))):
        return right, valid, details
    if np.any(np.ptp(pr[best], axis=0) < np.array([width, height]) * .25):
        return right, valid, details
    fit = np.linalg.lstsq(design[best], delta[best], rcond=None)[0]
    residual = np.abs(design[best] @ fit - delta[best])
    corners = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]) @ fit
    details.update({
        "inlier_count": count,
        "median_vertical_error_before_pixels": float(np.median(np.abs(delta[best]))),
        "median_vertical_error_after_pixels": float(np.median(residual)),
        "p90_vertical_error_after_pixels": float(np.percentile(residual, 90)),
    })
    if (np.max(np.abs(corners)) > max_shift or abs(fit[0] / width) > .01
            or abs(fit[1] / height) > .01 or np.median(residual) > .5):
        details["reason"] = "vertical fit exceeded the residual or small-correction limit"
        return right, valid, details
    if np.max(np.abs(corners)) < .25:
        details["reason"] = "already aligned within a quarter pixel"
        return right, valid, details
    matrix = np.array([[1., 0., 0.], [fit[0] / width, 1. + fit[1] / height, fit[2]]])
    aligned = cv2.warpAffine(right, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    # Only accept correspondences whose interpolation footprint is in the source.
    yy, xx = np.indices((height, width), dtype=np.float32)
    source_y = (yy - matrix[1, 0] * xx - matrix[1, 2]) / matrix[1, 1]
    valid = (source_y >= 0) & (source_y <= height - 1)
    details.update({
        "applied": True, "reason": "measured vertical misregistration",
        "right_to_aligned_affine": matrix.tolist(),
        "interpolation": "OpenCV INTER_LINEAR; uint8 inference input only",
        "invalid_border_pixels": int(np.count_nonzero(~valid)),
    })
    return aligned, valid, details


def correspondence_validity(signed_flow: np.ndarray, right_valid: np.ndarray) -> np.ndarray:
    """Reject off-image matches and interpolation across invalid registered borders."""
    height, width = signed_flow.shape
    xr = np.arange(width, dtype=np.float32)[None, :] + signed_flow
    valid = np.isfinite(xr) & (xr >= 0) & (xr <= width - 1)
    safe_x = np.where(valid, xr, 0)
    lo = np.floor(safe_x).astype(np.intp)
    hi = np.ceil(safe_x).astype(np.intp)
    rows = np.arange(height)[:, None]
    return valid & right_valid[rows, lo] & right_valid[rows, hi]


def stereo_photometric_support(
    left: np.ndarray, right: np.ndarray, disparity: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject unsupported classical matches without changing accepted disparities.

    Bidirectional SGBM can agree on a false match in flat patches or along a
    single edge (the aperture problem). Require local contrast in two directions
    in both views and a positive, exposure-independent patch correlation.
    """
    import cv2

    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float32)
    yy, xx = np.indices(disparity.shape, dtype=np.float32)
    xr = xx - np.where(np.isfinite(disparity), disparity, 0)
    warped = cv2.remap(right_gray, xr, yy, cv2.INTER_LINEAR)
    # Float64 moments avoid catastrophic cancellation on nearly flat uint8 data.
    a, b = left_gray.astype(np.float64), warped.astype(np.float64)
    def mean(value: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(value, -1, (9, 9), normalize=True)
    ma, mb = mean(a), mean(b)
    va = np.maximum(0, mean(a * a) - ma * ma)
    vb = np.maximum(0, mean(b * b) - mb * mb)
    covariance = mean(a * b) - ma * mb
    correlation = covariance / np.sqrt(np.maximum(va * vb, 1e-12))
    left_texture = cv2.cornerMinEigenVal(left_gray, 9, 3)
    right_texture = cv2.remap(cv2.cornerMinEigenVal(right_gray, 9, 3), xr, yy, cv2.INTER_LINEAR)
    supported = ((correlation >= .8) & (left_texture >= 1.0) & (right_texture >= 1.0)
                 & np.isfinite(disparity) & (xr >= 4) & (xr <= disparity.shape[1] - 5))
    supported[:4] = False
    supported[-4:] = False
    supported[:, :4] = False
    supported[:, -4:] = False
    return supported, {
        "method": "9x9 normalized patch correlation and two-direction structure tensor",
        "minimum_correlation": .8, "minimum_structure_eigenvalue_code_squared": 1.0,
        "policy": "unsupported estimates become NaN; accepted disparities are never smoothed or rescaled",
    }


def reverse_correspondence_support(forward: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    """Check x_right=x_left+forward and x_left=x_right+reverse at both neighbors."""
    xr = np.arange(forward.shape[1], dtype=np.float32)[None, :] + forward
    in_bounds = np.isfinite(xr) & (xr >= 0) & (xr <= forward.shape[1] - 1)
    safe_x = np.where(in_bounds, xr, 0)
    lo, hi = np.floor(safe_x).astype(np.intp), np.ceil(safe_x).astype(np.intp)
    rows = np.arange(forward.shape[0])[:, None]
    return (in_bounds & (np.abs(forward + reverse[rows, lo]) <= 1.0)
            & (np.abs(forward + reverse[rows, hi]) <= 1.0))


def linear_depth_displacement(
    disparity: np.ndarray, spatial: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Explicit near-high 0..1 derivative, linear in camera-axis distance, not 1/Z."""
    _, depth = derive_raft_height_and_depth(-disparity, {
        **spatial, "principal_point_delta_x_pixels": 0.0,
    })
    finite = np.isfinite(depth) & (depth > 0)
    if not np.any(finite):
        raise DisplacementMappingError("no finite positive depth for linear displacement")
    near, far = np.min(depth[finite]), np.max(depth[finite])
    span = np.float32(far - near)
    mapped = np.full(depth.shape, np.nan, dtype=np.float32)
    mapped[finite] = (far - depth[finite]) / (span if span > 0 else np.float32(1))
    details = {
        "policy": "explicit linear camera-axis depth mapping",
        "source_units": "pixels", "intermediate_units": "meters", "output_units": "normalized 0..1",
        "formula": "Z = float32(focal_px * baseline_m) / disparity; (far_m - Z) / (far_m - near_m)",
        "near_depth_meters_float32": float(near), "far_depth_meters_float32": float(far),
        "displacement_scale_meters_float32": float(span),
        "constant_map": bool(span == 0), "finite_input_count": int(np.count_nonzero(finite)),
        "nonfinite_policy": "nonpositive disparity and nonfinite depth remain NaN; no hole filling",
        "scientific_pixel_disparity_replaced": False,
        "precision_scope": "Explicit float32 derivative; linear in Z, preserves finite range without percentile clipping",
    }
    return mapped, details


def run_stereo_matching(
    left: np.ndarray,
    right: np.ndarray,
    spatial: Mapping[str, Any],
    options: StereoMatchingOptions | None = None,
) -> StereoMatchingResult:
    """Compute a full-resolution classical Semi-Global Block Matching height map."""
    settings = options or StereoMatchingOptions()
    left_array, right_array = _validate_stereo_matching_inputs(left, right, spatial)
    if (
        isinstance(settings.block_size, bool)
        or not isinstance(settings.block_size, int)
        or not 3 <= settings.block_size <= 21
        or settings.block_size % 2 == 0
    ):
        raise StereoMatchingError("StereoSGBM block size must be an odd integer in [3, 21]")
    num_disparities = _stereo_search_range(left_array.shape[1], settings.maximum_disparity)

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StereoMatchingError(
            "classical stereo matching requires opencv-python-headless>=4.13"
        ) from exc

    right_array, right_valid, registration = register_stereo_rows(left_array, right_array)

    channels = int(left_array.shape[2])
    block_area = settings.block_size * settings.block_size
    parameters = {
        "minDisparity": 0,
        "numDisparities": num_disparities,
        "blockSize": settings.block_size,
        "P1": 8 * channels * block_area,
        "P2": 32 * channels * block_area,
        "disp12MaxDiff": 1,
        "preFilterCap": 31,
        "uniquenessRatio": 5,
        "speckleWindowSize": 50,
        "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    }
    try:
        matcher = cv2.StereoSGBM.create(**parameters)
        fixed_disparity = np.asarray(matcher.compute(left_array, right_array))
        # OpenCV's disp12 check reuses the forward cost; independently match
        # the reverse pair as well. Flipping keeps the same positive search range.
        reverse_fixed = matcher.compute(np.ascontiguousarray(right_array[:, ::-1]),
                                        np.ascontiguousarray(left_array[:, ::-1]))[:, ::-1]
    except Exception as exc:
        raise StereoMatchingError(
            "OpenCV StereoSGBM failed at full "
            f"{left_array.shape[1]}x{left_array.shape[0]} resolution: {exc}"
        ) from exc
    if fixed_disparity.dtype != np.dtype("int16") or fixed_disparity.shape != left_array.shape[:2]:
        raise StereoMatchingError(
            "OpenCV StereoSGBM returned an unexpected disparity dtype or shape"
        )

    invalid_fixed_value = np.int16(-16)
    valid = fixed_disparity > invalid_fixed_value
    disparity = np.ascontiguousarray(
        fixed_disparity.astype(np.float32) / np.float32(16.0), dtype=np.float32
    )
    reverse = reverse_fixed.astype(np.float32) / np.float32(16.0)
    xr = np.arange(disparity.shape[1], dtype=np.float32)[None, :] - disparity
    # Conservative check of both bracketing pixels; no averaging through a
    # depth discontinuity and no modification of accepted 1/16-pixel estimates.
    lo = np.clip(np.floor(xr), 0, disparity.shape[1] - 1).astype(np.intp)
    hi = np.clip(np.ceil(xr), 0, disparity.shape[1] - 1).astype(np.intp)
    rows = np.arange(disparity.shape[0])[:, None]
    rev_lo, rev_hi = reverse[rows, lo], reverse[rows, hi]
    consistent = ((rev_lo >= 0) & (rev_hi >= 0)
                  & (np.abs(disparity - rev_lo) <= 1) & (np.abs(disparity - rev_hi) <= 1))
    try:
        principal_point_delta = np.float32(spatial["principal_point_delta_x_pixels"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise StereoMatchingError("spatial-photo principal-point calibration is incomplete") from exc
    if not np.isfinite(principal_point_delta):
        raise StereoMatchingError("spatial-photo principal-point calibration is nonfinite")
    height = np.ascontiguousarray(
        disparity + principal_point_delta, dtype=np.float32
    )
    in_bounds = correspondence_validity(-disparity, right_valid)
    photometric, support_details = stereo_photometric_support(left_array, right_array, disparity)
    accepted = valid & (height > 0) & in_bounds & consistent & photometric
    # Consistency rejection can split formerly connected false matches into
    # islands. Filter *after* that rejection, retaining accepted codes exactly.
    checked_fixed = np.where(accepted, fixed_disparity, invalid_fixed_value).astype(np.int16)
    cv2.filterSpeckles(checked_fixed, int(invalid_fixed_value), 200, 32)
    supported = checked_fixed > invalid_fixed_value
    height[~supported] = np.float32(np.nan)
    finite = np.isfinite(height)
    valid_count = int(np.count_nonzero(finite))
    details = {
        "engine": "OpenCV StereoSGBM",
        "algorithm": "classical semi-global block matching",
        "opencv_version": str(cv2.__version__),
        "input_shape": list(left_array.shape),
        "input_dtype": left_array.dtype.name,
        "output_dtype": height.dtype.name,
        "full_decoded_resolution": True,
        "resized_or_tiled": False,
        "color_conversion": False,
        "gamma_correction": False,
        "normalization": False,
        "vertical_registration": registration,
        "photometric_support": support_details,
        "photometrically_unsupported_pixel_count": int(np.count_nonzero(valid & ~photometric)),
        "out_of_view_pixel_count": int(np.count_nonzero(~in_bounds)),
        "left_right_consistency": "independent reverse match; both neighbors within 1 pixel",
        "inconsistent_pixel_count": int(np.count_nonzero(valid & ~consistent)),
        "post_consistency_speckle_filter": {
            "maximum_component_pixels": 200, "neighbor_disparity_difference_pixels": 2,
            "rejected_pixel_count": int(np.count_nonzero(accepted & ~supported)),
            "policy": "reject small disconnected disparity components as NaN; never smooth or fill",
        },
        "parameters": {
            **parameters,
            "mode": "STEREO_SGBM_MODE_SGBM_3WAY",
        },
        "fixed_point_scale": 16,
        "fixed_point_precision_pixels": 1.0 / 16.0,
        "invalid_fixed_point_value": int(invalid_fixed_value),
        "invalid_output_representation": "float32 NaN",
        "valid_pixel_count": valid_count,
        "invalid_pixel_count": int(height.size - valid_count),
        "valid_pixel_fraction": valid_count / int(height.size),
        "height_map_formula": "(stereo_sgbm_fixed_disparity / 16) + (cx_right - cx_left); negative/unmatched is NaN",
        "height_map_units": "pixels",
        "height_map_value_direction": "larger values generally indicate nearer geometry",
        "principal_point_delta_x_pixels": float(principal_point_delta),
        "apple_disparity_adjustment_applied_to_geometry": False,
        "apple_disparity_adjustment_reason": (
            "It is a presentation zero-parallax shift, not an optical calibration term."
        ),
        "precision_scope": (
            "The source RGB arrays remain untouched. The exported float32 values exactly preserve "
            "OpenCV's 1/16-pixel fixed-point StereoSGBM estimate; unmatched pixels are NaN. "
            "This is an inferred correspondence result, not measured source depth."
        ),
    }
    return StereoMatchingResult(height, details)


def derive_raft_height_and_depth(
    signed_flow_pixels: np.ndarray, spatial: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Derive near-high pixel disparity and metric depth in explicit float32."""
    flow = np.asarray(signed_flow_pixels)
    if flow.dtype != np.dtype("float32") or flow.ndim != 2 or flow.size == 0:
        raise RaftStereoError("RAFT signed flow must be a nonempty float32 HxW array")
    try:
        principal_point_delta = np.float32(spatial["principal_point_delta_x_pixels"])
        focal_length = np.float32(spatial["focal_length_pixels_for_depth"])
        baseline = np.float32(spatial["baseline_meters"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RaftStereoError("spatial-photo metric calibration is incomplete") from exc
    if (
        not np.isfinite(principal_point_delta)
        or not np.isfinite(focal_length)
        or not np.isfinite(baseline)
        or focal_length <= np.float32(0.0)
        or baseline <= np.float32(0.0)
    ):
        raise RaftStereoError("spatial-photo metric calibration is not positive and finite")
    height_disparity = np.ascontiguousarray(
        principal_point_delta - flow, dtype=np.float32
    )
    # RAFT predicts x_right - x_left; geometric disparity is the opposite sign.
    # abs() would fold impossible negative disparities into plausible near geometry.
    height_disparity[~np.isfinite(height_disparity) | (height_disparity < 0)] = np.float32(np.nan)
    numerator = np.float32(focal_length * baseline)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_meters = np.ascontiguousarray(
            numerator / height_disparity, dtype=np.float32
        )
    return height_disparity, depth_meters


def run_raft_stereo(
    left: np.ndarray,
    right: np.ndarray,
    spatial: Mapping[str, Any],
    options: RaftStereoOptions,
) -> RaftStereoResult:
    """Infer full-resolution signed flow, height disparity, and metric depth."""
    left_array, right_array = _validate_raft_inputs(
        left, right, spatial, options.iterations
    )
    right_array, right_valid, registration = register_stereo_rows(left_array, right_array)
    root, model_path, model_member = resolve_raft_resources(options)
    checkpoint_bytes, checkpoint_name = _checkpoint_bytes(model_path, model_member)
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()

    try:
        import torch
    except ImportError as exc:
        raise RaftStereoError(
            "PyTorch is required for RAFT-Stereo; install the project with the 'raft' extra"
        ) from exc

    root_text = str(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        from core.raft_stereo import RAFTStereo
        from core.utils.utils import InputPadder
    except Exception as exc:
        raise RaftStereoError(f"could not import RAFT-Stereo from {root}: {exc}") from exc
    finally:
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass

    device = _select_device(torch, options.device)
    configuration = _model_configuration(checkpoint_name)
    try:
        model = RAFTStereo(configuration)
        state = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, Mapping):
            raise RaftStereoError("checkpoint does not contain a model state dictionary")
        state_without_parallel_prefix = {
            str(key).removeprefix("module."): value for key, value in state.items()
        }
        model.load_state_dict(state_without_parallel_prefix, strict=True)
        model.to(device)
        model.eval()

        left_tensor = (
            torch.from_numpy(left_array.copy())
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        right_tensor = (
            torch.from_numpy(right_array.copy())
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        padder = InputPadder(left_tensor.shape, divis_by=32)
        left_padded, right_padded = padder.pad(left_tensor, right_tensor)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.cuda\.amp\.autocast.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"torch\.meshgrid:.*",
                category=UserWarning,
            )
            with torch.inference_mode():
                _, flow = model(
                    left_padded,
                    right_padded,
                    iters=options.iterations,
                    test_mode=True,
                )
                # Mirroring the reverse pair retains the disparity sign expected
                # by pretrained stereo models. Convert back to right coordinates.
                _, reverse_flow = model(
                    torch.flip(right_padded, dims=[3]),
                    torch.flip(left_padded, dims=[3]),
                    iters=options.iterations,
                    test_mode=True,
                )
        unpadded = padder.unpad(flow)
        signed_flow = np.ascontiguousarray(
            unpadded[0, 0].to(device="cpu", dtype=torch.float32).numpy(),
            dtype=np.float32,
        )
        reverse_signed_flow = np.ascontiguousarray(
            -padder.unpad(torch.flip(reverse_flow, dims=[3]))[0, 0]
            .to(device="cpu", dtype=torch.float32).numpy(), dtype=np.float32,
        )
        if signed_flow.shape != left_array.shape[:2] or reverse_signed_flow.shape != signed_flow.shape:
            raise RaftStereoError("RAFT-Stereo returned a flow grid that does not match the full input resolution")
        del (
            flow,
            reverse_flow,
            unpadded,
            left_padded,
            right_padded,
            left_tensor,
            right_tensor,
            model,
            state,
            state_without_parallel_prefix,
        )
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except RaftStereoError:
        raise
    except Exception as exc:
        raise RaftStereoError(
            f"RAFT-Stereo inference failed on {device} at full {left_array.shape[1]}x{left_array.shape[0]} resolution: {exc}"
        ) from exc

    height_disparity, depth_meters = derive_raft_height_and_depth(signed_flow, spatial)
    in_bounds = correspondence_validity(signed_flow, right_valid)
    consistent = reverse_correspondence_support(signed_flow, reverse_signed_flow)
    accepted = in_bounds & consistent & (height_disparity > 0)
    height_disparity[~accepted] = np.float32(np.nan)
    depth_meters[~accepted] = np.float32(np.nan)
    principal_point_delta = np.float32(spatial["principal_point_delta_x_pixels"])
    focal_length = np.float32(spatial["focal_length_pixels_for_depth"])
    baseline = np.float32(spatial["baseline_meters"])

    finite_depth = np.isfinite(depth_meters)
    details = {
        "engine": "RAFT-Stereo",
        "upstream_repository": "https://github.com/princeton-vl/RAFT-Stereo",
        "raft_source_root": str(root),
        "raft_source_commit": _git_head_if_available(root),
        "checkpoint_path": str(model_path),
        "checkpoint_member": model_member,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "torch_version": str(torch.__version__),
        "device": device,
        "iterations": options.iterations,
        "correlation_implementation": configuration.corr_implementation,
        "input_shape": list(left_array.shape),
        "input_dtype": left_array.dtype.name,
        "output_dtype": signed_flow.dtype.name,
        "full_decoded_resolution": True,
        "resized_or_tiled": False,
        "vertical_registration": registration,
        "out_of_view_pixel_count": int(np.count_nonzero(~in_bounds)),
        "left_right_consistency": "independent mirrored reverse inference; both neighbors within 1 pixel",
        "inconsistent_pixel_count": int(np.count_nonzero(in_bounds & ~consistent)),
        "raw_signed_flow_filtered": False,
        "input_left_sha256": _array_sha256(left_array),
        "input_right_sha256": _array_sha256(right_array),
        "model_internal_downsample_factor": 2 ** configuration.n_downsample,
        "output_upsampling": "RAFT learned convex upsampling; full output size does not imply independent per-pixel measurements",
        "signed_flow_semantics": "horizontal x_right - x_left correspondence displacement in pixels",
        "height_map_formula": "(cx_right - cx_left) - signed_flow_pixels; negative/nonfinite is NaN",
        "height_map_units": "pixels",
        "height_map_value_direction": "larger values generally indicate nearer geometry",
        "metric_depth_formula": "float32(focal_length_pixels * baseline_meters) / height_disparity_pixels",
        "metric_depth_units": "meters",
        "metric_depth_value_direction": "smaller values indicate nearer geometry",
        "focal_length_pixels": float(focal_length),
        "baseline_meters": float(baseline),
        "principal_point_delta_x_pixels": float(principal_point_delta),
        "finite_metric_depth_count": int(np.count_nonzero(finite_depth)),
        "nonfinite_metric_depth_count": int(depth_meters.size - np.count_nonzero(finite_depth)),
        "apple_disparity_adjustment_applied_to_geometry": False,
        "apple_disparity_adjustment_reason": (
            "It is a presentation zero-parallax shift, not an optical calibration term."
        ),
        "precision_scope": (
            "The float32 arrays preserve the model output and documented float32 derivations exactly. "
            "RAFT-Stereo is an inferred estimate, not a measured or mathematically exact source depth map."
        ),
    }
    return RaftStereoResult(signed_flow, height_disparity, depth_meters, details)


def _git_head_if_available(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref: "):
            reference = root / ".git" / text[5:]
            if reference.is_file():
                return reference.read_text(encoding="ascii").strip()
            packed = root / ".git" / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="ascii").splitlines():
                    if line and not line.startswith(("#", "^")):
                        value, name = line.split(" ", 1)
                        if name == text[5:]:
                            return value
            return None
        return text or None
    except (OSError, UnicodeError, ValueError):
        return None
