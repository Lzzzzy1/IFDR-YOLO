from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

from ifdr_yolo.data.kitti_types import TRAIN_CLASS_TO_ID


_NON_TRAINING_CLASSES = frozenset(
    {"Van", "Truck", "Person_sitting", "Tram", "Misc", "DontCare"}
)


@dataclass(frozen=True)
class NaturalDegradationRecord:
    image_id: str
    object_id: int
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    box_height: float
    depth_m: float | None
    depth_available: bool
    occlusion_level: int
    truncation: float
    sampling_score: float
    visibility_score: float


@dataclass(frozen=True)
class NaturalDegradationLoadResult:
    records: tuple[NaturalDegradationRecord, ...]
    skipped_non_training_count: int


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_sampling_score(box_height: float, depth_m: float | None) -> float:
    """Compute the natural sampling score from box height and optional depth."""
    height = _finite_number(box_height, "box_height")
    height_score = _clip((64.0 - height) / 60.0, 0.0, 1.0)
    if depth_m is None:
        depth_score = 0.0
    else:
        depth = _finite_number(depth_m, "depth_m")
        depth_score = _clip((depth - 15.0) / 45.0, 0.0, 1.0)
    return 1.0 - (1.0 - height_score) * (1.0 - depth_score)


def compute_visibility_score(occlusion_level: int, truncation: float) -> float:
    """Compute the natural visibility score from occlusion and truncation."""
    if isinstance(occlusion_level, bool) or not isinstance(occlusion_level, int):
        raise ValueError("occlusion_level must be an integer")
    if not 0 <= occlusion_level <= 3:
        raise ValueError("occlusion_level must be within [0, 3]")
    truncation_value = _finite_number(truncation, "truncation")
    if not 0.0 <= truncation_value <= 1.0:
        raise ValueError("truncation must be within [0, 1]")
    occlusion_score = occlusion_level / 3.0
    return 1.0 - (1.0 - occlusion_score) * (1.0 - truncation_value)


def _line_error(line_number: int, message: str) -> None:
    raise ValueError(f"JSONL line {line_number}: {message}")


def _require_text(value: Any, name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        _line_error(line_number, f"{name} must be non-empty text")
    return value


def _require_bbox(value: Any, line_number: int) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        _line_error(line_number, "bbox must be an object")
    try:
        coordinates = tuple(
            _finite_number(value[name], f"bbox.{name}")
            for name in ("x1", "y1", "x2", "y2")
        )
    except KeyError as exc:
        _line_error(line_number, f"bbox is missing {exc.args[0]}")
    except ValueError as exc:
        _line_error(line_number, str(exc))
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        _line_error(line_number, "bbox width and height must be positive")
    return coordinates


def _require_truncation(value: Any, line_number: int) -> float:
    try:
        truncation = _finite_number(value, "truncated")
    except ValueError as exc:
        _line_error(line_number, str(exc))
    if not 0.0 <= truncation <= 1.0:
        _line_error(line_number, "truncated must be within [0, 1]")
    return truncation


def _require_occlusion(value: Any, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _line_error(line_number, "occluded must be an integer")
    if not 0 <= value <= 3:
        _line_error(line_number, "occluded must be within [0, 3]")
    return value


def _optional_location(value: Any, line_number: int) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        _line_error(line_number, "location_xyz must be a numeric sequence of length 3")
    try:
        numbers = tuple(_finite_number(item, "location_xyz") for item in value)
    except ValueError as exc:
        _line_error(line_number, str(exc))
    return numbers


def load_natural_degradation_records(
    jsonl_path: str | Path,
) -> NaturalDegradationLoadResult:
    """Load auditable natural degradation records from KITTI object metadata."""
    records: list[NaturalDegradationRecord] = []
    next_object_ids: dict[str, int] = {}
    skipped_non_training_count = 0

    with Path(jsonl_path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                _line_error(line_number, f"malformed JSON ({exc})")
            if not isinstance(row, dict):
                _line_error(line_number, "JSON object required")

            image_id = _require_text(row.get("image_id"), "image_id", line_number)
            class_name = _require_text(row.get("kind"), "kind", line_number)
            if class_name not in TRAIN_CLASS_TO_ID and class_name not in _NON_TRAINING_CLASSES:
                _line_error(line_number, f"unknown class {class_name!r}")

            bbox_xyxy = _require_bbox(row.get("bbox"), line_number)
            truncation = _require_truncation(row.get("truncated"), line_number)
            occlusion_level = _require_occlusion(row.get("occluded"), line_number)
            location = (
                _optional_location(row["location_xyz"], line_number)
                if "location_xyz" in row
                else None
            )
            depth_m = None if location is None else location[2]
            if class_name in TRAIN_CLASS_TO_ID and depth_m is not None and depth_m <= 0.0:
                _line_error(line_number, "depth_m must be finite and positive")

            if "object_id" in row:
                object_id = row["object_id"]
                if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id < 0:
                    _line_error(line_number, "object_id must be a non-negative integer")
            else:
                object_id = next_object_ids.get(image_id, 0)
            next_object_ids[image_id] = next_object_ids.get(image_id, 0) + 1

            if class_name in _NON_TRAINING_CLASSES:
                skipped_non_training_count += 1
                continue

            box_height = bbox_xyxy[3] - bbox_xyxy[1]
            records.append(
                NaturalDegradationRecord(
                    image_id=image_id,
                    object_id=object_id,
                    class_id=TRAIN_CLASS_TO_ID[class_name],
                    class_name=class_name,
                    bbox_xyxy=bbox_xyxy,
                    box_height=box_height,
                    depth_m=depth_m,
                    depth_available=depth_m is not None,
                    occlusion_level=occlusion_level,
                    truncation=truncation,
                    sampling_score=compute_sampling_score(box_height, depth_m),
                    visibility_score=compute_visibility_score(occlusion_level, truncation),
                )
            )

    return NaturalDegradationLoadResult(
        records=tuple(records),
        skipped_non_training_count=skipped_non_training_count,
    )
