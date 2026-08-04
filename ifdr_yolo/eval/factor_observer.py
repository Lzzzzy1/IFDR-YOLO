"""Deterministic observation manifests and crash-safe JSONL storage.

This module intentionally owns only the geometry and evidence bookkeeping used
by the factor observer.  Model loading/forwarding and intervention rendering
are kept out of this foundation so the manifest can be validated before any
checkpoint is executed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ifdr_yolo.data.natural_degradation import NaturalDegradationRecord


DEFAULT_REQUIRED_NODES = (11, 14, 17, 20, 23, 26)
REGISTERED_SEVERITIES = (0.25, 0.5, 0.75, 1.0)
_FACTORS = ("sampling", "visibility")
_ROLES = ("target", "background")
_INTERVENTION_KINDS = ("natural", "clean", "sampling", "visibility")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    if value != value.lower():
        raise ValueError(f"{name} must be lowercase hexadecimal")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _registered_severity(value: object, name: str) -> float:
    number = _finite(value, name)
    matches = [candidate for candidate in REGISTERED_SEVERITIES if abs(number - candidate) <= 1e-9]
    if len(matches) != 1:
        raise ValueError(f"{name} must be one of the registered severities")
    return matches[0]


def _box(value: object, name: str = "bbox_xyxy") -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    result = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    x1, y1, x2, y2 = result
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have positive area")
    return result


def _box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _condition_sort_key(condition: "ObservationCondition") -> tuple[object, ...]:
    return (
        condition.object_id,
        condition.intervention_factor or "",
        condition.intervention_severity,
        condition.intervention_kind,
        condition.region_role,
    )


def _canonical_pair_id(
    *,
    image_id: str,
    object_id: int,
    factor: str,
    seed: int,
    source_sha256: str,
) -> str:
    return _digest(
        {
            "kind": "pair",
            "image_id": image_id,
            "object_id": object_id,
            "factor": factor,
            "seed": seed,
            "source_sha256": source_sha256,
        }
    )


def _canonical_condition_id(
    *,
    image_id: str,
    object_id: int,
    class_id: int,
    class_name: str,
    bbox_xyxy: Sequence[float],
    region_role: str,
    intervention_kind: str,
    intervention_factor: str | None,
    intervention_severity: float,
    pair_id: str | None,
    source_sha256: str,
    seed: int,
) -> str:
    return _digest(
        {
            "kind": "condition",
            "image_id": image_id,
            "object_id": object_id,
            "class_id": class_id,
            "class_name": class_name,
            "bbox_xyxy": list(bbox_xyxy),
            "region_role": region_role,
            "intervention_kind": intervention_kind,
            "intervention_factor": intervention_factor,
            "intervention_severity": intervention_severity,
            "pair_id": pair_id,
            "source_sha256": source_sha256,
            "seed": seed,
        }
    )


def _canonical_transform_id(
    *,
    image_id: str,
    source_sha256: str,
    seed: int,
    object_id: int,
    intervention_kind: str,
    intervention_factor: str | None,
    intervention_severity: float,
    region_role: str,
    bbox_xyxy: Sequence[float],
) -> str:
    """Return the stable pixel-transform identity for one condition.

    Natural and clean conditions all consume the untouched source image, so
    their transform identity is shared across every ROI/object in the image.
    Non-zero interventions identify every input that can change pixels,
    including the actual applied ROI, while deliberately excluding condition
    bookkeeping such as ``pair_id``.
    """

    if intervention_kind in {"natural", "clean"}:
        payload: dict[str, object] = {
            "kind": "source_transform",
            "image_id": image_id,
            "source_sha256": source_sha256,
        }
    else:
        payload = {
            "kind": "intervention_transform",
            "image_id": image_id,
            "source_sha256": source_sha256,
            "seed": seed,
            "object_id": object_id,
            "factor": intervention_factor,
            "severity": intervention_severity,
            "region_role": region_role,
            "bbox_xyxy": tuple(bbox_xyxy),
        }
    return _digest(payload)


@dataclass(frozen=True)
class LetterboxGeometry:
    """The exact affine geometry used by :func:`letterbox_image`."""

    original_width: int
    original_height: int
    input_size: int
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def __post_init__(self) -> None:
        _integer(self.original_width, "original_width", minimum=1)
        _integer(self.original_height, "original_height", minimum=1)
        _integer(self.input_size, "input_size", minimum=1)
        scale = _finite(self.scale, "scale")
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        _integer(self.resized_width, "resized_width", minimum=1)
        _integer(self.resized_height, "resized_height", minimum=1)
        for name in ("pad_left", "pad_top", "pad_right", "pad_bottom"):
            _integer(getattr(self, name), name)
        if self.resized_width + self.pad_left + self.pad_right != self.input_size:
            raise ValueError("horizontal letterbox geometry does not fill input_size")
        if self.resized_height + self.pad_top + self.pad_bottom != self.input_size:
            raise ValueError("vertical letterbox geometry does not fill input_size")
        object.__setattr__(self, "scale", scale)

    def to_dict(self) -> dict[str, object]:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "input_size": self.input_size,
            "scale": self.scale,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
        }


@dataclass(frozen=True)
class ObservationCondition:
    """One deterministic condition/ROI in an image observation plan."""

    image_id: str
    seed: int
    object_id: int
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    box_height: float
    natural_sampling: float
    natural_visibility: float
    region_role: str
    intervention_kind: str
    intervention_factor: str | None
    intervention_severity: float
    pair_id: str | None
    condition_id: str
    transform_id: str
    source_sha256: str
    matched_background_bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be non-empty text")
        _integer(self.seed, "seed")
        _integer(self.object_id, "object_id")
        _integer(self.class_id, "class_id")
        if self.class_id not in (0, 1, 2):
            raise ValueError("class_id must be one of 0, 1, or 2")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be non-empty text")
        expected_class_name = ("Car", "Pedestrian", "Cyclist")[self.class_id]
        if self.class_name != expected_class_name:
            raise ValueError("class_name must correspond to class_id")
        source = _sha256_hex(self.source_sha256, "source_sha256")
        object.__setattr__(self, "source_sha256", source)
        bbox = _box(self.bbox_xyxy)
        object.__setattr__(self, "bbox_xyxy", bbox)
        box_height = _finite(self.box_height, "box_height")
        if box_height <= 0.0 or not math.isclose(box_height, bbox[3] - bbox[1], rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("box_height must equal bbox height and be positive")
        object.__setattr__(self, "box_height", box_height)
        sampling = _finite(self.natural_sampling, "natural_sampling")
        visibility = _finite(self.natural_visibility, "natural_visibility")
        if not 0.0 <= sampling <= 1.0 or not 0.0 <= visibility <= 1.0:
            raise ValueError("natural targets must be within [0, 1]")
        object.__setattr__(self, "natural_sampling", sampling)
        object.__setattr__(self, "natural_visibility", visibility)
        if self.region_role not in _ROLES:
            raise ValueError("region_role must be target or background")
        if self.intervention_kind not in _INTERVENTION_KINDS:
            raise ValueError("invalid intervention_kind")
        if self.intervention_factor is not None and self.intervention_factor not in _FACTORS:
            raise ValueError("intervention_factor must be sampling or visibility")
        severity = _finite(self.intervention_severity, "intervention_severity")
        if severity == 0.0:
            canonical_severity = 0.0
        else:
            canonical_severity = _registered_severity(severity, "intervention_severity")
        severity = canonical_severity
        object.__setattr__(self, "intervention_severity", severity)
        matched_background = (
            None
            if self.matched_background_bbox is None
            else _box(self.matched_background_bbox, "matched_background_bbox")
        )
        object.__setattr__(self, "matched_background_bbox", matched_background)
        if self.intervention_kind == "natural":
            if self.intervention_factor is not None or severity != 0.0:
                raise ValueError("natural conditions do not have intervention metadata")
            if self.region_role != "target" or self.pair_id is not None or matched_background is not None:
                raise ValueError("natural conditions are target-only and unpaired")
        else:
            if self.intervention_factor not in _FACTORS:
                raise ValueError("controlled conditions require an intervention factor")
            if not isinstance(self.pair_id, str):
                raise ValueError("controlled conditions require a pair_id")
            _sha256_hex(self.pair_id, "pair_id")
            expected_pair_id = _canonical_pair_id(
                image_id=self.image_id,
                object_id=self.object_id,
                factor=self.intervention_factor,
                seed=self.seed,
                source_sha256=source,
            )
            if self.pair_id != expected_pair_id:
                raise ValueError("pair_id is inconsistent with condition metadata")
            if matched_background is None:
                raise ValueError("controlled conditions require a matched_background_bbox")
            if self.intervention_kind == "clean":
                if severity != 0.0:
                    raise ValueError("clean conditions must have severity 0")
            elif self.intervention_kind in _FACTORS:
                if self.intervention_kind != self.intervention_factor:
                    raise ValueError("intervention_kind must match intervention_factor")
                if severity <= 0.0:
                    raise ValueError("factor intervention severity must be positive")
            if self.region_role == "background" and self.bbox_xyxy != matched_background:
                raise ValueError("background bbox must equal matched_background_bbox")
        _sha256_hex(self.condition_id, "condition_id")
        expected_condition_id = _canonical_condition_id(
            image_id=self.image_id,
            object_id=self.object_id,
            class_id=self.class_id,
            class_name=self.class_name,
            bbox_xyxy=bbox,
            region_role=self.region_role,
            intervention_kind=self.intervention_kind,
            intervention_factor=self.intervention_factor,
            intervention_severity=severity,
            pair_id=self.pair_id,
            source_sha256=source,
            seed=self.seed,
        )
        if self.condition_id != expected_condition_id:
            raise ValueError("condition_id is inconsistent with condition metadata")
        _sha256_hex(self.transform_id, "transform_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "seed": self.seed,
            "object_id": self.object_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_xyxy": list(self.bbox_xyxy),
            "box_height": self.box_height,
            "natural_sampling": self.natural_sampling,
            "natural_visibility": self.natural_visibility,
            "region_role": self.region_role,
            "intervention_kind": self.intervention_kind,
            "intervention_factor": self.intervention_factor,
            "intervention_severity": self.intervention_severity,
            "pair_id": self.pair_id,
            "condition_id": self.condition_id,
            "transform_id": self.transform_id,
            "source_sha256": self.source_sha256,
            "matched_background_bbox": (
                None if self.matched_background_bbox is None else list(self.matched_background_bbox)
            ),
        }


@dataclass(frozen=True)
class ImageObservationPlan:
    image_id: str
    image_path: str
    width: int
    height: int
    source_sha256: str
    conditions: tuple[ObservationCondition, ...]
    expected_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be non-empty text")
        if not isinstance(self.image_path, str) or not self.image_path:
            raise ValueError("image_path must be non-empty text")
        _integer(self.width, "width", minimum=1)
        _integer(self.height, "height", minimum=1)
        source = _sha256_hex(self.source_sha256, "source_sha256")
        object.__setattr__(self, "source_sha256", source)
        conditions = tuple(self.conditions)
        if not conditions:
            raise ValueError("an image observation plan must contain conditions")
        if any(condition.image_id != self.image_id for condition in conditions):
            raise ValueError("condition image_id does not match plan")
        if any(condition.source_sha256 != source for condition in conditions):
            raise ValueError("condition source_sha256 does not match plan")
        if conditions != tuple(sorted(conditions, key=_condition_sort_key)):
            raise ValueError("conditions are not in canonical order")
        for condition in conditions:
            x1, y1, x2, y2 = condition.bbox_xyxy
            if x1 < 0.0 or y1 < 0.0 or x2 > self.width or y2 > self.height:
                raise ValueError("condition bbox is outside image bounds")
            if condition.matched_background_bbox is not None:
                bx1, by1, bx2, by2 = condition.matched_background_bbox
                if bx1 < 0.0 or by1 < 0.0 or bx2 > self.width or by2 > self.height:
                    raise ValueError("matched background bbox is outside image bounds")
        condition_ids = [condition.condition_id for condition in conditions]
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError("duplicate condition identity")
        expected = tuple(self.expected_observation_ids)
        if len(set(expected)) != len(expected):
            raise ValueError("expected_observation_ids must be unique SHA-256 digests")
        for observation_id in expected:
            _sha256_hex(observation_id, "observation_id")
        natural_by_object: dict[int, ObservationCondition] = {}
        for condition in conditions:
            if condition.intervention_kind == "natural":
                if condition.object_id in natural_by_object:
                    raise ValueError("each object must have one natural condition")
                natural_by_object[condition.object_id] = condition
        if not natural_by_object:
            raise ValueError("plan must contain natural conditions")
        for condition in conditions:
            natural = natural_by_object.get(condition.object_id)
            if natural is None:
                raise ValueError("controlled condition has no natural object condition")
            if (
                condition.seed != natural.seed
                or condition.class_id != natural.class_id
                or condition.class_name != natural.class_name
                or condition.box_height != natural.box_height
                or condition.natural_sampling != natural.natural_sampling
                or condition.natural_visibility != natural.natural_visibility
            ):
                raise ValueError("condition metadata is inconsistent with natural condition")
            if condition.intervention_kind != "natural":
                if condition.region_role == "target" and condition.bbox_xyxy != natural.bbox_xyxy:
                    raise ValueError("controlled target bbox is inconsistent with natural bbox")
                if condition.region_role == "background" and condition.bbox_xyxy != condition.matched_background_bbox:
                    raise ValueError("controlled background bbox is inconsistent with matched background")
        natural_boxes = [condition.bbox_xyxy for condition in natural_by_object.values()]
        pairs: dict[str, list[ObservationCondition]] = {}
        for condition in conditions:
            if condition.intervention_kind != "natural":
                assert condition.pair_id is not None
                pairs.setdefault(condition.pair_id, []).append(condition)
        for pair_id, pair_conditions in pairs.items():
            factors = {condition.intervention_factor for condition in pair_conditions}
            objects = {condition.object_id for condition in pair_conditions}
            matched = {condition.matched_background_bbox for condition in pair_conditions}
            if len(factors) != 1 or len(objects) != 1 or len(matched) != 1:
                raise ValueError("controlled pair background metadata is inconsistent")
            factor = next(iter(factors))
            background = next(iter(matched))
            assert factor is not None and background is not None
            target_boxes = {condition.bbox_xyxy for condition in pair_conditions if condition.region_role == "target"}
            if len(target_boxes) != 1:
                raise ValueError("controlled pair must have one target ROI")
            target = next(iter(target_boxes))
            if not math.isclose(background[2] - background[0], target[2] - target[0], rel_tol=0.0, abs_tol=1e-6) or not math.isclose(background[3] - background[1], target[3] - target[1], rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("matched background must match target dimensions")
            if any(_box_iou(background, natural_box) > 0.05 + 1e-12 for natural_box in natural_boxes):
                raise ValueError("matched background IoU exceeds 0.05")
            expected_pair = {
                ("clean", role, 0.0)
                for role in _ROLES
            }
            expected_pair.update(
                (factor, role, severity)
                for role in _ROLES
                for severity in REGISTERED_SEVERITIES
            )
            actual_pair = {
                (condition.intervention_kind, condition.region_role, condition.intervention_severity)
                for condition in pair_conditions
            }
            if len(pair_conditions) != 10 or actual_pair != expected_pair:
                raise ValueError("controlled pair must contain clean plus four severities per role")
        expected_for_plan = tuple(
            _digest({"condition_id": condition.condition_id, "node_id": node})
            for condition in conditions
            for node in DEFAULT_REQUIRED_NODES
        )
        if expected != expected_for_plan:
            raise ValueError("expected_observation_ids do not match conditions")
        for condition in conditions:
            expected_transform_id = _canonical_transform_id(
                image_id=condition.image_id,
                source_sha256=condition.source_sha256,
                seed=condition.seed,
                object_id=condition.object_id,
                intervention_kind=condition.intervention_kind,
                intervention_factor=condition.intervention_factor,
                intervention_severity=condition.intervention_severity,
                region_role=condition.region_role,
                bbox_xyxy=condition.bbox_xyxy,
            )
            if condition.transform_id != expected_transform_id:
                raise ValueError("condition transform_id is inconsistent with its pixel transform")
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "expected_observation_ids", expected)

    @property
    def expected_condition_ids(self) -> tuple[str, ...]:
        return tuple(condition.condition_id for condition in self.conditions)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "source_sha256": self.source_sha256,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "expected_condition_ids": list(self.expected_condition_ids),
            "expected_observation_ids": list(self.expected_observation_ids),
        }


@dataclass(frozen=True)
class FactorObservationManifest:
    plans: tuple[ImageObservationPlan, ...]
    checkpoint_sha256: str
    seed: int
    required_nodes: tuple[int, ...] = DEFAULT_REQUIRED_NODES
    input_size: int = 640

    def __post_init__(self) -> None:
        plans = tuple(self.plans)
        if not plans:
            raise ValueError("manifest must contain at least one image plan")
        image_ids = [plan.image_id for plan in plans]
        if image_ids != sorted(image_ids) or len(set(image_ids)) != len(image_ids):
            raise ValueError("plans must be sorted by unique image_id")
        nodes = tuple(_integer(node, "required node") for node in self.required_nodes)
        if nodes != DEFAULT_REQUIRED_NODES:
            raise ValueError(f"required_nodes must equal {DEFAULT_REQUIRED_NODES}")
        object.__setattr__(self, "plans", plans)
        object.__setattr__(self, "required_nodes", nodes)
        object.__setattr__(self, "checkpoint_sha256", _sha256_hex(self.checkpoint_sha256, "checkpoint_sha256"))
        manifest_seed = _integer(self.seed, "seed")
        object.__setattr__(self, "seed", manifest_seed)
        object.__setattr__(self, "input_size", _integer(self.input_size, "input_size", minimum=1))
        for plan in plans:
            if any(condition.seed != manifest_seed for condition in plan.conditions):
                raise ValueError(f"condition seed does not match manifest seed for {plan.image_id}")
            expected_for_plan = tuple(
                _digest({"condition_id": condition.condition_id, "node_id": node})
                for condition in plan.conditions
                for node in nodes
            )
            if plan.expected_observation_ids != expected_for_plan:
                raise ValueError(f"expected observation IDs do not match conditions for {plan.image_id}")
        expected = self.expected_observation_ids
        if len(set(expected)) != len(expected):
            raise ValueError("manifest contains duplicate expected observation IDs")

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(plan.image_id for plan in self.plans)

    @property
    def expected_condition_ids(self) -> tuple[str, ...]:
        return tuple(condition.condition_id for plan in self.plans for condition in plan.conditions)

    @property
    def expected_observation_ids(self) -> tuple[str, ...]:
        return tuple(observation_id for plan in self.plans for observation_id in plan.expected_observation_ids)

    @property
    def expected_observation_count(self) -> int:
        return len(self.expected_observation_ids)

    @property
    def manifest_sha256(self) -> str:
        return self.hash()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "checkpoint_sha256": self.checkpoint_sha256,
            "seed": self.seed,
            "registered_severities": list(REGISTERED_SEVERITIES),
            "required_nodes": list(self.required_nodes),
            "input_size": self.input_size,
            "plans": [plan.to_dict() for plan in self.plans],
            "expected_condition_ids": list(self.expected_condition_ids),
            "expected_observation_ids": list(self.expected_observation_ids),
            "expected_observation_count": self.expected_observation_count,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def _read_png(path: Path) -> tuple[int, int, str]:
    if not path.is_file():
        raise ValueError(f"image path does not exist: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read image path: {path}") from exc
    if not raw:
        raise ValueError(f"image is empty: {path}")
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[0] <= 0 or decoded.shape[1] <= 0:
        raise ValueError(f"image is not a readable PNG: {path}")
    return int(decoded.shape[1]), int(decoded.shape[0]), hashlib.sha256(raw).hexdigest()


def _background_bbox(
    *,
    image_id: str,
    object_id: int,
    target: tuple[float, float, float, float],
    boxes: Sequence[tuple[float, float, float, float]],
    width: int,
    height: int,
    seed: int,
) -> tuple[float, float, float, float]:
    box_width = target[2] - target[0]
    box_height = target[3] - target[1]
    if box_width > width or box_height > height:
        raise ValueError(f"no valid matched background bbox for {image_id}/{object_id}")
    x_max = width - box_width
    y_max = height - box_height
    candidates: set[tuple[float, float, float, float]] = set()
    # A fixed nine-by-nine grid covers corners and central regions.  Coordinates
    # are rounded to six decimals so equivalent floating candidates collapse.
    for y_index in range(9):
        y = min(y_max, max(0.0, round(y_max * y_index / 8.0, 6)))
        for x_index in range(9):
            x = min(x_max, max(0.0, round(x_max * x_index / 8.0, 6)))
            candidates.add((x, y, x + box_width, y + box_height))
    ordered = sorted(
        candidates,
        key=lambda candidate: _digest(
            {"image_id": image_id, "object_id": object_id, "seed": seed, "bbox": candidate}
        ),
    )
    for candidate in ordered:
        if all(_box_iou(candidate, existing) <= 0.05 + 1e-12 for existing in boxes):
            return candidate
    raise ValueError(f"no valid matched background bbox for {image_id}/{object_id}")


def _condition(
    *,
    record: NaturalDegradationRecord,
    source_sha256: str,
    region_role: str,
    bbox_xyxy: tuple[float, float, float, float],
    matched_background_bbox: tuple[float, float, float, float] | None,
    intervention_kind: str,
    intervention_factor: str | None,
    intervention_severity: float,
    pair_id: str | None,
    seed: int,
) -> ObservationCondition:
    condition_id = _canonical_condition_id(
        image_id=record.image_id,
        object_id=record.object_id,
        class_id=record.class_id,
        class_name=record.class_name,
        bbox_xyxy=bbox_xyxy,
        region_role=region_role,
        intervention_kind=intervention_kind,
        intervention_factor=intervention_factor,
        intervention_severity=intervention_severity,
        pair_id=pair_id,
        source_sha256=source_sha256,
        seed=seed,
    )
    transform_id = _canonical_transform_id(
        image_id=record.image_id,
        source_sha256=source_sha256,
        seed=seed,
        object_id=record.object_id,
        intervention_kind=intervention_kind,
        intervention_factor=intervention_factor,
        intervention_severity=intervention_severity,
        region_role=region_role,
        bbox_xyxy=bbox_xyxy,
    )
    return ObservationCondition(
        image_id=record.image_id,
        seed=seed,
        object_id=record.object_id,
        class_id=record.class_id,
        class_name=record.class_name,
        bbox_xyxy=bbox_xyxy,
        box_height=float(record.box_height),
        natural_sampling=float(record.sampling_score),
        natural_visibility=float(record.visibility_score),
        region_role=region_role,
        intervention_kind=intervention_kind,
        intervention_factor=intervention_factor,
        intervention_severity=intervention_severity,
        pair_id=pair_id,
        condition_id=condition_id,
        transform_id=transform_id,
        source_sha256=source_sha256,
        matched_background_bbox=matched_background_bbox,
    )


def build_factor_observation_manifest(
    records: Iterable[NaturalDegradationRecord],
    image_paths: Mapping[str, str | Path],
    selected_intervention_objects: Iterable[tuple[str, int]],
    checkpoint_sha256: str,
    seed: int,
    required_nodes: Sequence[int] = DEFAULT_REQUIRED_NODES,
    input_size: int = 640,
) -> FactorObservationManifest:
    """Build a deterministic, immutable expected-condition manifest."""

    checkpoint_sha256 = _sha256_hex(checkpoint_sha256, "checkpoint_sha256")
    seed = _integer(seed, "seed")
    input_size = _integer(input_size, "input_size", minimum=1)
    nodes = tuple(_integer(node, "required node") for node in required_nodes)
    if nodes != DEFAULT_REQUIRED_NODES:
        raise ValueError(f"required_nodes must equal {DEFAULT_REQUIRED_NODES}")
    if not isinstance(image_paths, Mapping):
        raise ValueError("image_paths must be a mapping from image id to PNG path")

    normalized_records = tuple(records)
    if not normalized_records:
        raise ValueError("records must not be empty")
    by_image: dict[str, list[NaturalDegradationRecord]] = {}
    identities: set[tuple[str, int]] = set()
    for record in normalized_records:
        if not isinstance(record, NaturalDegradationRecord):
            raise ValueError("records must contain NaturalDegradationRecord values")
        identity = (record.image_id, record.object_id)
        if identity in identities:
            raise ValueError(f"duplicate object identity: {identity}")
        identities.add(identity)
        _integer(record.object_id, "object_id")
        _integer(record.class_id, "class_id")
        bbox = _box(record.bbox_xyxy)
        if not math.isclose(record.box_height, bbox[3] - bbox[1], rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"invalid box_height for {identity}")
        _finite(record.sampling_score, "sampling_score")
        _finite(record.visibility_score, "visibility_score")
        if not 0.0 <= record.sampling_score <= 1.0 or not 0.0 <= record.visibility_score <= 1.0:
            raise ValueError(f"natural scores must be within [0, 1] for {identity}")
        by_image.setdefault(record.image_id, []).append(record)

    selected: set[tuple[str, int]] = set()
    for value in selected_intervention_objects:
        if isinstance(value, NaturalDegradationRecord):
            identity = (value.image_id, value.object_id)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            identity = (value[0], value[1])
        else:
            raise ValueError("selected_intervention_objects must contain (image_id, object_id) identities")
        if not isinstance(identity[0], str) or isinstance(identity[1], bool) or not isinstance(identity[1], int):
            raise ValueError("selected intervention identity is malformed")
        if identity not in identities:
            raise ValueError(f"selected intervention object is unknown: {identity}")
        if identity in selected:
            raise ValueError(f"duplicate selected intervention object: {identity}")
        selected.add(identity)

    image_meta: dict[str, tuple[int, int, str, str]] = {}
    for image_id in sorted(by_image):
        if image_id not in image_paths:
            raise ValueError(f"missing image path for image_id {image_id!r}")
        path = Path(image_paths[image_id])
        width, height, source_sha = _read_png(path)
        image_meta[image_id] = (width, height, source_sha, str(path))
        for record in by_image[image_id]:
            x1, y1, x2, y2 = _box(record.bbox_xyxy)
            if x1 < 0.0 or y1 < 0.0 or x2 > width or y2 > height:
                raise ValueError(f"bbox outside image for {(image_id, record.object_id)}")

    plans: list[ImageObservationPlan] = []
    for image_id in sorted(by_image):
        width, height, source_sha, image_path = image_meta[image_id]
        image_records = sorted(by_image[image_id], key=lambda item: item.object_id)
        all_boxes = [tuple(record.bbox_xyxy) for record in image_records]
        conditions: list[ObservationCondition] = []
        for record in image_records:
            identity = (record.image_id, record.object_id)
            conditions.append(
                _condition(
                    record=record,
                    source_sha256=source_sha,
                    region_role="target",
                    bbox_xyxy=tuple(record.bbox_xyxy),
                    matched_background_bbox=None,
                    intervention_kind="natural",
                    intervention_factor=None,
                    intervention_severity=0.0,
                    pair_id=None,
                    seed=seed,
                )
            )
            if identity not in selected:
                continue
            background = _background_bbox(
                image_id=image_id,
                object_id=record.object_id,
                target=tuple(record.bbox_xyxy),
                boxes=all_boxes,
                width=width,
                height=height,
                seed=seed,
            )
            for factor in _FACTORS:
                for kind, severity in (("clean", 0.0), *( (factor, level) for level in REGISTERED_SEVERITIES )):
                    pair_id = _canonical_pair_id(
                        image_id=image_id,
                        object_id=record.object_id,
                        factor=factor,
                        seed=seed,
                        source_sha256=source_sha,
                    )
                    for role, bbox in (("target", tuple(record.bbox_xyxy)), ("background", background)):
                        conditions.append(
                            _condition(
                                record=record,
                                source_sha256=source_sha,
                                region_role=role,
                                bbox_xyxy=bbox,
                                matched_background_bbox=background,
                                intervention_kind=kind,
                                intervention_factor=factor,
                                intervention_severity=severity,
                                pair_id=pair_id,
                                seed=seed,
                            )
                        )
        conditions.sort(key=_condition_sort_key)
        expected_ids = tuple(
            _digest({"condition_id": condition.condition_id, "node_id": node})
            for condition in conditions
            for node in nodes
        )
        plans.append(
            ImageObservationPlan(
                image_id=image_id,
                image_path=image_path,
                width=width,
                height=height,
                source_sha256=source_sha,
                conditions=tuple(conditions),
                expected_observation_ids=expected_ids,
            )
        )
    return FactorObservationManifest(
        plans=tuple(plans),
        checkpoint_sha256=checkpoint_sha256,
        seed=seed,
        required_nodes=nodes,
        input_size=input_size,
    )


def letterbox_image(
    image: np.ndarray,
    input_size: int = 640,
) -> tuple[torch.Tensor, LetterboxGeometry]:
    """Convert a BGR uint8 image to a padded RGB tensor and frozen geometry."""

    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] <= 0
        or image.shape[1] <= 0
    ):
        raise ValueError("image must be a non-empty uint8 BGR HWC array")
    input_size = _integer(input_size, "input_size", minimum=1)
    original_height, original_width = int(image.shape[0]), int(image.shape[1])
    scale = min(input_size / original_width, input_size / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    pad_width = input_size - resized_width
    pad_height = input_size - resized_height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    geometry = LetterboxGeometry(
        original_width=original_width,
        original_height=original_height,
        input_size=input_size,
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )
    resized = (
        image.copy()
        if (resized_width, resized_height) == (original_width, original_height)
        else cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    )
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    rgb = np.ascontiguousarray(canvas[..., ::-1])
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
    return tensor, geometry


def map_box_to_feature_roi(
    bbox_xyxy: Sequence[float],
    geometry: LetterboxGeometry,
    feature_height: int | tuple[int, int],
    feature_width: int | None = None,
) -> tuple[int, int, int, int]:
    """Map an original-image box through letterbox geometry to a feature map."""

    if not isinstance(geometry, LetterboxGeometry):
        raise ValueError("geometry must be a LetterboxGeometry")
    if feature_width is None:
        if not isinstance(feature_height, (tuple, list)) or len(feature_height) != 2:
            raise ValueError("feature dimensions must be height and width")
        feature_height, feature_width = feature_height
    height = _integer(feature_height, "feature_height", minimum=1)
    width = _integer(feature_width, "feature_width", minimum=1)
    x1, y1, x2, y2 = _box(bbox_xyxy)
    mapped = (
        x1 * geometry.scale + geometry.pad_left,
        y1 * geometry.scale + geometry.pad_top,
        x2 * geometry.scale + geometry.pad_left,
        y2 * geometry.scale + geometry.pad_top,
    )
    fx1 = math.floor(mapped[0] * width / geometry.input_size)
    fy1 = math.floor(mapped[1] * height / geometry.input_size)
    fx2 = math.ceil(mapped[2] * width / geometry.input_size)
    fy2 = math.ceil(mapped[3] * height / geometry.input_size)
    fx1 = min(max(fx1, 0), width - 1)
    fy1 = min(max(fy1, 0), height - 1)
    fx2 = min(max(fx2, fx1 + 1), width)
    fy2 = min(max(fy2, fy1 + 1), height)
    return int(fx1), int(fy1), int(fx2), int(fy2)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class FactorObservationJournal:
    """Exactly-once JSONL writer with one active writer per output pair.

    Startup and finalization perform full content validation.  Individual
    commits use cached prefix and progress-file signatures for O(1) external
    change detection; same-size middle-file edits are intentionally outside
    the single-writer, non-adversarial commit guarantee.
    """

    def __init__(
        self,
        manifest: FactorObservationManifest,
        output_jsonl: str | Path,
        progress_json: str | Path,
    ) -> None:
        if not isinstance(manifest, FactorObservationManifest):
            raise ValueError("manifest must be a FactorObservationManifest")
        self.manifest = manifest
        self._plan_by_image = {plan.image_id: plan for plan in manifest.plans}
        self.output_path = Path(output_jsonl)
        self.progress_path = Path(progress_json)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists():
            with self.output_path.open("wb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        if self.progress_path.exists():
            try:
                loaded = json.loads(self.progress_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError("progress JSON is malformed") from exc
            if not isinstance(loaded, dict):
                raise ValueError("progress JSON must contain an object")
            self._state: dict[str, Any] = loaded
            self._validate_binding()
        else:
            self._state = {
                "schema_version": 1,
                "manifest_sha256": manifest.hash(),
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "completed": {},
                "inflight": None,
                "status": "running",
            }
            _atomic_write_json(self.progress_path, self._state)
        last_completed_end = self._validate_completed_prefix(
            allow_suffix=self._state.get("inflight") is not None
        )
        if self._state.get("inflight") is not None:
            self._recover_inflight(last_completed_end)
        self._validate_file_and_progress()
        self._committed_end = self.output_path.stat().st_size
        self._progress_signature = self._progress_file_signature()

    def _progress_file_signature(self) -> tuple[int, int, int, int, int]:
        try:
            stat = self.progress_path.stat()
        except OSError as exc:
            raise ValueError("progress JSON is unavailable") from exc
        return (
            int(getattr(stat, "st_dev", 0)),
            int(getattr(stat, "st_ino", 0)),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(getattr(stat, "st_ctime_ns", 0)),
        )

    def _refresh_progress_signature(self) -> None:
        self._progress_signature = self._progress_file_signature()

    def _assert_commit_state(self) -> None:
        if self._state.get("inflight") is not None:
            raise ValueError("cannot commit while another transaction is inflight")
        try:
            output_size = self.output_path.stat().st_size
        except OSError as exc:
            raise ValueError("output JSONL is unavailable") from exc
        if output_size != self._committed_end:
            raise ValueError("output JSONL changed since journal validation")
        if self._progress_file_signature() != self._progress_signature:
            raise ValueError("progress JSON changed since journal validation")

    def _validate_binding(self) -> None:
        if self._state.get("schema_version") != 1:
            raise ValueError("unsupported progress schema_version")
        if self._state.get("manifest_sha256") != self.manifest.hash():
            raise ValueError("progress manifest hash does not match manifest")
        if self._state.get("checkpoint_sha256") != self.manifest.checkpoint_sha256:
            raise ValueError("progress checkpoint hash does not match manifest")
        if not isinstance(self._state.get("completed"), dict):
            raise ValueError("progress completed must be an object")

    @staticmethod
    def _expected_observation_hash(plan: ImageObservationPlan) -> str:
        return hashlib.sha256(
            _canonical_json(list(plan.expected_observation_ids)).encode("utf-8")
        ).hexdigest()

    def _validate_completed_prefix(self, *, allow_suffix: bool) -> int:
        completed = self._state.get("completed")
        if not isinstance(completed, dict):
            raise ValueError("progress completed must be an object")
        size = self.output_path.stat().st_size
        entries: list[tuple[int, int, str, dict[str, Any], ImageObservationPlan]] = []
        expected_entry_fields = {"start_offset", "end_offset", "expected_hash", "rows_sha256", "row_count"}
        for image_id, entry in completed.items():
            plan = self._plan_by_image.get(image_id)
            if plan is None:
                raise ValueError("progress contains an unknown completed image")
            if not isinstance(entry, dict) or set(entry) != expected_entry_fields:
                raise ValueError("completed image entry fields do not match schema")
            start_offset = entry["start_offset"]
            end_offset = entry["end_offset"]
            if (
                isinstance(start_offset, bool)
                or type(start_offset) is not int
                or isinstance(end_offset, bool)
                or type(end_offset) is not int
                or start_offset < 0
                or end_offset < start_offset
            ):
                raise ValueError("completed image offsets are invalid")
            if end_offset > size:
                raise ValueError("completed image extends beyond JSONL")
            expected_hash = _sha256_hex(entry["expected_hash"], "completed expected_hash")
            if expected_hash != self._expected_observation_hash(plan):
                raise ValueError("completed expected_hash does not match manifest")
            rows_sha256 = _sha256_hex(entry["rows_sha256"], "completed rows_sha256")
            row_count = entry["row_count"]
            if (
                isinstance(row_count, bool)
                or type(row_count) is not int
                or row_count != len(plan.expected_observation_ids)
            ):
                raise ValueError("completed row_count does not match manifest")
            entries.append((start_offset, end_offset, image_id, entry, plan))
        entries.sort(key=lambda item: item[0])
        cursor = 0
        with self.output_path.open("rb") as handle:
            for start_offset, end_offset, image_id, entry, plan in entries:
                if start_offset != cursor:
                    raise ValueError("completed image blocks are not contiguous")
                handle.seek(start_offset)
                block = handle.read(end_offset - start_offset)
                if len(block) != end_offset - start_offset:
                    raise ValueError("completed image block is truncated")
                if hashlib.sha256(block).hexdigest() != entry["rows_sha256"]:
                    raise ValueError("completed image rows hash does not match JSONL")
                seen: set[str] = set()
                rows: list[dict[str, Any]] = []
                for raw_line in block.splitlines(keepends=True):
                    if not raw_line.endswith(b"\n"):
                        raise ValueError("completed image contains an unterminated JSONL line")
                    try:
                        row = json.loads(
                            raw_line.decode("utf-8"),
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                        )
                    except (UnicodeDecodeError, ValueError, TypeError) as exc:
                        raise ValueError("completed image contains malformed JSON") from exc
                    if not isinstance(row, dict):
                        raise ValueError("completed image rows must be objects")
                    observation_id = row.get("observation_id")
                    if (
                        not isinstance(observation_id, str)
                        or observation_id not in plan.expected_observation_ids
                        or observation_id in seen
                        or row.get("image_id") != image_id
                    ):
                        raise ValueError("completed image rows do not match manifest identities")
                    seen.add(observation_id)
                    rows.append(row)
                if len(rows) != entry["row_count"] or seen != set(plan.expected_observation_ids):
                    raise ValueError("completed image rows do not match manifest")
                cursor = end_offset
        if allow_suffix:
            if size < cursor:
                raise ValueError("JSONL is shorter than completed prefix")
        elif size != cursor:
            raise ValueError("JSONL contains an uncommitted suffix")
        return cursor

    def _recover_inflight(self, last_completed_end: int) -> None:
        inflight = self._state.get("inflight")
        if not isinstance(inflight, dict):
            raise ValueError("progress inflight must be an object")
        expected_fields = {"image_id", "start_offset", "expected_hash", "expected_row_count"}
        if set(inflight) != expected_fields:
            raise ValueError("progress inflight fields do not match schema")
        image_id = inflight["image_id"]
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("progress inflight image_id is invalid")
        plan = self._plan_by_image.get(image_id)
        if plan is None:
            raise ValueError("progress inflight image_id is unknown")
        if image_id in self._state["completed"]:
            raise ValueError("progress inflight image_id is already completed")
        offset = inflight["start_offset"]
        if isinstance(offset, bool) or type(offset) is not int or offset < 0:
            raise ValueError("progress inflight start_offset is invalid")
        size = self.output_path.stat().st_size
        if offset > size:
            raise ValueError("output JSONL is shorter than inflight start offset")
        if offset != last_completed_end:
            raise ValueError("progress inflight start_offset does not follow completed prefix")
        expected_hash = inflight["expected_hash"]
        if expected_hash != self._expected_observation_hash(plan):
            _sha256_hex(expected_hash, "progress inflight expected_hash")
            raise ValueError("progress inflight expected_hash does not match manifest")
        else:
            _sha256_hex(expected_hash, "progress inflight expected_hash")
        expected_row_count = inflight["expected_row_count"]
        if (
            isinstance(expected_row_count, bool)
            or type(expected_row_count) is not int
            or expected_row_count != len(plan.expected_observation_ids)
        ):
            raise ValueError("progress inflight expected_row_count does not match manifest")
        with self.output_path.open("r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
        self._state["inflight"] = None
        self._state["status"] = "running"
        _atomic_write_json(self.progress_path, self._state)

    def _expected_by_observation(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for plan in self.manifest.plans:
            for observation_id in plan.expected_observation_ids:
                result[observation_id] = plan.image_id
        return result

    def _scan_file(self) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
        expected_by_observation = self._expected_by_observation()
        by_image: dict[str, list[dict[str, Any]]] = {}
        blocks: dict[str, dict[str, Any]] = {}
        seen_observation_ids: set[str] = set()
        with self.output_path.open("rb") as handle:
            offset = 0
            for raw_line in handle:
                end_offset = offset + len(raw_line)
                if not raw_line.endswith(b"\n"):
                    raise ValueError("JSONL contains an unterminated line")
                try:
                    row = json.loads(raw_line.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                except (UnicodeDecodeError, ValueError, TypeError) as exc:
                    raise ValueError("JSONL contains malformed JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be JSON objects")
                observation_id = row.get("observation_id")
                if not isinstance(observation_id, str) or observation_id not in expected_by_observation:
                    raise ValueError("JSONL contains unknown observation_id")
                image_id = row.get("image_id")
                if image_id != expected_by_observation[observation_id]:
                    raise ValueError("JSONL observation image_id does not match manifest")
                if observation_id in seen_observation_ids:
                    raise ValueError("JSONL contains duplicate observation_id")
                seen_observation_ids.add(observation_id)
                by_image.setdefault(image_id, []).append(row)
                if image_id not in blocks:
                    block = {"start_offset": offset, "end_offset": offset, "bytes": bytearray()}
                    blocks[image_id] = block
                else:
                    block = blocks[image_id]
                if block["end_offset"] != offset:
                    raise ValueError("JSONL rows for one image are not contiguous")
                block["end_offset"] = end_offset
                block["bytes"].extend(raw_line)
                offset = end_offset
        for image_id, block in blocks.items():
            block["rows_sha256"] = hashlib.sha256(bytes(block.pop("bytes"))).hexdigest()
        return by_image, blocks

    def _validate_file_and_progress(self) -> None:
        by_image, blocks = self._scan_file()
        completed = self._state["completed"]
        known_images = set(self.manifest.image_ids)
        if any(image_id not in known_images for image_id in completed):
            raise ValueError("progress contains an unknown completed image")
        for image_id, rows in by_image.items():
            if image_id not in completed:
                raise ValueError("JSONL contains rows for an image not marked completed")
            entry = completed[image_id]
            block = blocks[image_id]
            if not isinstance(entry, dict) or entry.get("rows_sha256") != block["rows_sha256"]:
                raise ValueError("completed image rows hash does not match JSONL")
            if entry.get("start_offset") != block["start_offset"] or entry.get("end_offset") != block["end_offset"]:
                raise ValueError("completed image offsets do not match JSONL")
            expected_ids = set(self._plan_by_image[image_id].expected_observation_ids)
            if {row["observation_id"] for row in rows} != expected_ids:
                raise ValueError("completed image rows do not match manifest")
        if self._state.get("status") == "complete":
            expected_images = set(self.manifest.image_ids)
            if set(completed) != expected_images:
                raise ValueError("complete progress is missing an image")

    @staticmethod
    def _invoke_crash_hook(crash_hook: object) -> None:
        if crash_hook is None:
            return
        if not callable(crash_hook):
            raise ValueError("crash_hook must be callable")
        try:
            signature = inspect.signature(crash_hook)
        except (TypeError, ValueError):
            # Some C-level callables have no inspectable signature; the
            # documented hook receives the transaction phase in that case.
            crash_hook("after_append")
            return
        accepts_argument = any(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.VAR_POSITIONAL)
            for parameter in signature.parameters.values()
        )
        if accepts_argument:
            crash_hook("after_append")
        else:
            crash_hook()

    def _canonical_rows(self, image_id: str, rows: Iterable[Mapping[str, object]]) -> tuple[bytes, tuple[str, ...]]:
        materialized = tuple(rows)
        if any(not isinstance(row, Mapping) for row in materialized):
            raise ValueError("rows must contain JSON objects")
        if any(any(not isinstance(key, str) for key in row) for row in materialized):
            raise ValueError("row object keys must be strings")
        observation_ids = [row.get("observation_id") for row in materialized]
        if any(not isinstance(item, str) for item in observation_ids):
            raise ValueError("every row requires an observation_id")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("rows contain duplicate observation_id")
        plan = self._plan_by_image.get(image_id)
        if plan is None:
            raise ValueError(f"unknown image_id: {image_id}")
        expected = set(plan.expected_observation_ids)
        actual = set(observation_ids)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"rows do not match manifest expected identities (missing={missing}, extra={extra})")
        canonical_lines: list[bytes] = []
        for row in sorted(materialized, key=lambda value: str(value["observation_id"])):
            if row.get("image_id") != image_id:
                raise ValueError("row image_id does not match commit image_id")
            try:
                encoded = (_canonical_json(dict(row)) + "\n").encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("rows must be finite JSON objects") from exc
            canonical_lines.append(encoded)
        return b"".join(canonical_lines), tuple(sorted(actual))

    def commit_image(
        self,
        image_id: str,
        rows: Iterable[Mapping[str, object]],
        crash_hook: object = None,
    ) -> bool:
        """Commit exactly one image transaction; return ``False`` when skipped."""

        payload, observation_ids = self._canonical_rows(image_id, rows)
        rows_hash = hashlib.sha256(payload).hexdigest()
        self._assert_commit_state()
        completed = self._state["completed"]
        if image_id in completed:
            entry = completed[image_id]
            if entry.get("rows_sha256") != rows_hash:
                raise ValueError("completed image commit conflicts with existing rows")
            return False
        plan = self._plan_by_image[image_id]
        start_offset = self._committed_end
        expected_hash = self._expected_observation_hash(plan)
        self._state["inflight"] = {
            "image_id": image_id,
            "start_offset": start_offset,
            "expected_hash": expected_hash,
            "expected_row_count": len(observation_ids),
        }
        _atomic_write_json(self.progress_path, self._state)
        self._refresh_progress_signature()
        with self.output_path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._invoke_crash_hook(crash_hook)
        end_offset = start_offset + len(payload)
        completed[image_id] = {
            "start_offset": start_offset,
            "end_offset": end_offset,
            "expected_hash": expected_hash,
            "rows_sha256": rows_hash,
            "row_count": len(observation_ids),
        }
        self._state["inflight"] = None
        self._state["status"] = "running"
        _atomic_write_json(self.progress_path, self._state)
        self._committed_end = end_offset
        self._refresh_progress_signature()
        return True

    def finalize(self) -> dict[str, object]:
        """Require exact manifest identity coverage and atomically mark complete."""

        self._assert_commit_state()
        by_image, _ = self._scan_file()
        expected_by_image = {plan.image_id: set(plan.expected_observation_ids) for plan in self.manifest.plans}
        if set(by_image) != set(expected_by_image):
            missing_images = sorted(set(expected_by_image) - set(by_image))
            extra_images = sorted(set(by_image) - set(expected_by_image))
            raise ValueError(f"finalize image coverage mismatch (missing={missing_images}, extra={extra_images})")
        for image_id, expected in expected_by_image.items():
            actual = {row["observation_id"] for row in by_image[image_id]}
            if actual != expected:
                raise ValueError(f"finalize observation coverage mismatch for image {image_id}")
        self._validate_file_and_progress()
        summary = {
            "status": "complete",
            "manifest_sha256": self.manifest.hash(),
            "checkpoint_sha256": self.manifest.checkpoint_sha256,
            "image_count": len(expected_by_image),
            "condition_count": len(self.manifest.expected_condition_ids),
            "expected_observation_count": self.manifest.expected_observation_count,
            "observed_observation_count": sum(len(rows) for rows in by_image.values()),
        }
        self._assert_commit_state()
        self._state["status"] = "complete"
        self._state["summary"] = summary
        _atomic_write_json(self.progress_path, self._state)
        self._refresh_progress_signature()
        return summary


__all__ = [
    "FactorObservationJournal",
    "FactorObservationManifest",
    "ImageObservationPlan",
    "LetterboxGeometry",
    "ObservationCondition",
    "build_factor_observation_manifest",
    "letterbox_image",
    "map_box_to_feature_roi",
]
