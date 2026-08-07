from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr

from ifdr_yolo.data.interventions.sampler import (
    DeterministicInterventionSampler,
    SamplingPolicy,
)
from ifdr_yolo.data.interventions.schema import InterventionSpec
from ifdr_yolo.data.interventions.targets import factor_target_for_spec
from ifdr_yolo.data.interventions.transforms import apply_intervention
from ifdr_yolo.data.metadata_index import (
    FactorMetadataIndex,
    KittiLabelCandidate,
    match_metadata_object,
)
from ifdr_yolo.data.kitti_types import TRAIN_CLASS_TO_ID
from ifdr_yolo.losses.factor_alignment import ObjectFactorTarget


FACTOR_TARGET_KEY = "ifdr_factor_target"
FACTOR_WEIGHT_KEY = "ifdr_factor_weight"
COUNTERFACTUAL_IMAGE_KEY = "ifdr_counterfactual_img"
COUNTERFACTUAL_DELTA_KEY = "ifdr_counterfactual_delta"
COUNTERFACTUAL_WEIGHT_KEY = "ifdr_counterfactual_weight"
CLEAN_IMAGE_KEY = "ifdr_clean_image"
TARGET_IMAGE_KEY = "ifdr_target_image"
BACKGROUND_IMAGE_KEY = "ifdr_background_image"
BACKGROUND_FACTOR_TARGET_KEY = "ifdr_background_factor_target"
FACTOR_OBJECT_TARGETS_KEY = "ifdr_factor_object_targets"
SPECIFICITY_PAIRS_KEY = "ifdr_specificity_pairs"


@dataclass
class SpecificityRejectionCounter:
    """Explicit, serializable rejection accounting for one data pipeline."""

    counts: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        key = str(reason).strip() or "malformed"
        self.counts[key] = self.counts.get(key, 0) + 1

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    @property
    def rejections(self) -> dict[str, int]:
        return dict(self.counts)

    def count(self, reason: str | None = None) -> int:
        return self.total if reason is None else int(self.counts.get(reason, 0))

    def __getitem__(self, reason: str) -> int:
        return int(self.counts.get(reason, 0))


def _reject(counter: SpecificityRejectionCounter | None, reason: str) -> None:
    if counter is not None:
        if not isinstance(counter, SpecificityRejectionCounter):
            raise ValueError("rejection_counter must be SpecificityRejectionCounter")
        counter.reject(reason)


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _strict_box(value: object, field_name: str = "box") -> tuple[float, float, float, float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"{field_name} must contain four coordinates")
    box = tuple(_finite_float(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    x1, y1, x2, y2 = box
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{field_name} must be finite, ordered, and normalized")
    return box


def _normalize_factor_kind(value: object, field_name: str) -> str:
    value = getattr(value, "value", value)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be sampling, visibility, or identity")
    normalized = value.strip().lower()
    if normalized not in {"sampling", "visibility"}:
        raise ValueError(f"{field_name} must be sampling or visibility")
    return normalized


@dataclass(frozen=True)
class SpecificityPair:
    """Minimum immutable target/background pair contract."""

    target_index: int
    target_box_xyxy_normalized: tuple[float, float, float, float]
    background_box_xyxy_normalized: tuple[float, float, float, float]
    factor_kind: str
    factor_channel: int
    severity: float
    transform_seed: int
    weight: float
    background_max_iou: float = 0.0
    batch_index: int | None = None
    target_spec: InterventionSpec | None = None
    background_spec: InterventionSpec | None = None

    def __post_init__(self) -> None:
        if isinstance(self.target_index, bool) or not isinstance(self.target_index, int) or self.target_index < 0:
            raise ValueError("target_index must be a non-negative integer")
        object.__setattr__(self, "target_box_xyxy_normalized", _strict_box(self.target_box_xyxy_normalized, "target_box"))
        object.__setattr__(self, "background_box_xyxy_normalized", _strict_box(self.background_box_xyxy_normalized, "background_box"))
        kind = _normalize_factor_kind(self.factor_kind, "factor_kind")
        if isinstance(self.factor_channel, bool) or not isinstance(self.factor_channel, int) or self.factor_channel not in {0, 1}:
            raise ValueError("factor_channel must be integer 0 (sampling) or 1 (visibility)")
        channel = self.factor_channel
        expected_channel = 0 if kind == "sampling" else 1
        if expected_channel != channel:
            raise ValueError("factor kind and channel must match")
        object.__setattr__(self, "factor_kind", kind)
        object.__setattr__(self, "factor_channel", channel)
        severity = _finite_float(self.severity, "severity")
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be within [0, 1]")
        object.__setattr__(self, "severity", severity)
        if isinstance(self.transform_seed, bool) or not isinstance(self.transform_seed, int) or self.transform_seed < 0:
            raise ValueError("transform_seed must be a non-negative integer")
        weight = _finite_float(self.weight, "weight")
        expected_weight = 0.0 if severity < 0.25 else severity
        if not math.isclose(weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("specificity weight must be zero below severity 0.25")
        object.__setattr__(self, "weight", weight)
        iou = _finite_float(self.background_max_iou, "background_max_iou")
        if iou != 0.0:
            raise ValueError("background_max_iou must be zero")
        object.__setattr__(self, "background_max_iou", 0.0)
        if self.batch_index is not None and (
            isinstance(self.batch_index, bool) or not isinstance(self.batch_index, int) or self.batch_index < 0
        ):
            raise ValueError("batch_index must be a non-negative integer")
        if (self.target_spec is not None and not isinstance(self.target_spec, InterventionSpec)) or (
            self.background_spec is not None and not isinstance(self.background_spec, InterventionSpec)
        ):
            raise ValueError("target/background specs must be InterventionSpec values")
        if (self.target_spec is None) != (self.background_spec is None):
            raise ValueError("target/background specs must be supplied together")
        if self.target_spec is not None and self.background_spec is not None:
            if self.target_spec.kind != self.background_spec.kind or self.target_spec.strength != self.background_spec.strength or self.target_spec.seed != self.background_spec.seed:
                raise ValueError("target/background intervention specs must share kind, strength, and seed")

    @property
    def target_box(self) -> tuple[float, float, float, float]:
        return self.target_box_xyxy_normalized

    @property
    def target_box_xyxy(self) -> tuple[float, float, float, float]:
        return self.target_box_xyxy_normalized

    @property
    def background_box(self) -> tuple[float, float, float, float]:
        return self.background_box_xyxy_normalized

    @property
    def background_box_xyxy(self) -> tuple[float, float, float, float]:
        return self.background_box_xyxy_normalized

    @property
    def kind(self) -> str:
        return self.factor_kind

    @property
    def channel(self) -> int:
        return self.factor_channel

    @property
    def seed(self) -> int:
        return self.transform_seed

    @property
    def batch_idx(self) -> int | None:
        return self.batch_index

    def __getitem__(self, key: str) -> object:
        aliases = {
            "target_box": self.target_box_xyxy_normalized,
            "background_box": self.background_box_xyxy_normalized,
            "kind": self.factor_kind,
            "channel": self.factor_channel,
            "seed": self.transform_seed,
            "batch_idx": self.batch_index,
        }
        if key in aliases:
            return aliases[key]
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def as_dict(self) -> dict[str, object]:
        return {
            "target_index": self.target_index,
            "target_box_xyxy_normalized": self.target_box_xyxy_normalized,
            "background_box_xyxy_normalized": self.background_box_xyxy_normalized,
            "factor_kind": self.factor_kind,
            "factor_channel": self.factor_channel,
            "severity": self.severity,
            "transform_seed": self.transform_seed,
            "weight": self.weight,
            "background_max_iou": self.background_max_iou,
            "batch_index": self.batch_index,
            "target_spec": self.target_spec,
            "background_spec": self.background_spec,
        }


def _labels_boxes(labels: Mapping[str, Any]) -> np.ndarray:
    value = labels.get("bboxes")
    if value is None:
        return _normalized_xyxy(dict(labels))
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("bboxes must be an Nx4 sequence")
    return array


def _specificity_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def build_specificity_pair(
    labels: Mapping[str, Any],
    *,
    target_index: int,
    background_box: Sequence[float],
    severity: float,
    transform_seed: int,
    factor_kind: object = "sampling",
    factor_channel: object | None = None,
    rejection_counter: SpecificityRejectionCounter | None = None,
    target_spec: InterventionSpec | None = None,
    background_spec: InterventionSpec | None = None,
) -> SpecificityPair:
    """Validate and construct one target-specific pair, failing closed."""

    try:
        if not isinstance(labels, Mapping):
            raise ValueError("labels must be a mapping")
        if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 0:
            raise ValueError("target_index must be a non-negative integer")
        boxes = _labels_boxes(labels)
        if target_index >= len(boxes):
            raise ValueError("target_index is out of bounds")
        target_boxes = tuple(_strict_box(row, "annotated box") for row in boxes)
        background = _strict_box(background_box, "background_box")
        overlaps = [_specificity_iou(background, box) for box in target_boxes]
        maximum = max(overlaps, default=0.0)
        if maximum != 0.0:
            raise ValueError("background overlaps annotated object")
        kind = _normalize_factor_kind(factor_kind, "factor_kind")
        expected_channel = 0 if kind == "sampling" else 1
        channel = expected_channel if factor_channel is None else factor_channel
        if isinstance(channel, bool) or not isinstance(channel, int) or channel not in {0, 1}:
            raise ValueError("factor_channel must be integer 0 (sampling) or 1 (visibility)")
        if channel != expected_channel:
            raise ValueError("factor kind and channel must match")
        severity_value = _finite_float(severity, "severity")
        if not 0.0 <= severity_value <= 1.0:
            raise ValueError("severity must be within [0, 1]")
        if isinstance(transform_seed, bool) or not isinstance(transform_seed, int) or transform_seed < 0:
            raise ValueError("transform_seed must be a non-negative integer")
        if (target_spec is None) != (background_spec is None):
            raise ValueError("target/background specs must be supplied together")
        if target_spec is not None and background_spec is not None:
            if target_spec.kind != background_spec.kind or target_spec.strength != background_spec.strength or target_spec.seed != background_spec.seed:
                raise ValueError("target/background intervention specs must share kind, strength, and seed")
        weight = 0.0 if severity_value < 0.25 else severity_value
        return SpecificityPair(
            target_index=target_index,
            target_box_xyxy_normalized=target_boxes[target_index],
            background_box_xyxy_normalized=background,
            factor_kind=kind,
            factor_channel=channel,
            severity=severity_value,
            transform_seed=transform_seed,
            weight=weight,
            background_max_iou=0.0,
            target_spec=target_spec,
            background_spec=background_spec,
        )
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        _reject(rejection_counter, "overlap" if "overlaps annotated" in str(exc) else "malformed")
        raise ValueError(str(exc)) from exc


def _epoch_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("epoch must be a non-negative integer")
    return value


class SharedEpoch:
    """Small process-shared epoch clock for persistent data workers."""

    def __init__(self, value: int = 0) -> None:
        self._value = multiprocessing.Value("q", _epoch_value(value))

    def get(self) -> int:
        with self._value.get_lock():
            return int(self._value.value)

    def set(self, value: int) -> None:
        value = _epoch_value(value)
        with self._value.get_lock():
            self._value.value = value


def _stable_index(size: int, *parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % size


def _normalized_xyxy(labels: dict[str, Any]) -> np.ndarray:
    image = labels["img"]
    height, width = image.shape[:2]
    instances = deepcopy(labels.get("instances"))
    if instances is None or len(instances) == 0:
        return np.empty((0, 4), dtype=np.float32)
    instances.convert_bbox(format="xyxy")
    boxes = instances.bboxes.astype(np.float32, copy=True)
    if not instances.normalized:
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
    return np.clip(boxes, 0.0, 1.0)


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(right - left, 0.0) * np.maximum(
        bottom - top,
        0.0,
    )
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1],
        0.0,
    )
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-9)


def _background_region(
    object_region: tuple[float, float, float, float],
    all_boxes: np.ndarray,
    *,
    selector: int,
) -> tuple[float, float, float, float]:
    width = max(object_region[2] - object_region[0], 0.05)
    height = max(object_region[3] - object_region[1], 0.05)
    width = min(width, 0.45)
    height = min(height, 0.45)
    candidates: list[tuple[float, float, float, float]] = []
    for center_y in np.linspace(height / 2, 1.0 - height / 2, 5):
        for center_x in np.linspace(width / 2, 1.0 - width / 2, 5):
            candidates.append(
                (
                    float(center_x - width / 2),
                    float(center_y - height / 2),
                    float(center_x + width / 2),
                    float(center_y + height / 2),
                )
            )
    overlaps = np.array(
        [
            float(_box_iou(np.asarray(candidate), all_boxes).max(initial=0.0))
            for candidate in candidates
        ]
    )
    minimum = overlaps.min()
    best = np.flatnonzero(np.isclose(overlaps, minimum, atol=1e-12))
    return candidates[int(best[selector % len(best)])]


def _sampling_proxy(box: tuple[float, float, float, float], height: int) -> float:
    box_height = max((box[3] - box[1]) * height, 0.0)
    return float(np.clip((64.0 - box_height) / 60.0, 0.0, 1.0))


class IFDRInterventionTransform:
    """Apply one reproducible object/background intervention to a YOLO sample."""

    def __init__(
        self,
        *,
        base_seed: int,
        epoch_state: SharedEpoch,
        enabled: bool,
        counterfactual_enabled: bool = True,
        policy: SamplingPolicy | None = None,
        calibration_enabled: bool = False,
        metadata_index: FactorMetadataIndex | None = None,
        rejection_counter: SpecificityRejectionCounter | None = None,
    ) -> None:
        if not isinstance(epoch_state, SharedEpoch):
            raise ValueError("epoch_state must be SharedEpoch")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(counterfactual_enabled, bool):
            raise ValueError("counterfactual_enabled must be a boolean")
        if not isinstance(calibration_enabled, bool):
            raise ValueError("calibration_enabled must be a boolean")
        if calibration_enabled and metadata_index is not None and not isinstance(metadata_index, FactorMetadataIndex):
            raise ValueError("metadata_index must be a FactorMetadataIndex")
        if rejection_counter is not None and not isinstance(rejection_counter, SpecificityRejectionCounter):
            raise ValueError("rejection_counter must be SpecificityRejectionCounter")
        self.epoch_state = epoch_state
        self.enabled = enabled
        self.counterfactual_enabled = counterfactual_enabled
        self.calibration_enabled = calibration_enabled
        self.metadata_index = metadata_index
        self.rejection_counter = rejection_counter or SpecificityRejectionCounter()
        self.sampler = DeterministicInterventionSampler(
            base_seed=base_seed,
            policy=policy,
        )

    def _empty_maps(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[:2]
        shape = (2, height, width)
        return (
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
        )

    def _counterfactual_image(self, image: np.ndarray) -> torch.Tensor:
        rgb_chw = np.ascontiguousarray(
            image[:, :, ::-1].transpose(2, 0, 1)
        )
        return torch.from_numpy(rgb_chw)

    def _empty_counterfactual(self, image: np.ndarray) -> tuple[torch.Tensor, ...]:
        delta, weight = self._empty_maps(image)
        return self._counterfactual_image(image), delta, weight

    def _set_empty_supervision(
        self,
        labels: dict[str, Any],
        image: np.ndarray,
        *,
        spec: str,
    ) -> dict[str, Any]:
        target, factor_weight = self._empty_maps(image)
        counterfactual, delta, counterfactual_weight = (
            self._empty_counterfactual(image)
        )
        labels[FACTOR_TARGET_KEY] = target
        labels[FACTOR_WEIGHT_KEY] = factor_weight
        if self.counterfactual_enabled:
            labels[COUNTERFACTUAL_IMAGE_KEY] = counterfactual
            labels[COUNTERFACTUAL_DELTA_KEY] = delta
            labels[COUNTERFACTUAL_WEIGHT_KEY] = counterfactual_weight
        labels["ifdr_spec"] = spec
        return labels

    def _set_empty_calibration(
        self,
        labels: dict[str, Any],
        image: np.ndarray,
        *,
        spec: str,
    ) -> dict[str, Any]:
        result = self._set_empty_supervision(labels, image, spec=spec)
        clean = self._counterfactual_image(image)
        result[CLEAN_IMAGE_KEY] = clean.clone()
        result[TARGET_IMAGE_KEY] = clean.clone()
        result[BACKGROUND_IMAGE_KEY] = clean.clone()
        result[BACKGROUND_FACTOR_TARGET_KEY] = torch.zeros_like(result[FACTOR_TARGET_KEY])
        result[FACTOR_OBJECT_TARGETS_KEY] = ()
        result[SPECIFICITY_PAIRS_KEY] = ()
        if spec in {"no_objects", "identity", "disabled"}:
            _reject(self.rejection_counter, spec)
        return result

    def _common_seed(self, image_id: str, object_id: int, epoch: int) -> int:
        return _stable_index(
            (1 << 63) - 1,
            "ifdr-specificity-common-seed-v1",
            self.sampler.base_seed,
            image_id,
            object_id,
            epoch,
        )

    @staticmethod
    def _record_target(record: Mapping[str, Any]) -> tuple[float, float]:
        return (
            float(record.get("sampling", record.get("target", (0.0, 0.0))[0])),
            float(record.get("visibility", record.get("target", (0.0, 0.0))[1])),
        )

    def _calibration_records(
        self,
        labels: dict[str, Any],
        boxes: np.ndarray,
    ) -> tuple[dict[str, Any], ...]:
        raw = labels.get("_ifdr_metadata_records", labels.get(FACTOR_OBJECT_TARGETS_KEY, ()))
        if raw is None:
            raw = ()
        if isinstance(raw, Mapping) or isinstance(raw, (str, bytes)):
            raise ValueError("factor metadata records must be a sequence")
        records = tuple(raw)
        try:
            records = tuple(sorted(records, key=lambda item: int(item.get("raw_label_index", 0)) if isinstance(item, Mapping) else 0))
        except (TypeError, ValueError, AttributeError):
            _reject(self.rejection_counter, "identity_drift")
            raise ValueError("malformed calibration object identity")
        classes = labels.get("cls")
        if isinstance(classes, torch.Tensor):
            classes = classes.detach().cpu().numpy()
        classes_array = np.asarray(classes if classes is not None else (), dtype=np.float64).reshape(-1)
        if len(records) != len(boxes) or len(classes_array) != len(boxes):
            _reject(self.rejection_counter, "identity_drift")
            raise ValueError("calibration object count or identity drift")
        normalized: list[dict[str, Any]] = []
        used: set[int] = set()
        for index, record in enumerate(records):
            if isinstance(record, ObjectFactorTarget):
                source = {
                    "class_id": record.class_id,
                    "sampling": record.target[0],
                    "visibility": record.target[1],
                    "sampling_valid": record.valid[0],
                    "visibility_valid": record.valid[1],
                }
            elif isinstance(record, Mapping):
                source = dict(record)
            else:
                _reject(self.rejection_counter, "identity_drift")
                raise ValueError("malformed calibration object record")
            class_id = source.get("class_id")
            if isinstance(class_id, bool) or not isinstance(class_id, (int, np.integer)) or int(class_id) != int(classes_array[index]):
                _reject(self.rejection_counter, "identity_drift")
                raise ValueError("calibration object class identity drift")
            raw_box = source.get("raw_box_xyxy_normalized", source.get("box_xyxy_normalized"))
            if raw_box is not None:
                raw_box = _strict_box(raw_box, "raw metadata box")
            current_box = tuple(float(value) for value in boxes[index])
            raw_index = source.get("raw_label_index", index)
            if isinstance(raw_index, bool) or not isinstance(raw_index, (int, np.integer)) or int(raw_index) != index:
                _reject(self.rejection_counter, "identity_drift")
                raise ValueError("calibration object raw label identity drift")
            target = self._record_target(source)
            valid = (
                bool(source.get("sampling_valid", source.get("valid", (True, True))[0])),
                bool(source.get("visibility_valid", source.get("valid", (True, True))[1])),
            )
            normalized.append({
                **source,
                "class_id": int(class_id),
                "raw_label_index": int(raw_index),
                "box_xyxy_normalized": current_box,
                "raw_box_xyxy_normalized": raw_box or current_box,
                "target": target,
                "sampling": target[0],
                "visibility": target[1],
                "valid": valid,
            })
            used.add(int(raw_index))
        if len(used) != len(records):
            _reject(self.rejection_counter, "identity_drift")
            raise ValueError("duplicate calibration object identity")
        return tuple(normalized)

    def _calibration_call(self, labels: dict[str, Any], image: np.ndarray, boxes: np.ndarray) -> dict[str, Any]:
        if len(boxes) == 0:
            return self._set_empty_calibration(labels, image, spec="no_objects")
        records = self._calibration_records(labels, boxes)
        epoch = self.epoch_state.get()
        image_id = Path(str(labels.get("im_file", "unknown"))).stem
        target_index = _stable_index(
            len(boxes), "ifdr-object-v1", self.sampler.base_seed, image_id, epoch,
        )
        object_region = tuple(float(value) for value in boxes[target_index])
        selector = _stable_index(
            1 << 31, "ifdr-background-v1", self.sampler.base_seed, image_id, target_index, epoch,
        )
        rejection_before = self.rejection_counter.total
        try:
            background_region = _background_region(object_region, boxes, selector=selector)
            # Validate every annotated box and fail closed when no empty patch exists.
            pair_boxes = {"bboxes": boxes}
            object_spec, background_spec = self.sampler.sample_matched_pair(
                image_id=image_id,
                object_id=target_index,
                epoch=epoch,
                slot=0,
                object_region=object_region,
                background_region=background_region,
            )
            if object_spec.kind.value == "identity":
                return self._set_empty_calibration(labels, image, spec="identity")
            common_seed = self._common_seed(image_id, target_index, epoch)
            object_spec = replace(object_spec, seed=common_seed)
            background_spec = replace(background_spec, seed=common_seed)
            kind = object_spec.kind.value
            channel = 0 if kind == "sampling" else 1
            pair = build_specificity_pair(
                pair_boxes,
                target_index=target_index,
                background_box=background_region,
                severity=object_spec.strength,
                transform_seed=common_seed,
                factor_kind=kind,
                factor_channel=channel,
                rejection_counter=self.rejection_counter,
                target_spec=object_spec,
                background_spec=background_spec,
            )
        except (ValueError, IndexError):
            if self.rejection_counter.total == rejection_before:
                _reject(self.rejection_counter, "empty_background")
            return self._set_empty_calibration(labels, image, spec="rejected")

        natural_sampling = _sampling_proxy(object_region, image.shape[0])
        target = factor_target_for_spec(object_spec, natural_sampling=natural_sampling, natural_occlusion=0.0)
        applied_target = apply_intervention(image, object_spec, target)
        background_target = factor_target_for_spec(background_spec)
        applied_background = apply_intervention(image, background_spec, background_target)
        clean_tensor = self._counterfactual_image(image)
        labels[FACTOR_TARGET_KEY] = torch.from_numpy(np.stack((applied_target.sampling_target, applied_target.visibility_target), axis=0))
        labels[FACTOR_WEIGHT_KEY] = torch.from_numpy(np.stack((applied_target.sampling_weight, applied_target.visibility_weight), axis=0))
        labels[BACKGROUND_FACTOR_TARGET_KEY] = torch.zeros_like(labels[FACTOR_TARGET_KEY])
        labels[CLEAN_IMAGE_KEY] = clean_tensor.clone()
        labels[TARGET_IMAGE_KEY] = self._counterfactual_image(applied_target.image)
        labels[BACKGROUND_IMAGE_KEY] = self._counterfactual_image(applied_background.image)
        labels["img"] = applied_target.image
        labels[FACTOR_OBJECT_TARGETS_KEY] = records
        labels[SPECIFICITY_PAIRS_KEY] = (pair,)
        labels["ifdr_spec"] = json.dumps(object_spec.to_payload(), sort_keys=True, separators=(",", ":"))
        if self.counterfactual_enabled:
            labels[COUNTERFACTUAL_IMAGE_KEY] = clean_tensor
            labels[COUNTERFACTUAL_DELTA_KEY] = torch.zeros_like(labels[FACTOR_TARGET_KEY])
            labels[COUNTERFACTUAL_WEIGHT_KEY] = torch.zeros_like(labels[FACTOR_WEIGHT_KEY])
        return labels

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        image = labels.get("img")
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("IFDR transform requires a uint8 HWC image")
        if not self.enabled:
            if self.calibration_enabled:
                return self._set_empty_calibration(labels, image, spec="disabled")
            return self._set_empty_supervision(
                labels,
                image,
                spec="disabled",
            )

        boxes = _normalized_xyxy(labels)
        if self.calibration_enabled:
            return self._calibration_call(labels, image, boxes)
        if len(boxes) == 0:
            return self._set_empty_supervision(
                labels,
                image,
                spec="no_objects",
            )

        epoch = self.epoch_state.get()
        image_id = Path(str(labels.get("im_file", "unknown"))).stem
        object_id = _stable_index(
            len(boxes),
            "ifdr-object-v1",
            self.sampler.base_seed,
            image_id,
            epoch,
        )
        object_region = tuple(float(value) for value in boxes[object_id])
        selector = _stable_index(
            1 << 31,
            "ifdr-background-v1",
            self.sampler.base_seed,
            image_id,
            object_id,
            epoch,
        )
        background_region = _background_region(
            object_region,
            boxes,
            selector=selector,
        )
        object_spec, background_spec = self.sampler.sample_matched_pair(
            image_id=image_id,
            object_id=object_id,
            epoch=epoch,
            slot=0,
            object_region=object_region,
            background_region=background_region,
        )
        spec = object_spec if object_spec.seed % 2 == 0 else background_spec
        natural_sampling = (
            _sampling_proxy(object_region, image.shape[0])
            if spec is object_spec
            else 0.0
        )
        target = factor_target_for_spec(
            spec,
            natural_sampling=natural_sampling,
            natural_occlusion=0.0,
        )
        clean_image = image.copy()
        applied = apply_intervention(image, spec, target)
        labels["img"] = applied.image
        labels[FACTOR_TARGET_KEY] = torch.from_numpy(
            np.stack(
                (applied.sampling_target, applied.visibility_target),
                axis=0,
            )
        )
        labels[FACTOR_WEIGHT_KEY] = torch.from_numpy(
            np.stack(
                (applied.sampling_weight, applied.visibility_weight),
                axis=0,
            )
        )
        counterfactual_delta = torch.zeros_like(labels[FACTOR_TARGET_KEY])
        counterfactual_weight = torch.zeros_like(labels[FACTOR_WEIGHT_KEY])
        if spec.strength > 0.0:
            support = torch.from_numpy(
                np.maximum(
                    applied.sampling_weight,
                    applied.visibility_weight,
                )
            )
            counterfactual_weight[0] = support
            counterfactual_weight[1] = support
            sampling_base = natural_sampling
            visibility_base = 0.0
            counterfactual_delta[0][support > 0] = (
                target.sampling - sampling_base
            )
            counterfactual_delta[1][support > 0] = (
                target.visibility - visibility_base
            )
        if self.counterfactual_enabled:
            labels[COUNTERFACTUAL_IMAGE_KEY] = self._counterfactual_image(
                clean_image
            )
            labels[COUNTERFACTUAL_DELTA_KEY] = counterfactual_delta
            labels[COUNTERFACTUAL_WEIGHT_KEY] = counterfactual_weight
        labels["ifdr_spec"] = json.dumps(
            spec.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return labels


def _factor_target_from_record(record: object, batch_index: int) -> ObjectFactorTarget:
    if isinstance(record, ObjectFactorTarget):
        source = {
            "class_id": record.class_id,
            "box_xyxy_normalized": record.box_xyxy_normalized,
            "target": record.target,
            "valid": record.valid,
        }
    elif isinstance(record, Mapping):
        source = record
    else:
        raise ValueError("factor object target must be a mapping or ObjectFactorTarget")
    class_id = source.get("class_id")
    if isinstance(class_id, bool) or not isinstance(class_id, (int, np.integer)) or int(class_id) < 0:
        raise ValueError("factor object target class_id must be non-negative")
    box = _strict_box(source.get("box_xyxy_normalized"), "box_xyxy_normalized")
    target_value = source.get("target", (source.get("sampling", 0.0), source.get("visibility", 0.0)))
    valid_value = source.get("valid", (source.get("sampling_valid", False), source.get("visibility_valid", False)))
    if isinstance(target_value, (str, bytes)) or not isinstance(target_value, Sequence) or len(target_value) != 2:
        raise ValueError("factor object target target must contain two values")
    if isinstance(valid_value, (str, bytes)) or not isinstance(valid_value, Sequence) or len(valid_value) != 2:
        raise ValueError("factor object target valid must contain two booleans")
    values = tuple(_finite_float(item, "factor target") for item in target_value)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("factor target values must be within [0, 1]")
    valid = tuple(valid_value)
    if any(type(item) is not bool for item in valid):
        raise ValueError("factor target valid must contain booleans")
    return ObjectFactorTarget(
        batch_index=batch_index,
        class_id=int(class_id),
        box_xyxy_normalized=box,
        target=(values[0], values[1]),
        valid=(valid[0], valid[1]),
    )


def _specificity_pair_from_record(record: object, batch_index: int) -> SpecificityPair:
    if isinstance(record, SpecificityPair):
        return replace(record, batch_index=batch_index)
    if not isinstance(record, Mapping):
        raise ValueError("specificity pair must be a mapping or SpecificityPair")
    source = dict(record)
    required = {
        "target_index",
        "target_box_xyxy_normalized",
        "background_box_xyxy_normalized",
        "factor_kind",
        "factor_channel",
        "severity",
        "transform_seed",
        "weight",
    }
    if not required.issubset(source):
        missing = sorted(required - set(source))
        raise ValueError(f"malformed specificity pair missing fields: {missing}")
    return SpecificityPair(
        target_index=source["target_index"],
        target_box_xyxy_normalized=source["target_box_xyxy_normalized"],
        background_box_xyxy_normalized=source["background_box_xyxy_normalized"],
        factor_kind=source["factor_kind"],
        factor_channel=source["factor_channel"],
        severity=source["severity"],
        transform_seed=source["transform_seed"],
        weight=source["weight"],
        background_max_iou=source.get("background_max_iou", 0.0),
        batch_index=batch_index,
        target_spec=source.get("target_spec"),
        background_spec=source.get("background_spec"),
    )


def _stack_optional_views(collated: dict[str, Any], key: str, batch: Sequence[Mapping[str, Any]]) -> None:
    values = [sample.get(key) for sample in batch]
    if not any(value is not None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(f"{key} must be present for every calibration sample")
    tensors = [value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in values]
    collated[key] = torch.stack(tensors, dim=0)


def collate_ifdr_batch(
    batch: list[dict[str, Any]],
    *,
    rejection_counter: SpecificityRejectionCounter | None = None,
) -> dict[str, Any]:
    collated = YOLODataset.collate_fn(batch)
    for key in (FACTOR_TARGET_KEY, FACTOR_WEIGHT_KEY):
        values = collated.get(key)
        if values is None:
            continue
        if not isinstance(values, tuple):
            raise RuntimeError(f"{key} was not preserved by collation")
        collated[key] = torch.stack(values, dim=0)
    for key in (
        COUNTERFACTUAL_IMAGE_KEY,
        COUNTERFACTUAL_DELTA_KEY,
        COUNTERFACTUAL_WEIGHT_KEY,
    ):
        values = collated.get(key)
        if values is None:
            continue
        if not isinstance(values, tuple):
            raise RuntimeError(f"{key} was not preserved by collation")
        collated[key] = torch.stack(values, dim=0)
    _stack_optional_views(collated, CLEAN_IMAGE_KEY, batch)
    _stack_optional_views(collated, TARGET_IMAGE_KEY, batch)
    _stack_optional_views(collated, BACKGROUND_IMAGE_KEY, batch)
    if any(BACKGROUND_FACTOR_TARGET_KEY in sample for sample in batch):
        _stack_optional_views(collated, BACKGROUND_FACTOR_TARGET_KEY, batch)
    object_records: list[ObjectFactorTarget] = []
    if any(FACTOR_OBJECT_TARGETS_KEY in sample for sample in batch):
        for batch_index, sample in enumerate(batch):
            values = sample.get(FACTOR_OBJECT_TARGETS_KEY, ())
            if isinstance(values, Mapping) or isinstance(values, (str, bytes)):
                _reject(rejection_counter, "malformed")
                raise ValueError("factor object targets must be a sequence")
            try:
                for value in values:
                    object_records.append(_factor_target_from_record(value, batch_index))
            except (TypeError, ValueError) as exc:
                _reject(rejection_counter, "malformed")
                raise ValueError(str(exc)) from exc
    if any(FACTOR_OBJECT_TARGETS_KEY in sample for sample in batch):
        collated[FACTOR_OBJECT_TARGETS_KEY] = tuple(object_records)
    pair_records: list[SpecificityPair] = []
    seen_pair_keys: set[tuple[int, int, str, int]] = set()
    if any(SPECIFICITY_PAIRS_KEY in sample for sample in batch):
        for batch_index, sample in enumerate(batch):
            values = sample.get(SPECIFICITY_PAIRS_KEY, ())
            if isinstance(values, Mapping) or isinstance(values, (str, bytes)):
                _reject(rejection_counter, "malformed")
                raise ValueError("specificity pairs must be a sequence")
            try:
                for value in values:
                    pair = _specificity_pair_from_record(value, batch_index)
                    identity = (batch_index, pair.target_index, pair.factor_kind, pair.transform_seed)
                    if identity in seen_pair_keys:
                        _reject(rejection_counter, "duplicate")
                        raise ValueError("duplicate specificity pair")
                    seen_pair_keys.add(identity)
                    pair_records.append(pair)
            except (TypeError, ValueError) as exc:
                _reject(rejection_counter, "malformed" if "duplicate" not in str(exc) else "duplicate")
                raise ValueError(str(exc)) from exc
    if any(SPECIFICITY_PAIRS_KEY in sample for sample in batch):
        collated[SPECIFICITY_PAIRS_KEY] = tuple(pair_records)
    return collated


class IFDRYOLODataset(YOLODataset):
    """YOLO detection dataset with project-owned dense factor targets."""

    collate_fn = staticmethod(collate_ifdr_batch)

    def __init__(
        self,
        *args,
        intervention_seed: int,
        interventions_enabled: bool,
        counterfactual_enabled: bool = False,
        intervention_policy: SamplingPolicy | None = None,
        calibration_enabled: bool = False,
        metadata_index: FactorMetadataIndex | None = None,
        specificity_rejection_counter: SpecificityRejectionCounter | None = None,
        **kwargs,
    ) -> None:
        self.epoch_state = SharedEpoch()
        self.calibration_enabled = calibration_enabled
        self.metadata_index = metadata_index
        self.specificity_rejection_counter = specificity_rejection_counter or SpecificityRejectionCounter()
        self.intervention_transform = IFDRInterventionTransform(
            base_seed=intervention_seed,
            epoch_state=self.epoch_state,
            enabled=interventions_enabled,
            counterfactual_enabled=counterfactual_enabled,
            policy=intervention_policy,
            calibration_enabled=calibration_enabled,
            metadata_index=metadata_index,
            rejection_counter=self.specificity_rejection_counter,
        )
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if getattr(self, "calibration_enabled", False):
            self._validate_calibration_transforms(transforms, hyp)
        transforms.insert(-1, self.intervention_transform)
        return transforms

    @staticmethod
    def _validate_calibration_transforms(transforms: object, hyp: object | None) -> None:
        prohibited = {
            "mosaic", "mixup", "copypaste", "copy_paste", "cutmix",
            "degrees", "translate", "scale", "shear", "perspective",
            "randomperspective", "randomflip", "randomcrop", "crop",
        }
        values = getattr(transforms, "transforms", None)
        if values is None and hasattr(transforms, "tolist"):
            values = transforms.tolist()
        if values is None:
            values = (transforms,)
        def visit(item: object) -> None:
            name = item.__class__.__name__.lower()
            if any(token in name for token in prohibited):
                raise ValueError(f"calibration geometry transform is not allowed: {name}")
            if name not in {"format", "letterbox", "compose", "pretransform", "randomhsv", "albumentations", "classifyletterbox"}:
                # Unknown transforms may alter geometry or identity.
                raise ValueError(f"unknown calibration transform: {name}")
            children = getattr(item, "transforms", None)
            if children is not None and children is not item:
                for child in children:
                    visit(child)
        for value in values:
            visit(value)
        if hyp is not None:
            for name in ("mosaic", "mixup", "copy_paste", "copy_paste_mode", "cutmix", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr"):
                setting = hyp.get(name, 0.0) if isinstance(hyp, Mapping) else getattr(hyp, name, 0.0)
                if isinstance(setting, (int, float)) and float(setting) != 0.0:
                    raise ValueError(f"calibration geometry setting is not allowed: {name}")

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        labels = super().get_image_and_label(index)
        if not getattr(self, "calibration_enabled", False):
            return labels
        metadata_index = getattr(self, "metadata_index", None)
        if not isinstance(metadata_index, FactorMetadataIndex):
            _reject(getattr(self, "specificity_rejection_counter", None), "missing_metadata")
            raise ValueError("calibration requires a FactorMetadataIndex")
        image_id = str(labels.get("image_id") or Path(str(labels.get("im_file", "unknown"))).stem)
        records = tuple(metadata_index.by_image.get(image_id, ()))
        original_shape = labels.get("ori_shape")
        if (
            isinstance(original_shape, (str, bytes))
            or not isinstance(original_shape, Sequence)
            or len(original_shape) != 2
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
                for value in original_shape
            )
        ):
            _reject(self.specificity_rejection_counter, "missing_metadata")
            raise ValueError("calibration metadata binding requires valid ori_shape")
        original_height, original_width = (int(value) for value in original_shape)
        instances = deepcopy(labels.get("instances"))
        if instances is None:
            _reject(self.specificity_rejection_counter, "missing_metadata")
            raise ValueError("calibration metadata binding requires instances")
        instances.convert_bbox(format="xyxy")
        boxes = np.asarray(instances.bboxes, dtype=np.float64)
        if instances.normalized:
            boxes[:, [0, 2]] *= original_width
            boxes[:, [1, 3]] *= original_height
        classes = np.asarray(labels.get("cls", ()), dtype=np.float64).reshape(-1)
        if len(records) != len(boxes) or len(classes) != len(boxes):
            _reject(self.specificity_rejection_counter, "missing_metadata")
            raise ValueError("metadata object count mismatch")
        class_names = {value: key for key, value in TRAIN_CLASS_TO_ID.items()}
        candidates = tuple(
            KittiLabelCandidate(
                image_id=image_id,
                object_index=object_index,
                class_id=int(classes[object_index]),
                class_name=class_names.get(int(classes[object_index]), ""),
                bbox_xyxy=tuple(float(value) for value in boxes[object_index]),
            )
            for object_index in range(len(boxes))
        )
        used: set[int] = set()
        attached: list[dict[str, Any]] = []
        try:
            for record in records:
                match = match_metadata_object(record=record, candidates=candidates)
                if match.object_index in used:
                    raise ValueError("duplicate metadata identity")
                used.add(match.object_index)
                raw_box = tuple(float(value) for value in record.bbox_xyxy)
                raw_normalized = (
                    raw_box[0] / original_width,
                    raw_box[1] / original_height,
                    raw_box[2] / original_width,
                    raw_box[3] / original_height,
                )
                attached.append({
                    "object_id": record.object_id,
                    "class_id": record.class_id,
                    "class_name": record.class_name,
                    "raw_label_index": match.object_index,
                    "raw_box_xyxy_normalized": raw_normalized,
                    "box_xyxy_normalized": raw_normalized,
                    "sampling": record.sampling,
                    "visibility": record.visibility,
                    "sampling_valid": record.sampling_valid,
                    "visibility_valid": record.visibility_valid,
                    "target": (record.sampling, record.visibility),
                    "valid": (record.sampling_valid, record.visibility_valid),
                })
        except (TypeError, ValueError) as exc:
            _reject(self.specificity_rejection_counter, "missing_metadata")
            raise ValueError(str(exc)) from exc
        if len(used) != len(candidates):
            _reject(self.specificity_rejection_counter, "identity_drift")
            raise ValueError("unused or duplicate metadata label identity")
        attached.sort(key=lambda item: item["raw_label_index"])
        labels["_ifdr_metadata_records"] = tuple(attached)
        labels["ifdr_raw_label_indices"] = tuple(item["raw_label_index"] for item in attached)
        labels[FACTOR_OBJECT_TARGETS_KEY] = tuple(attached)
        return labels

    def set_epoch(self, epoch: int) -> None:
        self.epoch_state.set(epoch)


def build_ifdr_dataset(
    cfg: object,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    *,
    mode: str,
    rect: bool,
    stride: int,
    intervention_seed: int,
    interventions_enabled: bool,
    counterfactual_enabled: bool = False,
    intervention_policy: SamplingPolicy | None = None,
    calibration_enabled: bool = False,
    metadata_index: FactorMetadataIndex | None = None,
    specificity_rejection_counter: SpecificityRejectionCounter | None = None,
) -> IFDRYOLODataset:
    """Build an IFDR dataset using the locked Ultralytics 8.4.98 contract."""

    if mode not in {"train", "val"}:
        raise ValueError("mode must be train or val")
    fraction = cfg.fraction if mode == "train" else 1.0
    return IFDRYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train" and not calibration_enabled,
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
        intervention_seed=intervention_seed,
        interventions_enabled=interventions_enabled,
        counterfactual_enabled=counterfactual_enabled,
        intervention_policy=intervention_policy,
        calibration_enabled=calibration_enabled,
        metadata_index=metadata_index,
        specificity_rejection_counter=specificity_rejection_counter,
    )
