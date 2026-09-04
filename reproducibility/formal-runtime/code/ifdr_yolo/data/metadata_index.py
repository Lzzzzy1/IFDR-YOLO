"""Immutable KITTI object metadata for factor-guided experiments.

The index deliberately keeps the raw KITTI identity and geometry intact.  It
is built before any geometry-changing augmentation, and therefore gives both
metadata replay and learned-factor calibration the same object binding.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import re
from pathlib import Path

from ifdr_yolo.data.kitti_types import TRAIN_CLASS_TO_ID
from ifdr_yolo.data.natural_degradation import (
    compute_sampling_score,
    compute_visibility_score,
)


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CLASS_ID_TO_NAME = {value: key for key, value in TRAIN_CLASS_TO_ID.items()}


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _validate_image_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("image_id must be exact non-empty text")
    if any(character.isspace() for character in value):
        raise ValueError("image_id must not contain whitespace")
    return value


def _validate_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    coordinates = tuple(
        _finite_float(item, f"bbox_xyxy[{index}]")
        for index, item in enumerate(value)
    )
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive area")
    return coordinates


@dataclass(frozen=True)
class KittiLabelCandidate:
    """Immutable label candidate used for exact metadata binding."""

    image_id: str
    object_index: int
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _validate_image_id(self.image_id))
        if isinstance(self.object_index, bool) or not isinstance(self.object_index, int):
            raise ValueError("object_index must be an integer")
        if self.object_index < 0:
            raise ValueError("object_index must be non-negative")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise ValueError("class_id must be an integer")
        class_name = _CLASS_ID_TO_NAME.get(self.class_id)
        if class_name is None or self.class_name != class_name:
            raise ValueError("class_id and class_name must match")
        object.__setattr__(self, "bbox_xyxy", _validate_bbox(self.bbox_xyxy))


@dataclass(frozen=True)
class KittiMetadataObject:
    """Raw, pre-augmentation KITTI object metadata."""

    image_id: str
    object_index: int
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    depth_m: float | None = None
    occlusion: int = 0
    truncation: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _validate_image_id(self.image_id))
        if isinstance(self.object_index, bool) or not isinstance(self.object_index, int):
            raise ValueError("object_index must be an integer")
        if self.object_index < 0:
            raise ValueError("object_index must be non-negative")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise ValueError("class_id must be an integer")
        class_name = _CLASS_ID_TO_NAME.get(self.class_id)
        if class_name is None or self.class_name != class_name:
            raise ValueError("class_id and class_name must match")
        object.__setattr__(self, "bbox_xyxy", _validate_bbox(self.bbox_xyxy))
        if self.depth_m is not None:
            object.__setattr__(self, "depth_m", _finite_float(self.depth_m, "depth_m"))
        if isinstance(self.occlusion, bool) or not isinstance(self.occlusion, int):
            raise ValueError("occlusion must be an integer")
        if not 0 <= self.occlusion <= 3:
            raise ValueError("occlusion must be within [0, 3]")
        truncation = _finite_float(self.truncation, "truncation")
        if not 0.0 <= truncation <= 1.0:
            raise ValueError("truncation must be within [0, 1]")
        object.__setattr__(self, "truncation", truncation)

    @property
    def object_id(self) -> str:
        return f"{self.image_id}:{self.object_index:06d}"

    def as_label(self) -> KittiLabelCandidate:
        return KittiLabelCandidate(
            image_id=self.image_id,
            object_index=self.object_index,
            class_id=self.class_id,
            class_name=self.class_name,
            bbox_xyxy=self.bbox_xyxy,
        )


@dataclass(frozen=True)
class FactorObjectRecord:
    image_id: str
    object_id: str
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    height: float
    depth_m: float | None
    occlusion: int
    truncation: float
    sampling: float
    visibility: float
    joint: float
    sampling_valid: bool
    visibility_valid: bool


def _record_payload(record: FactorObjectRecord) -> dict[str, Any]:
    return {
        "image_id": record.image_id,
        "object_id": record.object_id,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "bbox_xyxy": list(record.bbox_xyxy),
        "height": record.height,
        "depth_m": record.depth_m,
        "occlusion": record.occlusion,
        "truncation": record.truncation,
        "sampling": record.sampling,
        "visibility": record.visibility,
        "joint": record.joint,
        "sampling_valid": record.sampling_valid,
        "visibility_valid": record.visibility_valid,
    }


@dataclass(frozen=True)
class FactorMetadataIndex:
    by_image: Mapping[str, tuple[FactorObjectRecord, ...]]
    source_sha256: str
    split_sha256: str
    label_source_sha256: str
    sha256: str
    invalid_depth_count: int = 0

    def __post_init__(self) -> None:
        frozen_by_image = MappingProxyType(
            {
                image_id: tuple(records)
                for image_id, records in self.by_image.items()
            }
        )
        object.__setattr__(self, "by_image", frozen_by_image)

    def to_json_bytes(self) -> bytes:
        payload = {
            "by_image": {
                image_id: [_record_payload(record) for record in records]
                for image_id, records in sorted(self.by_image.items())
            },
            "source_sha256": self.source_sha256,
            "split_sha256": self.split_sha256,
            "label_source_sha256": self.label_source_sha256,
            "invalid_depth_count": self.invalid_depth_count,
            "sha256": self.sha256,
        }
        return _canonical_json(payload)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA256")
    return value


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_metadata_object(
    record: KittiMetadataObject,
    candidates: Sequence[KittiLabelCandidate],
    *,
    minimum_iou: float = 0.99,
) -> KittiLabelCandidate:
    if (
        type(minimum_iou) is not float
        or not isfinite(minimum_iou)
        or minimum_iou != 0.99
    ):
        raise ValueError("minimum_iou is registered at 0.99")
    matches = [
        candidate
        for candidate in candidates
        if candidate.image_id == record.image_id
        and candidate.class_id == record.class_id
        and box_iou(candidate.bbox_xyxy, record.bbox_xyxy) >= minimum_iou
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "ambiguous"
        raise ValueError(f"{reason} metadata match for {record.object_id}")
    return matches[0]


def _index_payload(
    records_by_image: Mapping[str, tuple[FactorObjectRecord, ...]],
    *,
    source_sha256: str,
    split_sha256: str,
    label_source_sha256: str,
    invalid_depth_count: int,
) -> dict[str, Any]:
    return {
        "by_image": {
            image_id: [_record_payload(record) for record in records]
            for image_id, records in sorted(records_by_image.items())
        },
        "source_sha256": source_sha256,
        "split_sha256": split_sha256,
        "label_source_sha256": label_source_sha256,
        "invalid_depth_count": invalid_depth_count,
    }


def build_metadata_index(
    objects: Sequence[KittiMetadataObject],
    *,
    labels: Mapping[str, Sequence[KittiLabelCandidate]],
    split_sha256: str,
    source_sha256: str,
    label_source_sha256: str,
) -> FactorMetadataIndex:
    """Bind every source object to exactly one unaugmented label candidate."""
    source_hash = _validate_hash(source_sha256, "source_sha256")
    split_hash = _validate_hash(split_sha256, "split_sha256")
    label_hash = _validate_hash(label_source_sha256, "label_source_sha256")
    if not isinstance(labels, Mapping):
        raise ValueError("labels must be a mapping")

    normalized_labels: dict[str, tuple[KittiLabelCandidate, ...]] = {}
    for image_id, image_candidates in labels.items():
        if not isinstance(image_id, str):
            raise ValueError("label mapping keys must be image_id text")
        if not isinstance(image_candidates, Sequence) or isinstance(
            image_candidates, (str, bytes)
        ):
            raise ValueError(f"labels for {image_id} must be a sequence")
        try:
            candidates = tuple(image_candidates)
        except TypeError as exc:
            raise ValueError(f"labels for {image_id} must be a sequence") from exc
        for candidate in candidates:
            if not isinstance(candidate, KittiLabelCandidate):
                raise ValueError("label candidates must be KittiLabelCandidate instances")
            if candidate.image_id != image_id:
                raise ValueError(
                    f"label candidate image_id does not match mapping key: {image_id}"
                )
        normalized_labels[image_id] = candidates

    normalized_objects = tuple(objects)
    seen_ids: set[tuple[str, int]] = set()
    records: list[FactorObjectRecord] = []
    invalid_depth_count = 0
    used_candidates: set[tuple[str, int]] = set()

    for source in normalized_objects:
        if not isinstance(source, KittiMetadataObject):
            raise ValueError("metadata objects must be KittiMetadataObject instances")
        identity = (source.image_id, source.object_index)
        if identity in seen_ids:
            raise ValueError(f"duplicate object identity for {source.object_id}")
        seen_ids.add(identity)

    for source in sorted(normalized_objects, key=lambda item: (item.image_id, item.object_index)):
        image_candidates = normalized_labels.get(source.image_id)
        if image_candidates is None:
            raise ValueError(f"missing metadata match for {source.object_id}")
        match = match_metadata_object(source, image_candidates)
        candidate_identity = (match.image_id, match.object_index)
        if candidate_identity in used_candidates:
            raise ValueError(f"duplicate label candidate identity for {source.object_id}")
        used_candidates.add(candidate_identity)

        depth_m = source.depth_m
        if depth_m is None or depth_m <= 0.0:
            invalid_depth_count += 1
            depth_m = None
        height = source.bbox_xyxy[3] - source.bbox_xyxy[1]
        sampling = compute_sampling_score(height, depth_m)
        visibility = compute_visibility_score(source.occlusion, source.truncation)
        joint = 1.0 - (1.0 - sampling) * (1.0 - visibility)
        records.append(
            FactorObjectRecord(
                image_id=source.image_id,
                object_id=source.object_id,
                class_id=source.class_id,
                class_name=source.class_name,
                bbox_xyxy=source.bbox_xyxy,
                height=height,
                depth_m=depth_m,
                occlusion=source.occlusion,
                truncation=source.truncation,
                sampling=sampling,
                visibility=visibility,
                joint=joint,
                sampling_valid=True,
                visibility_valid=True,
            )
        )

    for image_candidates in normalized_labels.values():
        for candidate in image_candidates:
            if (candidate.image_id, candidate.object_index) not in used_candidates:
                raise ValueError(
                    f"unused eligible label candidate for {candidate.image_id}:"
                    f"{candidate.object_index:06d}"
                )

    grouped: dict[str, tuple[FactorObjectRecord, ...]] = {}
    for record in records:
        grouped.setdefault(record.image_id, ())
        grouped[record.image_id] = grouped[record.image_id] + (record,)
    immutable_by_image: Mapping[str, tuple[FactorObjectRecord, ...]] = MappingProxyType(
        {image_id: grouped[image_id] for image_id in sorted(grouped)}
    )
    payload = _index_payload(
        immutable_by_image,
        source_sha256=source_hash,
        split_sha256=split_hash,
        label_source_sha256=label_hash,
        invalid_depth_count=invalid_depth_count,
    )
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return FactorMetadataIndex(
        by_image=immutable_by_image,
        source_sha256=source_hash,
        split_sha256=split_hash,
        label_source_sha256=label_hash,
        sha256=digest,
        invalid_depth_count=invalid_depth_count,
    )


def load_metadata_index(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> FactorMetadataIndex:
    """Deserialize and integrity-check a generated ``metadata_index.json``.

    The generated artifact stores the digest of the canonical index payload in
    ``sha256`` while the protocol identity stores that same digest.  Parsing is
    intentionally strict: malformed records, altered fields, and an altered
    artifact digest all fail before a dataset can consume the index.
    """

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"metadata index must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"metadata index does not exist: {resolved}")
    raw = resolved.read_bytes()
    if not raw:
        raise ValueError("metadata index is empty")
    expected_logical_sha256 = (
        _validate_hash(expected_sha256, "metadata index SHA256")
        if expected_sha256 is not None
        else None
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("metadata index must contain valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("metadata index root must be a mapping")
    by_image_raw = payload.get("by_image")
    if not isinstance(by_image_raw, Mapping):
        raise ValueError("metadata index by_image must be a mapping")

    records_by_image: dict[str, tuple[FactorObjectRecord, ...]] = {}
    for image_id, raw_records in by_image_raw.items():
        if not isinstance(image_id, str) or not image_id or image_id.strip() != image_id:
            raise ValueError("metadata index image IDs must be exact text")
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            raise ValueError(f"metadata index records for {image_id} must be a sequence")
        parsed: list[FactorObjectRecord] = []
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("metadata index object record must be a mapping")
            required = {
                "image_id", "object_id", "class_id", "class_name", "bbox_xyxy",
                "height", "depth_m", "occlusion", "truncation", "sampling",
                "visibility", "joint", "sampling_valid", "visibility_valid",
            }
            if set(raw_record) != required:
                raise ValueError("metadata index object record fields are invalid")
            record_image_id = raw_record["image_id"]
            object_id = raw_record["object_id"]
            class_id = raw_record["class_id"]
            class_name = raw_record["class_name"]
            if record_image_id != image_id or not isinstance(object_id, str):
                raise ValueError("metadata index object identity is invalid")
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ValueError("metadata index class_id is invalid")
            if not isinstance(class_name, str):
                raise ValueError("metadata index class_name is invalid")
            bbox = raw_record["bbox_xyxy"]
            if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
                raise ValueError("metadata index bbox is invalid")
            try:
                numbers = tuple(float(value) for value in bbox)
                height = float(raw_record["height"])
                depth = raw_record["depth_m"]
                depth = None if depth is None else float(depth)
                occlusion = int(raw_record["occlusion"])
                truncation = float(raw_record["truncation"])
                sampling = float(raw_record["sampling"])
                visibility = float(raw_record["visibility"])
                joint = float(raw_record["joint"])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("metadata index numeric field is invalid") from error
            if not all(isfinite(value) for value in numbers + (height, truncation, sampling, visibility, joint)):
                raise ValueError("metadata index numeric field must be finite")
            if depth is not None and not isfinite(depth):
                raise ValueError("metadata index depth_m must be finite")
            if not isinstance(raw_record["sampling_valid"], bool) or not isinstance(raw_record["visibility_valid"], bool):
                raise ValueError("metadata index validity flags must be boolean")
            parsed.append(
                FactorObjectRecord(
                    image_id=record_image_id,
                    object_id=object_id,
                    class_id=class_id,
                    class_name=class_name,
                    bbox_xyxy=numbers,
                    height=height,
                    depth_m=depth,
                    occlusion=occlusion,
                    truncation=truncation,
                    sampling=sampling,
                    visibility=visibility,
                    joint=joint,
                    sampling_valid=raw_record["sampling_valid"],
                    visibility_valid=raw_record["visibility_valid"],
                )
            )
        records_by_image[image_id] = tuple(parsed)

    source_sha256 = _validate_hash(payload.get("source_sha256"), "source_sha256")
    split_sha256 = _validate_hash(payload.get("split_sha256"), "split_sha256")
    label_source_sha256 = _validate_hash(
        payload.get("label_source_sha256"), "label_source_sha256"
    )
    invalid_depth_count = payload.get("invalid_depth_count", 0)
    if isinstance(invalid_depth_count, bool) or not isinstance(invalid_depth_count, int) or invalid_depth_count < 0:
        raise ValueError("invalid_depth_count must be a non-negative integer")
    supplied_sha256 = _validate_hash(payload.get("sha256"), "metadata index sha256")
    canonical_payload = _index_payload(
        records_by_image,
        source_sha256=source_sha256,
        split_sha256=split_sha256,
        label_source_sha256=label_source_sha256,
        invalid_depth_count=invalid_depth_count,
    )
    actual_sha256 = hashlib.sha256(_canonical_json(canonical_payload)).hexdigest()
    if supplied_sha256 != actual_sha256:
        raise ValueError(
            "metadata index SHA256 mismatch: "
            f"expected={supplied_sha256}, actual={actual_sha256}"
        )
    if expected_logical_sha256 is not None and supplied_sha256 != expected_logical_sha256:
        file_digest = hashlib.sha256(raw).hexdigest()
        if file_digest != expected_logical_sha256:
            raise ValueError("metadata index SHA256 does not match expected identity")
    return FactorMetadataIndex(
        by_image=records_by_image,
        source_sha256=source_sha256,
        split_sha256=split_sha256,
        label_source_sha256=label_source_sha256,
        sha256=supplied_sha256,
        invalid_depth_count=invalid_depth_count,
    )


deserialize_metadata_index = load_metadata_index


__all__ = [
    "FactorMetadataIndex",
    "FactorObjectRecord",
    "KittiLabelCandidate",
    "KittiMetadataObject",
    "build_metadata_index",
    "load_metadata_index",
    "deserialize_metadata_index",
    "box_iou",
    "compute_sampling_score",
    "compute_visibility_score",
    "match_metadata_object",
]
