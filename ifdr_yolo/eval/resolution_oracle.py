"""Pure geometry and candidate rules for the value-of-resolution oracle.

This module deliberately contains no model or filesystem code.  It defines the
frozen crop/mapping/NMS and candidate-pool rules used by the staged oracle
runner.  O2 candidate construction has no ground-truth argument by design:
the pool must be materialized before any oracle selection can inspect labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    Difficulty,
    EVAL_CLASSES,
    KittiObject,
)
from ifdr_yolo.eval.kitti_ap40 import (
    CLASS_IOU_THRESHOLDS,
    DIFFICULTY_RULES,
    box_iou,
    classify_ground_truth,
    GroundTruthStatus,
)


# The crop is read at its native crop resolution.  The fixed half-image
# window therefore supplies the intended 2x *effective target resolution*;
# there is no second coordinate scale in the mapping functions.
NMS_IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 300
PROPOSAL_CONF_MIN = 0.001
PROPOSAL_CONF_MAX = 0.25
GRID_COLUMNS = 3
GRID_ROWS = 2
PROPOSAL_LIMIT = 18
UTILITY_TP_WEIGHT = 1.0
UTILITY_MEAN_IOU_WEIGHT = 0.25
UTILITY_FP_WEIGHT = -0.25
UTILITY_DUPLICATE_WEIGHT = -0.10


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class ImageSize:
    """Immutable positive image dimensions in full-image pixel coordinates."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or isinstance(self.height, bool):
            raise ValueError("image dimensions must be integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True)
class CropWindow:
    """A clamped full-image crop rectangle."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = tuple(_finite(value, "crop coordinate") for value in self.as_xyxy())
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"crop window must have positive area: {values}")

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_bbox(self) -> BoundingBox:
        return BoundingBox(*self.as_xyxy())


@dataclass(frozen=True)
class OracleCandidate:
    """A deterministic crop candidate.

    ``source`` is either ``grid``, ``proposal`` or ``gt`` (O1 only).  The
    ground-truth source is intentionally impossible to produce through
    :func:`build_o2_candidate_pool`, whose signature contains no labels.
    """

    window: CropWindow
    source: str
    rank: int
    proposal_score: float | None = None

    def __post_init__(self) -> None:
        if self.source not in {"grid", "proposal", "gt"}:
            raise ValueError(f"unknown candidate source: {self.source}")
        if self.rank < 0:
            raise ValueError("candidate rank must be non-negative")
        if self.proposal_score is not None:
            _finite(self.proposal_score, "proposal score")


@dataclass(frozen=True)
class UtilityComponents:
    """Frozen per-crop utility components and the registered linear score."""

    delta_tp: float
    delta_mean_iou: float
    delta_fp: float
    delta_duplicates: float

    def __post_init__(self) -> None:
        for name in ("delta_tp", "delta_mean_iou", "delta_fp", "delta_duplicates"):
            _finite(getattr(self, name), name)

    @property
    def utility(self) -> float:
        return (
            UTILITY_TP_WEIGHT * self.delta_tp
            + UTILITY_MEAN_IOU_WEIGHT * self.delta_mean_iou
            + UTILITY_FP_WEIGHT * self.delta_fp
            + UTILITY_DUPLICATE_WEIGHT * self.delta_duplicates
        )


def _image_size(value: ImageSize | tuple[int, int]) -> ImageSize:
    if isinstance(value, ImageSize):
        return value
    if len(value) != 2:
        raise ValueError("image size must contain width and height")
    return ImageSize(int(value[0]), int(value[1]))


def _box(value: BoundingBox | CropWindow) -> BoundingBox:
    return value if isinstance(value, BoundingBox) else value.as_bbox()


def clamp_crop_window(
    window: BoundingBox | CropWindow,
    image_size: ImageSize | tuple[int, int],
) -> CropWindow:
    """Clamp a crop rectangle to image bounds without changing its type/role."""

    size = _image_size(image_size)
    box = _box(window)
    x1 = min(max(box.x1, 0.0), float(size.width))
    y1 = min(max(box.y1, 0.0), float(size.height))
    x2 = min(max(box.x2, 0.0), float(size.width))
    y2 = min(max(box.y2, 0.0), float(size.height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("clamping produced an empty crop window")
    return CropWindow(x1, y1, x2, y2)


def fixed_crop_window(
    target_box: BoundingBox,
    image_size: ImageSize | tuple[int, int],
) -> CropWindow:
    """Build the fixed half-image crop centered on a target."""
    size = _image_size(image_size)
    return _fixed_window_from_center(
        size,
        math.floor((target_box.x1 + target_box.x2) / 2.0),
        math.floor((target_box.y1 + target_box.y2) / 2.0),
    )


def _fixed_window_from_center(
    size: ImageSize,
    center_x: int,
    center_y: int,
) -> CropWindow:
    """Create an integer ceil-half window using floor centers and edge shifts."""

    crop_width = math.ceil(size.width / 2.0)
    crop_height = math.ceil(size.height / 2.0)
    x1 = min(max(center_x - crop_width // 2, 0), size.width - crop_width)
    y1 = min(max(center_y - crop_height // 2, 0), size.height - crop_height)
    return CropWindow(
        float(x1),
        float(y1),
        float(x1 + crop_width),
        float(y1 + crop_height),
    )


def full_to_crop_box(
    full_box: BoundingBox,
    crop_window: CropWindow,
) -> BoundingBox:
    """Map a full-image box into native crop pixel coordinates."""

    crop = crop_window
    return BoundingBox(
        min(max(full_box.x1 - crop.x1, 0.0), crop.width),
        min(max(full_box.y1 - crop.y1, 0.0), crop.height),
        min(max(full_box.x2 - crop.x1, 0.0), crop.width),
        min(max(full_box.y2 - crop.y1, 0.0), crop.height),
    )


def crop_to_full_box(
    crop_box: BoundingBox,
    crop_window: CropWindow,
    *,
    image_size: ImageSize | tuple[int, int] | None = None,
) -> BoundingBox:
    """Map native crop pixels back to full-image coordinates and clip them."""

    clipped_crop = BoundingBox(
        min(max(crop_box.x1, 0.0), crop_window.width),
        min(max(crop_box.y1, 0.0), crop_window.height),
        min(max(crop_box.x2, 0.0), crop_window.width),
        min(max(crop_box.y2, 0.0), crop_window.height),
    )
    mapped = BoundingBox(
        crop_window.x1 + clipped_crop.x1,
        crop_window.y1 + clipped_crop.y1,
        crop_window.x1 + clipped_crop.x2,
        crop_window.y1 + clipped_crop.y2,
    )
    if image_size is None:
        return mapped
    return _box(clamp_crop_window(mapped, image_size))


def box_to_normalized_yolo(
    box: BoundingBox,
    image_size: ImageSize | tuple[int, int],
) -> tuple[float, float, float, float]:
    """Convert a pixel box to clipped ``cx cy width height`` YOLO values."""

    size = _image_size(image_size)
    clipped = _box(clamp_crop_window(box, size))
    return (
        ((clipped.x1 + clipped.x2) / 2.0) / size.width,
        ((clipped.y1 + clipped.y2) / 2.0) / size.height,
        clipped.width / size.width,
        clipped.height / size.height,
    )


def normalized_yolo_to_box(
    values: Sequence[float],
    image_size: ImageSize | tuple[int, int],
) -> BoundingBox:
    """Convert normalized YOLO ``cx cy width height`` values to a clipped box."""

    if len(values) != 4:
        raise ValueError("YOLO box must contain four values")
    size = _image_size(image_size)
    cx, cy, width, height = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (cx, cy, width, height)):
        raise ValueError("YOLO box values must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("YOLO box dimensions must be positive")
    return _box(
        clamp_crop_window(
            BoundingBox(
                (cx - width / 2.0) * size.width,
                (cy - height / 2.0) * size.height,
                (cx + width / 2.0) * size.width,
                (cy + height / 2.0) * size.height,
            ),
            size,
        )
    )


def classwise_nms(
    detections: Sequence[Detection],
    *,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    max_det: int = MAX_DETECTIONS,
) -> tuple[Detection, ...]:
    """Stable classwise NMS; input order breaks score ties (base first)."""

    iou_threshold = _finite(iou_threshold, "IoU threshold")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be within [0, 1]")
    if isinstance(max_det, bool) or max_det <= 0:
        raise ValueError("max_det must be positive")
    ranked = sorted(
        enumerate(detections),
        key=lambda item: (-item[1].score, item[0]),
    )
    kept: list[Detection] = []
    for _, detection in ranked:
        if any(
            detection.kind == previous.kind
            and box_iou(detection.bbox, previous.bbox) > iou_threshold
            for previous in kept
        ):
            continue
        kept.append(detection)
        if len(kept) >= max_det:
            break
    return tuple(kept)


def fuse_detections(
    base_detections: Sequence[Detection],
    crop_detections: Sequence[Detection],
    *,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    max_det: int = MAX_DETECTIONS,
) -> tuple[Detection, ...]:
    """Fuse predictions with base predictions first for deterministic ties."""

    return classwise_nms(
        tuple(base_detections) + tuple(crop_detections),
        iou_threshold=iou_threshold,
        max_det=max_det,
    )


def _grid_windows(size: ImageSize) -> tuple[CropWindow, ...]:
    windows: list[CropWindow] = []
    for row in range(GRID_ROWS):
        center_y = math.floor(size.height * (0.25 if row == 0 else 0.75))
        for column in range(GRID_COLUMNS):
            center_x = math.floor(size.width * ((column + 1) / 4.0))
            windows.append(_fixed_window_from_center(size, center_x, center_y))
    return tuple(windows)


def build_o1_candidates(
    ground_truth_by_image: Mapping[str, Sequence[KittiObject]],
    image_sizes: Mapping[str, ImageSize | tuple[int, int]],
) -> dict[str, tuple[OracleCandidate, ...]]:
    """Build the optimistic O1 GT-centered windows for targets below 40px."""

    candidates: dict[str, tuple[OracleCandidate, ...]] = {}
    for image_id in sorted(image_sizes):
        size = _image_size(image_sizes[image_id])
        image_candidates: list[OracleCandidate] = []
        for index, target in enumerate(ground_truth_by_image.get(image_id, ())):
            if target.kind not in EVAL_CLASSES or target.bbox.height >= 40.0:
                continue
            image_candidates.append(
                OracleCandidate(
                    window=fixed_crop_window(target.bbox, size),
                    source="gt",
                    rank=index,
                )
            )
        candidates[image_id] = tuple(image_candidates)
    return candidates


def build_o2_candidate_pool(
    image_sizes: Mapping[str, ImageSize | tuple[int, int]],
    first_pass_predictions: Mapping[str, Sequence[Detection]],
    *,
    confidence_min: float = PROPOSAL_CONF_MIN,
    confidence_max: float = PROPOSAL_CONF_MAX,
    proposal_limit: int = PROPOSAL_LIMIT,
) -> dict[str, tuple[OracleCandidate, ...]]:
    """Build the frozen GT-free O2 pool: six grid + at most 18 proposals."""

    confidence_min = _finite(confidence_min, "confidence minimum")
    confidence_max = _finite(confidence_max, "confidence maximum")
    if confidence_min > confidence_max:
        raise ValueError("confidence bounds must be ordered")
    if proposal_limit < 0:
        raise ValueError("proposal limit must be non-negative")

    pools: dict[str, tuple[OracleCandidate, ...]] = {}
    for image_id in sorted(image_sizes):
        size = _image_size(image_sizes[image_id])
        candidates: list[OracleCandidate] = [
            OracleCandidate(window=window, source="grid", rank=rank)
            for rank, window in enumerate(_grid_windows(size))
        ]
        ranked_proposals = sorted(
            enumerate(first_pass_predictions.get(image_id, ())),
            key=lambda item: (
                item[1].bbox.area,
                item[1].score,
                item[1].kind,
                item[1].bbox.x1,
                item[1].bbox.y1,
                item[1].bbox.x2,
                item[1].bbox.y2,
                item[0],
            ),
        )
        seen: set[tuple[float, float, float, float]] = set()
        proposal_rank = 0
        for _, detection in ranked_proposals:
            if not confidence_min <= detection.score <= confidence_max:
                continue
            if detection.bbox.height >= 40.0:
                continue
            window = fixed_crop_window(detection.bbox, size)
            key = tuple(round(value, 9) for value in window.as_xyxy())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                OracleCandidate(
                    window=window,
                    source="proposal",
                    rank=proposal_rank,
                    proposal_score=detection.score,
                )
            )
            proposal_rank += 1
            if proposal_rank >= proposal_limit:
                break
        pools[image_id] = tuple(candidates)
    return pools


def select_one_crop(
    candidates: Sequence[OracleCandidate],
    utilities: Mapping[OracleCandidate, UtilityComponents],
) -> OracleCandidate | None:
    """Select at most one strictly positive-utility crop; ties keep pool order."""

    best_candidate: OracleCandidate | None = None
    best_utility = 0.0
    for candidate in candidates:
        components = utilities.get(candidate)
        if components is None:
            continue
        if components.utility > best_utility:
            best_candidate = candidate
            best_utility = components.utility
    return best_candidate


@dataclass(frozen=True)
class _ModerateCounts:
    true_positives: int
    false_positives: int
    duplicates: int
    mean_iou: float


def _intersection_over_detection(
    detection_box: BoundingBox,
    region_box: BoundingBox,
) -> float:
    intersection_width = max(
        0.0,
        min(detection_box.x2, region_box.x2)
        - max(detection_box.x1, region_box.x1),
    )
    intersection_height = max(
        0.0,
        min(detection_box.y2, region_box.y2)
        - max(detection_box.y1, region_box.y1),
    )
    if detection_box.area <= 0.0:
        return 0.0
    return intersection_width * intersection_height / detection_box.area


def _moderate_counts(
    ground_truth_by_image: Mapping[str, Sequence[KittiObject]],
    detections_by_image: Mapping[str, Sequence[Detection]],
) -> _ModerateCounts:
    true_positives = 0
    false_positives = 0
    duplicates = 0
    matched_ious: list[float] = []
    for class_name in EVAL_CLASSES:
        valid_by_image: dict[str, list[KittiObject]] = {}
        ignored_by_image: dict[str, list[KittiObject]] = {}
        dontcare_by_image: dict[str, list[KittiObject]] = {}
        matched_by_image: dict[str, list[bool]] = {}
        ignored_matched_by_image: dict[str, list[bool]] = {}
        for image_id, objects in ground_truth_by_image.items():
            valid: list[KittiObject] = []
            ignored: list[KittiObject] = []
            for target in objects:
                status = classify_ground_truth(
                    target,
                    class_name,
                    # This function intentionally fixes Moderate; callers
                    # cannot tune thresholds after seeing oracle outputs.
                    difficulty=Difficulty.MODERATE,
                )
                if status is GroundTruthStatus.VALID:
                    valid.append(target)
                elif status is GroundTruthStatus.IGNORED:
                    ignored.append(target)
            valid_by_image[image_id] = valid
            ignored_by_image[image_id] = ignored
            dontcare_by_image[image_id] = [
                target for target in objects if target.kind == "DontCare"
            ]
            matched_by_image[image_id] = [False] * len(valid)
            ignored_matched_by_image[image_id] = [False] * len(ignored)
        ranked = sorted(
            (
                detection
                for detections in detections_by_image.values()
                for detection in detections
                if detection.kind == class_name
            ),
            key=lambda detection: detection.score,
            reverse=True,
        )
        threshold = CLASS_IOU_THRESHOLDS[class_name]
        min_height = DIFFICULTY_RULES[Difficulty.MODERATE][0]
        for detection in ranked:
            valid = valid_by_image.get(detection.image_id, [])
            matched = matched_by_image.get(detection.image_id, [])
            best_index = -1
            best_iou = threshold
            for index, target in enumerate(valid):
                if matched[index]:
                    continue
                overlap = box_iou(detection.bbox, target.bbox)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0:
                matched[best_index] = True
                true_positives += 1
                matched_ious.append(best_iou)
                continue
            if any(
                box_iou(detection.bbox, target.bbox) > threshold
                for target in valid
            ):
                duplicates += 1
            ignored = ignored_by_image.get(detection.image_id, [])
            ignored_matched = ignored_matched_by_image.get(detection.image_id, [])
            ignored_index = -1
            ignored_iou = threshold
            for index, target in enumerate(ignored):
                if ignored_matched[index]:
                    continue
                overlap = box_iou(detection.bbox, target.bbox)
                if overlap > ignored_iou:
                    ignored_index, ignored_iou = index, overlap
            if ignored_index >= 0 or detection.bbox.height < min_height:
                if ignored_index >= 0:
                    ignored_matched[ignored_index] = True
                continue
            if any(
                _intersection_over_detection(detection.bbox, region.bbox)
                > threshold
                for region in dontcare_by_image.get(detection.image_id, [])
            ):
                continue
            false_positives += 1
    return _ModerateCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        duplicates=duplicates,
        mean_iou=(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0,
    )


def moderate_utility(
    base_predictions: Mapping[str, Sequence[Detection]],
    reobserved_predictions: Mapping[str, Sequence[Detection]],
    ground_truth_by_image: Mapping[str, Sequence[KittiObject]],
) -> UtilityComponents:
    """Return a frozen Moderate matching proxy for candidate selection.

    Final AP is always computed by :func:`evaluate_prediction_directory`.
    This helper exists only to rank a single crop before the exact evaluator
    runs and therefore is deliberately not presented as an AP implementation.
    """

    base = _moderate_counts(ground_truth_by_image, base_predictions)
    candidate = _moderate_counts(ground_truth_by_image, reobserved_predictions)
    return UtilityComponents(
        delta_tp=float(candidate.true_positives - base.true_positives),
        delta_mean_iou=candidate.mean_iou - base.mean_iou,
        delta_fp=float(candidate.false_positives - base.false_positives),
        delta_duplicates=float(candidate.duplicates - base.duplicates),
    )


__all__ = [
    "NMS_IOU_THRESHOLD",
    "MAX_DETECTIONS",
    "PROPOSAL_CONF_MIN",
    "PROPOSAL_CONF_MAX",
    "UTILITY_TP_WEIGHT",
    "UTILITY_MEAN_IOU_WEIGHT",
    "UTILITY_FP_WEIGHT",
    "UTILITY_DUPLICATE_WEIGHT",
    "UtilityComponents",
    "ImageSize",
    "CropWindow",
    "OracleCandidate",
    "clamp_crop_window",
    "fixed_crop_window",
    "full_to_crop_box",
    "crop_to_full_box",
    "box_to_normalized_yolo",
    "normalized_yolo_to_box",
    "classwise_nms",
    "fuse_detections",
    "build_o1_candidates",
    "build_o2_candidate_pool",
    "select_one_crop",
    "moderate_utility",
]
