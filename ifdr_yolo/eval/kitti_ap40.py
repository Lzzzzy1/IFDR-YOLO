from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    Difficulty,
    EVAL_CLASSES,
    KittiObject,
)


DIFFICULTY_RULES = {
    Difficulty.EASY: (40.0, 0, 0.15),
    Difficulty.MODERATE: (25.0, 1, 0.30),
    Difficulty.HARD: (25.0, 2, 0.50),
}
CLASS_IOU_THRESHOLDS = {
    "Car": 0.70,
    "Pedestrian": 0.50,
    "Cyclist": 0.50,
}
IGNORE_KIND = {
    "Car": {"Van"},
    "Pedestrian": {"Person_sitting"},
    "Cyclist": set(),
}


class GroundTruthStatus(Enum):
    VALID = "valid"
    IGNORED = "ignored"
    IRRELEVANT = "irrelevant"


@dataclass(frozen=True)
class ClassMetrics:
    ap40: float
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    scores: tuple[float, ...]
    num_valid_gt: int
    true_positives: int
    false_positives: int
    ignored_detections: int


def box_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection_width = max(
        0.0,
        min(left.x2, right.x2) - max(left.x1, right.x1),
    )
    intersection_height = max(
        0.0,
        min(left.y2, right.y2) - max(left.y1, right.y1),
    )
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def is_valid_ground_truth(
    obj: KittiObject,
    class_name: str,
    difficulty: Difficulty,
) -> bool:
    if obj.kind != class_name:
        return False
    min_height, max_occlusion, max_truncation = DIFFICULTY_RULES[difficulty]
    return (
        obj.bbox.height >= min_height
        and obj.occluded <= max_occlusion
        and obj.truncated <= max_truncation
    )


def classify_ground_truth(
    obj: KittiObject,
    class_name: str,
    difficulty: Difficulty,
) -> GroundTruthStatus:
    if obj.kind == class_name:
        if is_valid_ground_truth(obj, class_name, difficulty):
            return GroundTruthStatus.VALID
        return GroundTruthStatus.IGNORED
    if obj.kind in IGNORE_KIND[class_name]:
        return GroundTruthStatus.IGNORED
    return GroundTruthStatus.IRRELEVANT


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
    return (intersection_width * intersection_height) / detection_box.area


def _compute_ap40(
    precision: tuple[float, ...],
    recall: tuple[float, ...],
) -> float:
    sampled_precision: list[float] = []
    for recall_level in (index / 40.0 for index in range(1, 41)):
        candidates = [
            precision_value
            for precision_value, recall_value in zip(precision, recall)
            if recall_value >= recall_level
        ]
        sampled_precision.append(max(candidates, default=0.0))
    return 100.0 * sum(sampled_precision) / 40.0


def evaluate_class(
    gt_by_image: dict[str, tuple[KittiObject, ...]],
    detections_by_image: dict[str, tuple[Detection, ...]],
    class_name: str,
    difficulty: Difficulty,
) -> ClassMetrics:
    """Evaluate one KITTI class/difficulty.

    ``ap40`` is reported on a 0--100 percentage scale. Precision and recall
    curves remain fractions in the 0--1 range.
    """

    if class_name not in EVAL_CLASSES:
        raise ValueError(f"unknown KITTI evaluation class: {class_name}")
    for image_id, detections in detections_by_image.items():
        for detection in detections:
            if detection.image_id != image_id:
                raise ValueError(
                    "detection image ID mismatch: "
                    f"mapping={image_id}, detection={detection.image_id}"
                )

    valid_ground_truth: dict[str, tuple[KittiObject, ...]] = {}
    ignored_ground_truth: dict[str, tuple[KittiObject, ...]] = {}
    dontcare_regions: dict[str, tuple[KittiObject, ...]] = {}
    matched_valid: dict[str, list[bool]] = {}
    matched_ignored: dict[str, list[bool]] = {}
    for image_id, objects in gt_by_image.items():
        valid = tuple(
            obj
            for obj in objects
            if classify_ground_truth(obj, class_name, difficulty)
            is GroundTruthStatus.VALID
        )
        ignored = tuple(
            obj
            for obj in objects
            if classify_ground_truth(obj, class_name, difficulty)
            is GroundTruthStatus.IGNORED
        )
        valid_ground_truth[image_id] = valid
        ignored_ground_truth[image_id] = ignored
        dontcare_regions[image_id] = tuple(
            obj for obj in objects if obj.kind == "DontCare"
        )
        matched_valid[image_id] = [False] * len(valid)
        matched_ignored[image_id] = [False] * len(ignored)

    ranked_detections = sorted(
        (
            detection
            for detections in detections_by_image.values()
            for detection in detections
            if detection.kind == class_name
        ),
        key=lambda detection: detection.score,
        reverse=True,
    )
    num_valid_gt = sum(len(objects) for objects in valid_ground_truth.values())
    threshold = CLASS_IOU_THRESHOLDS[class_name]
    min_detection_height = DIFFICULTY_RULES[difficulty][0]
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    evaluated_scores: list[float] = []
    ignored_detection_count = 0

    for detection in ranked_detections:
        candidates = valid_ground_truth.get(detection.image_id, ())
        candidate_matches = matched_valid.get(detection.image_id, [])
        best_index = -1
        best_iou = threshold
        for index, obj in enumerate(candidates):
            if candidate_matches[index]:
                continue
            overlap = box_iou(detection.bbox, obj.bbox)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0:
            candidate_matches[best_index] = True
            tp_flags.append(1)
            fp_flags.append(0)
            evaluated_scores.append(detection.score)
            continue

        ignored_candidates = ignored_ground_truth.get(detection.image_id, ())
        ignored_matches = matched_ignored.get(detection.image_id, [])
        ignored_index = -1
        ignored_iou = threshold
        for index, obj in enumerate(ignored_candidates):
            if ignored_matches[index]:
                continue
            overlap = box_iou(detection.bbox, obj.bbox)
            if overlap >= ignored_iou:
                ignored_iou = overlap
                ignored_index = index
        if ignored_index >= 0:
            ignored_matches[ignored_index] = True
            ignored_detection_count += 1
            continue

        overlaps_dontcare = any(
            _intersection_over_detection(detection.bbox, region.bbox)
            >= threshold
            for region in dontcare_regions.get(detection.image_id, ())
        )
        if overlaps_dontcare or detection.bbox.height < min_detection_height:
            ignored_detection_count += 1
        else:
            tp_flags.append(0)
            fp_flags.append(1)
            evaluated_scores.append(detection.score)

    precision_values: list[float] = []
    recall_values: list[float] = []
    cumulative_tp = 0
    cumulative_fp = 0
    for is_tp, is_fp in zip(tp_flags, fp_flags):
        cumulative_tp += is_tp
        cumulative_fp += is_fp
        precision_values.append(cumulative_tp / (cumulative_tp + cumulative_fp))
        recall_values.append(
            cumulative_tp / num_valid_gt if num_valid_gt else 0.0
        )

    for index in range(len(precision_values) - 2, -1, -1):
        precision_values[index] = max(
            precision_values[index],
            precision_values[index + 1],
        )

    precision = tuple(precision_values)
    recall = tuple(recall_values)
    return ClassMetrics(
        ap40=_compute_ap40(precision, recall),
        precision=precision,
        recall=recall,
        scores=tuple(evaluated_scores),
        num_valid_gt=num_valid_gt,
        true_positives=sum(tp_flags),
        false_positives=sum(fp_flags),
        ignored_detections=ignored_detection_count,
    )
