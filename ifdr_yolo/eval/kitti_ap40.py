from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ClassMetrics:
    ap40: float
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    scores: tuple[float, ...]
    num_valid_gt: int
    true_positives: int
    false_positives: int


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

    valid_ground_truth: dict[str, tuple[KittiObject, ...]] = {}
    matched: dict[str, list[bool]] = {}
    for image_id, objects in gt_by_image.items():
        valid = tuple(
            obj
            for obj in objects
            if is_valid_ground_truth(obj, class_name, difficulty)
        )
        valid_ground_truth[image_id] = valid
        matched[image_id] = [False] * len(valid)

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
    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for detection in ranked_detections:
        candidates = valid_ground_truth.get(detection.image_id, ())
        candidate_matches = matched.get(detection.image_id, [])
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
        else:
            tp_flags.append(0)
            fp_flags.append(1)

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
        scores=tuple(detection.score for detection in ranked_detections),
        num_valid_gt=num_valid_gt,
        true_positives=sum(tp_flags),
        false_positives=sum(fp_flags),
    )
