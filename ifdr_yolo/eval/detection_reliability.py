from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import math

from ifdr_yolo.data.kitti_types import (
    Detection,
    Difficulty,
    KittiObject,
)
from ifdr_yolo.eval.kitti_ap40 import evaluate_class


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_score: float
    mean_iou: float
    absolute_gap: float


@dataclass(frozen=True)
class DetectionReliabilityMetrics:
    confidence_threshold: float
    matching_iou_threshold: float
    laece0: float | None
    lrp: float
    lrp_loc: float | None
    lrp_fp: float | None
    lrp_fn: float
    num_valid_gt: int
    true_positives: int
    false_positives: int
    false_negatives: int
    evaluated_detections: int
    bins: tuple[ReliabilityBin, ...]


def _validate_threshold(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be within [0, 1]")
    return float(value)


def _lrp_components(
    *,
    matched_ious: tuple[float, ...],
    num_valid_gt: int,
    matching_iou_threshold: float,
) -> tuple[float, float | None, float | None, float, int, int, int]:
    true_positives = sum(iou > matching_iou_threshold for iou in matched_ious)
    false_positives = len(matched_ious) - true_positives
    false_negatives = num_valid_gt - true_positives
    localization_error = sum(
        1.0 - iou
        for iou in matched_ious
        if iou > matching_iou_threshold
    )
    denominator = true_positives + false_positives + false_negatives
    lrp = (
        (
            localization_error / (1.0 - matching_iou_threshold)
            + false_positives
            + false_negatives
        )
        / denominator
        if denominator
        else 0.0
    )
    lrp_loc = (
        localization_error / true_positives if true_positives else None
    )
    lrp_fp = (
        false_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )
    lrp_fn = false_negatives / num_valid_gt if num_valid_gt else 0.0
    return (
        lrp,
        lrp_loc,
        lrp_fp,
        lrp_fn,
        true_positives,
        false_positives,
        false_negatives,
    )


def evaluate_detection_reliability(
    *,
    gt_by_image: dict[str, tuple[KittiObject, ...]],
    detections_by_image: dict[str, tuple[Detection, ...]],
    class_name: str,
    difficulty: Difficulty,
    confidence_threshold: float,
    bins: int = 25,
    valid_selector: Callable[[KittiObject], bool] | None = None,
    matching_iou_threshold: float = 0.0,
) -> DetectionReliabilityMetrics:
    confidence_threshold = _validate_threshold(
        confidence_threshold, "confidence threshold"
    )
    matching_iou_threshold = _validate_threshold(
        matching_iou_threshold, "matching IoU threshold"
    )
    if matching_iou_threshold >= 1.0:
        raise ValueError("matching IoU threshold must be less than 1")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    if set(gt_by_image) != set(detections_by_image):
        raise ValueError("ground truth and detections must use the same image IDs")
    filtered = {
        image_id: tuple(
            detection
            for detection in detections_by_image[image_id]
            if detection.score >= confidence_threshold
        )
        for image_id in gt_by_image
    }
    metrics = evaluate_class(
        gt_by_image,
        filtered,
        class_name,
        difficulty,
        valid_selector,
        iou_threshold=matching_iou_threshold,
    )
    if metrics.num_valid_gt == 0:
        raise ValueError("reliability evaluation requires a valid target")
    if any(not 0.0 <= score <= 1.0 for score in metrics.scores):
        raise ValueError("detection confidence must be within [0, 1]")

    reliability_bins: list[ReliabilityBin] = []
    weighted_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = tuple(
            item
            for item in zip(metrics.scores, metrics.matched_ious, strict=True)
            if (
                lower <= item[0] <= upper
                if index == 0
                else lower < item[0] <= upper
            )
        )
        if not selected:
            continue
        mean_score = sum(score for score, _ in selected) / len(selected)
        mean_iou = sum(iou for _, iou in selected) / len(selected)
        gap = abs(mean_score - mean_iou)
        weighted_error += gap * len(selected)
        reliability_bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(selected),
                mean_score=mean_score,
                mean_iou=mean_iou,
                absolute_gap=gap,
            )
        )
    lrp, lrp_loc, lrp_fp, lrp_fn, true_positives, false_positives, false_negatives = (
        _lrp_components(
            matched_ious=metrics.matched_ious,
            num_valid_gt=metrics.num_valid_gt,
            matching_iou_threshold=matching_iou_threshold,
        )
    )
    return DetectionReliabilityMetrics(
        confidence_threshold=confidence_threshold,
        matching_iou_threshold=matching_iou_threshold,
        laece0=(
            weighted_error / len(metrics.scores) if metrics.scores else None
        ),
        lrp=lrp,
        lrp_loc=lrp_loc,
        lrp_fp=lrp_fp,
        lrp_fn=lrp_fn,
        num_valid_gt=metrics.num_valid_gt,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        evaluated_detections=len(metrics.scores),
        bins=tuple(reliability_bins),
    )


def select_lrp_threshold(
    *,
    gt_by_image: dict[str, tuple[KittiObject, ...]],
    detections_by_image: dict[str, tuple[Detection, ...]],
    class_name: str,
    difficulty: Difficulty,
    valid_selector: Callable[[KittiObject], bool] | None = None,
    matching_iou_threshold: float = 0.0,
) -> float:
    matching_iou_threshold = _validate_threshold(
        matching_iou_threshold, "matching IoU threshold"
    )
    if matching_iou_threshold >= 1.0:
        raise ValueError("matching IoU threshold must be less than 1")
    metrics = evaluate_class(
        gt_by_image,
        detections_by_image,
        class_name,
        difficulty,
        valid_selector,
        iou_threshold=matching_iou_threshold,
    )
    if metrics.num_valid_gt == 0:
        raise ValueError("threshold selection requires a valid target")
    if not metrics.scores:
        return 1.0

    best_threshold = metrics.scores[0]
    best_lrp = math.inf
    for index, score in enumerate(metrics.scores):
        if index + 1 < len(metrics.scores) and metrics.scores[index + 1] == score:
            continue
        lrp = _lrp_components(
            matched_ious=metrics.matched_ious[: index + 1],
            num_valid_gt=metrics.num_valid_gt,
            matching_iou_threshold=matching_iou_threshold,
        )[0]
        if (lrp, -score) < (best_lrp, -best_threshold):
            best_lrp = lrp
            best_threshold = score
    return best_threshold


def deterministic_calibration_split(
    image_ids: tuple[str, ...],
    *,
    seed: int,
    calibration_fraction: float = 0.5,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(image_ids) < 2 or len(set(image_ids)) != len(image_ids):
        raise ValueError("image IDs must be unique and contain at least two items")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a non-negative integer")
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not 0.0 < float(calibration_fraction) < 1.0
    ):
        raise ValueError("calibration fraction must be within (0, 1)")
    ordered = sorted(
        image_ids,
        key=lambda image_id: sha256(
            f"{seed}:{image_id}".encode("utf-8")
        ).digest(),
    )
    calibration_count = max(
        1,
        min(len(ordered) - 1, round(len(ordered) * calibration_fraction)),
    )
    return (
        tuple(sorted(ordered[:calibration_count])),
        tuple(sorted(ordered[calibration_count:])),
    )
