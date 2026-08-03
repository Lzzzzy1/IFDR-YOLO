from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import random

from ifdr_yolo.data.kitti_types import (
    Detection,
    Difficulty,
    KittiObject,
)
from ifdr_yolo.eval.kitti_ap40 import evaluate_class


@dataclass(frozen=True)
class PairedBootstrapAP40:
    reference_ap40: float
    candidate_ap40: float
    difference_ap40: float
    ci_lower: float
    ci_upper: float
    confidence: float
    probability_improvement: float
    iterations: int
    seed: int


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _resample_by_image(
    image_ids: tuple[str, ...],
    sampled_ids: list[str],
    gt_by_image: dict[str, tuple[KittiObject, ...]],
    detections_by_image: dict[str, tuple[Detection, ...]],
) -> tuple[
    dict[str, tuple[KittiObject, ...]],
    dict[str, tuple[Detection, ...]],
]:
    ground_truth: dict[str, tuple[KittiObject, ...]] = {}
    detections: dict[str, tuple[Detection, ...]] = {}
    if not set(sampled_ids).issubset(image_ids):
        raise ValueError("bootstrap sample contains an unknown image ID")
    for index, source_id in enumerate(sampled_ids):
        sampled_id = f"bootstrap_{index:06d}"
        ground_truth[sampled_id] = gt_by_image[source_id]
        detections[sampled_id] = tuple(
            Detection(
                image_id=sampled_id,
                kind=detection.kind,
                score=detection.score,
                bbox=detection.bbox,
            )
            for detection in detections_by_image[source_id]
        )
    return ground_truth, detections


def paired_bootstrap_ap40(
    *,
    gt_by_image: dict[str, tuple[KittiObject, ...]],
    reference_by_image: dict[str, tuple[Detection, ...]],
    candidate_by_image: dict[str, tuple[Detection, ...]],
    class_name: str,
    difficulty: Difficulty,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
    valid_selector: Callable[[KittiObject], bool] | None = None,
) -> PairedBootstrapAP40:
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise ValueError("bootstrap iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("bootstrap confidence must be within (0, 1)")
    image_ids = tuple(sorted(gt_by_image))
    if (
        not image_ids
        or set(image_ids) != set(reference_by_image)
        or set(image_ids) != set(candidate_by_image)
    ):
        raise ValueError(
            "ground truth, reference and candidate must use the same image IDs"
        )

    reference = evaluate_class(
        gt_by_image,
        reference_by_image,
        class_name,
        difficulty,
        valid_selector,
    )
    candidate = evaluate_class(
        gt_by_image,
        candidate_by_image,
        class_name,
        difficulty,
        valid_selector,
    )
    if reference.num_valid_gt == 0:
        raise ValueError("paired bootstrap requires at least one valid target")

    generator = random.Random(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled_ids = generator.choices(image_ids, k=len(image_ids))
        sampled_gt, sampled_reference = _resample_by_image(
            image_ids,
            sampled_ids,
            gt_by_image,
            reference_by_image,
        )
        _, sampled_candidate = _resample_by_image(
            image_ids,
            sampled_ids,
            gt_by_image,
            candidate_by_image,
        )
        reference_metrics = evaluate_class(
            sampled_gt,
            sampled_reference,
            class_name,
            difficulty,
            valid_selector,
        )
        candidate_metrics = evaluate_class(
            sampled_gt,
            sampled_candidate,
            class_name,
            difficulty,
            valid_selector,
        )
        if reference_metrics.num_valid_gt:
            differences.append(
                candidate_metrics.ap40 - reference_metrics.ap40
            )
    if not differences:
        raise RuntimeError("bootstrap produced no samples with valid targets")

    tail = (1.0 - float(confidence)) / 2.0
    difference = candidate.ap40 - reference.ap40
    return PairedBootstrapAP40(
        reference_ap40=reference.ap40,
        candidate_ap40=candidate.ap40,
        difference_ap40=difference,
        ci_lower=_quantile(differences, tail),
        ci_upper=_quantile(differences, 1.0 - tail),
        confidence=float(confidence),
        probability_improvement=(
            sum(value > 0.0 for value in differences) / len(differences)
        ),
        iterations=len(differences),
        seed=seed,
    )
