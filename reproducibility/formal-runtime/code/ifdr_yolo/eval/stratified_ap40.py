from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from ifdr_yolo.data.kitti_types import (
    Detection,
    Difficulty,
    EVAL_CLASSES,
    KittiObject,
)
from ifdr_yolo.eval.kitti_ap40 import evaluate_class


@dataclass(frozen=True)
class TargetSlice:
    axis: str
    name: str
    lower: float
    upper: float | None
    include_lower: bool
    include_upper: bool

    def __post_init__(self) -> None:
        if self.axis not in {"height", "depth", "occlusion", "truncation"}:
            raise ValueError(f"unsupported target-slice axis: {self.axis}")
        if not self.name.strip():
            raise ValueError("target-slice name must not be empty")
        if not math.isfinite(self.lower):
            raise ValueError("target-slice lower bound must be finite")
        if self.upper is not None and (
            not math.isfinite(self.upper) or self.upper < self.lower
        ):
            raise ValueError("target-slice upper bound must be finite and ordered")

    def matches(self, obj: KittiObject) -> bool:
        if self.axis == "height":
            value = obj.bbox.height
        elif self.axis == "depth":
            value = obj.location_xyz[2]
        elif self.axis == "occlusion":
            value = float(obj.occluded)
        else:
            value = obj.truncated
        lower_match = value >= self.lower if self.include_lower else value > self.lower
        if self.upper is None:
            upper_match = True
        elif self.include_upper:
            upper_match = value <= self.upper
        else:
            upper_match = value < self.upper
        return lower_match and upper_match


KITTI_RESEARCH_SLICES = (
    TargetSlice("height", "small_25_40", 25.0, 40.0, False, True),
    TargetSlice("height", "medium_40_80", 40.0, 80.0, False, True),
    TargetSlice("height", "large_gt_80", 80.0, None, False, False),
    TargetSlice("depth", "near_0_20m", 0.0, 20.0, False, True),
    TargetSlice("depth", "mid_20_40m", 20.0, 40.0, False, True),
    TargetSlice("depth", "far_gt_40m", 40.0, None, False, False),
    TargetSlice("occlusion", "occlusion_0", 0.0, 0.0, True, True),
    TargetSlice("occlusion", "occlusion_1", 1.0, 1.0, True, True),
    TargetSlice("occlusion", "occlusion_2", 2.0, 2.0, True, True),
    TargetSlice("truncation", "truncation_000_015", 0.0, 0.15, True, True),
    TargetSlice("truncation", "truncation_015_030", 0.15, 0.30, False, True),
    TargetSlice("truncation", "truncation_030_050", 0.30, 0.50, False, True),
)


def evaluate_target_slices(
    *,
    gt_by_image: dict[str, tuple[KittiObject, ...]],
    detections_by_image: dict[str, tuple[Detection, ...]],
    target_slices: tuple[TargetSlice, ...] = KITTI_RESEARCH_SLICES,
) -> dict[str, object]:
    grouped: dict[str, dict[str, object]] = {}
    for target_slice in target_slices:
        classes: dict[str, object] = {}
        for class_name in EVAL_CLASSES:
            metrics = evaluate_class(
                gt_by_image=gt_by_image,
                detections_by_image=detections_by_image,
                class_name=class_name,
                difficulty=Difficulty.HARD,
                valid_selector=target_slice.matches,
            )
            classes[class_name] = {
                "ap40": metrics.ap40,
                "num_valid_gt": metrics.num_valid_gt,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "ignored_detections": metrics.ignored_detections,
            }
        grouped.setdefault(target_slice.axis, {})[target_slice.name] = {
            "definition": asdict(target_slice),
            "classes": classes,
        }
    return {
        "schema_version": 1,
        "metric": "KITTI_2D_CONDITIONAL_AP40",
        "base_difficulty": Difficulty.HARD.value,
        "slices": grouped,
    }
