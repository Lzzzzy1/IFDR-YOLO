from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SOURCE_ROOT = Path(
    r"E:\myyolo\stage11-v213-anchored-rerun-20260827e"
)
sys.path.insert(0, str(SOURCE_ROOT))

from ifdr_yolo.data.kitti_types import (  # noqa: E402
    BoundingBox,
    Detection,
    KittiObject,
)
from ifdr_yolo.eval.stratified_ap40 import (  # noqa: E402
    KITTI_RESEARCH_SLICES,
)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_moderate_stratified.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_moderate_stratified",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _object(
    kind: str,
    *,
    height: float,
    occluded: int,
    truncated: float,
    depth: float,
    x1: float,
) -> KittiObject:
    return KittiObject(
        kind=kind,
        truncated=truncated,
        occluded=occluded,
        alpha=0.0,
        bbox=BoundingBox(x1, 0.0, x1 + 20.0, height),
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, depth),
        rotation_y=0.0,
    )


def _detection(image_id: str, obj: KittiObject, score: float) -> Detection:
    return Detection(
        image_id=image_id,
        kind=obj.kind,
        score=score,
        bbox=obj.bbox,
    )


def test_moderate_slices_exclude_hard_only_ground_truth() -> None:
    image_id = "000001"
    moderate_small_ped = _object(
        "Pedestrian",
        height=30.0,
        occluded=1,
        truncated=0.30,
        depth=45.0,
        x1=0.0,
    )
    hard_only_small_ped = _object(
        "Pedestrian",
        height=30.0,
        occluded=2,
        truncated=0.50,
        depth=45.0,
        x1=30.0,
    )
    moderate_large_cyc = _object(
        "Cyclist",
        height=90.0,
        occluded=0,
        truncated=0.0,
        depth=10.0,
        x1=60.0,
    )
    ground_truth = {
        image_id: (
            moderate_small_ped,
            hard_only_small_ped,
            moderate_large_cyc,
        )
    }
    detections = {
        image_id: (
            _detection(image_id, moderate_small_ped, 0.9),
            _detection(image_id, hard_only_small_ped, 0.8),
            _detection(image_id, moderate_large_cyc, 0.7),
        )
    }

    report = MODULE.evaluate_moderate_slices(
        gt_by_image=ground_truth,
        detections_by_image=detections,
        classes=("Pedestrian", "Cyclist"),
        target_slices=KITTI_RESEARCH_SLICES,
    )

    assert report["metric"] == "KITTI_2D_MODERATE_CONDITIONAL_AP40"
    assert report["base_difficulty"] == "moderate"
    small = report["slices"]["height"]["small_25_40"]
    assert small["classes"]["Pedestrian"]["num_valid_gt"] == 1
    assert small["classes"]["Pedestrian"]["true_positives"] == 1
    assert small["classes"]["Pedestrian"]["ap40"] == 100.0
    assert small["macro_ap40"] == 50.0
    large = report["slices"]["height"]["large_gt_80"]
    assert large["classes"]["Cyclist"]["num_valid_gt"] == 1
    assert large["classes"]["Cyclist"]["ap40"] == 100.0


if __name__ == "__main__":
    test_moderate_slices_exclude_hard_only_ground_truth()
    print("PASS: Moderate-valid slice semantics")
