from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    Difficulty,
    KittiObject,
)
from ifdr_yolo.data.splits import sha256_file
from ifdr_yolo.eval.kitti_ap40 import CLASS_IOU_THRESHOLDS, evaluate_class


def _make_object(kind: str, box: BoundingBox) -> KittiObject:
    return KittiObject(
        kind=kind,
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox=box,
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 10.0),
        rotation_y=0.0,
    )


def build_controlled_suite() -> tuple[
    dict[str, tuple[KittiObject, ...]],
    dict[str, dict[str, tuple[Detection, ...]]],
]:
    image_ids = tuple(f"{index:06d}" for index in range(50))
    ground_truth = {
        image_id: (
            _make_object("Car", BoundingBox(0, 0, 100, 100)),
            _make_object("Pedestrian", BoundingBox(200, 0, 250, 80)),
            _make_object("Cyclist", BoundingBox(300, 0, 350, 80)),
            _make_object("Van", BoundingBox(400, 0, 500, 100)),
            _make_object("Person_sitting", BoundingBox(550, 0, 600, 80)),
            _make_object("DontCare", BoundingBox(700, 0, 800, 100)),
        )
        for image_id in image_ids
    }

    scenarios: dict[str, dict[str, tuple[Detection, ...]]] = {}
    for scenario_name in (
        "perfect",
        "duplicate",
        "high_fp",
        "half_missed",
        "ignore",
    ):
        by_image: dict[str, tuple[Detection, ...]] = {}
        for image_number, image_id in enumerate(image_ids):
            detections: list[Detection] = []
            for object_number, obj in enumerate(ground_truth[image_id]):
                if obj.kind not in ("Car", "Pedestrian", "Cyclist"):
                    continue
                if scenario_name == "half_missed" and image_number >= 25:
                    continue
                score = 1.0 - (image_number * 100 + object_number) * 1e-6
                if scenario_name == "high_fp":
                    false_box = BoundingBox(
                        obj.bbox.x1,
                        obj.bbox.y1 + 150,
                        obj.bbox.x2,
                        obj.bbox.y2 + 150,
                    )
                    detections.append(
                        Detection(image_id, obj.kind, score, false_box)
                    )
                    score -= 0.5
                detections.append(
                    Detection(image_id, obj.kind, score, obj.bbox)
                )
                if scenario_name == "duplicate":
                    detections.append(
                        Detection(image_id, obj.kind, score - 0.2, obj.bbox)
                    )
            if scenario_name == "ignore":
                detections.extend(
                    (
                        Detection(
                            image_id,
                            "Car",
                            0.95,
                            BoundingBox(400, 0, 500, 100),
                        ),
                        Detection(
                            image_id,
                            "Pedestrian",
                            0.94,
                            BoundingBox(550, 0, 600, 80),
                        ),
                        Detection(
                            image_id,
                            "Cyclist",
                            0.93,
                            BoundingBox(700, 0, 800, 100),
                        ),
                    )
                )
            by_image[image_id] = tuple(detections)
        scenarios[scenario_name] = by_image
    return ground_truth, scenarios


def to_reference_ground_truth_annotation(
    objects: tuple[KittiObject, ...],
) -> dict[str, np.ndarray]:
    count = len(objects)
    return {
        "name": np.array([obj.kind for obj in objects]),
        "truncated": np.array(
            [obj.truncated for obj in objects],
            dtype=float,
        ),
        "occluded": np.array([obj.occluded for obj in objects], dtype=int),
        "alpha": np.array([obj.alpha for obj in objects], dtype=float),
        "bbox": np.array(
            [obj.bbox.as_xyxy() for obj in objects],
            dtype=float,
        ).reshape(count, 4),
        "dimensions": np.array(
            [obj.dimensions_hwl for obj in objects],
            dtype=float,
        ).reshape(count, 3),
        "location": np.array(
            [obj.location_xyz for obj in objects],
            dtype=float,
        ).reshape(count, 3),
        "rotation_y": np.array(
            [obj.rotation_y for obj in objects],
            dtype=float,
        ),
    }


def to_reference_detection_annotation(
    detections: tuple[Detection, ...],
) -> dict[str, np.ndarray]:
    count = len(detections)
    return {
        "name": np.array([detection.kind for detection in detections]),
        "alpha": np.full((count,), -10.0),
        "bbox": np.array(
            [detection.bbox.as_xyxy() for detection in detections],
            dtype=float,
        ).reshape(count, 4),
        "dimensions": np.zeros((count, 3), dtype=float),
        "location": np.zeros((count, 3), dtype=float),
        "rotation_y": np.zeros((count,), dtype=float),
        "score": np.array(
            [detection.score for detection in detections],
            dtype=float,
        ),
    }


def _load_openpcdet_reference(reference_dir: Path):
    eval_path = reference_dir / "eval.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"reference eval.py not found: {eval_path}")

    numba_stub = types.ModuleType("numba")

    def jit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda function: function

    numba_stub.jit = jit
    sys.modules["numba"] = numba_stub

    package_name = "_ifdr_openpcdet_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(reference_dir)]
    sys.modules[package_name] = package
    rotate_stub = types.ModuleType(f"{package_name}.rotate_iou")

    def reject_rotated_iou(*args, **kwargs):
        raise RuntimeError("rotated IoU must not be called by the 2D check")

    rotate_stub.rotate_iou_gpu_eval = reject_rotated_iou
    sys.modules[f"{package_name}.rotate_iou"] = rotate_stub

    module_name = f"{package_name}.eval"
    spec = importlib.util.spec_from_file_location(module_name, eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference evaluator: {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_reference_check(reference_dir: Path) -> dict[str, object]:
    reference = _load_openpcdet_reference(reference_dir)
    ground_truth, scenarios = build_controlled_suite()
    image_ids = tuple(ground_truth)
    gt_annos = [
        to_reference_ground_truth_annotation(ground_truth[image_id])
        for image_id in image_ids
    ]
    class_names = ("Car", "Pedestrian", "Cyclist")
    case_results: list[dict[str, object]] = []
    max_absolute_difference = 0.0

    for scenario_name, detections_by_image in scenarios.items():
        dt_annos = [
            to_reference_detection_annotation(detections_by_image[image_id])
            for image_id in image_ids
        ]
        for class_index, class_name in enumerate(class_names):
            min_overlaps = np.full(
                (1, 3, 1),
                CLASS_IOU_THRESHOLDS[class_name],
                dtype=float,
            )
            reference_result = reference.eval_class(
                gt_annos,
                dt_annos,
                [class_index],
                [0, 1, 2],
                0,
                min_overlaps,
                False,
                num_parts=1,
            )
            reference_ap = reference.get_mAP_R40(
                reference_result["precision"]
            )[0, :, 0]
            ours = np.array(
                [
                    evaluate_class(
                        ground_truth,
                        detections_by_image,
                        class_name,
                        difficulty,
                    ).ap40
                    for difficulty in Difficulty
                ]
            )
            differences = np.abs(ours - reference_ap)
            max_absolute_difference = max(
                max_absolute_difference,
                float(np.max(differences)),
            )
            for index, difficulty in enumerate(Difficulty):
                case_results.append(
                    {
                        "scenario": scenario_name,
                        "class_name": class_name,
                        "difficulty": difficulty.value,
                        "ours_ap40": float(ours[index]),
                        "reference_ap40": float(reference_ap[index]),
                        "absolute_difference": float(differences[index]),
                    }
                )

    payload: dict[str, object] = {
        "reference_eval_sha256": sha256_file(reference_dir / "eval.py"),
        "reference_rotate_iou_sha256": (
            sha256_file(reference_dir / "rotate_iou.py")
            if (reference_dir / "rotate_iou.py").exists()
            else None
        ),
        "controlled_image_count": len(image_ids),
        "case_count": len(case_results),
        "max_absolute_difference": max_absolute_difference,
        "cases": case_results,
    }
    return payload
