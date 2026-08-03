from __future__ import annotations

import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image

from ifdr_yolo.data.kitti_types import Difficulty, EVAL_CLASSES
from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.eval.detection_reliability import (
    deterministic_calibration_split,
    evaluate_detection_reliability,
    select_lrp_threshold,
)
from ifdr_yolo.eval.prediction_io import (
    load_kitti_ground_truth,
    load_yolo_predictions,
)
from ifdr_yolo.eval.stratified_ap40 import TargetSlice
from ifdr_yolo.eval.stratified_report import _validate_prediction_ids


def _ids_sha256(image_ids: tuple[str, ...]) -> str:
    return sha256(("\n".join(image_ids) + "\n").encode("utf-8")).hexdigest()


def evaluate_reliability_runs(
    *,
    run_prediction_dirs: dict[str, Path],
    label_dir: Path,
    image_dir: Path,
    split_path: Path,
    split_seed: int,
    class_names: tuple[str, ...] = EVAL_CLASSES,
    target_slices: tuple[TargetSlice, ...] = (),
    bins: int = 25,
) -> dict[str, object]:
    if not run_prediction_dirs:
        raise ValueError("at least one prediction run is required")
    if not class_names or any(name not in EVAL_CLASSES for name in class_names):
        raise ValueError("reliability report contains an unknown class")
    target_names = tuple(target_slice.name for target_slice in target_slices)
    if len(set(target_names)) != len(target_names) or "overall" in target_names:
        raise ValueError("reliability target names must be unique")

    image_ids = load_ids(split_path)
    calibration_ids, test_ids = deterministic_calibration_split(
        image_ids,
        seed=split_seed,
    )
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"evaluation image does not exist: {image_path}"
            )
        with Image.open(image_path) as image:
            image_sizes[image_id] = image.size
    ground_truth = load_kitti_ground_truth(label_dir, image_ids)
    calibration_gt = {
        image_id: ground_truth[image_id] for image_id in calibration_ids
    }
    test_gt = {image_id: ground_truth[image_id] for image_id in test_ids}

    runs: dict[str, object] = {}
    targets = (("overall", None),) + tuple(
        (target_slice.name, target_slice) for target_slice in target_slices
    )
    for run_name, prediction_dir in sorted(run_prediction_dirs.items()):
        _validate_prediction_ids(prediction_dir, image_ids)
        predictions = load_yolo_predictions(prediction_dir, image_sizes)
        calibration_predictions = {
            image_id: predictions[image_id] for image_id in calibration_ids
        }
        test_predictions = {
            image_id: predictions[image_id] for image_id in test_ids
        }
        classes: dict[str, object] = {}
        for class_name in class_names:
            confidence_threshold = select_lrp_threshold(
                gt_by_image=calibration_gt,
                detections_by_image=calibration_predictions,
                class_name=class_name,
                difficulty=Difficulty.HARD,
            )
            target_results: dict[str, object] = {}
            for target_name, target_slice in targets:
                selector = (
                    target_slice.matches if target_slice is not None else None
                )
                try:
                    metrics = evaluate_detection_reliability(
                        gt_by_image=test_gt,
                        detections_by_image=test_predictions,
                        class_name=class_name,
                        difficulty=Difficulty.HARD,
                        confidence_threshold=confidence_threshold,
                        bins=bins,
                        valid_selector=selector,
                    )
                    target_results[target_name] = {
                        "supported": True,
                        **asdict(metrics),
                    }
                except ValueError as error:
                    if "valid target" not in str(error):
                        raise
                    target_results[target_name] = {
                        "supported": False,
                        "reason": "no_valid_target",
                    }
            classes[class_name] = {
                "confidence_threshold": confidence_threshold,
                "targets": target_results,
            }
        runs[run_name] = {
            "prediction_dir": str(prediction_dir.resolve()),
            "classes": classes,
        }
    return {
        "schema_version": 1,
        "metric": "KITTI_LAECE0_LRP_DISJOINT_CALIBRATION",
        "protocol": {
            "source_split_sha256": sha256_file(split_path),
            "source_count": len(image_ids),
            "split_seed": split_seed,
            "calibration_count": len(calibration_ids),
            "test_count": len(test_ids),
            "calibration_ids_sha256": _ids_sha256(calibration_ids),
            "test_ids_sha256": _ids_sha256(test_ids),
            "base_difficulty": Difficulty.HARD.value,
            "matching_iou_threshold": 0.0,
            "threshold_selection": "LRP-optimal on calibration split",
            "bins": bins,
        },
        "target_definitions": {
            "overall": None,
            **{
                target_slice.name: asdict(target_slice)
                for target_slice in target_slices
            },
        },
        "runs": runs,
    }


def write_reliability_report(
    output_dir: Path,
    report: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "detection_reliability.json"
    csv_path = output_dir / "detection_reliability.csv"
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_json.replace(json_path)

    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "run",
                "class",
                "target",
                "confidence_threshold",
                "laece0",
                "lrp",
                "lrp_loc",
                "lrp_fp",
                "lrp_fn",
                "num_valid_gt",
                "true_positives",
                "false_positives",
                "false_negatives",
                "evaluated_detections",
                "supported",
            )
        )
        for run_name, run in sorted(report["runs"].items()):
            for class_name, class_payload in sorted(run["classes"].items()):
                threshold = class_payload["confidence_threshold"]
                for target_name, metrics in sorted(
                    class_payload["targets"].items()
                ):
                    writer.writerow(
                        (
                            run_name,
                            class_name,
                            target_name,
                            threshold,
                            metrics.get("laece0"),
                            metrics.get("lrp"),
                            metrics.get("lrp_loc"),
                            metrics.get("lrp_fp"),
                            metrics.get("lrp_fn"),
                            metrics.get("num_valid_gt"),
                            metrics.get("true_positives"),
                            metrics.get("false_positives"),
                            metrics.get("false_negatives"),
                            metrics.get("evaluated_detections"),
                            metrics["supported"],
                        )
                    )
    temporary_csv.replace(csv_path)
    return json_path, csv_path
