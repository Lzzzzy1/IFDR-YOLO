from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.eval.prediction_io import (
    load_kitti_ground_truth,
    load_yolo_predictions,
)
from ifdr_yolo.eval.stratified_ap40 import evaluate_target_slices


def _validate_prediction_ids(
    prediction_dir: Path,
    image_ids: tuple[str, ...],
) -> None:
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"prediction directory does not exist: {prediction_dir}"
        )
    actual = {path.stem for path in prediction_dir.glob("*.txt")}
    expected = set(image_ids)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "prediction IDs do not match split: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def evaluate_stratified_runs(
    *,
    run_prediction_dirs: dict[str, Path],
    label_dir: Path,
    image_dir: Path,
    split_path: Path,
) -> dict[str, object]:
    if not run_prediction_dirs:
        raise ValueError("at least one prediction run is required")
    image_ids = load_ids(split_path)
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

    runs: dict[str, object] = {}
    for run_name, prediction_dir in sorted(run_prediction_dirs.items()):
        _validate_prediction_ids(prediction_dir, image_ids)
        predictions = load_yolo_predictions(prediction_dir, image_sizes)
        evaluation = evaluate_target_slices(
            gt_by_image=ground_truth,
            detections_by_image=predictions,
        )
        evaluation["prediction_dir"] = str(prediction_dir.resolve())
        runs[run_name] = evaluation
    return {
        "schema_version": 1,
        "evaluator": "ifdr_yolo.stratified_ap40",
        "split_sha256": sha256_file(split_path),
        "split_count": len(image_ids),
        "runs": runs,
    }


def write_stratified_report(
    output_dir: Path,
    report: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stratified_ap40.json"
    csv_path = output_dir / "stratified_ap40.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "run",
                "axis",
                "slice",
                "class",
                "ap40",
                "num_valid_gt",
                "true_positives",
                "false_positives",
                "ignored_detections",
            )
        )
        for run_name, run in sorted(report["runs"].items()):
            for axis, slices in sorted(run["slices"].items()):
                for slice_name, payload in sorted(slices.items()):
                    for class_name, metrics in sorted(
                        payload["classes"].items()
                    ):
                        writer.writerow(
                            (
                                run_name,
                                axis,
                                slice_name,
                                class_name,
                                metrics["ap40"],
                                metrics["num_valid_gt"],
                                metrics["true_positives"],
                                metrics["false_positives"],
                                metrics["ignored_detections"],
                            )
                        )
    return json_path, csv_path
