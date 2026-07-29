from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image

from ifdr_yolo.data.kitti_types import Difficulty, EVAL_CLASSES
from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.eval.kitti_ap40 import evaluate_class
from ifdr_yolo.eval.prediction_io import (
    load_kitti_ground_truth,
    load_yolo_predictions,
)


def evaluate_prediction_directory(
    *,
    prediction_dir: Path,
    label_dir: Path,
    image_dir: Path,
    split_path: Path,
) -> dict[str, object]:
    image_ids = load_ids(split_path)
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"evaluation image does not exist: {image_path}")
        with Image.open(image_path) as image:
            image_sizes[image_id] = image.size

    ground_truth = load_kitti_ground_truth(label_dir, image_ids)
    predictions = load_yolo_predictions(prediction_dir, image_sizes)
    class_payload: dict[str, dict[str, dict[str, object]]] = {}
    for class_name in EVAL_CLASSES:
        difficulty_payload: dict[str, dict[str, object]] = {}
        for difficulty in Difficulty:
            metrics = evaluate_class(
                gt_by_image=ground_truth,
                detections_by_image=predictions,
                class_name=class_name,
                difficulty=difficulty,
            )
            difficulty_payload[difficulty.value] = asdict(metrics)
        class_payload[class_name] = difficulty_payload

    return {
        "evaluator": "ifdr_yolo.kitti_ap40",
        "split_sha256": sha256_file(split_path),
        "split_count": len(image_ids),
        "classes": class_payload,
    }


def write_evaluation_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
