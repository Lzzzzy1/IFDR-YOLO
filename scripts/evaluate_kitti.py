from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ifdr_yolo.data.kitti_types import Difficulty, EVAL_CLASSES
from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.eval.kitti_ap40 import evaluate_class
from ifdr_yolo.eval.prediction_io import (
    load_kitti_ground_truth,
    load_yolo_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO predictions with KITTI 2D AP40."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("kitti_raw/training/label_2/training/label_2"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("kitti_raw/training/image_2/training/image_2"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/kitti_val.txt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_ids = load_ids(args.split)
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_id in image_ids:
        with Image.open(args.image_dir / f"{image_id}.png") as image:
            image_sizes[image_id] = image.size

    ground_truth = load_kitti_ground_truth(args.label_dir, image_ids)
    predictions = load_yolo_predictions(args.prediction_dir, image_sizes)
    class_payload: dict[str, dict[str, dict[str, object]]] = {}

    print("class difficulty ap40 valid_gt tp fp ignored")
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
            print(
                f"{class_name} {difficulty.value} {metrics.ap40:.4f} "
                f"{metrics.num_valid_gt} {metrics.true_positives} "
                f"{metrics.false_positives} {metrics.ignored_detections}"
            )
        class_payload[class_name] = difficulty_payload

    payload = {
        "evaluator": "ifdr_yolo.kitti_ap40",
        "split_sha256": sha256_file(args.split),
        "split_count": len(image_ids),
        "classes": class_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
