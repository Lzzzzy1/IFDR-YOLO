from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.kitti_types import Difficulty, EVAL_CLASSES
from ifdr_yolo.eval.evaluate import (
    evaluate_prediction_directory,
    write_evaluation_json,
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
    payload = evaluate_prediction_directory(
        prediction_dir=args.prediction_dir,
        label_dir=args.label_dir,
        image_dir=args.image_dir,
        split_path=args.split,
    )
    classes = payload["classes"]
    assert isinstance(classes, dict)
    print("class difficulty ap40 valid_gt tp fp ignored")
    for class_name in EVAL_CLASSES:
        class_metrics = classes[class_name]
        assert isinstance(class_metrics, dict)
        for difficulty in Difficulty:
            metrics = class_metrics[difficulty.value]
            assert isinstance(metrics, dict)
            print(
                f"{class_name} {difficulty.value} "
                f"{float(metrics['ap40']):.4f} "
                f"{metrics['num_valid_gt']} {metrics['true_positives']} "
                f"{metrics['false_positives']} "
                f"{metrics['ignored_detections']}"
            )
    write_evaluation_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
