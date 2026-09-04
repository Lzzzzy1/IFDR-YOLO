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
from ifdr_yolo.eval.paired_bootstrap import paired_bootstrap_ap40
from ifdr_yolo.eval.prediction_io import (
    load_kitti_ground_truth,
    load_yolo_predictions,
)
from ifdr_yolo.eval.stratified_ap40 import KITTI_RESEARCH_SLICES
from ifdr_yolo.eval.stratified_report import _validate_prediction_ids


SLICE_BY_NAME = {
    target_slice.name: target_slice
    for target_slice in KITTI_RESEARCH_SLICES
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired image-bootstrap CI for one conditional KITTI AP40 comparison.",
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--class-name", choices=EVAL_CLASSES, required=True)
    parser.add_argument("--slice", choices=tuple(SLICE_BY_NAME), required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_ids = load_ids(args.split.resolve())
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_id in image_ids:
        image_path = args.image_dir.resolve() / f"{image_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"evaluation image does not exist: {image_path}"
            )
        with Image.open(image_path) as image:
            image_sizes[image_id] = image.size

    reference_dir = args.reference_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    _validate_prediction_ids(reference_dir, image_ids)
    _validate_prediction_ids(candidate_dir, image_ids)
    ground_truth = load_kitti_ground_truth(args.label_dir.resolve(), image_ids)
    reference = load_yolo_predictions(reference_dir, image_sizes)
    candidate = load_yolo_predictions(candidate_dir, image_sizes)
    target_slice = SLICE_BY_NAME[args.slice]
    comparison = paired_bootstrap_ap40(
        gt_by_image=ground_truth,
        reference_by_image=reference,
        candidate_by_image=candidate,
        class_name=args.class_name,
        difficulty=Difficulty.HARD,
        iterations=args.iterations,
        seed=args.seed,
        valid_selector=target_slice.matches,
    )
    payload = {
        "schema_version": 1,
        "metric": "KITTI_2D_CONDITIONAL_AP40_PAIRED_IMAGE_BOOTSTRAP",
        "base_difficulty": Difficulty.HARD.value,
        "split_sha256": sha256_file(args.split.resolve()),
        "split_count": len(image_ids),
        "reference": {
            "name": args.reference_name,
            "prediction_dir": str(reference_dir),
        },
        "candidate": {
            "name": args.candidate_name,
            "prediction_dir": str(candidate_dir),
        },
        "class_name": args.class_name,
        "target_slice": asdict(target_slice),
        "comparison": asdict(comparison),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_output.replace(output)
    print(f"bootstrap_json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
