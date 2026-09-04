from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.bdd100k_prepare import build_bdd100k_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert official BDD100K JSON annotations to auditable YOLO data."
    )
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--git-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_bdd100k_dataset(
        train_annotations_path=args.train_annotations,
        val_annotations_path=args.val_annotations,
        image_dir=args.images,
        output_dir=args.output,
        overwrite_generated=args.overwrite,
        git_commit=args.git_commit,
    )
    print(
        f"train_images={summary.train.image_count} "
        f"train_objects={summary.train.object_count}"
    )
    print(
        f"val_images={summary.val.image_count} "
        f"val_objects={summary.val.object_count}"
    )
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
