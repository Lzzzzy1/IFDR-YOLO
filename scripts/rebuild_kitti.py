from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.audit import audit_fixed_size_assumption, write_audit_reports
from ifdr_yolo.data.build_dataset import build_dataset
from ifdr_yolo.data.splits import load_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild KITTI YOLO labels with actual image dimensions."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("kitti_raw/training/image_2/training/image_2"),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("kitti_raw/training/label_2/training/label_2"),
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=Path("configs/splits/kitti_train.txt"),
    )
    parser.add_argument(
        "--val-ids",
        type=Path,
        default=Path("configs/splits/kitti_val.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/kitti_yolo_v2"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("docs/reports/kitti_label_rebuild_audit.json"),
    )
    parser.add_argument(
        "--audit-markdown",
        type=Path,
        default=Path("docs/reports/kitti_label_rebuild_audit.md"),
    )
    parser.add_argument("--git-commit")
    parser.add_argument("--overwrite-generated", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("image_dir", "label_dir", "train_ids", "val_ids", "output_dir"):
        print(f"{name}={getattr(args, name).resolve()}")

    train_ids = load_ids(args.train_ids)
    val_ids = load_ids(args.val_ids)
    summary = build_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        output_dir=args.output_dir,
        overwrite_generated=args.overwrite_generated,
        git_commit=args.git_commit,
    )
    audit = audit_fixed_size_assumption(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        image_ids=train_ids + val_ids,
    )
    write_audit_reports(audit, args.audit_json, args.audit_markdown)
    print(
        f"images={summary.image_count} "
        f"train={summary.train_count} val={summary.val_count}"
    )
    print(
        f"invalid_boxes={summary.invalid_box_count} "
        f"clipped_boxes={summary.clipped_box_count}"
    )
    print(f"fixed_size_affected_targets={audit.affected_target_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
