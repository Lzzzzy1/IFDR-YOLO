from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.phase1_audit import audit_generated_dataset
from ifdr_yolo.data.splits import load_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all IFDR-YOLO Phase 1 acceptance checks."
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
        "--generated-dir",
        type=Path,
        default=Path("data/processed/kitti_yolo_v2"),
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=Path("docs/reports/ap40-reference-check.json"),
    )
    parser.add_argument(
        "--sample-source-hashes",
        action="store_true",
        help="Verify five deterministic source hashes instead of all 7481.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_ids = load_ids(args.train_ids)
    val_ids = load_ids(args.val_ids)
    summary = audit_generated_dataset(
        source_image_dir=args.image_dir,
        source_label_dir=args.label_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        generated_dir=args.generated_dir,
        verify_all_source_hashes=not args.sample_source_hashes,
    )

    reference = json.loads(
        args.reference_report.read_text(encoding="utf-8")
    )
    if reference["max_absolute_difference"] > reference["tolerance"]:
        raise ValueError("AP40 reference consistency check failed")
    if reference["case_count"] != 45:
        raise ValueError("AP40 reference report does not contain 45 cases")

    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError("unit tests failed")

    print("PHASE 1 ACCEPTED")
    print(
        f"images={summary.image_count} "
        f"train={len(train_ids)} val={len(val_ids)}"
    )
    print(f"labels={summary.label_count} yolo_rows={summary.yolo_row_count}")
    print(
        f"metadata_images={summary.metadata_image_count} "
        f"metadata_objects={summary.metadata_object_count}"
    )
    print(
        f"verified_source_hashes={summary.verified_source_hash_count}"
    )
    print("unit_tests=passed")
    print("split_integrity=passed")
    print("yolo_coordinates=passed")
    print("ap40_reference_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
