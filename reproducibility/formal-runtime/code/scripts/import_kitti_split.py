from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.splits import (
    discover_ids,
    load_ids,
    sha256_file,
    validate_split,
)


TRAIN_URL = (
    "https://raw.githubusercontent.com/traveller59/second.pytorch/"
    "master/second/data/ImageSets/train.txt"
)
VAL_URL = (
    "https://raw.githubusercontent.com/traveller59/second.pytorch/"
    "master/second/data/ImageSets/val.txt"
)


@dataclass(frozen=True)
class SplitImportSummary:
    name: str
    train_count: int
    val_count: int
    train_url: str
    val_url: str
    train_sha256: str
    val_sha256: str


def import_split_files(
    *,
    train_source: Path,
    val_source: Path,
    output_dir: Path,
    available_ids: set[str],
    expected_train_count: int,
    expected_val_count: int,
    train_url: str,
    val_url: str,
) -> SplitImportSummary:
    train_ids = load_ids(train_source)
    val_ids = load_ids(val_source)
    if len(train_ids) != expected_train_count:
        raise ValueError(
            f"expected {expected_train_count} training IDs, got {len(train_ids)}"
        )
    if len(val_ids) != expected_val_count:
        raise ValueError(
            f"expected {expected_val_count} validation IDs, got {len(val_ids)}"
        )
    validate_split(train_ids, val_ids, available_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "kitti_train.txt"
    val_output = output_dir / "kitti_val.txt"
    train_output.write_bytes(train_source.read_bytes())
    val_output.write_bytes(val_source.read_bytes())

    summary = SplitImportSummary(
        name="Chen common KITTI train/val split",
        train_count=len(train_ids),
        val_count=len(val_ids),
        train_url=train_url,
        val_url=val_url,
        train_sha256=sha256_file(train_output),
        val_sha256=sha256_file(val_output),
    )
    (output_dir / "source.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and lock a fixed KITTI train/validation split."
    )
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--val-source", type=Path, required=True)
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
        "--output-dir",
        type=Path,
        default=Path("configs/splits"),
    )
    parser.add_argument("--expected-train-count", type=int, default=3712)
    parser.add_argument("--expected-val-count", type=int, default=3769)
    parser.add_argument("--train-url", default=TRAIN_URL)
    parser.add_argument("--val-url", default=VAL_URL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available_ids = discover_ids(args.image_dir, args.label_dir)
    summary = import_split_files(
        train_source=args.train_source,
        val_source=args.val_source,
        output_dir=args.output_dir,
        available_ids=available_ids,
        expected_train_count=args.expected_train_count,
        expected_val_count=args.expected_val_count,
        train_url=args.train_url,
        val_url=args.val_url,
    )
    print(f"train={summary.train_count} val={summary.val_count}")
    print(f"train_sha256={summary.train_sha256}")
    print(f"val_sha256={summary.val_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
