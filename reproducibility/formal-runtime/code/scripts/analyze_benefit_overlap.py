"""CLI for fixed-image P2/A/B object-identity benefit overlap."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.splits import load_ids
from ifdr_yolo.eval.benefit_overlap import (
    DEFAULT_CLASSES,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    analyze_benefit_overlap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify object-identity rescue/harm overlap for KITTI Moderate "
            "P2, fusion-only A, and DCLI-only B."
        )
    )
    parser.add_argument(
        "--split", "--image-ids", dest="split", type=Path, required=True,
        help="fixed image-ID file",
    )
    parser.add_argument("--image-dir", "--raw-image-dir", dest="image_dir", type=Path, required=True)
    parser.add_argument("--label-dir", "--gt-label-dir", dest="label_dir", type=Path, required=True)
    parser.add_argument(
        "--p2-dir", "--p2-label-dir", "--prediction-p2-dir",
        dest="p2_dir", type=Path, required=True,
    )
    parser.add_argument(
        "--a-dir", "--a-label-dir", "--fusion-only-dir", "--prediction-a-dir",
        dest="a_dir", type=Path, required=True,
    )
    parser.add_argument(
        "--b-dir", "--b-label-dir", "--dcli-only-dir", "--prediction-b-dir",
        dest="b_dir", type=Path, required=True,
    )
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        choices=("Car", "Pedestrian", "Cyclist"),
        help="class to include (repeat; defaults to Pedestrian and Cyclist)",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument("--bootstrap-seed", "--seed", dest="bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--journal", type=Path, help="per-image JSONL journal")
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--max-images",
        type=int,
        help="interrupt after this many newly processed images (recovery test only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split = args.split.expanduser().resolve()
    image_ids = load_ids(split)
    output_json = args.output_json.expanduser().resolve()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else output_json.with_suffix(".csv")
    )
    journal = (
        args.journal.expanduser().resolve()
        if args.journal is not None
        else output_json.with_suffix(".journal.jsonl")
    )
    class_names = tuple(args.class_names) if args.class_names else DEFAULT_CLASSES
    result = analyze_benefit_overlap(
        image_ids=image_ids,
        image_ids_path=split,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        p2_dir=args.p2_dir,
        a_dir=args.a_dir,
        b_dir=args.b_dir,
        class_names=class_names,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        journal_path=journal,
        output_json=output_json,
        output_csv=output_csv,
        max_images=args.max_images,
    )
    print(f"benefit_overlap_json={output_json}")
    print(f"benefit_overlap_csv={output_csv}")
    print(f"journal={journal}")
    print(f"manifest_sha256={result['manifest']['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
