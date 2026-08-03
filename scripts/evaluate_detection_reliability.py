from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.eval.reliability_report import (
    evaluate_reliability_runs,
    write_reliability_report,
)
from ifdr_yolo.eval.stratified_ap40 import KITTI_RESEARCH_SLICES


RELIABILITY_TARGET_NAMES = {
    "small_25_40",
    "far_gt_40m",
    "occlusion_2",
}
RELIABILITY_TARGET_SLICES = tuple(
    target_slice
    for target_slice in KITTI_RESEARCH_SLICES
    if target_slice.name in RELIABILITY_TARGET_NAMES
)


def _run_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("run must use NAME=PREDICTION_DIR")
    return name.strip(), Path(path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate KITTI LaECE0 and LRP on a disjoint calibration/test split."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_run_spec,
        required=True,
        metavar="NAME=PREDICTION_DIR",
    )
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260803)
    parser.add_argument("--bins", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_prediction_dirs: dict[str, Path] = {}
    for name, path in args.run:
        if name in run_prediction_dirs:
            raise ValueError(f"duplicate run name: {name}")
        run_prediction_dirs[name] = path.resolve()
    report = evaluate_reliability_runs(
        run_prediction_dirs=run_prediction_dirs,
        label_dir=args.label_dir.resolve(),
        image_dir=args.image_dir.resolve(),
        split_path=args.split.resolve(),
        split_seed=args.split_seed,
        target_slices=RELIABILITY_TARGET_SLICES,
        bins=args.bins,
    )
    json_path, csv_path = write_reliability_report(
        args.output_dir.resolve(),
        report,
    )
    print(f"reliability_json={json_path}")
    print(f"reliability_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
