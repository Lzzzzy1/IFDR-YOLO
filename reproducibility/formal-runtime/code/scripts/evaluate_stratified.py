from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.eval.stratified_report import (
    evaluate_stratified_runs,
    write_stratified_report,
)


def _run_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("run must use NAME=PREDICTION_DIR")
    return name.strip(), Path(path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI AP40 by size, depth and occlusion slices.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_prediction_dirs: dict[str, Path] = {}
    for name, path in args.run:
        if name in run_prediction_dirs:
            raise ValueError(f"duplicate run name: {name}")
        run_prediction_dirs[name] = path.resolve()
    report = evaluate_stratified_runs(
        run_prediction_dirs=run_prediction_dirs,
        label_dir=args.label_dir.resolve(),
        image_dir=args.image_dir.resolve(),
        split_path=args.split.resolve(),
    )
    json_path, csv_path = write_stratified_report(
        args.output_dir.resolve(),
        report,
    )
    print(f"stratified_json={json_path}")
    print(f"stratified_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
