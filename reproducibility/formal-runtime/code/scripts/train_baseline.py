from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.baseline import run_baseline
from ifdr_yolo.experiments.config import load_baseline_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible YOLOv8m KITTI baseline.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "smoke", "full"),
        required=True,
    )
    parser.add_argument(
        "--device",
        help="Explicit device override such as 0 or cpu.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config = load_baseline_config(
        config_path.resolve(),
        repository_root=repository_root,
    )
    result = run_baseline(
        config,
        mode=args.mode,
        repository_root=repository_root,
        device_override=args.device,
    )
    if result.mode == "dry-run":
        print("BASELINE PREFLIGHT PASSED")
    else:
        print(f"BASELINE {result.mode.upper()} COMPLETE")
        print(f"run_dir={result.run_dir}")
        print(f"metrics_ap40={result.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
