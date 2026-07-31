from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.config import load_ifdr_config
from ifdr_yolo.experiments.recovery import recover_ifdr_run
from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume a failed formal IFDR run in place and finish KITTI AP40."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    bootstrap_ultralytics_config(repository_root)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = repository_root / run_dir
    config = load_ifdr_config(
        config_path.resolve(),
        repository_root=repository_root,
    )
    result = recover_ifdr_run(
        config,
        run_dir=run_dir,
        repository_root=repository_root,
        device=args.device,
    )
    print("IFDR RECOVERY COMPLETE")
    print(f"completed_epochs={result.completed_epochs}")
    print(f"run_dir={result.run_dir}")
    print(f"metrics_ap40={result.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
