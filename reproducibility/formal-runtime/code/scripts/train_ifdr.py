from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.baseline import run_baseline
from ifdr_yolo.experiments.config import load_ifdr_config
from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reproducible IFDR-YOLO KITTI experiment.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "smoke", "full"),
        required=True,
    )
    parser.add_argument("--device", help="Explicit device such as 0 or cpu.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    bootstrap_ultralytics_config(repository_root)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config = load_ifdr_config(
        config_path.resolve(),
        repository_root=repository_root,
    )
    from ifdr_yolo.experiments.ifdr_runtime import IFDRRuntimeAdapter

    result = run_baseline(
        config,
        mode=args.mode,
        repository_root=repository_root,
        adapter=IFDRRuntimeAdapter(config),
        device_override=args.device,
    )
    if result.mode == "dry-run":
        print("IFDR PREFLIGHT PASSED")
    else:
        print(f"IFDR {result.mode.upper()} COMPLETE")
        print(f"run_dir={result.run_dir}")
        print(f"metrics_ap40={result.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
