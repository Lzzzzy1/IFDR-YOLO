"""Prepare or run the leakage-free seed-17 plain P3--P5 reference job."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.config import load_baseline_config
from ifdr_yolo.experiments.p2_fit_reference import (
    REFERENCE_EXECUTION_PURPOSE,
    STAGE9_REFERENCE_EXECUTION_PURPOSE,
    STAGE11_REFERENCE_EXECUTION_PURPOSE,
    run_p3p5_fit_reference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the leakage-free fit-only YOLOv8m plain P3-P5 reference.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-ids", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "smoke", "full"), default="dry-run")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution-purpose",
        choices=(REFERENCE_EXECUTION_PURPOSE, STAGE9_REFERENCE_EXECUTION_PURPOSE, STAGE11_REFERENCE_EXECUTION_PURPOSE),
        default=REFERENCE_EXECUTION_PURPOSE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repository_root / args.config
    config = load_baseline_config(config_path.resolve(), repository_root=repository_root)
    job = run_p3p5_fit_reference(
        config,
        repository_root=repository_root,
        output_dir=args.output_dir,
        mirror_dir=args.mirror_dir,
        fit_ids=args.fit_ids,
        development_ids=args.development_ids,
        mode=args.mode,
        device=args.device,
        resume=args.resume,
        execution_purpose=args.execution_purpose,
    )
    print(f"P3-P5 REFERENCE {args.mode.upper()} READY")
    print(f"output_dir={job.output_dir}")
    print(f"mirror_dir={job.mirror_dir}")
    print(f"identity_sha256={job.identity_sha256}")
    print(f"model_role={job.identity.model_role}")
    print(f"run_mode={job.identity.run_mode}")
    print(f"resumable={job.resumable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
