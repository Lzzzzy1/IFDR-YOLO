"""Explicit, non-training CLI for the approved seed-0 benchmark gate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.kitti_seed0_training_benchmark import run_preflight, run_registered_benchmark_stage, run_synthetic_recovery_probe


STAGES = (
    "preflight", "timing", "recovery-uninterrupted", "recovery-stop1",
    "recovery-resume", "compare-recovery",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed-0 benchmark gate. Training launch is intentionally not implemented here.",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--arm", choices=("P3P5_CONTROL", "DCLI"))
    parser.add_argument("--execution-role", choices=("timing_one_epoch", "recovery_uninterrupted_two_epoch", "recovery_interrupted_two_epoch"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fit-ids", type=Path)
    parser.add_argument("--development-ids", type=Path)
    parser.add_argument("--resolved-data", type=Path)
    parser.add_argument("--raw-label-dir", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--preflight-mirror-dir", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--synthetic", action="store_true", help="Run only the deterministic local recovery fixture.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "preflight":
        required = (args.arm, args.execution_role, args.config, args.fit_ids, args.development_ids, args.resolved_data, args.raw_label_dir)
        if any(value is None for value in required):
            raise SystemExit("preflight requires --arm --execution-role --config --fit-ids --development-ids --resolved-data --raw-label-dir")
        run_preflight(arm=args.arm, execution_role=args.execution_role, config_path=args.config, fit_ids=args.fit_ids, development_ids=args.development_ids, resolved_data=args.resolved_data, raw_label_dir=args.raw_label_dir, repository_root=args.repository_root, output_dir=args.output_dir, mirror_dir=args.mirror_dir, device=args.device or "0")
        return 0
    if not args.synthetic and args.stage in {"timing", "recovery-uninterrupted", "recovery-stop1", "recovery-resume"}:
        required = (args.arm, args.execution_role, args.config, args.fit_ids, args.development_ids,
                    args.preflight_dir, args.preflight_mirror_dir)
        if any(value is None for value in required):
            raise SystemExit("real benchmark stage requires arm, role, config, splits, and paired preflight roots")
        expected = {
            "timing": ("timing_one_epoch", False, None),
            "recovery-uninterrupted": ("recovery_uninterrupted_two_epoch", False, None),
            "recovery-stop1": ("recovery_interrupted_two_epoch", False, 1),
            "recovery-resume": ("recovery_interrupted_two_epoch", True, None),
        }[args.stage]
        if args.execution_role != expected[0]:
            raise SystemExit("stage and execution role do not match the frozen benchmark contract")
        run_registered_benchmark_stage(
            arm=args.arm, execution_role=args.execution_role, config_path=args.config,
            fit_ids=args.fit_ids, development_ids=args.development_ids,
            repository_root=args.repository_root, output_dir=args.output_dir, mirror_dir=args.mirror_dir,
            preflight_dir=args.preflight_dir, preflight_mirror_dir=args.preflight_mirror_dir,
            device=args.device or "0", resume=expected[1], stop_after_epoch=expected[2],
        )
        return 0
    if not args.synthetic:
        raise SystemExit("real training is blocked: this CLI currently permits only --synthetic contract checks")
    if args.stage == "recovery-stop1":
        run_synthetic_recovery_probe(args.output_dir, args.mirror_dir, stop_after_epoch=1)
    elif args.stage == "recovery-resume":
        run_synthetic_recovery_probe(args.output_dir, args.mirror_dir, stop_after_epoch=None, resume=True, ambient_seed=999)
    else:
        run_synthetic_recovery_probe(args.output_dir, args.mirror_dir, stop_after_epoch=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
