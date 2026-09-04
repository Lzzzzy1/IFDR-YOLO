"""Run the fit-only P2 candidate-survival assignment audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.p2_candidate_survival_audit import run_fit_assignment_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resolved-data", type=Path, required=True)
    parser.add_argument("--fit-ids", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--raw-label-dir", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--mirror", "--mirror-dir", dest="mirror_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_fit_assignment_audit(
        config_path=args.config,
        resolved_data_path=args.resolved_data,
        fit_ids_path=args.fit_ids,
        development_ids_path=args.development_ids,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        raw_label_dir=args.raw_label_dir,
        output_dir=args.output_dir,
        mirror_dir=args.mirror_dir,
        mode=args.mode,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
