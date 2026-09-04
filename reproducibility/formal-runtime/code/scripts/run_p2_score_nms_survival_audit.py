"""Run the fit-only score/NMS survival audit or its deterministic local smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.p2_score_nms_survival_audit import run_fit_score_nms_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resolved-data", type=Path, required=True)
    parser.add_argument("--fit-ids", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--raw-label-dir", type=Path, required=True)
    parser.add_argument("--expected-raw-label-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "benchmark32", "full"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    return parser


def _read_ids(path: Path) -> tuple[str, ...]:
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_fit_score_nms_audit(
        config_path=args.config,
        resolved_data_path=args.resolved_data,
        fit_ids_path=args.fit_ids,
        development_ids_path=args.development_ids,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        raw_label_dir=args.raw_label_dir,
        expected_raw_label_sha256=args.expected_raw_label_sha256,
        output_dir=args.output,
        mirror_dir=args.mirror,
        mode=args.mode,
        device=args.device,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
