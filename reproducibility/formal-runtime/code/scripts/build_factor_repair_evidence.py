"""CLI bridge from persisted calibration/observer/audit artifacts to evidence.

The command only loads files, validates scientific identity, invokes the pure
evidence builder, and writes JSON.  It never loads a model or runs the GPU
observer; observer execution remains an explicit upstream step.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.eval.factor_repair_evidence import build_factor_repair_evidence


def _read_image_ids(path: Path) -> tuple[str, ...]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"image IDs file is missing or empty: {path}")
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"image IDs JSON is malformed: {path}") from error
        if isinstance(value, dict):
            value = value.get("image_ids", value.get("development_ids"))
        if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
            raise ValueError("image IDs JSON must contain an array")
        image_ids = tuple(str(item) for item in value)
    else:
        image_ids = tuple(path.read_text(encoding="utf-8").splitlines())
    if not image_ids or any(not item or item != item.strip() for item in image_ids):
        raise ValueError("image IDs must be non-empty lines")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image IDs must be unique")
    return image_ids


def _checkpoint_arg(value: str) -> Path:
    """Accept either a direct path or the audit CLI's ``17=PATH`` spelling."""

    if "=" in value:
        seed, raw_path = value.split("=", 1)
        if seed != "17":
            raise argparse.ArgumentTypeError("development checkpoint must be seed 17")
        value = raw_path
    if not value.strip():
        raise argparse.ArgumentTypeError("checkpoint path must not be empty")
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build development FactorRepairEvidence from persisted artifacts."
    )
    parser.add_argument("--condition", required=True, choices=("F0", "F1", "F2", "F3"))
    parser.add_argument("--stage", default="development", choices=("development",))
    parser.add_argument("--checkpoint", "--last-pt", dest="checkpoint", required=True, type=_checkpoint_arg)
    parser.add_argument("--checkpoint-roles", "--checkpoint-roles-json", dest="checkpoint_roles", required=True, type=Path)
    parser.add_argument(
        "--observations",
        "--observer-observations",
        "--observer-jsonl",
        dest="observations",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--audit",
        "--audit-evidence",
        "--audit-json",
        dest="audit",
        required=True,
        type=Path,
    )
    parser.add_argument("--image-ids", "--development-image-ids", dest="image_ids", type=Path)
    parser.add_argument("--image-ids-hash", "--development-image-ids-hash", dest="image_ids_hash")
    parser.add_argument(
        "--output",
        "--output-dir",
        "--output-evidence",
        dest="output",
        required=True,
        type=Path,
        help="JSON output path or directory (directory writes evidence/gate files)",
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def _output_paths(value: Path) -> tuple[Path, Path]:
    value = value.expanduser().resolve(strict=False)
    if value.suffix.lower() == ".json":
        return value, value.with_name("absolute_gate.json")
    return value / "factor_repair_evidence.json", value / "absolute_gate.json"


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    image_ids = _read_image_ids(args.image_ids) if args.image_ids is not None else None
    evidence = build_factor_repair_evidence(
        condition=args.condition,
        stage=args.stage,
        checkpoint=args.checkpoint,
        checkpoint_roles=args.checkpoint_roles,
        observations=args.observations,
        audit=args.audit,
        image_ids=image_ids,
        image_ids_hash=args.image_ids_hash,
    )
    evidence_path, gate_path = _output_paths(args.output)
    _write_json(evidence_path, evidence.to_dict())
    _write_json(gate_path, evidence.absolute_gate.to_dict())
    return evidence_path, gate_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence_path, gate_path = run(args)
    except Exception as error:
        print(f"factor repair evidence build failed: {error}", file=sys.stderr)
        return 1
    print(f"evidence={evidence_path}")
    print(f"absolute_gate={gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run"]
