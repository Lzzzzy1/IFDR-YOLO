"""Build the immutable factor metadata development split."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.development_split import (
    REGISTERED_FRACTION,
    REGISTERED_SEED,
    build_development_split,
    write_split_outputs,
)


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on input-jsonl line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"input-jsonl line {line_number} must be an object"
                )
            rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic KITTI factor development split."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=REGISTERED_SEED)
    parser.add_argument("--fraction", type=float, default=REGISTERED_FRACTION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split = build_development_split(
        _load_rows(args.input_jsonl),
        seed=args.seed,
        fraction=args.fraction,
    )
    write_split_outputs(split, args.output_dir)
    print(
        f"fit={len(split.fit_ids)} "
        f"development={len(split.development_ids)} "
        f"sha256={split.sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
