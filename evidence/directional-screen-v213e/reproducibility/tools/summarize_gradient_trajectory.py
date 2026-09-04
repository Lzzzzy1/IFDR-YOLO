from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object, field: str, line_number: int) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"line {line_number}: {field} must be finite")
    return float(value)


def _mapping(value: object, field: str, line_number: int) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"line {line_number}: {field} must be an object")
    return value


def summarize_run(
    *,
    name: str,
    path: Path,
    epoch_start: int,
    epoch_end: int,
) -> dict[str, object]:
    path = Path(path)
    if not name:
        raise ValueError("run name must not be empty")
    if epoch_start <= 0 or epoch_end < epoch_start:
        raise ValueError("invalid epoch range")
    if not path.is_file():
        raise FileNotFoundError(path)

    selected: dict[int, list[tuple[float, float, float, bool]]] = defaultdict(list)
    total_records: dict[int, int] = defaultdict(int)
    skipped_null_cosine: dict[int, int] = defaultdict(int)
    process_ids: set[int] = set()
    source_line_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        source_line_count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {line_number}: invalid JSON") from error
        row = _mapping(row, "row", line_number)
        epoch_value = row.get("epoch")
        if isinstance(epoch_value, bool) or not isinstance(epoch_value, int):
            raise ValueError(f"line {line_number}: epoch must be an integer")
        epoch = epoch_value
        process_id = row.get("process_id")
        if isinstance(process_id, int) and not isinstance(process_id, bool):
            process_ids.add(process_id)
        if epoch < epoch_start or epoch > epoch_end:
            continue

        total_records[epoch] += 1
        parameter_groups = _mapping(
            row.get("parameter_groups"), "parameter_groups", line_number
        )
        anchor = _mapping(
            parameter_groups.get("semantic_anchor"),
            "parameter_groups.semantic_anchor",
            line_number,
        )
        pairs = _mapping(anchor.get("pairs"), "semantic_anchor.pairs", line_number)
        pair = _mapping(
            pairs.get("counterfactual::factor"),
            "semantic_anchor.pairs.counterfactual::factor",
            line_number,
        )
        cosine_value = pair.get("cosine")
        if cosine_value is None:
            skipped_null_cosine[epoch] += 1
            continue
        cosine = _finite_number(cosine_value, "cosine", line_number)
        if not -1.0 <= cosine <= 1.0:
            raise ValueError(f"line {line_number}: cosine must be within [-1, 1]")
        conflict = pair.get("conflict")
        if not isinstance(conflict, bool):
            raise ValueError(f"line {line_number}: conflict flag must be boolean")
        if conflict != (cosine < 0.0):
            raise ValueError(
                f"line {line_number}: conflict flag does not match cosine sign"
            )
        norms = _mapping(
            anchor.get("gradient_norms"), "semantic_anchor.gradient_norms", line_number
        )
        counterfactual = _finite_number(
            norms.get("counterfactual"), "counterfactual norm", line_number
        )
        factor = _finite_number(norms.get("factor"), "factor norm", line_number)
        if counterfactual < 0.0 or factor < 0.0:
            raise ValueError(f"line {line_number}: gradient norms must be non-negative")
        selected[epoch].append((cosine, counterfactual, factor, conflict))

    epoch_reports: dict[str, object] = {}
    missing_epochs: list[int] = []
    for epoch in range(epoch_start, epoch_end + 1):
        rows = selected.get(epoch, [])
        if not rows:
            missing_epochs.append(epoch)
            continue
        valid_count = len(rows)
        mean_cosine = sum(row[0] for row in rows) / valid_count
        mean_counterfactual = sum(row[1] for row in rows) / valid_count
        mean_factor = sum(row[2] for row in rows) / valid_count
        epoch_reports[str(epoch)] = {
            "total_records": total_records[epoch],
            "valid_records": valid_count,
            "skipped_null_cosine": skipped_null_cosine[epoch],
            "mean_cosine": mean_cosine,
            "conflicts": sum(row[3] for row in rows),
            "mean_counterfactual_norm": mean_counterfactual,
            "mean_factor_norm": mean_factor,
            "norm_ratio": (
                mean_counterfactual / mean_factor if mean_factor > 0.0 else None
            ),
        }

    return {
        "name": name,
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "source_line_count": source_line_count,
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "coverage_complete": not missing_epochs,
        "missing_epochs": missing_epochs,
        "process_ids": sorted(process_ids),
        "epochs": epoch_reports,
    }


def _parse_run(specification: str) -> tuple[str, Path]:
    try:
        name, raw_path = specification.split("=", 1)
    except ValueError as error:
        raise ValueError("run must use NAME=GRADIENT_JSONL") from error
    if not name or not raw_path:
        raise ValueError("run must use non-empty NAME=GRADIENT_JSONL")
    return name, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen factor/counterfactual gradient statistic."
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--epoch-start", type=int, default=6)
    parser.add_argument("--epoch-end", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs: dict[str, object] = {}
    for specification in args.run:
        name, path = _parse_run(specification)
        if name in runs:
            raise ValueError(f"duplicate run name: {name}")
        runs[name] = summarize_run(
            name=name,
            path=path,
            epoch_start=args.epoch_start,
            epoch_end=args.epoch_end,
        )
    report = {
        "schema_version": 1,
        "metric": "IFDR_FACTOR_COUNTERFACTUAL_GRADIENT_TRAJECTORY",
        "statistic": {
            "record_filter": "finite semantic_anchor counterfactual::factor cosine",
            "mean_cosine": "arithmetic mean across retained epoch records",
            "conflicts": "stored conflict flag, validated as cosine < 0",
            "norm_ratio": "epoch mean counterfactual norm / epoch mean factor norm",
        },
        "runs": runs,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    print(f"gradient_trajectory={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
