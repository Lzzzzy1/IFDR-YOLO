from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.eval.bootstrap_summary import summarize_bootstrap_directory


CSV_FIELDS = (
    "reference",
    "candidate",
    "class_name",
    "slice_name",
    "iterations",
    "bootstrap_seed",
    "seed_count",
    "mean_reference_ap40",
    "mean_candidate_ap40",
    "mean_difference_ap40",
    "sample_std_difference_ap40",
    "positive_seed_count",
    "negative_seed_count",
    "positive_ci_seed_count",
    "negative_ci_seed_count",
    "direction_consistency",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize complete paired-bootstrap results across seeds.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-seed", type=int, action="append", required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_seeds = tuple(sorted(args.expected_seed))
    groups = summarize_bootstrap_directory(
        args.input_dir,
        expected_seeds=expected_seeds,
    )
    payload = {
        "schema_version": 1,
        "metric": "KITTI_PAIRED_BOOTSTRAP_CROSS_SEED_SUMMARY",
        "source_directory": str(args.input_dir.resolve()),
        "expected_seeds": expected_seeds,
        "group_count": len(groups),
        "groups": [asdict(group) for group in groups],
    }
    _atomic_write_text(
        args.output_json.resolve(),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )

    rows = []
    for group in groups:
        seed_summary = group.seed_summary
        rows.append(
            {
                "reference": group.reference,
                "candidate": group.candidate,
                "class_name": group.class_name,
                "slice_name": group.slice_name,
                "iterations": group.iterations,
                "bootstrap_seed": group.bootstrap_seed,
                "seed_count": seed_summary.seed_count,
                "mean_reference_ap40": seed_summary.mean_reference_ap40,
                "mean_candidate_ap40": seed_summary.mean_candidate_ap40,
                "mean_difference_ap40": seed_summary.mean_difference_ap40,
                "sample_std_difference_ap40": (
                    seed_summary.sample_std_difference_ap40
                ),
                "positive_seed_count": seed_summary.positive_seed_count,
                "negative_seed_count": seed_summary.negative_seed_count,
                "positive_ci_seed_count": seed_summary.positive_ci_seed_count,
                "negative_ci_seed_count": seed_summary.negative_ci_seed_count,
                "direction_consistency": seed_summary.direction_consistency,
            }
        )
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(output_csv)
    print(f"bootstrap_summary_json={args.output_json.resolve()}")
    print(f"bootstrap_summary_csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
