from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


METRIC = "KITTI_2D_CONDITIONAL_AP40_PAIRED_IMAGE_BOOTSTRAP"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _task_id(
    reference_name: str,
    candidate_name: str,
    run_seed: int,
    class_name: str,
    slice_name: str,
) -> str:
    return (
        f"{reference_name}_vs_{candidate_name}__s{run_seed}__"
        f"{class_name}__{slice_name}"
    )


def validate_matrix(
    *,
    input_dir: Path,
    reference_name: str,
    candidate_name: str,
    run_seed: int,
    class_names: tuple[str, ...],
    slice_names: tuple[str, ...],
    requested_iterations: int,
    bootstrap_seed: int,
    expected_split_sha256: str,
    paired_source: Path | None = None,
    wrapper_source: Path | None = None,
) -> dict[str, Any]:
    if requested_iterations <= 0:
        raise ValueError("requested iterations must be positive")
    if run_seed < 0 or bootstrap_seed < 0:
        raise ValueError("seeds must be non-negative")
    if not class_names or not slice_names:
        raise ValueError("class and slice dimensions must be non-empty")

    input_dir = input_dir.resolve()
    status_path = input_dir / "status.json"
    status = _read_json(status_path)
    expected_ids = tuple(
        _task_id(reference_name, candidate_name, run_seed, class_name, slice_name)
        for class_name in class_names
        for slice_name in slice_names
    )
    raw_tasks = status.get("tasks")
    if not isinstance(raw_tasks, dict) or set(raw_tasks) != set(expected_ids):
        raise ValueError("status task set does not exactly match expected task set")
    if status.get("schema_version") != 1:
        raise ValueError("status schema version mismatch")
    if status.get("iterations") != requested_iterations:
        raise ValueError("status requested iteration count mismatch")
    if status.get("bootstrap_seed") != bootstrap_seed:
        raise ValueError("status bootstrap seed mismatch")
    if status.get("total") != len(expected_ids):
        raise ValueError("status total does not match expected task count")

    task_rows: list[dict[str, Any]] = []
    sparse_false_negatives = 0
    for task_id in expected_ids:
        class_name, slice_name = task_id.split("__")[-2:]
        output_path = input_dir / f"{task_id}.json"
        payload = _read_json(output_path)
        comparison = payload.get("comparison")
        if not isinstance(comparison, dict):
            raise ValueError(f"{task_id}: missing comparison object")
        try:
            identity_ok = (
                payload["schema_version"] == 1
                and payload["metric"] == METRIC
                and payload["base_difficulty"] == "hard"
                and payload["reference"]["name"] == reference_name
                and payload["candidate"]["name"] == candidate_name
                and payload["class_name"] == class_name
                and payload["target_slice"]["name"] == slice_name
                and payload["split_sha256"] == expected_split_sha256
                and comparison["seed"] == bootstrap_seed
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"{task_id}: incomplete identity fields") from error
        if not identity_ok:
            raise ValueError(f"{task_id}: identity or scientific contract mismatch")

        effective = _integer(
            comparison.get("iterations"),
            field=f"{task_id} effective iterations",
        )
        if not 0 < effective <= requested_iterations:
            raise ValueError(
                f"{task_id}: effective iterations must be within "
                f"[1, {requested_iterations}]"
            )
        reference_ap40 = _number(
            comparison.get("reference_ap40"), field=f"{task_id} reference AP40"
        )
        candidate_ap40 = _number(
            comparison.get("candidate_ap40"), field=f"{task_id} candidate AP40"
        )
        difference_ap40 = _number(
            comparison.get("difference_ap40"), field=f"{task_id} AP40 difference"
        )
        if not math.isclose(
            candidate_ap40 - reference_ap40,
            difference_ap40,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{task_id}: point AP40 difference is inconsistent")
        ci_lower = _number(comparison.get("ci_lower"), field=f"{task_id} CI lower")
        ci_upper = _number(comparison.get("ci_upper"), field=f"{task_id} CI upper")
        if ci_lower > ci_upper:
            raise ValueError(f"{task_id}: confidence interval is reversed")
        probability = _number(
            comparison.get("probability_improvement"),
            field=f"{task_id} probability improvement",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{task_id}: probability improvement is outside [0, 1]")

        status_record = raw_tasks[task_id]
        if not isinstance(status_record, dict):
            raise ValueError(f"{task_id}: invalid status task record")
        wrapper_state = status_record.get("state")
        skipped = requested_iterations - effective
        if skipped == 0:
            if wrapper_state != "complete":
                raise ValueError(
                    f"{task_id}: full output is not complete in wrapper status"
                )
            artifact_status = "valid_full_effective_replicates"
        else:
            command = status_record.get("command")
            if (
                wrapper_state != "failed"
                or status_record.get("return_code") != 0
                or not isinstance(command, str)
                or f"--iterations {requested_iterations}" not in command
            ):
                raise ValueError(
                    f"{task_id}: sparse output lacks the expected wrapper false-negative evidence"
                )
            sparse_false_negatives += 1
            artifact_status = "valid_sparse_effective_replicates"

        task_rows.append(
            {
                "task_id": task_id,
                "class_name": class_name,
                "slice_name": slice_name,
                "requested_iterations": requested_iterations,
                "effective_iterations": effective,
                "skipped_no_valid_target_resamples": skipped,
                "bootstrap_seed": bootstrap_seed,
                "reference_ap40": reference_ap40,
                "candidate_ap40": candidate_ap40,
                "difference_ap40": difference_ap40,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "probability_improvement": probability,
                "wrapper_state": wrapper_state,
                "artifact_status": artifact_status,
                "artifact": str(output_path),
                "artifact_sha256": _sha256(output_path),
            }
        )

    expected_complete = sum(row["wrapper_state"] == "complete" for row in task_rows)
    expected_failed = sum(row["wrapper_state"] == "failed" for row in task_rows)
    if (
        status.get("complete") != expected_complete
        or status.get("failed") != expected_failed
        or status.get("pending") != 0
    ):
        raise ValueError("status counters do not match task records")
    if sparse_false_negatives:
        if status.get("state") != "failed":
            raise ValueError("matrix wrapper state must preserve sparse false-negative failure")
        classification = "valid_with_wrapper_false_negative"
    else:
        if status.get("state") != "complete":
            raise ValueError("matrix wrapper state is not complete")
        classification = "valid_complete"

    source_evidence: dict[str, dict[str, str]] = {}
    for name, source in (
        ("paired_bootstrap_core", paired_source),
        ("matrix_completion_gate", wrapper_source),
    ):
        if source is not None:
            resolved = source.resolve()
            source_evidence[name] = {
                "path": str(resolved),
                "sha256": _sha256(resolved),
            }

    return {
        "schema_version": 1,
        "scientific_outputs_valid": True,
        "classification": classification,
        "matrix_wrapper_state": status.get("state"),
        "matrix_wrapper_status_preserved": True,
        "reference_name": reference_name,
        "candidate_name": candidate_name,
        "run_seed": run_seed,
        "requested_iterations_per_task": requested_iterations,
        "bootstrap_seed": bootstrap_seed,
        "expected_split_sha256": expected_split_sha256,
        "task_count": len(task_rows),
        "sparse_wrapper_false_negative_count": sparse_false_negatives,
        "code_semantics": {
            "core": (
                "The paired bootstrap attempts the requested number of image resamples, "
                "skips resamples with zero valid reference targets, and serializes the "
                "number of retained differences as comparison.iterations."
            ),
            "wrapper": (
                "The matrix completion gate requires comparison.iterations to equal the "
                "requested count, so a valid sparse-slice artifact can be marked failed "
                "despite child return_code=0."
            ),
        },
        "source_evidence": source_evidence,
        "status_artifact": str(status_path),
        "status_sha256": _sha256(status_path),
        "tasks": task_rows,
    }


def _write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "class_name",
        "slice_name",
        "requested_iterations",
        "effective_iterations",
        "skipped_no_valid_target_resamples",
        "reference_ap40",
        "candidate_ap40",
        "difference_ap40",
        "ci_lower",
        "ci_upper",
        "probability_improvement",
        "wrapper_state",
        "artifact_status",
        "artifact_sha256",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate paired-bootstrap artifacts without rewriting matrix state."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--class-name", action="append", required=True)
    parser.add_argument("--slice", dest="slice_names", action="append", required=True)
    parser.add_argument("--requested-iterations", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--paired-source", type=Path)
    parser.add_argument("--wrapper-source", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_matrix(
        input_dir=args.input_dir,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
        run_seed=args.run_seed,
        class_names=tuple(args.class_name),
        slice_names=tuple(args.slice_names),
        requested_iterations=args.requested_iterations,
        bootstrap_seed=args.bootstrap_seed,
        expected_split_sha256=args.expected_split_sha256,
        paired_source=args.paired_source,
        wrapper_source=args.wrapper_source,
    )
    _write_json_new(args.output_json.resolve(), report)
    _write_csv_new(args.output_csv.resolve(), report["tasks"])
    print(
        f"bootstrap_validation={report['classification']} "
        f"tasks={report['task_count']} "
        f"wrapper_false_negatives={report['sparse_wrapper_false_negative_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
