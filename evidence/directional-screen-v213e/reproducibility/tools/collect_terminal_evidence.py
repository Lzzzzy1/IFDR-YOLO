from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence


FROZEN_SPLIT_SHA256 = (
    "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
)
FROZEN_FIT_COUNT = 3341
FROZEN_DEVELOPMENT_COUNT = 371
FROZEN_EPOCHS = 15
FROZEN_EXECUTION_PURPOSE = "local_low_memory_seed0_diagnostic"
CONTROL_FILES = (
    "screen_manifest.json",
    "status.json",
    "results.csv",
    "gradient_diagnostics.jsonl",
    "assignment_diagnostics.jsonl",
    "post_training_leakage_audit.json",
    "checkpoint_provenance.json",
    "metrics_ap40.json",
)
OPTIONAL_CONTROL_FILES = frozenset(("assignment_diagnostics.jsonl",))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_ids(path: Path) -> tuple[str, ...]:
    values = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not values:
        raise ValueError(f"ID manifest is empty: {path}")
    return values


def _load_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"JSONL contains no records: {path}")
    return tuple(rows)


def _semantic_anchor(row: Mapping[str, object]) -> Mapping[str, object] | None:
    groups = row.get("parameter_groups")
    if not isinstance(groups, Mapping):
        return None
    anchor = groups.get("semantic_anchor")
    return anchor if isinstance(anchor, Mapping) else None


def _counterfactual_norm(row: Mapping[str, object]) -> float | None:
    anchor = _semantic_anchor(row)
    norms = anchor.get("gradient_norms") if isinstance(anchor, Mapping) else None
    value = norms.get("counterfactual") if isinstance(norms, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _counterfactual_cosine(row: Mapping[str, object]) -> float | None:
    anchor = _semantic_anchor(row)
    pairs = anchor.get("pairs") if isinstance(anchor, Mapping) else None
    pair = pairs.get("counterfactual::factor") if isinstance(pairs, Mapping) else None
    value = pair.get("cosine") if isinstance(pair, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def collect_terminal_evidence(
    *,
    run_root: Path,
    mirror_root: Path,
    launcher_stdout: Path,
    launcher_stderr: Path,
    fit_ids_path: Path,
    development_ids_path: Path,
    expected_identity: str,
    expected_split_sha256: str = FROZEN_SPLIT_SHA256,
) -> dict[str, object]:
    run_root = Path(run_root)
    mirror_root = Path(mirror_root)
    launcher_stdout = Path(launcher_stdout)
    launcher_stderr = Path(launcher_stderr)
    fit_ids_path = Path(fit_ids_path)
    development_ids_path = Path(development_ids_path)
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    def record(name: str, function: Callable[[], object]) -> None:
        try:
            result = function()
            if isinstance(result, tuple) and len(result) == 2:
                passed, detail = result
            else:
                passed, detail = result, result
            checks[name] = bool(passed)
            details[name] = detail
        except Exception as error:
            checks[name] = False
            details[name] = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

    screen_path = run_root / "screen_manifest.json"
    status_path = run_root / "status.json"
    checkpoint_path = run_root / "weights" / "last.pt"
    best_path = run_root / "weights" / "best.pt"
    provenance_path = run_root / "checkpoint_provenance.json"
    metrics_path = run_root / "metrics_ap40.json"
    results_path = run_root / "results.csv"
    gradient_path = run_root / "gradient_diagnostics.jsonl"
    prediction_root = run_root / "predictions" / "labels"

    def split_contract() -> tuple[bool, object]:
        fit_ids = _load_ids(fit_ids_path)
        development_ids = _load_ids(development_ids_path)
        screen = _load_json(screen_path)
        payload = {
            "fit_count": len(fit_ids),
            "fit_unique": len(set(fit_ids)),
            "development_count": len(development_ids),
            "development_unique": len(set(development_ids)),
            "overlap_count": len(set(fit_ids) & set(development_ids)),
            "fit_sha256": sha256_file(fit_ids_path),
            "development_sha256": sha256_file(development_ids_path),
        }
        passed = (
            payload["fit_count"] == FROZEN_FIT_COUNT
            and payload["fit_unique"] == FROZEN_FIT_COUNT
            and payload["development_count"] == FROZEN_DEVELOPMENT_COUNT
            and payload["development_unique"] == FROZEN_DEVELOPMENT_COUNT
            and payload["overlap_count"] == 0
            and payload["development_sha256"] == expected_split_sha256
            and screen.get("fit_ids_sha256") == payload["fit_sha256"]
            and screen.get("development_ids_sha256") == payload["development_sha256"]
        )
        return passed, payload

    record("split_contract", split_contract)

    def screen_contract() -> tuple[bool, object]:
        screen = _load_json(screen_path)
        fields = {
            "identity_sha256": screen.get("identity_sha256"),
            "epochs": screen.get("epochs"),
            "seed": screen.get("seed"),
            "fit_count": screen.get("fit_count"),
            "development_count": screen.get("development_count"),
            "execution_purpose": screen.get("execution_purpose"),
            "primary_checkpoint_role": screen.get("primary_checkpoint_role"),
        }
        passed = fields == {
            "identity_sha256": expected_identity,
            "epochs": FROZEN_EPOCHS,
            "seed": 0,
            "fit_count": FROZEN_FIT_COUNT,
            "development_count": FROZEN_DEVELOPMENT_COUNT,
            "execution_purpose": FROZEN_EXECUTION_PURPOSE,
            "primary_checkpoint_role": "last.pt",
        }
        return passed, fields

    record("screen_contract", screen_contract)

    def status_complete() -> tuple[bool, object]:
        status = _load_json(status_path)
        fields = {
            "state": status.get("state"),
            "epoch": status.get("epoch"),
            "pid": status.get("pid"),
            "identity_sha256": status.get("identity_sha256"),
            "checkpoint_role": status.get("checkpoint_role"),
        }
        passed = (
            fields["state"] == "complete"
            and int(fields["epoch"]) == FROZEN_EPOCHS
            and fields["identity_sha256"] == expected_identity
            and fields["checkpoint_role"] == "last.pt"
            and isinstance(fields["pid"], int)
            and fields["pid"] > 0
        )
        return passed, fields

    record("status_complete_epoch15", status_complete)

    def checkpoint_status_sha() -> tuple[bool, object]:
        status = _load_json(status_path)
        actual = sha256_file(checkpoint_path)
        payload = {"size": checkpoint_path.stat().st_size, "sha256": actual}
        return (
            payload["size"] > 0 and status.get("checkpoint_sha256") == actual,
            payload,
        )

    record("checkpoint_status_sha", checkpoint_status_sha)

    def checkpoint_provenance() -> tuple[bool, object]:
        provenance = _load_json(provenance_path)
        actual_last = sha256_file(checkpoint_path)
        actual_best = sha256_file(best_path)
        payload = {
            "checkpoint_sha256": actual_last,
            "best_checkpoint_sha256": actual_best,
            "identity_sha256": provenance.get("identity_sha256"),
            "checkpoint_role": provenance.get("checkpoint_role"),
        }
        passed = (
            provenance.get("checkpoint_sha256") == actual_last
            and provenance.get("best_checkpoint_sha256") == actual_best
            and provenance.get("identity_sha256") == expected_identity
            and provenance.get("checkpoint_role") == "last.pt"
        )
        return passed, payload

    record("checkpoint_provenance", checkpoint_provenance)

    def results_exact() -> tuple[bool, object]:
        with results_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(float(row["epoch"])) for row in rows]
        payload = {"row_count": len(rows), "epochs": epochs}
        return epochs == list(range(1, FROZEN_EPOCHS + 1)), payload

    record("results_exact_15_epochs", results_exact)

    def predictions_exact() -> tuple[bool, object]:
        expected = set(_load_ids(development_ids_path))
        actual = {path.stem for path in prediction_root.glob("*.txt") if path.is_file()}
        non_txt = [str(path) for path in prediction_root.iterdir() if path.is_file() and path.suffix != ".txt"]
        payload = {
            "expected_count": len(expected),
            "actual_count": len(actual),
            "missing": sorted(expected - actual)[:10],
            "extra": sorted(actual - expected)[:10],
            "non_txt": non_txt[:10],
        }
        return actual == expected and not non_txt, payload

    record("predictions_exact_dev371", predictions_exact)

    def metrics_contract() -> tuple[bool, object]:
        metrics = _load_json(metrics_path)
        classes = metrics["classes"]
        pedestrian = float(classes["Pedestrian"]["moderate"]["ap40"])
        cyclist = float(classes["Cyclist"]["moderate"]["ap40"])
        calculated = (pedestrian + cyclist) / 2.0
        explicit = float(metrics["moderate_macro_ap_r40"])
        payload = {
            "evaluator": metrics.get("evaluator"),
            "split_count": metrics.get("split_count"),
            "split_sha256": metrics.get("split_sha256"),
            "identity_sha256": metrics.get("identity_sha256"),
            "pedestrian_moderate_ap_r40": pedestrian,
            "cyclist_moderate_ap_r40": cyclist,
            "calculated_macro_ap_r40": calculated,
            "explicit_macro_ap_r40": explicit,
        }
        passed = (
            metrics.get("evaluator") == "ifdr_yolo.kitti_ap40"
            and int(metrics.get("split_count", -1)) == FROZEN_DEVELOPMENT_COUNT
            and metrics.get("split_sha256") == expected_split_sha256
            and metrics.get("identity_sha256") == expected_identity
            and math.isclose(calculated, explicit, rel_tol=0.0, abs_tol=1e-12)
        )
        return passed, payload

    record("metrics_identity_split_macro", metrics_contract)

    def gradient_contract() -> tuple[bool, object]:
        rows = _load_jsonl(gradient_path)
        status = _load_json(status_path)
        epochs = sorted({int(row["epoch"]) for row in rows})
        pids = sorted({int(row["process_id"]) for row in rows})
        active_counts = {
            epoch: sum(
                1
                for row in rows
                if int(row["epoch"]) == epoch
                and (_counterfactual_norm(row) or 0.0) > 0.0
                and _counterfactual_cosine(row) is not None
            )
            for epoch in range(6, 16)
        }
        frozen_positive = sum(
            1
            for row in rows
            if int(row["epoch"]) <= 5 and (_counterfactual_norm(row) or 0.0) > 0.0
        )
        payload = {
            "record_count": len(rows),
            "epochs": epochs,
            "process_ids": pids,
            "active_valid_counts": active_counts,
            "frozen_positive_counterfactual_records": frozen_positive,
        }
        passed = (
            epochs == list(range(1, 16))
            and pids == [int(status["pid"])]
            and all(count > 0 for count in active_counts.values())
            and frozen_positive == 0
        )
        return passed, payload

    record("gradient_schedule_and_pid", gradient_contract)

    record(
        "launcher_stdout_terminal_marker",
        lambda: (
            "metrics_ap40=" in launcher_stdout.read_text(encoding="utf-8", errors="replace"),
            {"size": launcher_stdout.stat().st_size},
        ),
    )
    record(
        "launcher_stderr_empty",
        lambda: (launcher_stderr.stat().st_size == 0, {"size": launcher_stderr.stat().st_size}),
    )

    def mirror_controls() -> tuple[bool, object]:
        mismatches: list[str] = []
        optional_absent: list[str] = []
        hashes: dict[str, str] = {}
        for name in CONTROL_FILES:
            primary = run_root / name
            mirrored = mirror_root / name
            primary_exists = primary.is_file()
            mirrored_exists = mirrored.is_file()
            if (
                name in OPTIONAL_CONTROL_FILES
                and not primary_exists
                and not mirrored_exists
            ):
                optional_absent.append(name)
                continue
            if not primary_exists or not mirrored_exists:
                mismatches.append(name)
                continue
            primary_sha = sha256_file(primary)
            hashes[name] = primary_sha
            if sha256_file(mirrored) != primary_sha:
                mismatches.append(name)
        return not mismatches, {
            "mismatches": mismatches,
            "optional_absent": optional_absent,
            "sha256": hashes,
        }

    record("primary_mirror_control_files", mirror_controls)

    def mirror_predictions() -> tuple[bool, object]:
        primary_root = run_root / "predictions" / "labels"
        mirrored_root = mirror_root / "predictions" / "labels"
        primary = {path.name: sha256_file(path) for path in primary_root.glob("*.txt") if path.is_file()}
        mirrored = {path.name: sha256_file(path) for path in mirrored_root.glob("*.txt") if path.is_file()}
        mismatches = sorted(name for name in set(primary) | set(mirrored) if primary.get(name) != mirrored.get(name))
        return (
            primary == mirrored and len(primary) == FROZEN_DEVELOPMENT_COUNT,
            {"primary_count": len(primary), "mirror_count": len(mirrored), "mismatches": mismatches[:10]},
        )

    record("primary_mirror_predictions", mirror_predictions)

    def mirror_checkpoint() -> tuple[bool, object]:
        actual = sha256_file(checkpoint_path)
        sidecar = (mirror_root / "weights" / "last.pt.sha256").read_text(encoding="utf-8").split()
        manifest = _load_json(mirror_root / "manifest.json")
        records = [record for record in manifest["files"] if isinstance(record, Mapping) and record.get("path") == "weights/last.pt"]
        payload = {"actual_sha256": actual, "sidecar": sidecar, "manifest_records": records}
        passed = (
            len(sidecar) >= 1
            and sidecar[0] == actual
            and len(records) == 1
            and records[0].get("sha256") == actual
            and int(records[0].get("size", -1)) == checkpoint_path.stat().st_size
        )
        return passed, payload

    record("mirror_checkpoint_publication", mirror_checkpoint)

    def manifest_records() -> tuple[bool, object]:
        manifest = _load_json(mirror_root / "manifest.json")
        records = manifest.get("files")
        if not isinstance(records, list):
            raise ValueError("mirror manifest files is not a list")
        mismatches: list[str] = []
        seen: set[str] = set()
        for item in records:
            if not isinstance(item, Mapping):
                mismatches.append("non_mapping_record")
                continue
            relative = str(item.get("path", ""))
            if not relative or relative in seen:
                mismatches.append(relative or "empty_path")
                continue
            seen.add(relative)
            target = checkpoint_path if relative == "weights/last.pt" else mirror_root / Path(relative)
            if not target.is_file():
                mismatches.append(relative)
                continue
            if int(item.get("size", -1)) != target.stat().st_size or item.get("sha256") != sha256_file(target):
                mismatches.append(relative)
        expected_paths = {
            name for name in CONTROL_FILES if (run_root / name).is_file()
        }
        expected_paths.update(
            path.relative_to(run_root).as_posix()
            for path in (run_root / "predictions" / "labels").glob("*.txt")
            if path.is_file()
        )
        expected_paths.add("weights/last.pt")
        missing_records = sorted(expected_paths - seen)
        extra_records = sorted(seen - expected_paths)
        return (
            not mismatches and not missing_records and not extra_records,
            {
                "record_count": len(records),
                "expected_count": len(expected_paths),
                "mismatches": mismatches[:10],
                "missing_records": missing_records[:10],
                "extra_records": extra_records[:10],
            },
        )

    record("mirror_manifest_records", manifest_records)

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "engineering_pass": not failed,
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "expected_identity_sha256": expected_identity,
        "expected_split_sha256": expected_split_sha256,
        "run_root": str(run_root.resolve()),
        "mirror_root": str(mirror_root.resolve()),
        "collector_sha256": sha256_file(Path(__file__).resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect fail-closed v213 terminal engineering evidence.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--launcher-stdout", type=Path, required=True)
    parser.add_argument("--launcher-stderr", type=Path, required=True)
    parser.add_argument("--fit-ids", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--expected-split-sha256", default=FROZEN_SPLIT_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = collect_terminal_evidence(
        run_root=args.run_root,
        mirror_root=args.mirror_root,
        launcher_stdout=args.launcher_stdout,
        launcher_stderr=args.launcher_stderr,
        fit_ids_path=args.fit_ids,
        development_ids_path=args.development_ids,
        expected_identity=args.expected_identity,
        expected_split_sha256=args.expected_split_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"engineering_pass={report['engineering_pass']}")
    print(f"failed_checks={','.join(report['failed_checks']) or 'none'}")
    print(f"terminal_evidence={args.output}")
    return 0 if report["engineering_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
