from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "collect_terminal_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("collect_terminal_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _build_terminal_tree(root: Path) -> dict[str, Path | str]:
    run = root / "run"
    mirror = root / "mirror"
    run.mkdir()
    mirror.mkdir()
    identity = "a" * 64
    fit_ids = tuple(f"fit-{index:04d}" for index in range(3341))
    dev_ids = tuple(f"dev-{index:04d}" for index in range(371))
    fit_path = root / "fit_ids.txt"
    dev_path = root / "development_ids.txt"
    fit_path.write_text("\n".join(fit_ids) + "\n", encoding="utf-8")
    dev_path.write_text("\n".join(dev_ids) + "\n", encoding="utf-8")

    checkpoint = run / "weights" / "last.pt"
    best = run / "weights" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"last-checkpoint")
    best.write_bytes(b"best-checkpoint")
    checkpoint_sha = _sha256(checkpoint)
    status = {
        "state": "complete",
        "epoch": 15,
        "pid": 123,
        "identity_sha256": identity,
        "checkpoint_role": "last.pt",
        "checkpoint_sha256": checkpoint_sha,
    }
    screen = {
        "identity_sha256": identity,
        "epochs": 15,
        "seed": 0,
        "fit_count": 3341,
        "development_count": 371,
        "fit_ids_sha256": MODULE.sha256_file(fit_path),
        "development_ids_sha256": MODULE.sha256_file(dev_path),
        "execution_purpose": "local_low_memory_seed0_diagnostic",
        "primary_checkpoint_role": "last.pt",
    }
    _json(run / "status.json", status)
    _json(run / "screen_manifest.json", screen)
    _json(
        run / "checkpoint_provenance.json",
        {
            "checkpoint_role": "last.pt",
            "checkpoint_sha256": checkpoint_sha,
            "best_checkpoint_sha256": _sha256(best),
            "identity_sha256": identity,
        },
    )
    metrics = {
        "evaluator": "ifdr_yolo.kitti_ap40",
        "split_count": 371,
        "split_sha256": MODULE.sha256_file(dev_path),
        "identity_sha256": identity,
        "classes": {
            "Pedestrian": {"moderate": {"ap40": 80.0}},
            "Cyclist": {"moderate": {"ap40": 90.0}},
        },
        "moderate_macro_ap_r40": 85.0,
    }
    _json(run / "metrics_ap40.json", metrics)
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "time"))
        writer.writeheader()
        for epoch in range(1, 16):
            writer.writerow({"epoch": epoch, "time": epoch * 10})
    with (run / "gradient_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(1, 16):
            active = epoch >= 6
            stream.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "process_id": 123,
                        "parameter_groups": {
                            "semantic_anchor": {
                                "gradient_norms": {
                                    "counterfactual": 0.1 if active else 0.0,
                                    "factor": 1.0,
                                },
                                "pairs": {
                                    "counterfactual::factor": {
                                        "cosine": 0.8 if active else None,
                                        "conflict": False,
                                    }
                                },
                            }
                        },
                    }
                )
                + "\n"
            )
    (run / "assignment_diagnostics.jsonl").write_text("{}\n", encoding="utf-8")
    _json(run / "post_training_leakage_audit.json", {"status": "PASS"})
    pred = run / "predictions" / "labels"
    pred.mkdir(parents=True)
    for image_id in dev_ids:
        (pred / f"{image_id}.txt").write_text("", encoding="utf-8")

    controls = (
        "screen_manifest.json",
        "status.json",
        "results.csv",
        "gradient_diagnostics.jsonl",
        "assignment_diagnostics.jsonl",
        "post_training_leakage_audit.json",
        "checkpoint_provenance.json",
        "metrics_ap40.json",
    )
    records: list[dict[str, object]] = []
    for name in controls:
        target = mirror / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / name, target)
        records.append({"path": name, "size": target.stat().st_size, "sha256": _sha256(target)})
    mirror_pred = mirror / "predictions" / "labels"
    shutil.copytree(pred, mirror_pred)
    for path in sorted(mirror_pred.glob("*.txt")):
        relative = path.relative_to(mirror).as_posix()
        records.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    records.append(
        {
            "path": "weights/last.pt",
            "size": checkpoint.stat().st_size,
            "sha256": checkpoint_sha,
        }
    )
    sidecar = mirror / "weights" / "last.pt.sha256"
    sidecar.parent.mkdir()
    sidecar.write_text(f"{checkpoint_sha}  last.pt\n", encoding="utf-8")
    _json(mirror / "manifest.json", {"files": records})

    stdout = root / "launcher.stdout.log"
    stderr = root / "launcher.stderr.log"
    stdout.write_text("metrics_ap40=run/metrics_ap40.json\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return {
        "run": run,
        "mirror": mirror,
        "fit": fit_path,
        "dev": dev_path,
        "stdout": stdout,
        "stderr": stderr,
        "identity": identity,
    }


def test_complete_terminal_tree_passes_all_checks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        paths = _build_terminal_tree(Path(temporary))
        report = MODULE.collect_terminal_evidence(
            run_root=paths["run"],
            mirror_root=paths["mirror"],
            launcher_stdout=paths["stdout"],
            launcher_stderr=paths["stderr"],
            fit_ids_path=paths["fit"],
            development_ids_path=paths["dev"],
            expected_identity=paths["identity"],
            expected_split_sha256=MODULE.sha256_file(paths["dev"]),
        )
        assert report["engineering_pass"] is True, report
        assert all(report["checks"].values())


def test_nonempty_stderr_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        paths = _build_terminal_tree(Path(temporary))
        paths["stderr"].write_text("traceback\n", encoding="utf-8")
        report = MODULE.collect_terminal_evidence(
            run_root=paths["run"],
            mirror_root=paths["mirror"],
            launcher_stdout=paths["stdout"],
            launcher_stderr=paths["stderr"],
            fit_ids_path=paths["fit"],
            development_ids_path=paths["dev"],
            expected_identity=paths["identity"],
            expected_split_sha256=MODULE.sha256_file(paths["dev"]),
        )
        assert report["engineering_pass"] is False
        assert report["checks"]["launcher_stderr_empty"] is False


def test_optional_assignment_diagnostics_may_be_absent_on_both_sides() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        paths = _build_terminal_tree(Path(temporary))
        (paths["run"] / "assignment_diagnostics.jsonl").unlink()
        (paths["mirror"] / "assignment_diagnostics.jsonl").unlink()
        manifest_path = paths["mirror"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = [
            row
            for row in manifest["files"]
            if row["path"] != "assignment_diagnostics.jsonl"
        ]
        _json(manifest_path, manifest)

        report = MODULE.collect_terminal_evidence(
            run_root=paths["run"],
            mirror_root=paths["mirror"],
            launcher_stdout=paths["stdout"],
            launcher_stderr=paths["stderr"],
            fit_ids_path=paths["fit"],
            development_ids_path=paths["dev"],
            expected_identity=paths["identity"],
            expected_split_sha256=MODULE.sha256_file(paths["dev"]),
        )
        assert report["engineering_pass"] is True, report
        assert report["details"]["primary_mirror_control_files"][
            "optional_absent"
        ] == ["assignment_diagnostics.jsonl"]


def test_optional_control_file_still_requires_existence_parity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        paths = _build_terminal_tree(Path(temporary))
        (paths["mirror"] / "assignment_diagnostics.jsonl").unlink()
        report = MODULE.collect_terminal_evidence(
            run_root=paths["run"],
            mirror_root=paths["mirror"],
            launcher_stdout=paths["stdout"],
            launcher_stderr=paths["stderr"],
            fit_ids_path=paths["fit"],
            development_ids_path=paths["dev"],
            expected_identity=paths["identity"],
            expected_split_sha256=MODULE.sha256_file(paths["dev"]),
        )
        assert report["engineering_pass"] is False
        assert report["checks"]["primary_mirror_control_files"] is False


if __name__ == "__main__":
    test_complete_terminal_tree_passes_all_checks()
    test_nonempty_stderr_fails_closed()
    test_optional_assignment_diagnostics_may_be_absent_on_both_sides()
    test_optional_control_file_still_requires_existence_parity()
    print("PASS: terminal evidence fail-closed semantics")
