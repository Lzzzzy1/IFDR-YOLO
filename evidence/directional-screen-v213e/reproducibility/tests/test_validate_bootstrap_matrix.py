from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "validate_bootstrap_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("validate_bootstrap_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload(*, effective_iterations: int = 868) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric": "KITTI_2D_CONDITIONAL_AP40_PAIRED_IMAGE_BOOTSTRAP",
        "base_difficulty": "hard",
        "split_count": 371,
        "split_sha256": "a" * 64,
        "reference": {"name": "PLAIN", "prediction_dir": "plain"},
        "candidate": {"name": "ANCHORED", "prediction_dir": "anchored"},
        "class_name": "Cyclist",
        "target_slice": {"name": "far_gt_40m"},
        "comparison": {
            "reference_ap40": 20.0,
            "candidate_ap40": 3.0,
            "difference_ap40": -17.0,
            "ci_lower": -40.0,
            "ci_upper": -5.0,
            "confidence": 0.95,
            "probability_improvement": 0.01,
            "iterations": effective_iterations,
            "seed": 17,
        },
    }


def _write_fixture(root: Path, *, effective_iterations: int = 868) -> None:
    task_id = "PLAIN_vs_ANCHORED__s0__Cyclist__far_gt_40m"
    (root / f"{task_id}.json").write_text(
        json.dumps(_payload(effective_iterations=effective_iterations)),
        encoding="utf-8",
    )
    status = {
        "schema_version": 1,
        "state": "failed",
        "iterations": 1000,
        "bootstrap_seed": 17,
        "total": 1,
        "complete": 0,
        "failed": 1,
        "pending": 0,
        "tasks": {
            task_id: {
                "state": "failed",
                "return_code": 0,
                "output": str(root / f"{task_id}.json"),
                "command": "python evaluate.py --iterations 1000 --seed 17",
            }
        },
    }
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")


def test_sparse_effective_replicates_are_valid_but_wrapper_state_is_preserved() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_fixture(root)
        report = MODULE.validate_matrix(
            input_dir=root,
            reference_name="PLAIN",
            candidate_name="ANCHORED",
            run_seed=0,
            class_names=("Cyclist",),
            slice_names=("far_gt_40m",),
            requested_iterations=1000,
            bootstrap_seed=17,
            expected_split_sha256="a" * 64,
        )

        assert report["scientific_outputs_valid"] is True
        assert report["matrix_wrapper_state"] == "failed"
        assert report["classification"] == "valid_with_wrapper_false_negative"
        row = report["tasks"][0]
        assert row["requested_iterations"] == 1000
        assert row["effective_iterations"] == 868
        assert row["skipped_no_valid_target_resamples"] == 132
        assert row["artifact_status"] == "valid_sparse_effective_replicates"


def test_effective_iterations_cannot_exceed_requested_iterations() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_fixture(root, effective_iterations=1001)
        try:
            MODULE.validate_matrix(
                input_dir=root,
                reference_name="PLAIN",
                candidate_name="ANCHORED",
                run_seed=0,
                class_names=("Cyclist",),
                slice_names=("far_gt_40m",),
                requested_iterations=1000,
                bootstrap_seed=17,
                expected_split_sha256="a" * 64,
            )
        except ValueError as error:
            assert "effective iterations" in str(error)
        else:
            raise AssertionError("invalid effective iteration count was accepted")


def test_missing_expected_task_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_fixture(root)
        try:
            MODULE.validate_matrix(
                input_dir=root,
                reference_name="PLAIN",
                candidate_name="ANCHORED",
                run_seed=0,
                class_names=("Pedestrian", "Cyclist"),
                slice_names=("far_gt_40m",),
                requested_iterations=1000,
                bootstrap_seed=17,
                expected_split_sha256="a" * 64,
            )
        except ValueError as error:
            assert "task set" in str(error)
        else:
            raise AssertionError("missing task was accepted")


if __name__ == "__main__":
    test_sparse_effective_replicates_are_valid_but_wrapper_state_is_preserved()
    test_effective_iterations_cannot_exceed_requested_iterations()
    test_missing_expected_task_fails_closed()
    print("PASS: bootstrap matrix validator")
