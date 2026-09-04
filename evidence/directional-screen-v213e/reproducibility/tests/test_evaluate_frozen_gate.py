from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_frozen_gate.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_frozen_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _overall(pedestrian: float, cyclist: float) -> dict[str, object]:
    return {
        "evaluator": "ifdr_yolo.kitti_ap40",
        "split_count": 371,
        "split_sha256": MODULE.FROZEN_SPLIT_SHA256,
        "classes": {
            "Pedestrian": {"moderate": {"ap40": pedestrian}},
            "Cyclist": {"moderate": {"ap40": cyclist}},
        },
    }


def _run(small: float, far: float, near: float, large: float) -> dict[str, object]:
    def payload(value: float) -> dict[str, object]:
        return {
            "classes": {
                "Pedestrian": {"ap40": value},
                "Cyclist": {"ap40": value},
            }
        }

    return {
        "metric": "KITTI_2D_CONDITIONAL_AP40",
        "base_difficulty": "hard",
        "slices": {
            "height": {
                "small_25_40": payload(small),
                "large_gt_80": payload(large),
            },
            "depth": {
                "far_gt_40m": payload(far),
                "near_0_20m": payload(near),
            },
        },
    }


def _stratified(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    return {
        "evaluator": "ifdr_yolo.stratified_ap40",
        "split_count": 371,
        "split_sha256": MODULE.FROZEN_SPLIT_SHA256,
        "runs": {"PLAIN": baseline, "CANDIDATE": candidate},
    }


def test_strict_and_nonstrict_gate_boundaries() -> None:
    report = MODULE.compute_gate(
        baseline_metrics=_overall(80.0, 90.0),
        candidate_metrics=_overall(81.1, 91.1),
        stratified_report=_stratified(
            _run(10.0, 20.0, 30.0, 40.0),
            _run(10.0, 20.1, 30.0, 40.0),
        ),
        baseline_run="PLAIN",
        candidate_run="CANDIDATE",
        engineering_checks={"terminal_evidence": True},
        enforce_frozen_baseline=False,
    )

    assert report["gates"]["overall"]["pass"] is True
    assert report["gates"]["small"]["pass"] is False
    assert report["gates"]["far"]["pass"] is True
    assert report["gates"]["near"]["pass"] is True
    assert report["gates"]["large"]["pass"] is True
    assert report["decision"] == "NO_GO"
    assert report["failed_gates"] == ["small"]


def test_engineering_failure_vetoes_scientific_pass() -> None:
    report = MODULE.compute_gate(
        baseline_metrics=_overall(80.0, 90.0),
        candidate_metrics=_overall(82.0, 92.5),
        stratified_report=_stratified(
            _run(10.0, 20.0, 30.0, 40.0),
            _run(11.0, 21.0, 31.0, 41.0),
        ),
        baseline_run="PLAIN",
        candidate_run="CANDIDATE",
        engineering_checks={"checkpoint_sha": True, "mirror": False},
        enforce_frozen_baseline=False,
    )

    assert report["scientific_pass"] is True
    assert report["engineering_pass"] is False
    assert report["decision"] == "NO_GO"
    assert report["failed_engineering_checks"] == ["mirror"]


if __name__ == "__main__":
    test_strict_and_nonstrict_gate_boundaries()
    test_engineering_failure_vetoes_scientific_pass()
    print("PASS: frozen gate semantics")
