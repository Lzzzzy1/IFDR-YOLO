from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "compare_local_endpoints.py"
)
SPEC = importlib.util.spec_from_file_location("compare_local_endpoints", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _run(overall: float, small: float, far: float, near: float, large: float) -> dict:
    return {
        "overall": {
            "macro_ap_r40": overall,
            "classes": {
                "Pedestrian": {"ap_r40": overall - 1.0},
                "Cyclist": {"ap_r40": overall + 1.0},
            },
        },
        "slices_macro_ap_r40": {
            "small": small,
            "far": far,
            "near": near,
            "large": large,
        },
    }


def test_comparison_applies_frozen_gates_and_b1_ablation() -> None:
    historical = {
        "scope": "LOCAL_SEED0_BATCH2_15EPOCH_DIRECTION_SCREEN_ONLY",
        "runs": {
            "PLAIN": _run(80.0, 30.0, 20.0, 90.0, 95.0),
            "B0": _run(82.0, 31.0, 19.0, 90.0, 95.0),
            "B1": _run(83.0, 29.0, 21.0, 91.0, 96.0),
        },
    }
    gate = {
        "split_count": 371,
        "split_sha256": "a" * 64,
        "baseline_run": "PLAIN",
        "candidate_run": "ANCHORED",
        "baseline_overall_classes": {"Pedestrian": 79.0, "Cyclist": 81.0},
        "candidate_overall_classes": {"Pedestrian": 80.0, "Cyclist": 82.0},
        "gates": {
            "overall": {"baseline": 80.0, "candidate": 81.0},
            "small": {"baseline": 30.0, "candidate": 32.0},
            "far": {"baseline": 20.0, "candidate": 22.0},
            "near": {"baseline": 90.0, "candidate": 90.0},
            "large": {"baseline": 95.0, "candidate": 95.0},
        },
    }

    report = MODULE.compare_endpoints(historical=historical, frozen_gate=gate)

    assert report["runs"]["B0"]["decision"] == "NO_GO"
    assert report["runs"]["B0"]["failed_gates"] == ["far"]
    assert report["runs"]["B1"]["failed_gates"] == ["small"]
    assert report["runs"]["ANCHORED"]["failed_gates"] == ["overall"]
    assert report["anchored_minus_b1"]["overall"] == -2.0
    assert report["ranks"]["overall"][0]["run"] == "B1"


def test_plain_binding_mismatch_fails_closed() -> None:
    historical = {
        "scope": "LOCAL",
        "runs": {
            "PLAIN": _run(80.1, 30.0, 20.0, 90.0, 95.0),
            "B0": _run(82.0, 31.0, 19.0, 90.0, 95.0),
            "B1": _run(83.0, 29.0, 21.0, 91.0, 96.0),
        },
    }
    gate = {
        "split_count": 371,
        "split_sha256": "a" * 64,
        "baseline_run": "PLAIN",
        "candidate_run": "ANCHORED",
        "baseline_overall_classes": {"Pedestrian": 79.0, "Cyclist": 81.0},
        "candidate_overall_classes": {"Pedestrian": 80.0, "Cyclist": 82.0},
        "gates": {
            "overall": {"baseline": 80.0, "candidate": 81.0},
            "small": {"baseline": 30.0, "candidate": 32.0},
            "far": {"baseline": 20.0, "candidate": 22.0},
            "near": {"baseline": 90.0, "candidate": 90.0},
            "large": {"baseline": 95.0, "candidate": 95.0},
        },
    }
    try:
        MODULE.compare_endpoints(historical=historical, frozen_gate=gate)
    except ValueError as error:
        assert "PLAIN binding" in str(error)
    else:
        raise AssertionError("mismatched PLAIN binding was accepted")


if __name__ == "__main__":
    test_comparison_applies_frozen_gates_and_b1_ablation()
    test_plain_binding_mismatch_fails_closed()
    print("PASS: local endpoint comparison")
