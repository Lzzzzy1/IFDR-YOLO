from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "summarize_training_stability.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_training_stability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


HEADER = (
    "epoch,time,train/box_loss,metrics/precision(B),metrics/recall(B),"
    "metrics/mAP50(B),metrics/mAP50-95(B),lr/pg0\n"
)


def _write(path: Path, rows: list[str]) -> None:
    path.write_text(HEADER + "".join(rows), encoding="utf-8")


def test_complete_finite_trajectories_and_final_delta() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        b1 = root / "b1.csv"
        anchored = root / "anchored.csv"
        _write(
            b1,
            [
                "1,1,2,0.5,0.4,0.6,0.3,0.01\n",
                "2,2,1,0.6,0.5,0.7,0.4,0.001\n",
            ],
        )
        _write(
            anchored,
            [
                "1,1.1,2,0.5,0.4,0.6,0.3,0.01\n",
                "2,2.1,1.1,0.55,0.45,0.65,0.35,0.001\n",
            ],
        )
        report = MODULE.summarize_trajectories(
            runs={"B1": b1, "ANCHORED": anchored},
            expected_epochs=2,
            reference_name="B1",
            candidate_name="ANCHORED",
        )

        assert report["all_training_stability_checks_pass"] is True
        assert report["lr_schedule_identical"] is True
        assert abs(
            report["comparison"]["final_delta"]["metrics/mAP50-95(B)"] + 0.05
        ) < 1e-12


def test_nonfinite_metric_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "bad.csv"
        _write(path, ["1,1,2,nan,0.4,0.6,0.3,0.01\n"])
        try:
            MODULE.summarize_trajectories(
                runs={"BAD": path},
                expected_epochs=1,
                reference_name=None,
                candidate_name=None,
            )
        except ValueError as error:
            assert "non-finite" in str(error)
        else:
            raise AssertionError("non-finite metric was accepted")


def test_lr_schedule_mismatch_is_reported() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        left = root / "left.csv"
        right = root / "right.csv"
        _write(left, ["1,1,2,0.5,0.4,0.6,0.3,0.01\n"])
        _write(right, ["1,1,2,0.5,0.4,0.6,0.3,0.02\n"])
        report = MODULE.summarize_trajectories(
            runs={"LEFT": left, "RIGHT": right},
            expected_epochs=1,
            reference_name=None,
            candidate_name=None,
        )
        assert report["lr_schedule_identical"] is False


if __name__ == "__main__":
    test_complete_finite_trajectories_and_final_delta()
    test_nonfinite_metric_fails_closed()
    test_lr_schedule_mismatch_is_reported()
    print("PASS: training stability summary")
