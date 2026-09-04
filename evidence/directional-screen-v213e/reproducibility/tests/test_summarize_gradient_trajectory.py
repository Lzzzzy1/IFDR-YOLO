from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "summarize_gradient_trajectory.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_gradient_trajectory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
    epoch: int,
    cosine: float | None,
    conflict: bool,
    counterfactual: float,
    factor: float,
    process_id: int = 123,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "process_id": process_id,
        "parameter_groups": {
            "semantic_anchor": {
                "gradient_norms": {
                    "counterfactual": counterfactual,
                    "factor": factor,
                },
                "pairs": {
                    "counterfactual::factor": {
                        "cosine": cosine,
                        "conflict": conflict,
                    }
                },
            }
        },
    }


def test_epoch_means_use_only_rows_with_valid_cosine() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "gradient.jsonl"
        rows = [
            _row(5, 0.9, False, 99.0, 1.0),
            _row(6, 0.5, False, 0.2, 0.1),
            _row(6, -0.5, True, 0.4, 0.3),
            _row(6, None, False, 100.0, 1.0),
            _row(7, 0.25, False, 0.5, 1.0, process_id=456),
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        report = MODULE.summarize_run(
            name="TEST",
            path=path,
            epoch_start=6,
            epoch_end=8,
        )

        epoch6 = report["epochs"]["6"]
        assert epoch6["mean_cosine"] == 0.0
        assert epoch6["conflicts"] == 1
        assert epoch6["valid_records"] == 2
        assert abs(epoch6["norm_ratio"] - 1.5) < 1e-12
        assert report["epochs"]["7"]["norm_ratio"] == 0.5
        assert report["missing_epochs"] == [8]
        assert report["coverage_complete"] is False
        assert report["process_ids"] == [123, 456]


def test_conflict_flag_must_match_negative_cosine() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "gradient.jsonl"
        path.write_text(
            json.dumps(_row(6, -0.25, False, 0.2, 0.1)) + "\n",
            encoding="utf-8",
        )
        try:
            MODULE.summarize_run(
                name="BAD",
                path=path,
                epoch_start=6,
                epoch_end=6,
            )
        except ValueError as error:
            assert "conflict flag" in str(error)
        else:
            raise AssertionError("inconsistent conflict flag was accepted")


if __name__ == "__main__":
    test_epoch_means_use_only_rows_with_valid_cosine()
    test_conflict_flag_must_match_negative_cosine()
    print("PASS: gradient trajectory statistic")
