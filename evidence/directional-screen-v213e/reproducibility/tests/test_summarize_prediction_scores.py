from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "summarize_prediction_scores.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_prediction_scores", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_score_summary_and_comparison() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        left = root / "left"
        right = root / "right"
        left.mkdir()
        right.mkdir()
        (root / "split.txt").write_text("000001\n000002\n", encoding="utf-8")
        (left / "000001.txt").write_text(
            "1 0.5 0.5 0.1 0.1 0.2\n2 0.5 0.5 0.1 0.1 0.4\n",
            encoding="utf-8",
        )
        (left / "000002.txt").write_text(
            "2 0.5 0.5 0.1 0.1 0.8\n",
            encoding="utf-8",
        )
        (right / "000001.txt").write_text(
            "1 0.5 0.5 0.1 0.1 0.3\n2 0.5 0.5 0.1 0.1 0.6\n",
            encoding="utf-8",
        )
        (right / "000002.txt").write_text(
            "2 0.5 0.5 0.1 0.1 1.0\n",
            encoding="utf-8",
        )

        report = MODULE.summarize_runs(
            runs={"B1": left, "ANCHORED": right},
            split_path=root / "split.txt",
            reference_name="B1",
            candidate_name="ANCHORED",
            thresholds=(0.5,),
        )

        cyclist = report["runs"]["B1"]["classes"]["Cyclist"]
        assert cyclist["detections"] == 2
        assert abs(cyclist["median"] - 0.6) < 1e-12
        assert cyclist["count_at_or_above"]["0.5"] == 1
        delta = report["comparison"]["classes"]["Cyclist"]
        assert delta["detection_count_delta"] == 0
        assert abs(delta["mean_score_delta"] - 0.2) < 1e-12


def test_missing_split_file_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "run"
        run.mkdir()
        (root / "split.txt").write_text("000001\n000002\n", encoding="utf-8")
        (run / "000001.txt").write_text("2 0.5 0.5 0.1 0.1 0.8\n", encoding="utf-8")
        try:
            MODULE.summarize_runs(
                runs={"ONLY": run},
                split_path=root / "split.txt",
                reference_name=None,
                candidate_name=None,
                thresholds=(0.5,),
            )
        except ValueError as error:
            assert "file set" in str(error)
        else:
            raise AssertionError("incomplete prediction set was accepted")


def test_csv_writer_handles_median_without_duplicate_q50_field() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "run"
        run.mkdir()
        split = root / "split.txt"
        split.write_text("000001\n", encoding="utf-8")
        (run / "000001.txt").write_text(
            "2 0.5 0.5 0.1 0.1 0.8\n", encoding="utf-8"
        )
        report = MODULE.summarize_runs(
            runs={"ONLY": run},
            split_path=split,
            reference_name=None,
            candidate_name=None,
            thresholds=(0.5,),
        )
        output = root / "summary.csv"
        MODULE._write_csv_new(output, report)
        text = output.read_text(encoding="utf-8")
        assert "median" in text
        assert "Cyclist" in text


if __name__ == "__main__":
    test_score_summary_and_comparison()
    test_missing_split_file_fails_closed()
    test_csv_writer_handles_median_without_duplicate_q50_field()
    print("PASS: prediction score summary")
