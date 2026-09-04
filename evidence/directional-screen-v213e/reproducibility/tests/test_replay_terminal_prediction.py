from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "replay_terminal_prediction.py"
)
SPEC = importlib.util.spec_from_file_location("replay_terminal_prediction", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_predictions(root: Path, values: dict[str, str]) -> Path:
    labels = root / "labels"
    labels.mkdir(parents=True)
    for image_id, text in values.items():
        (labels / f"{image_id}.txt").write_text(text, encoding="utf-8")
    return labels


def test_prediction_comparison_requires_exact_ids_and_hashes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        values = {"000001": "a\n", "000002": "", "000003": "c\n"}
        reference = _make_predictions(root / "reference", values)
        replay = _make_predictions(root / "replay", values)

        report = MODULE.compare_prediction_dirs(
            reference_dir=reference,
            replay_dir=replay,
            expected_ids=tuple(values),
        )
        assert report["exact_match"] is True
        assert report["mismatches"] == []
        assert report["reference_aggregate_sha256"] == report["replay_aggregate_sha256"]

        (replay / "000002.txt").write_text("changed\n", encoding="utf-8")
        changed = MODULE.compare_prediction_dirs(
            reference_dir=reference,
            replay_dir=replay,
            expected_ids=tuple(values),
        )
        assert changed["exact_match"] is False
        assert changed["mismatches"] == ["000002"]


if __name__ == "__main__":
    test_prediction_comparison_requires_exact_ids_and_hashes()
    print("PASS: terminal prediction replay comparison")
