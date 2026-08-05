from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from ifdr_yolo.data.development_split import build_development_split


SEED = 20260805
FRACTION = 0.10
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_factor_metadata.py"


def _rows_with_four_strata(count: int) -> list[dict[str, object]]:
    """Build equal-sized no-cyclist and cyclist tertile strata."""
    if count % 4:
        raise ValueError("count must be divisible by four")

    per_stratum = count // 4
    rows: list[dict[str, object]] = []
    for index in range(per_stratum):
        rows.append(
            {
                "image_id": f"image_{index:04d}",
                "cyclist": False,
                "cyclist_joint": 0.0,
            }
        )

    cyclist_count = count - per_stratum
    for index in range(cyclist_count):
        score = index / (cyclist_count - 1)
        rows.append(
            {
                "image_id": f"image_{per_stratum + index:04d}",
                "cyclist": True,
                "cyclist_joint": score,
            }
        )
    return rows


def _rows_without_cyclists(count: int) -> list[dict[str, object]]:
    return [
        {
            "image_id": f"image_{index:04d}",
            "cyclist": False,
            "cyclist_joint": 0.0,
        }
        for index in range(count)
    ]


def _write_rows_jsonl(path: Path, input_rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in input_rows),
        encoding="utf-8",
    )


def _run_cli(input_jsonl: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-jsonl",
            str(input_jsonl),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(SEED),
            "--fraction",
            str(FRACTION),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


class DevelopmentSplitTest(unittest.TestCase):
    def test_joint_score_is_finite_and_bounded(self) -> None:
        for score in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(score=score):
                input_rows = _rows_with_four_strata(40)
                input_rows[10]["cyclist_joint"] = score
                split = build_development_split(
                    input_rows, seed=SEED, fraction=FRACTION
                )
                self.assertEqual(len(split.fit_ids) + len(split.development_ids), 40)

        for score in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01):
            with self.subTest(score=score):
                input_rows = _rows_with_four_strata(40)
                input_rows[10]["cyclist_joint"] = score
                with self.assertRaisesRegex(ValueError, "finite|between|bounded"):
                    build_development_split(
                        input_rows, seed=SEED, fraction=FRACTION
                    )

    def test_no_cyclist_joint_score_is_zero(self) -> None:
        split = build_development_split(
            _rows_with_four_strata(40), seed=SEED, fraction=FRACTION
        )
        self.assertIn("no_cyclist", split.strata)

        invalid_rows = _rows_with_four_strata(40)
        invalid_rows[0]["cyclist_joint"] = 0.1
        with self.assertRaisesRegex(ValueError, "without Cyclist|no_cyclist|0.0"):
            build_development_split(invalid_rows, seed=SEED, fraction=FRACTION)

    def test_rejects_unregistered_seed_and_fraction(self) -> None:
        input_rows = _rows_with_four_strata(40)
        with self.assertRaisesRegex(ValueError, "seed=20260805"):
            build_development_split(input_rows, seed=17, fraction=FRACTION)
        with self.assertRaisesRegex(ValueError, "fraction=0.10"):
            build_development_split(input_rows, seed=SEED, fraction=0.20)

    def test_reversed_input_order_keeps_ids_and_sha256(self) -> None:
        input_rows = _rows_with_four_strata(120)
        first = build_development_split(input_rows, seed=SEED, fraction=FRACTION)
        second = build_development_split(
            list(reversed(input_rows)), seed=SEED, fraction=FRACTION
        )
        self.assertEqual(first.fit_ids, second.fit_ids)
        self.assertEqual(first.development_ids, second.development_ids)
        self.assertEqual(first.sha256, second.sha256)

    def test_round_half_up_count(self) -> None:
        split = build_development_split(
            _rows_without_cyclists(15), seed=SEED, fraction=FRACTION
        )
        self.assertEqual(len(split.development_ids), 2)

    def test_fit_development_are_disjoint_and_complete(self) -> None:
        input_rows = _rows_with_four_strata(120)
        split = build_development_split(input_rows, seed=SEED, fraction=FRACTION)
        fit_ids = set(split.fit_ids)
        development_ids = set(split.development_ids)
        self.assertTrue(fit_ids.isdisjoint(development_ids))
        self.assertEqual(
            fit_ids | development_ids,
            {row["image_id"] for row in input_rows},
        )

    def test_strata_are_immutable_after_build(self) -> None:
        split = build_development_split(
            _rows_with_four_strata(40), seed=SEED, fraction=FRACTION
        )
        with self.assertRaises(TypeError):
            split.strata["no_cyclist"] = ()  # type: ignore[index]

    def test_hash_is_stable_for_same_rows(self) -> None:
        input_rows = _rows_with_four_strata(40)
        first = build_development_split(input_rows, seed=SEED, fraction=FRACTION)
        second = build_development_split(
            [dict(row) for row in input_rows], seed=SEED, fraction=FRACTION
        )
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(len(first.sha256), 64)

    def test_feasible_small_strata_have_both_seats(self) -> None:
        input_rows = _rows_with_four_strata(40)
        split = build_development_split(input_rows, seed=SEED, fraction=FRACTION)
        self.assertEqual(
            set(split.strata),
            {"no_cyclist", "cyclist_lower", "cyclist_middle", "cyclist_upper"},
        )
        fit_ids = set(split.fit_ids)
        development_ids = set(split.development_ids)
        for stratum_ids in split.strata.values():
            self.assertGreaterEqual(len(stratum_ids), 2)
            self.assertTrue(set(stratum_ids) & fit_ids)
            self.assertTrue(set(stratum_ids) & development_ids)

    def test_infeasible_small_strata_raise_quota_constraints(self) -> None:
        with self.assertRaisesRegex(ValueError, "quota constraints"):
            build_development_split(
                _rows_with_four_strata(20), seed=SEED, fraction=FRACTION
            )

    def test_duplicate_ids_are_rejected(self) -> None:
        input_rows = _rows_with_four_strata(40)
        input_rows[1]["image_id"] = input_rows[0]["image_id"]
        with self.assertRaisesRegex(ValueError, "duplicate image_id"):
            build_development_split(input_rows, seed=SEED, fraction=FRACTION)

    def test_cli_outputs_are_atomic_and_idempotent(self) -> None:
        input_rows = _rows_with_four_strata(40)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_jsonl = root / "rows.jsonl"
            output_dir = root / "split"
            _write_rows_jsonl(input_jsonl, input_rows)

            first = _run_cli(input_jsonl, output_dir)
            self.assertEqual(first.returncode, 0, first.stderr)
            output_names = (
                "fit_ids.txt",
                "development_ids.txt",
                "development_split.json",
            )
            first_bytes = {
                name: (output_dir / name).read_bytes() for name in output_names
            }

            second = _run_cli(input_jsonl, output_dir)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                first_bytes,
                {name: (output_dir / name).read_bytes() for name in output_names},
            )

    def test_cli_refuses_non_identical_overwrite(self) -> None:
        input_rows = _rows_with_four_strata(40)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_jsonl = root / "rows.jsonl"
            output_dir = root / "split"
            _write_rows_jsonl(input_jsonl, input_rows)
            first = _run_cli(input_jsonl, output_dir)
            self.assertEqual(first.returncode, 0, first.stderr)

            fit_path = output_dir / "fit_ids.txt"
            fit_path.write_bytes(b"tampered\n")
            refused = _run_cli(input_jsonl, output_dir)
            self.assertNotEqual(refused.returncode, 0)
            self.assertRegex(
                refused.stdout + refused.stderr,
                "existing output|overwrite",
            )
            self.assertEqual(fit_path.read_bytes(), b"tampered\n")

    def test_cli_help_is_available(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("input-jsonl", result.stdout)


if __name__ == "__main__":
    unittest.main()
