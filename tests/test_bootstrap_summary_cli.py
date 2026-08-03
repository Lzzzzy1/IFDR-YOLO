import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.summarize_paired_bootstrap import main


class BootstrapSummaryCliTest(unittest.TestCase):
    def test_writes_json_and_flat_csv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "results"
            input_dir.mkdir()
            (input_dir / "status.json").write_text(
                json.dumps({"state": "complete"}), encoding="utf-8"
            )
            for seed, difference in ((17, 1.0), (29, 2.0), (41, 3.0)):
                payload = {
                    "schema_version": 1,
                    "metric": (
                        "KITTI_2D_CONDITIONAL_AP40_"
                        "PAIRED_IMAGE_BOOTSTRAP"
                    ),
                    "reference": {"name": "baseline"},
                    "candidate": {"name": "p2"},
                    "class_name": "Car",
                    "target_slice": {"name": "small_25_40"},
                    "comparison": {
                        "reference_ap40": 60.0,
                        "candidate_ap40": 60.0 + difference,
                        "difference_ap40": difference,
                        "ci_lower": difference - 0.5,
                        "ci_upper": difference + 0.5,
                        "probability_improvement": 0.99,
                        "iterations": 1000,
                        "seed": 20260803,
                    },
                }
                path = input_dir / (
                    f"baseline_vs_p2__s{seed}__Car__small_25_40.json"
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
            output_json = root / "summary.json"
            output_csv = root / "summary.csv"

            exit_code = main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--expected-seed",
                    "17",
                    "--expected-seed",
                    "29",
                    "--expected-seed",
                    "41",
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            summary = json.loads(output_json.read_text(encoding="utf-8"))
            with output_csv.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["group_count"], 1)
        self.assertEqual(
            summary["groups"][0]["seed_summary"]["mean_difference_ap40"],
            2.0,
        )
        self.assertEqual(rows[0]["mean_difference_ap40"], "2.0")


if __name__ == "__main__":
    unittest.main()
