import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ifdr_yolo.eval.bootstrap_summary import (
    SeedBootstrapResult,
    parse_task_id,
    summarize_bootstrap_directory,
    summarize_seed_results,
)


class BootstrapSummaryTest(unittest.TestCase):
    def test_parses_task_id_without_confusing_method_underscores(self) -> None:
        task = parse_task_id(
            "p2_vs_reliable_fusion__s29__Pedestrian__far_gt_40m"
        )

        self.assertEqual(task.reference, "p2")
        self.assertEqual(task.candidate, "reliable_fusion")
        self.assertEqual(task.seed, 29)
        self.assertEqual(task.class_name, "Pedestrian")
        self.assertEqual(task.slice_name, "far_gt_40m")

    def test_summarizes_seed_variation_without_pooled_significance(self) -> None:
        results = tuple(
            SeedBootstrapResult(
                seed=seed,
                reference_ap40=reference,
                candidate_ap40=reference + difference,
                difference_ap40=difference,
                ci_lower=lower,
                ci_upper=upper,
                probability_improvement=probability,
            )
            for seed, reference, difference, lower, upper, probability in (
                (17, 60.0, 1.0, 0.2, 1.8, 0.99),
                (29, 61.0, 2.0, -0.1, 4.1, 0.94),
                (41, 62.0, 3.0, 0.5, 5.5, 0.995),
            )
        )

        summary = summarize_seed_results(results, expected_seeds=(17, 29, 41))

        self.assertEqual(summary.seed_count, 3)
        self.assertEqual(summary.mean_difference_ap40, 2.0)
        self.assertEqual(summary.sample_std_difference_ap40, 1.0)
        self.assertEqual(summary.positive_seed_count, 3)
        self.assertEqual(summary.positive_ci_seed_count, 2)
        self.assertEqual(summary.direction_consistency, "positive")

    def test_rejects_missing_expected_seed(self) -> None:
        result = SeedBootstrapResult(
            seed=17,
            reference_ap40=60.0,
            candidate_ap40=61.0,
            difference_ap40=1.0,
            ci_lower=0.1,
            ci_upper=1.9,
            probability_improvement=0.99,
        )

        with self.assertRaisesRegex(ValueError, "expected seeds"):
            summarize_seed_results((result,), expected_seeds=(17, 29, 41))

    def test_summarizes_complete_result_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory)
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
                output = input_dir / (
                    f"baseline_vs_p2__s{seed}__Car__small_25_40.json"
                )
                output.write_text(json.dumps(payload), encoding="utf-8")

            groups = summarize_bootstrap_directory(
                input_dir,
                expected_seeds=(17, 29, 41),
            )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].iterations, 1000)
        self.assertEqual(groups[0].bootstrap_seed, 20260803)
        self.assertEqual(groups[0].seed_summary.mean_difference_ap40, 2.0)

    def test_rejects_directory_before_matrix_is_complete(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory)
            (input_dir / "status.json").write_text(
                json.dumps({"state": "running"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "not complete"):
                summarize_bootstrap_directory(
                    input_dir,
                    expected_seeds=(17, 29, 41),
                )


if __name__ == "__main__":
    unittest.main()
