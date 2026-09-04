from pathlib import Path
import unittest

from ifdr_yolo.eval.bootstrap_matrix import (
    BootstrapTask,
    build_bootstrap_tasks,
    parse_comparison_spec,
    parse_run_spec,
    result_is_complete,
)


class BootstrapMatrixTest(unittest.TestCase):
    def test_builds_paired_seed_class_and_slice_product(self) -> None:
        tasks = build_bootstrap_tasks(
            run_dirs={
                "baseline": {17: Path("baseline17"), 29: Path("baseline29")},
                "p2": {17: Path("p217"), 29: Path("p229")},
            },
            comparisons=(("baseline", "p2"),),
            class_names=("Pedestrian", "Cyclist"),
            slice_names=("small_25_40", "far_gt_40m"),
        )

        self.assertEqual(len(tasks), 8)
        self.assertEqual(
            tasks[0].task_id,
            "baseline_vs_p2__s17__Pedestrian__small_25_40",
        )
        self.assertEqual(tasks[0].reference_dir, Path("baseline17"))
        self.assertEqual(tasks[0].candidate_dir, Path("p217"))

    def test_rejects_unpaired_method_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "same seeds"):
            build_bootstrap_tasks(
                run_dirs={
                    "baseline": {17: Path("baseline17")},
                    "p2": {29: Path("p229")},
                },
                comparisons=(("baseline", "p2"),),
                class_names=("Pedestrian",),
                slice_names=("small_25_40",),
            )

    def test_parses_run_and_comparison_specs(self) -> None:
        method, seed, prediction_dir = parse_run_spec(
            "baseline:17=/runs/baseline/predictions/labels"
        )
        self.assertEqual((method, seed), ("baseline", 17))
        self.assertEqual(
            prediction_dir,
            Path("/runs/baseline/predictions/labels"),
        )
        self.assertEqual(
            parse_comparison_spec("baseline=p2"),
            ("baseline", "p2"),
        )

    def test_complete_result_requires_matching_task_and_settings(self) -> None:
        task = BootstrapTask(
            reference="baseline",
            candidate="p2",
            seed=17,
            class_name="Car",
            slice_name="small_25_40",
            reference_dir=Path("baseline17"),
            candidate_dir=Path("p217"),
        )
        payload = {
            "schema_version": 1,
            "metric": "KITTI_2D_CONDITIONAL_AP40_PAIRED_IMAGE_BOOTSTRAP",
            "reference": {"name": "baseline"},
            "candidate": {"name": "p2"},
            "class_name": "Car",
            "target_slice": {"name": "small_25_40"},
            "comparison": {"iterations": 1000, "seed": 20260803},
        }

        self.assertTrue(
            result_is_complete(
                payload,
                task=task,
                iterations=1000,
                bootstrap_seed=20260803,
            )
        )
        payload["comparison"]["iterations"] = 20
        self.assertFalse(
            result_is_complete(
                payload,
                task=task,
                iterations=1000,
                bootstrap_seed=20260803,
            )
        )


if __name__ == "__main__":
    unittest.main()
