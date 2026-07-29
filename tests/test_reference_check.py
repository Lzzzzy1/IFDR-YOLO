from pathlib import Path
import subprocess
import sys
import unittest

from ifdr_yolo.eval.reference_check import (
    build_controlled_suite,
    to_reference_detection_annotation,
    to_reference_ground_truth_annotation,
)


class ReferenceCheckTest(unittest.TestCase):
    def test_controlled_suite_has_fifty_images_and_five_scenarios(self) -> None:
        ground_truth, scenarios = build_controlled_suite()
        self.assertEqual(len(ground_truth), 50)
        self.assertEqual(
            tuple(scenarios),
            ("perfect", "duplicate", "high_fp", "half_missed", "ignore"),
        )
        self.assertEqual(len(scenarios["perfect"]), 50)

    def test_reference_annotations_have_official_array_shapes(self) -> None:
        ground_truth, scenarios = build_controlled_suite()
        image_id = "000000"

        gt_annotation = to_reference_ground_truth_annotation(
            ground_truth[image_id]
        )
        detection_annotation = to_reference_detection_annotation(
            scenarios["perfect"][image_id]
        )

        self.assertEqual(gt_annotation["bbox"].shape, (6, 4))
        self.assertEqual(gt_annotation["dimensions"].shape, (6, 3))
        self.assertEqual(detection_annotation["bbox"].shape, (3, 4))
        self.assertEqual(detection_annotation["score"].shape, (3,))

    def test_reference_check_script_has_direct_cli(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "scripts/check_ap40_reference.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--reference-dir", completed.stdout)


if __name__ == "__main__":
    unittest.main()
