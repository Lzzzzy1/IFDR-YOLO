import unittest

from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    Difficulty,
    KittiObject,
)
from ifdr_yolo.eval.detection_reliability import (
    deterministic_calibration_split,
    evaluate_detection_reliability,
    select_lrp_threshold,
)


def ground_truth(box: BoundingBox) -> KittiObject:
    return KittiObject(
        kind="Car",
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox=box,
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 10.0),
        rotation_y=0.0,
    )


class DetectionReliabilityTest(unittest.TestCase):
    def test_perfect_localization_reports_score_to_iou_calibration_gap(
        self,
    ) -> None:
        box = BoundingBox(0, 0, 100, 100)
        result = evaluate_detection_reliability(
            gt_by_image={"000001": (ground_truth(box),)},
            detections_by_image={
                "000001": (Detection("000001", "Car", 0.8, box),)
            },
            class_name="Car",
            difficulty=Difficulty.HARD,
            confidence_threshold=0.0,
            bins=5,
        )

        self.assertAlmostEqual(result.laece0, 0.2)
        self.assertEqual(result.lrp, 0.0)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.false_negatives, 0)

    def test_false_positive_affects_both_laece_and_lrp(self) -> None:
        box = BoundingBox(0, 0, 100, 100)
        false_box = BoundingBox(200, 200, 300, 300)
        result = evaluate_detection_reliability(
            gt_by_image={"000001": (ground_truth(box),)},
            detections_by_image={
                "000001": (
                    Detection("000001", "Car", 0.8, box),
                    Detection("000001", "Car", 0.2, false_box),
                )
            },
            class_name="Car",
            difficulty=Difficulty.HARD,
            confidence_threshold=0.0,
            bins=2,
        )

        self.assertAlmostEqual(result.laece0, 0.2)
        self.assertAlmostEqual(result.lrp, 0.5)
        self.assertAlmostEqual(result.lrp_fp, 0.5)

    def test_selects_lrp_optimal_threshold_on_calibration_images(self) -> None:
        box = BoundingBox(0, 0, 100, 100)
        high_false = BoundingBox(200, 200, 300, 300)
        low_false = BoundingBox(400, 400, 500, 500)

        threshold = select_lrp_threshold(
            gt_by_image={"000001": (ground_truth(box),)},
            detections_by_image={
                "000001": (
                    Detection("000001", "Car", 0.9, high_false),
                    Detection("000001", "Car", 0.8, box),
                    Detection("000001", "Car", 0.1, low_false),
                )
            },
            class_name="Car",
            difficulty=Difficulty.HARD,
        )

        self.assertEqual(threshold, 0.8)

    def test_calibration_split_is_deterministic_disjoint_and_complete(self) -> None:
        image_ids = tuple(f"{index:06d}" for index in range(10))

        calibration_ids, test_ids = deterministic_calibration_split(
            image_ids,
            seed=20260803,
        )

        self.assertEqual(
            (calibration_ids, test_ids),
            deterministic_calibration_split(image_ids, seed=20260803),
        )
        self.assertFalse(set(calibration_ids) & set(test_ids))
        self.assertEqual(set(calibration_ids) | set(test_ids), set(image_ids))
        self.assertEqual((len(calibration_ids), len(test_ids)), (5, 5))


if __name__ == "__main__":
    unittest.main()
