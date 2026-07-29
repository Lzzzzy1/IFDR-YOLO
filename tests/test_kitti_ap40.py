import unittest

from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    Difficulty,
    KittiObject,
)
from ifdr_yolo.eval.kitti_ap40 import (
    box_iou,
    evaluate_class,
    is_valid_ground_truth,
)


def ground_truth(
    kind: str,
    box: BoundingBox,
    truncated: float = 0.0,
    occluded: int = 0,
) -> KittiObject:
    return KittiObject(
        kind=kind,
        truncated=truncated,
        occluded=occluded,
        alpha=0.0,
        bbox=box,
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 0.0),
        rotation_y=0.0,
    )


class KittiAP40Test(unittest.TestCase):
    def test_iou_identity_and_disjoint(self) -> None:
        box = BoundingBox(0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(box_iou(box, box), 1.0)
        self.assertAlmostEqual(
            box_iou(box, BoundingBox(20.0, 20.0, 30.0, 30.0)),
            0.0,
        )

    def test_difficulty_height_threshold(self) -> None:
        short = ground_truth("Pedestrian", BoundingBox(0, 0, 20, 24))
        boundary = ground_truth("Pedestrian", BoundingBox(0, 0, 20, 25))
        tall = ground_truth("Pedestrian", BoundingBox(0, 0, 20, 25.01))
        self.assertFalse(
            is_valid_ground_truth(short, "Pedestrian", Difficulty.MODERATE)
        )
        self.assertFalse(
            is_valid_ground_truth(boundary, "Pedestrian", Difficulty.MODERATE)
        )
        self.assertTrue(
            is_valid_ground_truth(tall, "Pedestrian", Difficulty.MODERATE)
        )

    def test_perfect_detection_has_one_hundred_ap(self) -> None:
        box = BoundingBox(10, 10, 50, 80)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Pedestrian", box),)},
            detections_by_image={
                "000001": (Detection("000001", "Pedestrian", 0.9, box),)
            },
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertAlmostEqual(result.ap40, 100.0)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 0)

    def test_no_detection_has_zero_ap(self) -> None:
        box = BoundingBox(10, 10, 50, 80)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Pedestrian", box),)},
            detections_by_image={"000001": ()},
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertAlmostEqual(result.ap40, 0.0)
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 0)

    def test_car_detection_matching_van_is_ignored(self) -> None:
        box = BoundingBox(0, 0, 100, 100)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Van", box),)},
            detections_by_image={
                "000001": (Detection("000001", "Car", 0.9, box),)
            },
            class_name="Car",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.ignored_detections, 1)

    def test_pedestrian_detection_matching_person_sitting_is_ignored(self) -> None:
        box = BoundingBox(0, 0, 50, 80)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Person_sitting", box),)},
            detections_by_image={
                "000001": (Detection("000001", "Pedestrian", 0.9, box),)
            },
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.ignored_detections, 1)

    def test_detection_inside_dontcare_is_ignored(self) -> None:
        dontcare = ground_truth("DontCare", BoundingBox(0, 0, 100, 100))
        detection = Detection(
            "000001",
            "Pedestrian",
            0.9,
            BoundingBox(10, 10, 50, 80),
        )
        result = evaluate_class(
            gt_by_image={"000001": (dontcare,)},
            detections_by_image={"000001": (detection,)},
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.ignored_detections, 1)

    def test_dontcare_overlap_equal_to_threshold_does_not_suppress(self) -> None:
        dontcare = ground_truth("DontCare", BoundingBox(0, 0, 50, 100))
        detection = Detection(
            "000001",
            "Pedestrian",
            0.9,
            BoundingBox(0, 0, 100, 100),
        )
        result = evaluate_class(
            gt_by_image={"000001": (dontcare,)},
            detections_by_image={"000001": (detection,)},
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.ignored_detections, 0)

    def test_detection_matching_difficulty_ignored_ground_truth_is_ignored(
        self,
    ) -> None:
        box = BoundingBox(0, 0, 20, 20)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Pedestrian", box),)},
            detections_by_image={
                "000001": (Detection("000001", "Pedestrian", 0.9, box),)
            },
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.num_valid_gt, 0)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.ignored_detections, 1)

    def test_detection_below_difficulty_minimum_height_is_ignored(self) -> None:
        detection = Detection(
            "000001",
            "Pedestrian",
            0.9,
            BoundingBox(0, 0, 20, 20),
        )
        result = evaluate_class(
            gt_by_image={"000001": ()},
            detections_by_image={"000001": (detection,)},
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.ignored_detections, 1)

    def test_ordinary_detection_without_ground_truth_is_false_positive(self) -> None:
        detection = Detection(
            "000001",
            "Pedestrian",
            0.9,
            BoundingBox(0, 0, 30, 80),
        )
        result = evaluate_class(
            gt_by_image={"000001": ()},
            detections_by_image={"000001": (detection,)},
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.ignored_detections, 0)

    def test_duplicate_detection_is_false_positive(self) -> None:
        box = BoundingBox(0, 0, 50, 80)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Pedestrian", box),)},
            detections_by_image={
                "000001": (
                    Detection("000001", "Pedestrian", 0.9, box),
                    Detection("000001", "Pedestrian", 0.8, box),
                )
            },
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 1)

    def test_iou_equal_to_class_threshold_is_not_a_match(self) -> None:
        truth = BoundingBox(0, 0, 100, 100)
        detection_box = BoundingBox(0, 0, 50, 100)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Pedestrian", truth),)},
            detections_by_image={
                "000001": (
                    Detection("000001", "Pedestrian", 0.9, detection_box),
                )
            },
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 1)

    def test_high_score_false_positive_precedes_low_score_true_positive(
        self,
    ) -> None:
        truth = BoundingBox(0, 0, 50, 80)
        false_box = BoundingBox(100, 100, 150, 180)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Pedestrian", truth),)},
            detections_by_image={
                "000001": (
                    Detection("000001", "Pedestrian", 0.2, truth),
                    Detection("000001", "Pedestrian", 0.9, false_box),
                )
            },
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertEqual(result.scores, (0.9, 0.2))
        self.assertAlmostEqual(result.ap40, 50.0)

    def test_unknown_evaluation_class_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            evaluate_class({}, {}, "Truck", Difficulty.MODERATE)

    def test_detection_image_id_must_match_mapping_key(self) -> None:
        detection = Detection(
            "000002",
            "Pedestrian",
            0.9,
            BoundingBox(0, 0, 30, 80),
        )
        with self.assertRaisesRegex(ValueError, "image ID mismatch"):
            evaluate_class(
                gt_by_image={"000001": ()},
                detections_by_image={"000001": (detection,)},
                class_name="Pedestrian",
                difficulty=Difficulty.MODERATE,
            )


if __name__ == "__main__":
    unittest.main()
