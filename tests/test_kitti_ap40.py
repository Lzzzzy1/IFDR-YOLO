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
        tall = ground_truth("Pedestrian", BoundingBox(0, 0, 20, 25))
        self.assertFalse(
            is_valid_ground_truth(short, "Pedestrian", Difficulty.MODERATE)
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


if __name__ == "__main__":
    unittest.main()
