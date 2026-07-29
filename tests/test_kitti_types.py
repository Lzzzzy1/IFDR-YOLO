import unittest

from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    Difficulty,
    EVAL_CLASSES,
    TRAIN_CLASS_TO_ID,
)


class KittiTypesTest(unittest.TestCase):
    def test_bounding_box_geometry(self) -> None:
        box = BoundingBox(10.0, 20.0, 35.0, 60.0)
        self.assertEqual(box.width, 25.0)
        self.assertEqual(box.height, 40.0)
        self.assertEqual(box.area, 1000.0)

    def test_bounding_box_rejects_non_finite_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            BoundingBox(0.0, 0.0, float("nan"), 10.0)

    def test_core_evaluation_types_and_classes_are_stable(self) -> None:
        box = BoundingBox(0.0, 0.0, 10.0, 20.0)
        detection = Detection("000001", "Pedestrian", 0.75, box)
        self.assertEqual(detection.image_id, "000001")
        self.assertEqual(Difficulty.MODERATE.value, "moderate")
        self.assertEqual(EVAL_CLASSES, ("Car", "Pedestrian", "Cyclist"))
        self.assertEqual(
            TRAIN_CLASS_TO_ID,
            {"Car": 0, "Pedestrian": 1, "Cyclist": 2},
        )


if __name__ == "__main__":
    unittest.main()
