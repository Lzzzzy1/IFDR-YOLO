import unittest

from ifdr_yolo.data.kitti_types import BoundingBox, KittiObject
from ifdr_yolo.data.yolo_export import object_to_yolo


def make_object(kind: str, bbox: BoundingBox) -> KittiObject:
    return KittiObject(
        kind=kind,
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox=bbox,
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 0.0),
        rotation_y=0.0,
    )


class YoloExportTest(unittest.TestCase):
    def test_uses_actual_image_size(self) -> None:
        obj = make_object("Pedestrian", BoundingBox(0.0, 0.0, 122.4, 37.0))
        row = object_to_yolo(obj, image_width=1224, image_height=370)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.class_id, 1)
        self.assertAlmostEqual(row.x_center, 0.05)
        self.assertAlmostEqual(row.y_center, 0.05)
        self.assertAlmostEqual(row.width, 0.10)
        self.assertAlmostEqual(row.height, 0.10)

    def test_non_training_class_is_not_exported(self) -> None:
        obj = make_object("Van", BoundingBox(0.0, 0.0, 100.0, 100.0))
        self.assertIsNone(object_to_yolo(obj, 1242, 375))

    def test_clips_box_to_image(self) -> None:
        obj = make_object("Car", BoundingBox(-10.0, 10.0, 1250.0, 400.0))
        row = object_to_yolo(obj, 1242, 375)
        self.assertIsNotNone(row)
        assert row is not None
        for value in row.as_tuple()[1:]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_drops_zero_area_after_clipping(self) -> None:
        obj = make_object("Cyclist", BoundingBox(1300.0, 10.0, 1400.0, 20.0))
        self.assertIsNone(object_to_yolo(obj, 1242, 375))

    def test_serializes_with_stable_precision(self) -> None:
        obj = make_object("Car", BoundingBox(0.0, 0.0, 100.0, 100.0))
        row = object_to_yolo(obj, 1000, 500)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(
            row.serialize(),
            "0 0.05000000 0.10000000 0.10000000 0.20000000",
        )

    def test_rejects_non_positive_image_dimensions(self) -> None:
        obj = make_object("Car", BoundingBox(0.0, 0.0, 10.0, 10.0))
        with self.assertRaisesRegex(ValueError, "image dimensions"):
            object_to_yolo(obj, 0, 375)


if __name__ == "__main__":
    unittest.main()
