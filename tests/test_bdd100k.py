import unittest

from ifdr_yolo.data.bdd100k import (
    BDD100K_CLASS_TO_ID,
    parse_bdd100k_frame,
)


class Bdd100kParserTest(unittest.TestCase):
    def test_parses_environment_attributes_and_detection_rows(self) -> None:
        frame = {
            "name": "sample.jpg",
            "attributes": {
                "weather": "rainy",
                "scene": "city street",
                "timeofday": "night",
            },
            "labels": [
                {
                    "id": "1",
                    "category": "pedestrian",
                    "attributes": {"occluded": True, "truncated": False},
                    "box2d": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0},
                },
                {
                    "id": "2",
                    "category": "car",
                    "attributes": {"occluded": False, "truncated": True},
                    "box2d": {"x1": -10.0, "y1": 700.0, "x2": 1300.0, "y2": 740.0},
                },
            ],
        }

        parsed = parse_bdd100k_frame(frame, image_width=1280, image_height=720)

        self.assertEqual(parsed.name, "sample.jpg")
        self.assertEqual(parsed.weather, "rainy")
        self.assertEqual(parsed.scene, "city street")
        self.assertEqual(parsed.timeofday, "night")
        self.assertEqual(len(parsed.objects), 2)
        pedestrian, car = parsed.objects
        self.assertEqual(pedestrian.class_id, BDD100K_CLASS_TO_ID["pedestrian"])
        self.assertEqual(pedestrian.size_bin, "small")
        self.assertTrue(pedestrian.occluded)
        self.assertEqual(car.xyxy, (0.0, 700.0, 1280.0, 720.0))
        self.assertTrue(car.truncated)
        self.assertEqual(car.yolo_row.class_id, BDD100K_CLASS_TO_ID["car"])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in car.yolo_row.as_tuple()[1:]))

    def test_rejects_unknown_detection_category(self) -> None:
        frame = {
            "name": "sample.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "highway",
                "timeofday": "daytime",
            },
            "labels": [
                {
                    "id": "x",
                    "category": "spaceship",
                    "attributes": {},
                    "box2d": {"x1": 1, "y1": 1, "x2": 2, "y2": 2},
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "unknown BDD100K category"):
            parse_bdd100k_frame(frame, image_width=1280, image_height=720)

    def test_rejects_box_without_positive_clipped_area(self) -> None:
        frame = {
            "name": "sample.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "highway",
                "timeofday": "daytime",
            },
            "labels": [
                {
                    "id": "x",
                    "category": "car",
                    "attributes": {},
                    "box2d": {"x1": 1300, "y1": 1, "x2": 1400, "y2": 2},
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "positive clipped area"):
            parse_bdd100k_frame(frame, image_width=1280, image_height=720)


if __name__ == "__main__":
    unittest.main()
