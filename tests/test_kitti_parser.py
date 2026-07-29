from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.data.kitti_parser import parse_kitti_file, parse_kitti_line


class KittiParserTest(unittest.TestCase):
    def test_parses_ground_truth_line(self) -> None:
        line = (
            "Pedestrian 0.25 1 -0.20 10.0 20.0 30.0 80.0 "
            "1.70 0.60 0.80 1.0 2.0 15.0 0.10"
        )
        obj = parse_kitti_line(line)
        self.assertEqual(obj.kind, "Pedestrian")
        self.assertEqual(obj.occluded, 1)
        self.assertEqual(obj.bbox.as_xyxy(), (10.0, 20.0, 30.0, 80.0))
        self.assertEqual(obj.dimensions_hwl, (1.70, 0.60, 0.80))
        self.assertIsNone(obj.score)

    def test_parses_detection_score(self) -> None:
        line = "Car 0 0 0 1 2 11 22 1 2 3 4 5 6 0.5 0.91"
        self.assertAlmostEqual(parse_kitti_line(line).score or 0.0, 0.91)

    def test_rejects_wrong_field_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "15 or 16"):
            parse_kitti_line("Car 0 0")

    def test_rejects_non_finite_truncation(self) -> None:
        line = "Car nan 0 0 1 2 11 22 1 2 3 4 5 6 0.5"
        with self.assertRaisesRegex(ValueError, "truncated.*finite"):
            parse_kitti_line(line)

    def test_rejects_inverted_box(self) -> None:
        line = "Car 0 0 0 20 10 5 30 1 1 1 0 0 0 0"
        with self.assertRaisesRegex(ValueError, "invalid bounding box"):
            parse_kitti_line(line)

    def test_parse_file_returns_empty_tuple_for_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text("", encoding="utf-8")
            self.assertEqual(parse_kitti_file(path), ())

    def test_parse_file_error_contains_file_and_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text(
                "Car 0 0 0 1 2 11 22 1 2 3 4 5 6 0\nCar 0 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"bad\.txt:2"):
                parse_kitti_file(path)


if __name__ == "__main__":
    unittest.main()
