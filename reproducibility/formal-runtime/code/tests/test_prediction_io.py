from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.eval.prediction_io import (
    load_kitti_ground_truth,
    load_yolo_predictions,
)
from tests.test_dataset_builder import write_kitti_label


class PredictionIOTest(unittest.TestCase):
    def test_loads_normalized_prediction_with_actual_image_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction_dir = Path(directory)
            (prediction_dir / "000001.txt").write_text(
                "1 0.5 0.5 0.1 0.2 0.9\n",
                encoding="utf-8",
            )

            predictions = load_yolo_predictions(
                prediction_dir,
                {"000001": (1224, 370)},
            )

            detection = predictions["000001"][0]
            self.assertEqual(detection.kind, "Pedestrian")
            self.assertAlmostEqual(detection.score, 0.9)
            self.assertAlmostEqual(detection.bbox.x1, 550.8)
            self.assertAlmostEqual(detection.bbox.y1, 148.0)
            self.assertAlmostEqual(detection.bbox.x2, 673.2)
            self.assertAlmostEqual(detection.bbox.y2, 222.0)

    def test_rejects_prediction_without_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction_dir = Path(directory)
            (prediction_dir / "000001.txt").write_text(
                "1 0.5 0.5 0.1 0.2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "6 fields"):
                load_yolo_predictions(
                    prediction_dir,
                    {"000001": (1242, 375)},
                )

    def test_rejects_unknown_class_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction_dir = Path(directory)
            (prediction_dir / "000001.txt").write_text(
                "9 0.5 0.5 0.1 0.2 0.9\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "class ID"):
                load_yolo_predictions(
                    prediction_dir,
                    {"000001": (1242, 375)},
                )

    def test_clips_prediction_to_image_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction_dir = Path(directory)
            (prediction_dir / "000001.txt").write_text(
                "0 0.0 0.5 0.4 0.4 0.9\n",
                encoding="utf-8",
            )
            detection = load_yolo_predictions(
                prediction_dir,
                {"000001": (100, 50)},
            )["000001"][0]
            self.assertEqual(detection.bbox.x1, 0.0)
            self.assertEqual(detection.bbox.x2, 20.0)

    def test_missing_prediction_file_yields_empty_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = load_yolo_predictions(
                Path(directory),
                {"000001": (1242, 375)},
            )
            self.assertEqual(predictions, {"000001": ()})

    def test_loads_ground_truth_for_requested_ids_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_dir = Path(directory)
            write_kitti_label(
                label_dir / "000001.txt",
                kind="Cyclist",
                x1=1.0,
                y1=2.0,
                x2=31.0,
                y2=82.0,
            )
            write_kitti_label(
                label_dir / "000002.txt",
                kind="Car",
                x1=1.0,
                y1=2.0,
                x2=31.0,
                y2=82.0,
            )

            ground_truth = load_kitti_ground_truth(label_dir, ("000001",))

            self.assertEqual(tuple(ground_truth), ("000001",))
            self.assertEqual(ground_truth["000001"][0].kind, "Cyclist")


if __name__ == "__main__":
    unittest.main()
