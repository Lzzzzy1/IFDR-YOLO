import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ifdr_yolo.eval.evaluate import (
    evaluate_prediction_directory,
    write_evaluation_json,
)
from tests.test_dataset_builder import write_kitti_label


class EvaluatePipelineTest(unittest.TestCase):
    def test_perfect_car_prediction_produces_one_hundred_easy_ap40(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            prediction_dir = root / "predictions"
            image_dir.mkdir()
            label_dir.mkdir()
            prediction_dir.mkdir()
            Image.new("RGB", (100, 100)).save(image_dir / "000001.png")
            write_kitti_label(
                label_dir / "000001.txt",
                kind="Car",
                x1=10.0,
                y1=10.0,
                x2=60.0,
                y2=60.0,
            )
            (prediction_dir / "000001.txt").write_text(
                "0 0.35 0.35 0.5 0.5 0.99\n",
                encoding="utf-8",
                newline="\n",
            )
            split_path = root / "val.txt"
            split_path.write_text("000001", encoding="utf-8", newline="\n")

            result = evaluate_prediction_directory(
                prediction_dir=prediction_dir,
                label_dir=label_dir,
                image_dir=image_dir,
                split_path=split_path,
            )

            car_easy = result["classes"]["Car"]["easy"]
            self.assertEqual(car_easy["ap40"], 100.0)
            self.assertEqual(result["split_count"], 1)

    def test_writes_stable_json_with_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metrics.json"

            write_evaluation_json(path, {"z": 1, "a": 2})

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                json.dumps({"z": 1, "a": 2}, indent=2, sort_keys=True) + "\n",
            )


if __name__ == "__main__":
    unittest.main()
