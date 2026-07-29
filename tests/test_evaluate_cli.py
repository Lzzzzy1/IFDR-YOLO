import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from tests.test_dataset_builder import write_kitti_label


class EvaluateKittiCliTest(unittest.TestCase):
    def test_empty_predictions_produce_zero_ap_json(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_dir = root / "predictions"
            image_dir = root / "images"
            label_dir = root / "labels"
            prediction_dir.mkdir()
            image_dir.mkdir()
            label_dir.mkdir()
            Image.new("RGB", (1242, 375)).save(image_dir / "000001.png")
            write_kitti_label(
                label_dir / "000001.txt",
                kind="Pedestrian",
                x1=10.0,
                y1=10.0,
                x2=50.0,
                y2=90.0,
            )
            split_path = root / "val.txt"
            split_path.write_text("000001\n", encoding="utf-8")
            output_path = root / "metrics.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_kitti.py",
                    "--prediction-dir",
                    str(prediction_dir),
                    "--label-dir",
                    str(label_dir),
                    "--image-dir",
                    str(image_dir),
                    "--split",
                    str(split_path),
                    "--output",
                    str(output_path),
                ],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["evaluator"], "ifdr_yolo.kitti_ap40")
            self.assertEqual(
                payload["classes"]["Pedestrian"]["moderate"]["ap40"],
                0.0,
            )
            self.assertEqual(len(payload["split_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
