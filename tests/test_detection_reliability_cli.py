import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


class DetectionReliabilityCliTest(unittest.TestCase):
    def test_writes_disjoint_reliability_report_for_named_run(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            prediction_dir = root / "predictions"
            output_dir = root / "output"
            image_dir.mkdir()
            label_dir.mkdir()
            prediction_dir.mkdir()
            image_ids = tuple(f"{index:06d}" for index in range(4))
            for image_id in image_ids:
                Image.new("RGB", (100, 100)).save(image_dir / f"{image_id}.png")
                (label_dir / f"{image_id}.txt").write_text(
                    "Car 0 0 0 10 10 40 60 1 1 1 0 0 15 0\n"
                    "Pedestrian 0 2 0 50 10 70 45 1 1 1 0 0 45 0\n"
                    "Cyclist 0 2 0 20 55 45 95 1 1 1 0 0 45 0\n",
                    encoding="utf-8",
                )
                (prediction_dir / f"{image_id}.txt").write_text(
                    "0 0.25 0.35 0.30 0.50 0.9\n"
                    "1 0.60 0.275 0.20 0.35 0.8\n"
                    "2 0.325 0.75 0.25 0.40 0.7\n",
                    encoding="utf-8",
                )
            split_path = root / "val.txt"
            split_path.write_text(
                "\n".join(image_ids) + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_detection_reliability.py",
                    "--run",
                    f"p2_s17={prediction_dir}",
                    "--label-dir",
                    str(label_dir),
                    "--image-dir",
                    str(image_dir),
                    "--split",
                    str(split_path),
                    "--output-dir",
                    str(output_dir),
                    "--split-seed",
                    "20260803",
                    "--bins",
                    "5",
                ],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            json_path = output_dir / "detection_reliability.json"
            csv_path = output_dir / "detection_reliability.csv"
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            report = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(report["protocol"]["calibration_count"], 2)
        self.assertEqual(report["protocol"]["test_count"], 2)
        self.assertEqual(
            set(report["target_definitions"]),
            {"overall", "small_25_40", "far_gt_40m", "occlusion_2"},
        )
        self.assertEqual(
            set(report["runs"]["p2_s17"]["classes"]),
            {"Car", "Pedestrian", "Cyclist"},
        )
        self.assertIn("reliability_json=", completed.stdout)
        self.assertIn("reliability_csv=", completed.stdout)


if __name__ == "__main__":
    unittest.main()
