import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


class PairedBootstrapCliTest(unittest.TestCase):
    def test_writes_one_reproducible_slice_comparison(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            reference_dir = root / "reference"
            candidate_dir = root / "candidate"
            for path in (
                image_dir,
                label_dir,
                reference_dir,
                candidate_dir,
            ):
                path.mkdir()
            Image.new("RGB", (100, 100)).save(image_dir / "000001.png")
            (label_dir / "000001.txt").write_text(
                "Pedestrian 0 0 0 10 10 30 40 1 1 1 0 0 45 0\n",
                encoding="utf-8",
            )
            (reference_dir / "000001.txt").write_text("", encoding="utf-8")
            (candidate_dir / "000001.txt").write_text(
                "1 0.2 0.25 0.2 0.3 0.99\n",
                encoding="utf-8",
            )
            split_path = root / "val.txt"
            split_path.write_text("000001\n", encoding="utf-8")
            output_path = root / "bootstrap.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_paired_bootstrap.py",
                    "--reference-dir",
                    str(reference_dir),
                    "--candidate-dir",
                    str(candidate_dir),
                    "--reference-name",
                    "baseline_s17",
                    "--candidate-name",
                    "p2_s17",
                    "--class-name",
                    "Pedestrian",
                    "--slice",
                    "far_gt_40m",
                    "--iterations",
                    "10",
                    "--seed",
                    "17",
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
            self.assertEqual(payload["comparison"]["difference_ap40"], 100.0)
            self.assertEqual(payload["comparison"]["iterations"], 10)
            self.assertEqual(payload["target_slice"]["name"], "far_gt_40m")


if __name__ == "__main__":
    unittest.main()
