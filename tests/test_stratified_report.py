import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from ifdr_yolo.eval.stratified_report import (
    evaluate_stratified_runs,
    write_stratified_report,
)


class StratifiedReportTest(unittest.TestCase):
    def _dataset(self, root: Path) -> tuple[Path, Path, Path, Path]:
        image_dir = root / "images"
        label_dir = root / "labels"
        prediction_dir = root / "predictions"
        image_dir.mkdir()
        label_dir.mkdir()
        prediction_dir.mkdir()
        Image.new("RGB", (100, 100)).save(image_dir / "000001.png")
        (label_dir / "000001.txt").write_text(
            "Pedestrian 0 0 0 10 10 30 40 1 1 1 0 0 45 0\n",
            encoding="utf-8",
        )
        (prediction_dir / "000001.txt").write_text(
            "1 0.2 0.25 0.2 0.3 0.99\n",
            encoding="utf-8",
        )
        split_path = root / "val.txt"
        split_path.write_text("000001\n", encoding="utf-8")
        return image_dir, label_dir, prediction_dir, split_path

    def test_evaluates_small_and_far_slices_from_one_loaded_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir, label_dir, prediction_dir, split_path = self._dataset(root)

            report = evaluate_stratified_runs(
                run_prediction_dirs={"p2_s17": prediction_dir},
                label_dir=label_dir,
                image_dir=image_dir,
                split_path=split_path,
            )

            run = report["runs"]["p2_s17"]
            small = run["slices"]["height"]["small_25_40"]["classes"]
            far = run["slices"]["depth"]["far_gt_40m"]["classes"]
            self.assertEqual(report["split_count"], 1)
            self.assertEqual(small["Pedestrian"]["ap40"], 100.0)
            self.assertEqual(far["Pedestrian"]["ap40"], 100.0)

    def test_rejects_incomplete_prediction_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir, label_dir, prediction_dir, split_path = self._dataset(root)
            (prediction_dir / "000001.txt").unlink()

            with self.assertRaisesRegex(ValueError, "prediction IDs"):
                evaluate_stratified_runs(
                    run_prediction_dirs={"p2_s17": prediction_dir},
                    label_dir=label_dir,
                    image_dir=image_dir,
                    split_path=split_path,
                )

    def test_writes_json_and_flat_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir, label_dir, prediction_dir, split_path = self._dataset(root)
            report = evaluate_stratified_runs(
                run_prediction_dirs={"p2_s17": prediction_dir},
                label_dir=label_dir,
                image_dir=image_dir,
                split_path=split_path,
            )

            json_path, csv_path = write_stratified_report(
                root / "output",
                report,
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["split_count"], 1)
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("run,axis,slice,class,ap40,num_valid_gt", csv_text)
            self.assertIn("p2_s17,depth,far_gt_40m,Pedestrian,100.0,1", csv_text)

    def test_cli_accepts_named_run_and_writes_both_outputs(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir, label_dir, prediction_dir, split_path = self._dataset(root)
            output_dir = root / "output"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_stratified.py",
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
                ],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output_dir / "stratified_ap40.json").is_file())
            self.assertTrue((output_dir / "stratified_ap40.csv").is_file())


if __name__ == "__main__":
    unittest.main()
