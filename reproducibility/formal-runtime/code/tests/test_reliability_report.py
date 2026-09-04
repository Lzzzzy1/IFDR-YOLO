import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from ifdr_yolo.eval.reliability_report import (
    evaluate_reliability_runs,
    write_reliability_report,
)


class ReliabilityReportTest(unittest.TestCase):
    def _dataset(self, root: Path) -> tuple[Path, Path, Path, Path]:
        image_dir = root / "images"
        label_dir = root / "labels"
        prediction_dir = root / "predictions"
        image_dir.mkdir()
        label_dir.mkdir()
        prediction_dir.mkdir()
        image_ids = tuple(f"{index:06d}" for index in range(4))
        for image_id in image_ids:
            Image.new("RGB", (100, 100)).save(
                image_dir / f"{image_id}.png"
            )
            (label_dir / f"{image_id}.txt").write_text(
                "Pedestrian 0 0 0 10 10 30 50 1 1 1 0 0 45 0\n",
                encoding="utf-8",
            )
            (prediction_dir / f"{image_id}.txt").write_text(
                "1 0.2 0.3 0.2 0.4 0.8\n",
                encoding="utf-8",
            )
        split_path = root / "val.txt"
        split_path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
        return image_dir, label_dir, prediction_dir, split_path

    def test_selects_threshold_on_disjoint_half_and_evaluates_other_half(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir, label_dir, prediction_dir, split_path = self._dataset(root)

            report = evaluate_reliability_runs(
                run_prediction_dirs={"p2_s17": prediction_dir},
                label_dir=label_dir,
                image_dir=image_dir,
                split_path=split_path,
                split_seed=20260803,
                class_names=("Pedestrian",),
                target_slices=(),
                bins=5,
            )

        protocol = report["protocol"]
        metrics = report["runs"]["p2_s17"]["classes"]["Pedestrian"]
        self.assertEqual(protocol["calibration_count"], 2)
        self.assertEqual(protocol["test_count"], 2)
        self.assertNotEqual(
            protocol["calibration_ids_sha256"],
            protocol["test_ids_sha256"],
        )
        self.assertEqual(metrics["confidence_threshold"], 0.8)
        self.assertAlmostEqual(metrics["targets"]["overall"]["laece0"], 0.2)
        self.assertEqual(metrics["targets"]["overall"]["lrp"], 0.0)

    def test_writes_atomic_json_and_flat_csv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir, label_dir, prediction_dir, split_path = self._dataset(root)
            report = evaluate_reliability_runs(
                run_prediction_dirs={"p2_s17": prediction_dir},
                label_dir=label_dir,
                image_dir=image_dir,
                split_path=split_path,
                split_seed=20260803,
                class_names=("Pedestrian",),
                target_slices=(),
                bins=5,
            )

            json_path, csv_path = write_reliability_report(
                root / "output",
                report,
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            csv_text = csv_path.read_text(encoding="utf-8")

        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("run,class,target,confidence_threshold,laece0,lrp", csv_text)
        self.assertIn("p2_s17,Pedestrian,overall,0.8", csv_text)


if __name__ == "__main__":
    unittest.main()
