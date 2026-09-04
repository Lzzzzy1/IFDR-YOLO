from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ifdr_yolo.eval.benefit_overlap import analyze_benefit_overlap


def _label(path: Path, *, x1: float = 10.0, y1: float = 10.0) -> None:
    path.write_text(
        f"Pedestrian 0.0 0 0.0 {x1} {y1} {x1 + 20.0} {y1 + 50.0} "
        "1.0 1.0 1.0 0.0 0.0 20.0 0.0\n",
        encoding="utf-8",
    )


def _prediction(path: Path, *, count: int = 0, score: float = 0.9) -> None:
    line = f"1 0.2 0.3 0.2 0.5 {score}\n"
    path.write_text(line * count, encoding="utf-8")


class BenefitOverlapTest(unittest.TestCase):
    def test_missing_prediction_txt_is_treated_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            dirs = [root / name for name in ("p2", "a", "b")]
            for path in [image_dir, label_dir, *dirs]:
                path.mkdir()
            image_id = "000001"
            Image.new("RGB", (100, 100)).save(image_dir / f"{image_id}.png")
            _label(label_dir / f"{image_id}.txt")
            result = analyze_benefit_overlap(
                image_ids=(image_id,),
                image_dir=image_dir,
                label_dir=label_dir,
                p2_dir=dirs[0],
                a_dir=dirs[1],
                b_dir=dirs[2],
                class_names=("Pedestrian",),
                bootstrap_iterations=10,
            )
            metrics = result["classes"]["Pedestrian"]
            self.assertEqual(metrics["base"]["tp"], 0)
            self.assertEqual(metrics["A"]["fp"], 0)
            self.assertEqual(metrics["B"]["fp"], 0)

    def test_rescue_overlap_and_duplicate_changes_use_gt_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            p2_dir = root / "p2"
            a_dir = root / "a"
            b_dir = root / "b"
            for path in (image_dir, label_dir, p2_dir, a_dir, b_dir):
                path.mkdir()
            ids = ("000001", "000002", "000003")
            for image_id in ids:
                Image.new("RGB", (100, 100)).save(image_dir / f"{image_id}.png")
                _label(label_dir / f"{image_id}.txt")
            # P2 owns image 1; A and B independently rescue image 2.
            _prediction(p2_dir / "000001.txt", count=1)
            _prediction(a_dir / "000001.txt", count=2)
            _prediction(a_dir / "000002.txt", count=1)
            _prediction(b_dir / "000002.txt", count=1)
            for path in (p2_dir, a_dir, b_dir):
                for image_id in ids:
                    target = path / f"{image_id}.txt"
                    if not target.exists():
                        _prediction(target)

            result = analyze_benefit_overlap(
                image_ids=ids,
                image_dir=image_dir,
                label_dir=label_dir,
                p2_dir=p2_dir,
                a_dir=a_dir,
                b_dir=b_dir,
                class_names=("Pedestrian",),
                bootstrap_iterations=100,
                bootstrap_seed=17,
            )
            metrics = result["classes"]["Pedestrian"]
            self.assertEqual(metrics["base"]["tp"], 1)
            self.assertEqual(metrics["A"]["tp"], 2)
            self.assertEqual(metrics["B"]["tp"], 1)
            self.assertEqual(metrics["rescue"]["A_only"], 0)
            self.assertEqual(metrics["rescue"]["B_only"], 0)
            self.assertEqual(metrics["rescue"]["overlap"], 1)
            self.assertEqual(metrics["rescue"]["jaccard"], 1.0)
            self.assertEqual(metrics["A"]["duplicates"], 1)
            self.assertEqual(metrics["changes"]["A"]["duplicate_delta"], 1)
            self.assertIn("AP cannot be computed by summing object counts", result["ap_note"])

    def test_interrupted_journal_resumes_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            dirs = [root / name for name in ("p2", "a", "b")]
            for path in [image_dir, label_dir, *dirs]:
                path.mkdir()
            ids = ("000001", "000002")
            for image_id in ids:
                Image.new("RGB", (100, 100)).save(image_dir / f"{image_id}.png")
                _label(label_dir / f"{image_id}.txt")
                for path in dirs:
                    _prediction(path / f"{image_id}.txt", count=1)
            journal = root / "journal.jsonl"
            kwargs = dict(
                image_ids=ids,
                image_dir=image_dir,
                label_dir=label_dir,
                p2_dir=dirs[0],
                a_dir=dirs[1],
                b_dir=dirs[2],
                class_names=("Pedestrian",),
                journal_path=journal,
                bootstrap_iterations=25,
                bootstrap_seed=9,
            )
            with self.assertRaises(InterruptedError):
                analyze_benefit_overlap(**kwargs, max_images=1)
            first = analyze_benefit_overlap(**kwargs)
            second = analyze_benefit_overlap(**kwargs)
            self.assertEqual(first, second)
            records = [
                json.loads(line)
                for line in journal.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(sum(record.get("kind") == "image" for record in records), 2)


if __name__ == "__main__":
    unittest.main()
