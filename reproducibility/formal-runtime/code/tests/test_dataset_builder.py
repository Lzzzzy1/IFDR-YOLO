import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from ifdr_yolo.data.build_dataset import build_dataset
from ifdr_yolo.data.splits import sha256_file


def write_kitti_label(
    path: Path,
    *,
    kind: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> None:
    path.write_text(
        f"{kind} 0 0 0 {x1} {y1} {x2} {y2} "
        "1 1 1 0 0 10 0\n",
        encoding="utf-8",
    )


class DatasetBuilderTest(unittest.TestCase):
    def test_builds_two_splits_using_each_images_actual_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "source-images"
            label_dir = root / "source-labels"
            output_dir = root / "generated"
            image_dir.mkdir()
            label_dir.mkdir()
            Image.new("RGB", (1242, 375)).save(image_dir / "000001.png")
            Image.new("RGB", (1224, 370)).save(image_dir / "000002.png")
            write_kitti_label(
                label_dir / "000001.txt",
                kind="Car",
                x1=0.0,
                y1=0.0,
                x2=124.2,
                y2=37.5,
            )
            write_kitti_label(
                label_dir / "000002.txt",
                kind="Pedestrian",
                x1=0.0,
                y1=0.0,
                x2=122.4,
                y2=37.0,
            )

            result = build_dataset(
                image_dir=image_dir,
                label_dir=label_dir,
                train_ids=("000001",),
                val_ids=("000002",),
                output_dir=output_dir,
            )

            self.assertEqual(result.image_count, 2)
            self.assertEqual(result.train_count, 1)
            self.assertEqual(result.val_count, 1)
            self.assertEqual(result.invalid_box_count, 0)
            self.assertTrue((output_dir / "labels/train/000001.txt").exists())
            self.assertTrue((output_dir / "metadata/objects.jsonl").exists())
            val_row = (
                output_dir / "labels/val/000002.txt"
            ).read_text(encoding="utf-8").split()
            self.assertEqual(val_row[0], "1")
            self.assertAlmostEqual(float(val_row[3]), 0.1)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["image_count"], 2)
            image_record = json.loads(
                (output_dir / "metadata/images.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                image_record["source_label_sha256"],
                sha256_file(label_dir / "000001.txt"),
            )

    def test_rebuild_script_runs_directly_with_explicit_paths(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "source-images"
            label_dir = root / "source-labels"
            output_dir = root / "generated"
            image_dir.mkdir()
            label_dir.mkdir()
            for image_id, size in (
                ("000001", (1242, 375)),
                ("000002", (1224, 370)),
            ):
                Image.new("RGB", size).save(image_dir / f"{image_id}.png")
                write_kitti_label(
                    label_dir / f"{image_id}.txt",
                    kind="Car",
                    x1=0.0,
                    y1=0.0,
                    x2=100.0,
                    y2=100.0,
                )
            train_ids = root / "train.txt"
            val_ids = root / "val.txt"
            audit_json = root / "audit.json"
            audit_markdown = root / "audit.md"
            train_ids.write_text("000001\n", encoding="utf-8")
            val_ids.write_text("000002\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/rebuild_kitti.py",
                    "--image-dir",
                    str(image_dir),
                    "--label-dir",
                    str(label_dir),
                    "--train-ids",
                    str(train_ids),
                    "--val-ids",
                    str(val_ids),
                    "--output-dir",
                    str(output_dir),
                    "--audit-json",
                    str(audit_json),
                    "--audit-markdown",
                    str(audit_markdown),
                ],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("images=2 train=1 val=1", completed.stdout)
            self.assertTrue(audit_json.exists())
            self.assertTrue(audit_markdown.exists())


if __name__ == "__main__":
    unittest.main()
