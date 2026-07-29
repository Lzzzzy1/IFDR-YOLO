import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.import_kitti_split import import_split_files, main


class ImportKittiSplitTest(unittest.TestCase):
    def test_imports_validated_split_and_records_actual_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_source = root / "train-source.txt"
            val_source = root / "val-source.txt"
            output_dir = root / "output"
            train_source.write_text("000001\n", encoding="utf-8")
            val_source.write_text("000002\n", encoding="utf-8")

            summary = import_split_files(
                train_source=train_source,
                val_source=val_source,
                output_dir=output_dir,
                available_ids={"000001", "000002"},
                expected_train_count=1,
                expected_val_count=1,
                train_url="https://example.test/train.txt",
                val_url="https://example.test/val.txt",
            )

            source = json.loads(
                (output_dir / "source.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary.train_count, 1)
            self.assertEqual(summary.val_count, 1)
            self.assertEqual(
                source["train_sha256"],
                summary.train_sha256,
            )
            self.assertEqual(
                source["val_sha256"],
                summary.val_sha256,
            )
            self.assertEqual(source["train_url"], "https://example.test/train.txt")
            self.assertEqual(
                (output_dir / "kitti_train.txt").read_text(encoding="utf-8"),
                "000001\n",
            )

    def test_cli_imports_split_from_explicit_dataset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            output_dir = root / "splits"
            image_dir.mkdir()
            label_dir.mkdir()
            for image_id in ("000001", "000002"):
                (image_dir / f"{image_id}.png").touch()
                (label_dir / f"{image_id}.txt").touch()
            train_source = root / "train.txt"
            val_source = root / "val.txt"
            train_source.write_text("000001\n", encoding="utf-8")
            val_source.write_text("000002\n", encoding="utf-8")

            exit_code = main(
                [
                    "--train-source",
                    str(train_source),
                    "--val-source",
                    str(val_source),
                    "--image-dir",
                    str(image_dir),
                    "--label-dir",
                    str(label_dir),
                    "--output-dir",
                    str(output_dir),
                    "--expected-train-count",
                    "1",
                    "--expected-val-count",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "source.json").exists())

    def test_script_can_run_directly_from_repository_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            output_dir = root / "splits"
            image_dir.mkdir()
            label_dir.mkdir()
            for image_id in ("000001", "000002"):
                (image_dir / f"{image_id}.png").touch()
                (label_dir / f"{image_id}.txt").touch()
            train_source = root / "train.txt"
            val_source = root / "val.txt"
            train_source.write_text("000001\n", encoding="utf-8")
            val_source.write_text("000002\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/import_kitti_split.py",
                    "--train-source",
                    str(train_source),
                    "--val-source",
                    str(val_source),
                    "--image-dir",
                    str(image_dir),
                    "--label-dir",
                    str(label_dir),
                    "--output-dir",
                    str(output_dir),
                    "--expected-train-count",
                    "1",
                    "--expected-val-count",
                    "1",
                ],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
