from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from ifdr_yolo.data.build_dataset import build_dataset
from ifdr_yolo.data.phase1_audit import audit_generated_dataset
from tests.test_dataset_builder import write_kitti_label


class Phase1AuditTest(unittest.TestCase):
    def test_accepts_consistent_generated_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            generated_dir = root / "generated"
            image_dir.mkdir()
            label_dir.mkdir()
            for image_id in ("000001", "000002"):
                Image.new("RGB", (100, 50)).save(
                    image_dir / f"{image_id}.png"
                )
                write_kitti_label(
                    label_dir / f"{image_id}.txt",
                    kind="Car",
                    x1=0.0,
                    y1=0.0,
                    x2=50.0,
                    y2=40.0,
                )
            train_ids = ("000001",)
            val_ids = ("000002",)
            build_dataset(
                image_dir=image_dir,
                label_dir=label_dir,
                train_ids=train_ids,
                val_ids=val_ids,
                output_dir=generated_dir,
            )

            result = audit_generated_dataset(
                source_image_dir=image_dir,
                source_label_dir=label_dir,
                train_ids=train_ids,
                val_ids=val_ids,
                generated_dir=generated_dir,
                verify_all_source_hashes=True,
            )

            self.assertEqual(result.image_count, 2)
            self.assertEqual(result.label_count, 2)
            self.assertEqual(result.yolo_row_count, 2)
            self.assertEqual(result.verified_source_hash_count, 2)

    def test_audit_script_has_direct_cli(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "scripts/audit_phase1.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--generated-dir", completed.stdout)


if __name__ == "__main__":
    unittest.main()
