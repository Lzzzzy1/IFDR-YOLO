import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ifdr_yolo.data.audit import audit_fixed_size_assumption, write_audit_reports
from tests.test_dataset_builder import write_kitti_label


class FixedSizeAuditTest(unittest.TestCase):
    def test_quantifies_only_targets_changed_by_fixed_size_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            Image.new("RGB", (1242, 375)).save(image_dir / "000001.png")
            Image.new("RGB", (1224, 370)).save(image_dir / "000002.png")
            for image_id in ("000001", "000002"):
                write_kitti_label(
                    label_dir / f"{image_id}.txt",
                    kind="Pedestrian",
                    x1=0.0,
                    y1=0.0,
                    x2=122.4,
                    y2=37.0,
                )

            audit = audit_fixed_size_assumption(
                image_dir=image_dir,
                label_dir=label_dir,
                image_ids=("000001", "000002"),
            )

            self.assertEqual(audit.image_count, 2)
            self.assertEqual(audit.target_count, 2)
            self.assertEqual(audit.affected_target_count, 1)
            self.assertGreater(audit.max_absolute_normalized_error, 0.0)
            self.assertEqual(audit.by_image_size["1242x375"]["affected_targets"], 0)
            self.assertEqual(audit.by_image_size["1224x370"]["affected_targets"], 1)

    def test_writes_machine_and_human_readable_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            Image.new("RGB", (1224, 370)).save(image_dir / "000001.png")
            write_kitti_label(
                label_dir / "000001.txt",
                kind="Car",
                x1=0.0,
                y1=0.0,
                x2=100.0,
                y2=100.0,
            )
            audit = audit_fixed_size_assumption(
                image_dir=image_dir,
                label_dir=label_dir,
                image_ids=("000001",),
            )
            json_path = root / "audit.json"
            markdown_path = root / "audit.md"

            write_audit_reports(audit, json_path, markdown_path)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["image_count"], 1)
            self.assertIn("Fixed 1242x375", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
