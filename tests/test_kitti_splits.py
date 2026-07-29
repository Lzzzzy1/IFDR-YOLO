import json
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.data.splits import (
    discover_ids,
    load_ids,
    sha256_file,
    validate_split,
)


class KittiSplitTest(unittest.TestCase):
    def test_rejects_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split(("000001",), ("000001",), {"000001"})

    def test_rejects_missing_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_split(("000001",), (), {"000001", "000002"})

    def test_load_ids_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.txt"
            path.write_text("000001\n000001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_ids(path)

    def test_load_ids_rejects_invalid_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.txt"
            path.write_text("12\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid KITTI image ID"):
                load_ids(path)

    def test_sha256_file_matches_known_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223"
                "b00361a396177a9cb410ff61f20015ad",
            )

    def test_repository_split_counts_and_hashes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        split_dir = root / "configs" / "splits"
        train_path = split_dir / "kitti_train.txt"
        val_path = split_dir / "kitti_val.txt"
        train_ids = load_ids(train_path)
        val_ids = load_ids(val_path)
        source = json.loads(
            (split_dir / "source.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(train_ids), 3712)
        self.assertEqual(len(val_ids), 3769)
        self.assertEqual(source["train_sha256"], sha256_file(train_path))
        self.assertEqual(source["val_sha256"], sha256_file(val_path))

    def test_discovers_matching_image_and_label_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            (image_dir / "000001.png").touch()
            (label_dir / "000001.txt").touch()
            self.assertEqual(discover_ids(image_dir, label_dir), {"000001"})

    def test_discovery_rejects_image_label_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            (image_dir / "000001.png").touch()
            with self.assertRaisesRegex(ValueError, "image/label"):
                discover_ids(image_dir, label_dir)


if __name__ == "__main__":
    unittest.main()
