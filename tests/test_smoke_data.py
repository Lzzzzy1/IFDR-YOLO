import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from ifdr_yolo.experiments.smoke_data import (
    build_smoke_view,
    select_smoke_ids,
)


class SmokeDataTest(unittest.TestCase):
    def test_selects_first_sixteen_ids_without_overlap(self) -> None:
        train = tuple(f"{value:06d}" for value in range(20))
        val = tuple(f"{value:06d}" for value in range(100, 120))

        selection = select_smoke_ids(train, val, count=16)

        self.assertEqual(selection.train_ids, train[:16])
        self.assertEqual(selection.val_ids, val[:16])

    def test_rejects_insufficient_or_overlapping_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            select_smoke_ids(("000001",), ("000002", "000003"), count=2)
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_smoke_ids(
                ("000001", "000002"),
                ("000002", "000003"),
                count=2,
            )

    def test_builds_absolute_lists_yaml_and_traceable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated"
            for split, image_ids in (
                ("train", ("000001", "000002")),
                ("val", ("000101", "000102")),
            ):
                image_dir = generated / "images" / split
                image_dir.mkdir(parents=True)
                for image_id in image_ids:
                    Image.new("RGB", (20, 10)).save(
                        image_dir / f"{image_id}.png"
                    )

            view = build_smoke_view(
                output_dir=root / "smoke",
                generated_dir=generated,
                train_ids=("000001", "000002"),
                val_ids=("000101", "000102"),
                train_source_sha256="a" * 64,
                val_source_sha256="b" * 64,
                count=2,
            )

            train_paths = (view.root / "train.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(
                train_paths,
                [
                    str((generated / "images/train/000001.png").resolve()),
                    str((generated / "images/train/000002.png").resolve()),
                ],
            )
            data = yaml.safe_load(view.data_yaml.read_text(encoding="utf-8"))
            self.assertEqual(Path(data["train"]), view.root / "train.txt")
            self.assertEqual(Path(data["val"]), view.root / "val.txt")
            self.assertEqual(
                data["names"],
                {0: "Car", 1: "Pedestrian", 2: "Cyclist"},
            )
            manifest = json.loads(
                (view.root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["train_ids"], ["000001", "000002"])
            self.assertEqual(manifest["train_source_sha256"], "a" * 64)

    def test_identical_rebuild_is_idempotent_but_changed_view_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated"
            for split, image_ids in (
                ("train", ("000001", "000002")),
                ("val", ("000101", "000102")),
            ):
                image_dir = generated / "images" / split
                image_dir.mkdir(parents=True)
                for image_id in image_ids:
                    Image.new("RGB", (20, 10)).save(
                        image_dir / f"{image_id}.png"
                    )
            arguments = {
                "output_dir": root / "smoke",
                "generated_dir": generated,
                "train_ids": ("000001", "000002"),
                "val_ids": ("000101", "000102"),
                "train_source_sha256": "a" * 64,
                "val_source_sha256": "b" * 64,
            }

            first = build_smoke_view(**arguments, count=2)
            second = build_smoke_view(**arguments, count=2)

            self.assertEqual(first, second)
            with self.assertRaisesRegex(FileExistsError, "different content"):
                build_smoke_view(**arguments, count=1)


if __name__ == "__main__":
    unittest.main()
