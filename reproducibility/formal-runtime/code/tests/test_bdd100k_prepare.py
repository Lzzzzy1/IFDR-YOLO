import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ifdr_yolo.data.bdd100k_prepare import (
    build_bdd100k_dataset,
    build_bdd100k_split,
)
from scripts.prepare_bdd100k import main


class Bdd100kPrepareTest(unittest.TestCase):
    def test_cli_builds_train_val_dataset_from_explicit_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            for name in ("train.jpg", "val.jpg"):
                Image.new("RGB", (16, 16)).save(image_dir / name)

            def write(path: Path, name: str) -> None:
                path.write_text(
                    json.dumps(
                        [
                            {
                                "name": name,
                                "attributes": {
                                    "weather": "clear",
                                    "scene": "city street",
                                    "timeofday": "daytime",
                                },
                                "labels": [],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

            train_annotations = root / "train.json"
            val_annotations = root / "val.json"
            write(train_annotations, "train.jpg")
            write(val_annotations, "val.jpg")
            output_dir = root / "prepared"

            self.assertEqual(
                main(
                    [
                        "--train-annotations",
                        str(train_annotations),
                        "--val-annotations",
                        str(val_annotations),
                        "--images",
                        str(image_dir),
                        "--output",
                        str(output_dir),
                    ]
                ),
                0,
            )
            self.assertTrue((output_dir / "dataset.yaml").exists())

    def test_builds_disjoint_train_val_tree_with_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            for name in ("train.jpg", "val.jpg"):
                Image.new("RGB", (32, 24), color="black").save(image_dir / name)

            def write_annotations(path: Path, name: str) -> None:
                path.write_text(
                    json.dumps(
                        [
                            {
                                "name": name,
                                "attributes": {
                                    "weather": "clear",
                                    "scene": "city street",
                                    "timeofday": "daytime",
                                },
                                "labels": [],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

            train_annotations = root / "train.json"
            val_annotations = root / "val.json"
            write_annotations(train_annotations, "train.jpg")
            write_annotations(val_annotations, "val.jpg")

            output_dir = root / "prepared"
            summary = build_bdd100k_dataset(
                train_annotations_path=train_annotations,
                val_annotations_path=val_annotations,
                image_dir=image_dir,
                output_dir=output_dir,
            )

            self.assertEqual(summary.train.image_count, 1)
            self.assertEqual(summary.val.image_count, 1)
            self.assertTrue((output_dir / "images/train/train.jpg").exists())
            self.assertTrue((output_dir / "images/val/val.jpg").exists())
            self.assertIn("names:", (output_dir / "dataset.yaml").read_text())
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(manifest["splits"]), {"train", "val"})

    def test_builds_yolo_labels_and_auditable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            image_path = image_dir / "frame-1.jpg"
            Image.new("RGB", (100, 80), color="black").save(image_path)
            annotations = root / "labels.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "name": "frame-1.jpg",
                            "attributes": {
                                "weather": "clear",
                                "scene": "city street",
                                "timeofday": "daytime",
                            },
                            "labels": [
                                {
                                    "category": "pedestrian",
                                    "attributes": {
                                        "occluded": True,
                                        "truncated": False,
                                    },
                                    "box2d": {
                                        "x1": 10,
                                        "y1": 20,
                                        "x2": 30,
                                        "y2": 60,
                                    },
                                },
                                {
                                    "category": "bus",
                                    "attributes": {},
                                    "box2d": {
                                        "x1": 1,
                                        "y1": 1,
                                        "x2": 20,
                                        "y2": 20,
                                    },
                                },
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_dir = root / "prepared"

            summary = build_bdd100k_split(
                annotations_path=annotations,
                image_dir=image_dir,
                output_dir=output_dir,
                split="train",
            )

            self.assertEqual(summary.image_count, 1)
            self.assertEqual(summary.object_count, 1)
            self.assertEqual(summary.ignored_category_counts, {"bus": 1})
            self.assertTrue((output_dir / "images/train/frame-1.jpg").exists())
            label = (output_dir / "labels/train/frame-1.txt").read_text(
                encoding="utf-8"
            )
            self.assertEqual(label, "1 0.20000000 0.50000000 0.20000000 0.50000000\n")
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["split"], "train")
            self.assertEqual(manifest["image_count"], 1)
            self.assertEqual(manifest["class_counts"], {"pedestrian": 1})
            self.assertEqual(
                len(
                    (output_dir / "metadata/images_train.jsonl")
                    .read_text()
                    .splitlines()
                ),
                1,
            )

    def test_rejects_duplicate_frame_names_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (10, 10)).save(image_dir / "same.jpg")
            annotations = root / "labels.json"
            base = {
                "name": "same.jpg",
                "attributes": {
                    "weather": "clear",
                    "scene": "city street",
                    "timeofday": "daytime",
                },
                "labels": [],
            }
            annotations.write_text(json.dumps([base, base]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate BDD100K frame"):
                build_bdd100k_split(
                    annotations_path=annotations,
                    image_dir=image_dir,
                    output_dir=root / "prepared",
                    split="val",
                )

    def test_rejects_missing_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            annotations = root / "labels.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "name": "missing.jpg",
                            "attributes": {
                                "weather": "clear",
                                "scene": "city street",
                                "timeofday": "daytime",
                            },
                            "labels": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "missing BDD100K image"):
                build_bdd100k_split(
                    annotations_path=annotations,
                    image_dir=image_dir,
                    output_dir=root / "prepared",
                    split="val",
                )


if __name__ == "__main__":
    unittest.main()
