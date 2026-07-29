from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ifdr_yolo.data.build_dataset import build_dataset
from ifdr_yolo.data.splits import sha256_file
from ifdr_yolo.experiments.config import (
    BaselineConfig,
    ExperimentConfig,
    PathsConfig,
    PredictionConfig,
    TrainingConfig,
)
from ifdr_yolo.experiments.provenance import (
    canonical_ids_sha256,
    classify_porcelain_status,
    collect_environment,
    collect_git_provenance,
    find_repository_root,
    verify_dataset,
    verify_file_sha256,
)
from tests.test_dataset_builder import write_kitti_label


def make_config(root: Path) -> BaselineConfig:
    return BaselineConfig(
        schema_version=1,
        experiment=ExperimentConfig(
            dataset="kitti",
            model="yolov8m",
            variant="baseline",
            seed=17,
        ),
        paths=PathsConfig(
            model=root / "yolov8m.pt",
            model_sha256="0" * 64,
            data=root / "kitti.yaml",
            generated_data=root / "generated",
            raw_images=root / "raw_images",
            raw_labels=root / "raw_labels",
            train_ids=root / "splits" / "train.txt",
            val_ids=root / "splits" / "val.txt",
        ),
        training=TrainingConfig(
            epochs=300,
            imgsz=640,
            batch=16,
            workers=8,
            device="0",
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            patience=0,
            amp=True,
            deterministic=True,
            cache=False,
        ),
        prediction=PredictionConfig(
            conf=0.001,
            iou=0.7,
            max_det=300,
            half=False,
        ),
    )


def build_minimal_dataset(root: Path) -> BaselineConfig:
    config = make_config(root)
    config.paths.raw_images.mkdir()
    config.paths.raw_labels.mkdir()
    config.paths.train_ids.parent.mkdir()
    train_ids = ("000001",)
    val_ids = ("000002",)
    for image_id in train_ids + val_ids:
        Image.new("RGB", (100, 50)).save(
            config.paths.raw_images / f"{image_id}.png"
        )
        write_kitti_label(
            config.paths.raw_labels / f"{image_id}.txt",
            kind="Car",
            x1=0.0,
            y1=0.0,
            x2=50.0,
            y2=40.0,
        )
    config.paths.train_ids.write_text("000001", encoding="utf-8", newline="\n")
    config.paths.val_ids.write_text("000002", encoding="utf-8", newline="\n")
    source = {
        "name": "test split",
        "train_count": 1,
        "train_sha256": sha256_file(config.paths.train_ids),
        "val_count": 1,
        "val_sha256": sha256_file(config.paths.val_ids),
    }
    (config.paths.train_ids.parent / "source.json").write_text(
        json.dumps(source) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    build_dataset(
        image_dir=config.paths.raw_images,
        label_dir=config.paths.raw_labels,
        train_ids=train_ids,
        val_ids=val_ids,
        output_dir=config.paths.generated_data,
    )
    config.paths.model.write_bytes(b"weights")
    return replace(
        config,
        paths=replace(
            config.paths,
            model_sha256=sha256_file(config.paths.model),
        ),
    )


class ProvenanceTest(unittest.TestCase):
    def test_finds_repository_from_nested_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            nested = root / "a" / "b"
            nested.mkdir(parents=True)

            self.assertEqual(find_repository_root(nested), root.resolve())

    def test_collects_current_git_commit_and_change_classes(self) -> None:
        root = Path(__file__).resolve().parents[1]

        result = collect_git_provenance(root)

        self.assertEqual(len(str(result["commit"])), 40)
        self.assertEqual(result["branch"], "feature/phase2a-trusted-baseline")
        self.assertIsInstance(result["tracked_changes"], list)
        self.assertIsInstance(result["untracked_files"], list)

    def test_collects_environment_without_importing_ultralytics(self) -> None:
        result = collect_environment()

        self.assertEqual(result["ultralytics"], "8.4.98")
        self.assertIn("torch", result)
        self.assertIsInstance(result["cuda_available"], bool)

    def test_canonical_id_hash_includes_one_newline_per_id(self) -> None:
        image_ids = ("000001", "000002")
        expected = sha256(b"000001\n000002\n").hexdigest()

        self.assertEqual(canonical_ids_sha256(image_ids), expected)

    def test_verify_file_sha_reports_expected_and_actual_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"weights")

            with self.assertRaisesRegex(
                ValueError,
                "model SHA256 mismatch.*expected=.*actual=",
            ):
                verify_file_sha256(path, "0" * 64, label="model")

    def test_classifies_tracked_and_untracked_porcelain_lines(self) -> None:
        tracked, untracked = classify_porcelain_status(
            (" M README.md", "?? data/local.txt", "A  tracked.py")
        )

        self.assertEqual(tracked, (" M README.md", "A  tracked.py"))
        self.assertEqual(untracked, ("?? data/local.txt",))

    def test_verifies_byte_and_canonical_split_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = build_minimal_dataset(Path(directory))

            result = verify_dataset(config, verify_all_hashes=True)

            self.assertEqual(result["image_count"], 2)
            self.assertEqual(
                result["train_file_sha256"],
                sha256_file(config.paths.train_ids),
            )
            self.assertEqual(
                result["train_ids_sha256"],
                canonical_ids_sha256(("000001",)),
            )

    def test_rejects_generated_manifest_with_wrong_canonical_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = build_minimal_dataset(Path(directory))
            manifest_path = config.paths.generated_data / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["train_split_sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            with self.assertRaisesRegex(
                ValueError,
                "generated train split SHA256 mismatch",
            ):
                verify_dataset(config, verify_all_hashes=True)


if __name__ == "__main__":
    unittest.main()
