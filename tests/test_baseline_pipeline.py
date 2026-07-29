from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from ifdr_yolo.data.splits import sha256_file
from ifdr_yolo.experiments.baseline import (
    BaselineServices,
    ensure_prediction_files,
    run_baseline,
)
from tests.test_provenance import make_config


class FakeAdapter:
    def __init__(self, *, fail_training: bool = False) -> None:
        self.fail_training = fail_training
        self.train_calls: list[dict[str, object]] = []
        self.predict_calls: list[dict[str, object]] = []

    def runtime_info(self) -> dict[str, object]:
        return {
            "ultralytics": "8.4.98",
            "cuda_available": True,
            "cuda_device_count": 1,
        }

    def train(self, **kwargs: object) -> Path:
        self.train_calls.append(kwargs)
        if self.fail_training:
            raise RuntimeError("synthetic training failure")
        run_dir = Path(str(kwargs["run_dir"]))
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        best = weights / "best.pt"
        best.write_bytes(b"best")
        return best

    def predict(self, **kwargs: object) -> Path:
        self.predict_calls.append(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        labels = output_dir / "labels"
        labels.mkdir(parents=True)
        image_paths = kwargs["image_paths"]
        assert isinstance(image_paths, tuple)
        first_id = Path(image_paths[0]).stem
        (labels / f"{first_id}.txt").write_text(
            "0 0.5 0.5 0.2 0.2 0.9\n",
            encoding="utf-8",
            newline="\n",
        )
        return labels


def make_pipeline_config(
    root: Path,
    *,
    train_count: int = 1,
    val_count: int = 2,
):
    config = make_config(root)
    config.paths.model.write_bytes(b"model")
    config.paths.data.write_text("names: []\n", encoding="utf-8")
    config.paths.train_ids.parent.mkdir()
    train_ids = tuple(f"{value:06d}" for value in range(1, train_count + 1))
    val_ids = tuple(f"{value:06d}" for value in range(101, 101 + val_count))
    config.paths.train_ids.write_text(
        "\n".join(train_ids),
        encoding="utf-8",
        newline="\n",
    )
    config.paths.val_ids.write_text(
        "\n".join(val_ids),
        encoding="utf-8",
        newline="\n",
    )
    config.paths.raw_images.mkdir()
    config.paths.raw_labels.mkdir()
    for split, image_ids in (("train", train_ids), ("val", val_ids)):
        for image_id in image_ids:
            image = (
                config.paths.generated_data
                / "images"
                / split
                / f"{image_id}.png"
            )
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"image")
            label = (
                config.paths.generated_data
                / "labels"
                / split
                / f"{image_id}.txt"
            )
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
    source_path = root / "experiment.yaml"
    source_path.write_text("schema_version: 1\n", encoding="utf-8")
    return replace(
        config,
        paths=replace(
            config.paths,
            model_sha256=sha256_file(config.paths.model),
        ),
        source_path=source_path,
    )


def make_services(root: Path) -> BaselineServices:
    def verify_dataset(config, *, verify_all_hashes):
        return {
            "image_count": 2,
            "train_count": 1,
            "val_count": 2,
            "train_file_sha256": "a" * 64,
            "val_file_sha256": "b" * 64,
        }

    def collect_git(repository_root):
        return {
            "commit": "1" * 40,
            "branch": "feature/test",
            "tracked_changes": [],
            "untracked_files": [],
            "tracked_clean": True,
        }

    def evaluate(**kwargs):
        prediction_dir = kwargs["prediction_dir"]
        split_path = kwargs["split_path"]
        ids = split_path.read_text(encoding="utf-8").splitlines()
        self_files = sorted(path.stem for path in prediction_dir.glob("*.txt"))
        if self_files != sorted(ids):
            raise AssertionError("prediction set is incomplete")
        return {"evaluator": "test", "split_count": len(ids), "classes": {}}

    return BaselineServices(
        verify_dataset=verify_dataset,
        collect_git=collect_git,
        verify_file_sha256=lambda path, expected, label: expected,
        evaluate=evaluate,
        now=lambda: datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
    )


class BaselinePipelineTest(unittest.TestCase):
    def test_dry_run_never_calls_train_or_predict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_pipeline_config(root)
            adapter = FakeAdapter()

            result = run_baseline(
                config,
                mode="dry-run",
                adapter=adapter,
                repository_root=root,
                services=make_services(root),
            )

            self.assertEqual(result.mode, "dry-run")
            self.assertIsNone(result.run_dir)
            self.assertEqual(adapter.train_calls, [])
            self.assertEqual(adapter.predict_calls, [])

    def test_full_run_records_complete_lifecycle_and_all_predictions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_pipeline_config(root)
            adapter = FakeAdapter()

            result = run_baseline(
                config,
                mode="full",
                adapter=adapter,
                repository_root=root,
                services=make_services(root),
            )

            assert result.run_dir is not None
            status = json.loads(
                (result.run_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "complete")
            labels = result.run_dir / "predictions" / "labels"
            self.assertEqual(
                sorted(path.name for path in labels.glob("*.txt")),
                ["000101.txt", "000102.txt"],
            )
            self.assertEqual(
                (labels / "000102.txt").read_text(encoding="utf-8"),
                "",
            )
            self.assertTrue((result.run_dir / "metrics_ap40.json").is_file())
            self.assertTrue((result.run_dir / "config.input.yaml").is_file())
            self.assertEqual(len(adapter.train_calls), 1)
            self.assertEqual(len(adapter.predict_calls), 1)

    def test_smoke_resolved_config_records_effective_training_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_pipeline_config(root, train_count=16, val_count=16)
            adapter = FakeAdapter()

            result = run_baseline(
                config,
                mode="smoke",
                adapter=adapter,
                repository_root=root,
                services=make_services(root),
                device_override="cpu",
            )

            assert result.run_dir is not None
            resolved = yaml.safe_load(
                (result.run_dir / "config.resolved.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(resolved["mode"], "smoke")
            self.assertEqual(resolved["training"]["epochs"], 1)
            self.assertEqual(resolved["training"]["imgsz"], 320)
            self.assertEqual(resolved["training"]["batch"], 2)
            self.assertEqual(resolved["training"]["workers"], 0)
            self.assertEqual(resolved["training"]["device"], "cpu")
            image_paths = adapter.predict_calls[0]["image_paths"]
            assert isinstance(image_paths, tuple)
            self.assertTrue(
                image_paths[0].is_relative_to(root / "tmp" / "smoke-kitti")
            )

    def test_training_failure_is_recorded_and_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_pipeline_config(root)
            adapter = FakeAdapter(fail_training=True)

            with self.assertRaisesRegex(
                RuntimeError,
                "synthetic training failure",
            ):
                run_baseline(
                    config,
                    mode="full",
                    adapter=adapter,
                    repository_root=root,
                    services=make_services(root),
                )

            run_dirs = tuple((root / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            status = json.loads(
                (run_dirs[0] / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["stage"], "training")

    def test_prediction_completion_creates_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            (labels / "000001.txt").write_text(
                "0 0.5 0.5 0.2 0.2 0.9\n",
                encoding="utf-8",
            )

            ensure_prediction_files(labels, ("000001", "000002"))

            self.assertTrue((labels / "000002.txt").is_file())

    def test_prediction_completion_rejects_extra_or_five_field_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            (labels / "999999.txt").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected prediction IDs"):
                ensure_prediction_files(labels, ("000001",))
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            (labels / "000001.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must contain 6 fields"):
                ensure_prediction_files(labels, ("000001",))


if __name__ == "__main__":
    unittest.main()
