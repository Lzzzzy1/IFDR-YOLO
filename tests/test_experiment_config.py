from pathlib import Path
import tempfile
import unittest

import yaml

from ifdr_yolo.experiments.config import load_baseline_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = (
    "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
)


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": {
            "dataset": "kitti",
            "model": "yolov8m",
            "variant": "baseline",
            "seed": 17,
        },
        "paths": {
            "model": "yolov8m.pt",
            "model_sha256": MODEL_SHA256,
            "data": "configs/data/kitti_v2.yaml",
            "generated_data": "data/processed/kitti_yolo_v2",
            "raw_images": "kitti_raw/training/image_2/training/image_2",
            "raw_labels": "kitti_raw/training/label_2/training/label_2",
            "train_ids": "configs/splits/kitti_train.txt",
            "val_ids": "configs/splits/kitti_val.txt",
        },
        "training": {
            "epochs": 300,
            "imgsz": 640,
            "batch": 16,
            "workers": 8,
            "device": "0",
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": 0,
            "amp": True,
            "deterministic": True,
            "cache": False,
        },
        "prediction": {
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
        },
    }


class BaselineConfigTest(unittest.TestCase):
    def write_payload(
        self,
        directory: str,
        payload: dict[str, object],
    ) -> Path:
        path = Path(directory) / "experiment.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        return path

    def test_loads_valid_config_and_resolves_paths_from_repository(self) -> None:
        config = load_baseline_config(
            ROOT / "configs/experiments/kitti_yolov8m_baseline_s17.yaml",
            repository_root=ROOT,
        )

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.experiment.seed, 17)
        self.assertEqual(config.training.epochs, 300)
        self.assertEqual(config.paths.data, ROOT / "configs/data/kitti_v2.yaml")
        self.assertEqual(config.paths.model_sha256, MODEL_SHA256)
        self.assertEqual(
            config.source_path,
            ROOT / "configs/experiments/kitti_yolov8m_baseline_s17.yaml",
        )

    def test_rejects_unknown_training_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_payload()
            training = payload["training"]
            assert isinstance(training, dict)
            training["epochz"] = 300
            path = self.write_payload(directory, payload)

            with self.assertRaisesRegex(ValueError, "unknown training fields"):
                load_baseline_config(path, repository_root=ROOT)

    def test_rejects_boolean_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_payload()
            training = payload["training"]
            assert isinstance(training, dict)
            training["batch"] = True
            path = self.write_payload(directory, payload)

            with self.assertRaisesRegex(ValueError, "training.batch"):
                load_baseline_config(path, repository_root=ROOT)

    def test_rejects_missing_path_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_payload()
            paths = payload["paths"]
            assert isinstance(paths, dict)
            del paths["raw_labels"]
            path = self.write_payload(directory, payload)

            with self.assertRaisesRegex(ValueError, "missing paths fields"):
                load_baseline_config(path, repository_root=ROOT)

    def test_rejects_prediction_iou_above_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_payload()
            prediction = payload["prediction"]
            assert isinstance(prediction, dict)
            prediction["iou"] = 1.1
            path = self.write_payload(directory, payload)

            with self.assertRaisesRegex(ValueError, "prediction.iou"):
                load_baseline_config(path, repository_root=ROOT)


if __name__ == "__main__":
    unittest.main()
