import os
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.experiments.ultralytics_runtime import (
    UltralyticsAdapter,
    bootstrap_ultralytics_config,
    validate_runtime,
)


class FakeYOLO:
    instances: list["FakeYOLO"] = []

    def __init__(self, source: str) -> None:
        self.source = source
        self.train_kwargs: dict[str, object] | None = None
        self.predict_kwargs: dict[str, object] | None = None
        self.__class__.instances.append(self)

    def train(self, **kwargs: object) -> None:
        self.train_kwargs = kwargs
        run_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"best")

    def predict(self, **kwargs: object) -> None:
        self.predict_kwargs = kwargs
        output_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        (output_dir / "labels").mkdir(parents=True)


class UltralyticsRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeYOLO.instances.clear()

    def test_bootstrap_sets_config_dir_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            path = bootstrap_ultralytics_config(root)

            self.assertEqual(
                os.environ["YOLO_CONFIG_DIR"],
                str(root / "tmp" / "yolo-config"),
            )
            self.assertTrue(path.is_dir())

    def test_validate_runtime_accepts_expected_cuda_device(self) -> None:
        validate_runtime(
            actual_ultralytics="8.4.98",
            expected_ultralytics="8.4.98",
            cuda_available=True,
            device_count=1,
            requested_device="0",
            require_cuda=True,
        )

    def test_validate_runtime_rejects_version_and_device_mismatch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Ultralytics version"):
            validate_runtime(
                actual_ultralytics="8.4.97",
                expected_ultralytics="8.4.98",
                cuda_available=True,
                device_count=1,
                requested_device="0",
                require_cuda=True,
            )
        with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
            validate_runtime(
                actual_ultralytics="8.4.98",
                expected_ultralytics="8.4.98",
                cuda_available=False,
                device_count=0,
                requested_device="0",
                require_cuda=True,
            )
        with self.assertRaisesRegex(RuntimeError, "CUDA device 1"):
            validate_runtime(
                actual_ultralytics="8.4.98",
                expected_ultralytics="8.4.98",
                cuda_available=True,
                device_count=1,
                requested_device="1",
                require_cuda=True,
            )

    def test_adapter_trains_into_owned_run_and_returns_best_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "yolov8m.pt"
            data = root / "data.yaml"
            model.write_bytes(b"model")
            data.write_text("names: []\n", encoding="utf-8")
            run_dir = root / "runs" / "experiment"
            adapter = UltralyticsAdapter(yolo_factory=FakeYOLO)

            best = adapter.train(
                model_path=model,
                data_path=data,
                run_dir=run_dir,
                args={"epochs": 1, "device": "cpu"},
            )

            self.assertEqual(best, run_dir / "weights" / "best.pt")
            instance = FakeYOLO.instances[-1]
            assert instance.train_kwargs is not None
            self.assertEqual(instance.source, str(model))
            self.assertEqual(instance.train_kwargs["data"], str(data))
            self.assertTrue(instance.train_kwargs["exist_ok"])

    def test_adapter_predicts_with_confidence_text_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.pt"
            weights.write_bytes(b"best")
            image = root / "000001.png"
            image.write_bytes(b"image")
            output_dir = root / "predictions"
            adapter = UltralyticsAdapter(yolo_factory=FakeYOLO)

            labels = adapter.predict(
                weights=weights,
                image_paths=(image,),
                output_dir=output_dir,
                args={"conf": 0.001, "device": "cpu"},
            )

            self.assertEqual(labels, output_dir / "labels")
            instance = FakeYOLO.instances[-1]
            assert instance.predict_kwargs is not None
            self.assertEqual(instance.predict_kwargs["source"], [str(image)])
            self.assertTrue(instance.predict_kwargs["save_txt"])
            self.assertTrue(instance.predict_kwargs["save_conf"])


if __name__ == "__main__":
    unittest.main()
