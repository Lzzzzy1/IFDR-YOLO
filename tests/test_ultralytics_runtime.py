import os
from pathlib import Path
import tempfile
import unittest

import torch

from ifdr_yolo.experiments.config import InitializationConfig
from ifdr_yolo.experiments.ultralytics_runtime import (
    PreparedModel,
    UltralyticsAdapter,
    bootstrap_ultralytics_config,
    validate_runtime,
)
from ifdr_yolo.models.initialization import InitializationReport


class FakeYOLO:
    instances: list["FakeYOLO"] = []

    def __init__(self, source: str) -> None:
        self.source = source
        self.model = torch.nn.Linear(1, 1)
        self.overrides = {"model": source}
        self.callbacks: dict[str, object] = {}
        self.train_kwargs: dict[str, object] | None = None
        self.predict_kwargs: dict[str, object] | None = None
        self.__class__.instances.append(self)

    def train(self, **kwargs: object) -> None:
        raise AssertionError("prepared training must bypass YOLO.train rebuild")

    def _smart_load(self, name: str):
        if name != "trainer":
            raise AssertionError(f"unexpected component: {name}")
        return FakeTrainer

    def predict(self, **kwargs: object) -> None:
        self.predict_kwargs = kwargs
        output_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        (output_dir / "labels").mkdir(parents=True)


class FakeTrainer:
    instances: list["FakeTrainer"] = []

    def __init__(
        self,
        *,
        overrides: dict[str, object],
        _callbacks: dict[str, object],
    ) -> None:
        self.overrides = overrides
        self.callbacks = _callbacks
        self.model: object | None = None
        self.__class__.instances.append(self)

    def train(self) -> None:
        if self.model is None:
            raise AssertionError("prepared model was not attached to trainer")
        run_dir = (
            Path(str(self.overrides["project"]))
            / str(self.overrides["name"])
        )
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"best")


class UltralyticsRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeYOLO.instances.clear()
        FakeTrainer.instances.clear()

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
            prepared = PreparedModel(
                handle=FakeYOLO(str(model)),
                initialization=None,
            )

            best = adapter.train(
                prepared_model=prepared,
                data_path=data,
                run_dir=run_dir,
                args={"epochs": 1, "device": "cpu"},
            )

            self.assertEqual(best, run_dir / "weights" / "best.pt")
            trainer = FakeTrainer.instances[-1]
            self.assertIs(trainer.model, prepared.handle.model)
            self.assertEqual(trainer.overrides["model"], str(model))
            self.assertEqual(trainer.overrides["data"], str(data))
            self.assertTrue(trainer.overrides["exist_ok"])

    def test_adapter_prepares_seeded_model_without_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.yaml"
            model.write_text("nc: 3\n", encoding="utf-8")
            seed_calls: list[tuple[int, bool]] = []
            adapter = UltralyticsAdapter(
                yolo_factory=FakeYOLO,
                seed_initializer=lambda seed, deterministic: seed_calls.append(
                    (seed, deterministic)
                ),
            )

            prepared = adapter.prepare_model(
                model_path=model,
                model_sha256="a" * 64,
                initialization=None,
                seed=17,
                deterministic=True,
            )

            self.assertIs(prepared.handle, FakeYOLO.instances[-1])
            self.assertIsNone(prepared.initialization)
            self.assertEqual(seed_calls, [(17, True)])

    def test_adapter_prepares_target_before_semantic_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.yaml"
            pretrained = root / "yolov8m.pt"
            model.write_text("nc: 3\n", encoding="utf-8")
            pretrained.write_bytes(b"pretrained")
            initialization = InitializationConfig(
                pretrained=pretrained,
                pretrained_sha256="b" * 64,
                strategy="semantic_prefix",
                max_layer=15,
                expected_items=306,
            )
            initializer_calls: list[tuple[object, object, int, int]] = []
            inspected: list[object] = []

            def initialize(
                target: object,
                source: object,
                *,
                max_layer: int,
                expected_items: int,
            ) -> InitializationReport:
                initializer_calls.append(
                    (target, source, max_layer, expected_items)
                )
                return InitializationReport(
                    strategy="semantic_prefix",
                    max_layer=max_layer,
                    expected_items=expected_items,
                    transferred_items=expected_items,
                    source_items=355,
                    target_items=581,
                    untransferred_items=275,
                    transferred_keys=("model.0.weight",),
                    transferred_shapes={"model.0.weight": [1, 1]},
                )

            def inspect(model_handle: object) -> dict[str, object]:
                inspected.append(model_handle)
                return {"strides": [4.0, 8.0, 16.0, 32.0]}

            adapter = UltralyticsAdapter(
                yolo_factory=FakeYOLO,
                seed_initializer=lambda _seed, _deterministic: None,
                model_initializer=initialize,
                p2_inspector=inspect,
            )

            prepared = adapter.prepare_model(
                model_path=model,
                model_sha256="a" * 64,
                initialization=initialization,
                seed=17,
                deterministic=True,
            )

            self.assertEqual(
                [instance.source for instance in FakeYOLO.instances],
                [str(model), str(pretrained)],
            )
            target, source = FakeYOLO.instances
            self.assertEqual(
                initializer_calls,
                [(target.model, source.model, 15, 306)],
            )
            self.assertEqual(inspected, [target.model])
            self.assertIs(prepared.handle, target)
            assert prepared.initialization is not None
            self.assertEqual(prepared.initialization["transferred_items"], 306)
            self.assertEqual(prepared.initialization["architecture"], str(model))
            self.assertEqual(
                prepared.initialization["pretrained"],
                str(pretrained),
            )
            self.assertEqual(prepared.initialization["seed"], 17)
            self.assertTrue(prepared.initialization["deterministic"])

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
            self.assertEqual(instance.predict_kwargs["source"], str(root))
            self.assertTrue(instance.predict_kwargs["save_txt"])
            self.assertTrue(instance.predict_kwargs["save_conf"])


if __name__ == "__main__":
    unittest.main()
