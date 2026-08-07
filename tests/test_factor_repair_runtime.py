"""Task11 formal F0--F3 runtime contract tests (RED first)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import yaml

from ifdr_yolo.data.metadata_index import (
    KittiLabelCandidate,
    KittiMetadataObject,
    build_metadata_index,
)
from ifdr_yolo.data.ifdr_dataset import (
    BACKGROUND_IMAGE_KEY,
    CLEAN_IMAGE_KEY,
    TARGET_IMAGE_KEY,
)
from ifdr_yolo.experiments.factor_repair_runtime import (
    FactorRepairRuntime,
    build_factor_repair_runtime,
)
from ifdr_yolo.experiments.ifdr_trainer import (
    FactorCalibrationTrainer,
    factor_amp_preflight,
    validate_amp_outputs,
)
from scripts.train_factor_repair import run_registered_condition


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path, *, overlap: bool = False) -> SimpleNamespace:
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model.yaml"
    model.write_text("nc: 3\n", encoding="utf-8")
    image_root = root / "images"
    (image_root / "train").mkdir(parents=True)
    (image_root / "val").mkdir(parents=True)
    for image_id in ("fit-a", "fit-b"):
        (image_root / "train" / f"{image_id}.png").write_bytes(b"png")
    (image_root / "train" / "dev-a.png").write_bytes(b"png")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "Car"}}
        ),
        encoding="utf-8",
    )
    checkpoint = root / "init.pt"
    checkpoint.write_bytes(b"initialization")

    source = KittiMetadataObject(
        image_id="fit-a",
        object_index=0,
        class_id=0,
        class_name="Car",
        bbox_xyxy=(1.0, 1.0, 10.0, 10.0),
        depth_m=10.0,
    )
    index = build_metadata_index(
        [source],
        labels={"fit-a": [KittiLabelCandidate(**source.as_label().__dict__) ]},
        source_sha256="a" * 64,
        split_sha256="b" * 64,
        label_source_sha256="c" * 64,
    )
    config_dir = root / "protocol"
    config_dir.mkdir()
    (config_dir / "metadata_index.json").write_bytes(index.to_json_bytes())
    fit_ids = "fit-a\nfit-b\n"
    dev_ids = "fit-a\n" if overlap else "dev-a\n"
    (config_dir / "fit_ids.txt").write_bytes(fit_ids.encode())
    (config_dir / "development_ids.txt").write_bytes(dev_ids.encode())
    config_path = config_dir / "factor.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    identity = SimpleNamespace(
        metadata_sha256=index.sha256,
        fit_ids_sha256=hashlib.sha256(fit_ids.encode()).hexdigest(),
        development_ids_sha256=hashlib.sha256(dev_ids.encode()).hexdigest(),
        initialization_checkpoint_sha256=_sha(checkpoint),
    )
    config = SimpleNamespace(
        source_path=config_path,
        identity=identity,
        condition="F0",
        paths=SimpleNamespace(initialization_checkpoint=checkpoint),
        training=SimpleNamespace(imgsz=640),
    )
    return SimpleNamespace(
        config=config,
        model=model,
        data=data,
        checkpoint=checkpoint,
        protocol=config_dir,
    )


class FactorRepairRuntimeTest(unittest.TestCase):
    def test_calibration_runtime_and_trainer_disable_geometry_augmentation(self) -> None:
        required = (
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "close_mosaic",
        )
        expected = {name: 0.0 for name in required}
        expected["close_mosaic"] = 0
        with TemporaryDirectory() as directory:
            runtime = self._runtime(directory)
            self.assertEqual(dict(runtime.geometry_overrides), expected)
            provenance = runtime.static_provenance(trainable=(), frozen=())
            self.assertEqual(provenance["geometry_overrides"], expected)
            overrides = FactorCalibrationTrainer.ultralytics_overrides(runtime)
            self.assertEqual({name: overrides[name] for name in required}, expected)
            self.assertIsInstance(overrides["close_mosaic"], int)
            self.assertEqual(overrides["close_mosaic"], 0)

    def test_amp_preflight_rejects_cpu_unit_model(self) -> None:
        model = torch.nn.Linear(4, 4)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            factor_amp_preflight(model)

    def test_amp_output_validation_rejects_nonfinite_or_shape_drift(self) -> None:
        valid = (torch.zeros(1, 2),)
        validate_amp_outputs(valid, (torch.full((1, 2), 0.1),))
        with self.assertRaisesRegex(RuntimeError, "finite"):
            validate_amp_outputs((torch.full((1, 2), float("nan")),), valid)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            validate_amp_outputs(valid, (torch.zeros(2, 2),))

    def test_setup_train_temporarily_uses_local_amp_check_and_restores(self) -> None:
        import ultralytics.engine.trainer as ultralytics_trainer

        observed: list[object] = []
        original = ultralytics_trainer.check_amp

        def parent_setup(_trainer: object) -> str:
            observed.append(ultralytics_trainer.check_amp)
            return "ready"

        trainer = object.__new__(FactorCalibrationTrainer)
        with patch("ifdr_yolo.experiments.ifdr_trainer.factor_amp_preflight", return_value=True) as local:
            with patch.object(FactorCalibrationTrainer.__mro__[1], "_setup_train", parent_setup):
                self.assertEqual(trainer._setup_train(), "ready")
            self.assertIs(observed[0], local)
        self.assertIs(ultralytics_trainer.check_amp, original)

    def _runtime(self, directory: str, **kwargs: object) -> FactorRepairRuntime:
        fixture = _write_bundle(Path(directory), overlap=bool(kwargs.pop("overlap", False)))
        return build_factor_repair_runtime(
            fixture.config,
            condition=kwargs.pop("condition", "F0"),
            model_yaml=fixture.model,
            data_yaml=fixture.data,
            run_dir=Path(directory) / "run",
            **kwargs,
        )

    def test_formal_budget_is_30_and_unknown_condition_or_illegal_smoke_fails(self) -> None:
        with TemporaryDirectory() as directory:
            runtime = self._runtime(directory)
            self.assertEqual(runtime.epochs, 30)
            self.assertEqual(runtime.registered_epochs, 30)
            with self.assertRaises(ValueError):
                self._runtime(directory + "-unknown", condition="M1")
            with self.assertRaises((TypeError, ValueError)):
                self._runtime(directory + "-bad", smoke_mode=True, epochs=2)  # type: ignore[call-arg]

    def test_runtime_is_immutable_and_smoke_budget_is_explicit(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = _write_bundle(Path(directory))
            runtime = build_factor_repair_runtime(
                fixture.config,
                condition="F0",
                model_yaml=fixture.model,
                data_yaml=fixture.data,
                run_dir=Path(directory) / "run",
                smoke_mode=True,
            )
            self.assertIsInstance(runtime, FactorRepairRuntime)
            self.assertEqual(runtime.epochs, 1)
            self.assertEqual(runtime.registered_epochs, 30)
            self.assertEqual(runtime.run_mode, "nonformal")
            with self.assertRaises((AttributeError, TypeError)):
                runtime.epochs = 30  # type: ignore[misc]

    def test_bundle_hash_and_fit_development_leakage_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = _write_bundle(Path(directory), overlap=True)
            with self.assertRaisesRegex(ValueError, "overlap|leakage"):
                build_factor_repair_runtime(
                    fixture.config,
                    condition="F0",
                    model_yaml=fixture.model,
                    data_yaml=fixture.data,
                    run_dir=Path(directory) / "run",
                )

    def test_each_protocol_hash_is_checked(self) -> None:
        for field, filename in (
            ("metadata_sha256", "metadata_index.json"),
            ("fit_ids_sha256", "fit_ids.txt"),
            ("development_ids_sha256", "development_ids.txt"),
        ):
            with self.subTest(field=field):
                with TemporaryDirectory() as directory:
                    fixture = _write_bundle(Path(directory))
                    fixture.config.identity.__dict__[field] = "0" * 64
                    with self.assertRaisesRegex(ValueError, "SHA256|hash"):
                        build_factor_repair_runtime(
                            fixture.config,
                            condition="F0",
                            model_yaml=fixture.model,
                            data_yaml=fixture.data,
                            run_dir=Path(directory) / "run",
                        )

    def test_dataset_build_wires_calibration_train_only(self) -> None:
        trainer = object.__new__(FactorCalibrationTrainer)
        trainer.model = SimpleNamespace(stride=torch.tensor([32]))
        trainer.args = SimpleNamespace()
        trainer.data = {"nc": 1}
        trainer.intervention_seed = 17
        trainer.condition = "F0"
        trainer.metadata_index = object()
        trainer.specificity_rejection_counter = object()
        trainer.component_switches = SimpleNamespace(interventions=True, counterfactual_consistency=False)
        with patch("ifdr_yolo.experiments.ifdr_trainer.build_ifdr_dataset", return_value=object()) as builder:
            trainer.build_dataset("images.txt", mode="train", batch=2)
            trainer.build_dataset("images.txt", mode="val", batch=2)
        self.assertTrue(builder.call_args_list[0].kwargs["calibration_enabled"])
        self.assertIs(builder.call_args_list[0].kwargs["metadata_index"], trainer.metadata_index)
        self.assertIs(builder.call_args_list[0].kwargs["specificity_rejection_counter"], trainer.specificity_rejection_counter)
        self.assertFalse(builder.call_args_list[1].kwargs["calibration_enabled"])
        self.assertFalse(builder.call_args_list[1].kwargs["interventions_enabled"])

    def test_runtime_writes_absolute_manifests_and_key_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            runtime = self._runtime(directory)
            resolved = runtime.run_dir / "resolved_data.yaml"
            self.assertTrue(resolved.is_file())
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["train"]).resolve(), runtime.fit_manifest)
            self.assertEqual(Path(payload["val"]).resolve(), runtime.development_manifest)
            for name in ("resolved_runtime.json", "provenance.json"):
                record = json.loads((runtime.run_dir / name).read_text(encoding="utf-8"))
                for field in (
                    "run_mode", "registered_epochs", "actual_epochs", "model_sha256",
                    "data_sha256", "initialization_checkpoint_sha256", "fit_ids_sha256",
                    "development_ids_sha256", "metadata_index_sha256",
                    "trainable_parameter_names", "frozen_parameter_names",
                ):
                    self.assertIn(field, record)

    def test_runtime_provenance_merges_without_dropping_runner_identity(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = _write_bundle(Path(directory))
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            initialization = {"path": "init.pt", "sha256": "c" * 64}
            (run_dir / "provenance.json").write_text(
                json.dumps({"git": {"commit": "a" * 40}, "identity": {"metadata_sha256": fixture.config.identity.metadata_sha256}, "initialization_checkpoint": initialization}),
                encoding="utf-8",
            )
            runtime = build_factor_repair_runtime(
                fixture.config,
                condition="F0",
                model_yaml=fixture.model,
                data_yaml=fixture.data,
                run_dir=run_dir,
            )
            payload = json.loads((runtime.run_dir / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["git"]["commit"], "a" * 40)
            self.assertEqual(payload["initialization_checkpoint"], initialization)
            self.assertIn("runtime", payload)

    def test_resolved_manifests_refuse_divergent_resume(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = _write_bundle(Path(directory))
            run_dir = Path(directory) / "run"
            runtime = build_factor_repair_runtime(
                fixture.config,
                condition="F0",
                model_yaml=fixture.model,
                data_yaml=fixture.data,
                run_dir=run_dir,
            )
            runtime.fit_manifest.write_bytes(runtime.fit_manifest.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(ValueError, "not identical"):
                build_factor_repair_runtime(
                    fixture.config,
                    condition="F0",
                    model_yaml=fixture.model,
                    data_yaml=fixture.data,
                    run_dir=run_dir,
                )

    def test_evaluator_rejects_non_last_path(self) -> None:
        trainer = object.__new__(FactorCalibrationTrainer)
        trainer.runtime = SimpleNamespace(run_dir=Path("run"))
        with self.assertRaises(ValueError):
            trainer.evaluate_primary_last(Path("run/weights/best.pt"))

    def test_evaluator_passes_verified_last_path_to_validator(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            last = run_dir / "weights" / "last.pt"
            last.parent.mkdir(parents=True)
            last.write_bytes(b"last")
            seen: list[object] = []
            trainer = object.__new__(FactorCalibrationTrainer)
            trainer.runtime = SimpleNamespace(run_dir=run_dir)
            trainer.validator = lambda **kwargs: seen.append(kwargs["model"]) or {"ok": True}
            self.assertEqual(trainer.evaluate_primary_last(last), {"ok": True})
            self.assertEqual(seen, [last.resolve()])

    def test_epoch_draw_key_is_deterministic(self) -> None:
        keys: list[tuple[int, str]] = []
        trainer = object.__new__(FactorCalibrationTrainer)
        trainer.condition = "F0"
        trainer.seed = 17
        trainer.epoch = 3
        trainer.draw_callback = lambda epoch, key: keys.append((epoch, key))
        trainer._record_epoch_draw(trainer)
        trainer._record_epoch_draw(trainer)
        self.assertEqual(keys[0], keys[1])

    def test_complete_ultralytics_override_mapping_is_not_dropped(self) -> None:
        runtime = SimpleNamespace(
            model_yaml=Path("model.yaml"),
            resolved_data_yaml=Path("resolved_data.yaml"),
            initialization_checkpoint=Path("init.pt"),
            run_dir=Path("run"),
            epochs=30,
            imgsz=640,
            batch=16,
            workers=8,
            device="cpu",
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            seed=17,
            amp=True,
            deterministic=True,
            cache=False,
        )
        overrides = FactorCalibrationTrainer.ultralytics_overrides(runtime)
        self.assertEqual(
            overrides,
            {
                "model": str(Path("model.yaml").resolve()),
                "data": str(Path("resolved_data.yaml").resolve()),
                "pretrained": str(Path("init.pt").resolve()),
                "epochs": 30,
                "imgsz": 640,
                "batch": 16,
                "workers": 8,
                "device": "cpu",
                "optimizer": "SGD",
                "lr0": 0.01,
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 3.0,
                "seed": 17,
                "amp": True,
                "deterministic": True,
                "cache": False,
                "mosaic": 0.0,
                "mixup": 0.0,
                "cutmix": 0.0,
                "copy_paste": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "close_mosaic": 0,
                "patience": 0,
                "save_dir": str(Path("run").resolve()),
            },
        )

    def test_optimizer_filter_is_exactly_phase_trainable_names(self) -> None:
        model = torch.nn.Module()
        model.trainable = torch.nn.Parameter(torch.ones(1))
        model.frozen = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = object.__new__(FactorCalibrationTrainer)
        trainer._filter_optimizer_for_phase(
            optimizer,
            SimpleNamespace(trainable_parameter_names=("trainable",)),
            model,
        )
        self.assertEqual(
            {id(param) for group in optimizer.param_groups for param in group["params"]},
            {id(model.trainable)},
        )
        self.assertFalse(model.frozen.requires_grad)

    def test_phase_is_not_applied_while_model_is_still_a_yaml_string(self) -> None:
        trainer = object.__new__(FactorCalibrationTrainer)
        trainer.model = "models/kitti-p2-m.yaml"
        with patch("ifdr_yolo.experiments.ifdr_trainer.IFDRDetectionTrainer._setup_train", return_value="ready") as parent_setup:
            with patch("ifdr_yolo.experiments.factor_repair.semantic_calibration_phase", side_effect=AssertionError("phase too early")):
                # The phase is bound by build_optimizer after Ultralytics has
                # materialized ``self.model``; setup itself must not touch it.
                result = super(FactorCalibrationTrainer, trainer)._setup_train()
        self.assertEqual(result, "ready")
        parent_setup.assert_called_once()

    def test_custom_factory_internal_typeerror_is_not_retried(self) -> None:
        calls: list[dict[str, object]] = []

        class Store:
            state = "prepared"

            def transition(self, state: str) -> None:
                self.state = state

            def fail(self, **_: object) -> None:
                self.state = "failed"

        run = SimpleNamespace(
            config=object(),
            condition="F0",
            run_dir=Path("run"),
            journal=object(),
            record_epoch_draw=lambda *_: None,
            store=Store(),
            release=lambda: None,
        )

        def factory(**kwargs: object) -> object:
            calls.append(kwargs)
            raise TypeError("factory body failure")

        with self.assertRaisesRegex(TypeError, "factory body failure"):
            run_registered_condition(run, trainer_factory=factory)
        self.assertEqual(len(calls), 1)

    def test_factor_trainer_has_path_bound_evaluator_and_three_view_preprocess(self) -> None:
        parameters = inspect.signature(FactorCalibrationTrainer.evaluate_primary_last).parameters
        self.assertEqual(tuple(parameters), ("self", "path"))
        trainer = object.__new__(FactorCalibrationTrainer)
        trainer.device = torch.device("cpu")
        base = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        batch = {
            "img": base,
            CLEAN_IMAGE_KEY: torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
            TARGET_IMAGE_KEY: torch.full((1, 3, 2, 2), 128, dtype=torch.uint8),
            BACKGROUND_IMAGE_KEY: torch.full((1, 3, 2, 2), 64, dtype=torch.uint8),
        }
        with patch("ifdr_yolo.experiments.ifdr_trainer.IFDRDetectionTrainer.preprocess_batch", return_value=batch):
            result = trainer.preprocess_batch(batch)
        for key, expected in ((CLEAN_IMAGE_KEY, 1.0), (TARGET_IMAGE_KEY, 128 / 255), (BACKGROUND_IMAGE_KEY, 64 / 255)):
            self.assertEqual(result[key].shape, base.shape)
            self.assertEqual(result[key].dtype, base.dtype)
            self.assertEqual(result[key].device, base.device)
            self.assertTrue(torch.allclose(result[key], torch.full_like(base, expected)))


if __name__ == "__main__":
    unittest.main()
