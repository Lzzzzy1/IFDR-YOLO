from pathlib import Path
from types import SimpleNamespace
import unittest
import json
import tempfile
from unittest.mock import patch

import torch

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kitti-p2-m.yaml"
bootstrap_ultralytics_config(ROOT)


class FusionScheduleTest(unittest.TestCase):
    def test_keeps_baseline_then_ramps_to_full_strength(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import FusionSchedule

        schedule = FusionSchedule(frozen_epochs=5, ramp_epochs=10)

        expected = {
            0: 0.0,
            4: 0.0,
            5: 0.1,
            9: 0.5,
            14: 1.0,
            15: 1.0,
            299: 1.0,
        }
        for epoch, value in expected.items():
            self.assertAlmostEqual(schedule.value_at(epoch), value)

    def test_rejects_invalid_epoch_or_configuration(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import FusionSchedule

        for kwargs in (
            {"frozen_epochs": -1, "ramp_epochs": 10},
            {"frozen_epochs": 5, "ramp_epochs": 0},
            {"frozen_epochs": True, "ramp_epochs": 10},
        ):
            with self.assertRaises(ValueError):
                FusionSchedule(**kwargs)
        with self.assertRaises(ValueError):
            FusionSchedule().value_at(-1)

    def test_component_switches_produce_independent_epoch_schedules(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRComponentSwitches,
            FusionSchedule,
            apply_fusion_schedule,
        )

        class Recorder:
            values = None

            def set_component_schedules(self, **values) -> None:
                self.values = values

        model = Recorder()
        trainer = SimpleNamespace(
            epoch=14,
            model=model,
            fusion_schedule=FusionSchedule(),
            component_switches=IFDRComponentSwitches(
                fusion_gate=False,
                dcli=True,
                factor_supervision=False,
                interventions=True,
            ),
            fusion_schedule_value=None,
        )

        apply_fusion_schedule(trainer)

        self.assertEqual(
            model.values,
            {
                "fusion": 0.0,
                "dcli": 1.0,
                "factor_supervision": 0.0,
            },
        )


class IFDRDetectionTrainerTest(unittest.TestCase):
    def test_flush_gradient_diagnostics_appends_jsonl_records(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            flush_gradient_diagnostics,
        )

        record = {
            "schema_version": 1,
            "step": 2,
            "gradient_norms": {"detection": 1.0, "factor": 2.0},
            "pairs": {
                "detection::factor": {
                    "cosine": -0.5,
                    "conflict": True,
                }
            },
        }

        class Model:
            def __init__(self) -> None:
                self.records = (record,)

            def drain_gradient_diagnostics(self):
                records, self.records = self.records, ()
                return records

        with tempfile.TemporaryDirectory() as directory:
            trainer = type(
                "Trainer",
                (),
                {"model": Model(), "save_dir": Path(directory)},
            )()

            flush_gradient_diagnostics(trainer)
            flush_gradient_diagnostics(trainer)

            path = Path(directory) / "gradient_diagnostics.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), record)

    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_ultralytics_config(ROOT)

    def test_get_model_builds_project_owned_detector(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRDetectionTrainer,
        )
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        trainer = object.__new__(IFDRDetectionTrainer)
        trainer.data = {
            "nc": 3,
            "channels": 3,
            "names": {0: "Car", 1: "Pedestrian", 2: "Cyclist"},
        }
        trainer.set_model_names_for_load = lambda model: model

        model = trainer.get_model(str(MODEL_PATH), verbose=False)

        self.assertIsInstance(model, IFDRDetectionModel)
        self.assertEqual(model.yaml["nc"], 3)
        self.assertEqual(model.fusion_node_indices, (11, 14, 17, 20, 23, 26))

    def test_epoch_callback_updates_all_fusion_nodes(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            FusionSchedule,
            apply_fusion_schedule,
        )

        class ScheduleRecorder:
            def __init__(self) -> None:
                self.values: list[float] = []

            def set_reliability_schedule(self, value: float) -> None:
                self.values.append(value)

        model = ScheduleRecorder()
        dataset = SimpleNamespace(values=[])
        dataset.set_epoch = dataset.values.append
        trainer = SimpleNamespace(
            epoch=9,
            model=model,
            train_loader=SimpleNamespace(dataset=dataset),
            fusion_schedule=FusionSchedule(
                frozen_epochs=5,
                ramp_epochs=10,
            ),
            fusion_schedule_value=None,
        )

        value = apply_fusion_schedule(trainer)

        self.assertAlmostEqual(value, 0.5)
        self.assertEqual(model.values, [0.5])
        self.assertEqual(dataset.values, [9])
        self.assertAlmostEqual(trainer.fusion_schedule_value, 0.5)

    def test_epoch_callback_accepts_wrapped_model(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            FusionSchedule,
            apply_fusion_schedule,
        )

        class ScheduleRecorder:
            def __init__(self) -> None:
                self.value = None

            def set_reliability_schedule(self, value: float) -> None:
                self.value = value

        model = ScheduleRecorder()
        trainer = SimpleNamespace(
            epoch=5,
            model=SimpleNamespace(module=model),
            fusion_schedule=FusionSchedule(),
            fusion_schedule_value=None,
        )

        apply_fusion_schedule(trainer)

        self.assertAlmostEqual(model.value, 0.1)

    def test_build_dataset_uses_ifdr_dataset_for_train_and_validation(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        trainer = object.__new__(IFDRDetectionTrainer)
        trainer.model = SimpleNamespace(stride=torch.tensor([4, 8, 16, 32]))
        trainer.args = SimpleNamespace()
        trainer.data = {"nc": 3}
        trainer.intervention_seed = 17
        sentinel = object()

        with patch(
            "ifdr_yolo.experiments.ifdr_trainer.build_ifdr_dataset",
            return_value=sentinel,
        ) as builder:
            train = trainer.build_dataset("train.txt", mode="train", batch=8)
            validation = trainer.build_dataset("val.txt", mode="val", batch=8)

        self.assertIs(train, sentinel)
        self.assertIs(validation, sentinel)
        self.assertTrue(builder.call_args_list[0].kwargs["interventions_enabled"])
        self.assertFalse(builder.call_args_list[1].kwargs["interventions_enabled"])

    def test_component_switch_can_disable_training_interventions(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRComponentSwitches,
            IFDRDetectionTrainer,
        )

        trainer = object.__new__(IFDRDetectionTrainer)
        trainer.model = SimpleNamespace(stride=torch.tensor([4, 8, 16, 32]))
        trainer.args = SimpleNamespace()
        trainer.data = {"nc": 3}
        trainer.intervention_seed = 17
        trainer.component_switches = IFDRComponentSwitches(
            fusion_gate=True,
            dcli=True,
            factor_supervision=True,
            interventions=False,
        )

        with patch(
            "ifdr_yolo.experiments.ifdr_trainer.build_ifdr_dataset",
            return_value=object(),
        ) as builder:
            trainer.build_dataset("train.txt", mode="train", batch=8)

        self.assertFalse(builder.call_args.kwargs["interventions_enabled"])

    def test_preprocess_normalizes_and_resizes_counterfactual_view(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import COUNTERFACTUAL_IMAGE_KEY
        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        trainer = object.__new__(IFDRDetectionTrainer)
        trainer.device = torch.device("cpu")
        trainer.args = SimpleNamespace(multi_scale=0.0)
        batch = {
            "img": torch.full((1, 3, 8, 8), 255, dtype=torch.uint8),
            COUNTERFACTUAL_IMAGE_KEY: torch.full(
                (1, 3, 8, 8),
                127,
                dtype=torch.uint8,
            ),
        }

        result = trainer.preprocess_batch(batch)

        self.assertEqual(float(result["img"].max()), 1.0)
        self.assertAlmostEqual(
            float(result[COUNTERFACTUAL_IMAGE_KEY].max()),
            127.0 / 255.0,
        )

    def test_build_dataset_forwards_counterfactual_switch(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRComponentSwitches,
            IFDRDetectionTrainer,
        )

        trainer = object.__new__(IFDRDetectionTrainer)
        trainer.model = SimpleNamespace(stride=torch.tensor([4, 8, 16, 32]))
        trainer.args = SimpleNamespace()
        trainer.data = {"nc": 3}
        trainer.intervention_seed = 17
        trainer.component_switches = IFDRComponentSwitches(
            counterfactual_consistency=True,
        )

        with patch(
            "ifdr_yolo.experiments.ifdr_trainer.build_ifdr_dataset",
            return_value=object(),
        ) as builder:
            trainer.build_dataset("train.txt", mode="train", batch=8)

        self.assertTrue(
            builder.call_args.kwargs["counterfactual_enabled"]
        )

    def test_loss_names_expose_factor_and_counterfactual_terms(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        self.assertEqual(
            IFDRDetectionTrainer.IFDR_LOSS_NAMES,
            (
                "box_loss",
                "cls_loss",
                "dfl_loss",
                "factor_loss",
                "counterfactual_loss",
            ),
        )

    def test_validator_creation_restores_ifdr_loss_names(self) -> None:
        from ultralytics.models.yolo.detect import DetectionTrainer

        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        trainer = object.__new__(IFDRDetectionTrainer)
        sentinel = object()
        with patch.object(
            DetectionTrainer,
            "get_validator",
            return_value=sentinel,
        ):
            validator = trainer.get_validator()

        self.assertIs(validator, sentinel)
        self.assertEqual(
            trainer.loss_names,
            IFDRDetectionTrainer.IFDR_LOSS_NAMES,
        )

    def test_factor_calibration_uses_three_semantic_training_loss_labels(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            FactorCalibrationTrainer,
            IFDRDetectionTrainer,
        )

        self.assertEqual(
            FactorCalibrationTrainer.IFDR_LOSS_NAMES,
            (
                "synthetic_factor_loss",
                "natural_factor_loss",
                "specificity_loss",
            ),
        )
        self.assertEqual(len(FactorCalibrationTrainer.IFDR_LOSS_NAMES), 3)
        self.assertEqual(len(IFDRDetectionTrainer.IFDR_LOSS_NAMES), 5)

    def test_factor_calibration_validate_temporarily_uses_detection_loss_shape(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            FactorCalibrationTrainer,
            IFDRDetectionTrainer,
        )

        trainer = object.__new__(FactorCalibrationTrainer)
        calibration_names = FactorCalibrationTrainer.IFDR_LOSS_NAMES
        trainer.loss_names = calibration_names
        trainer.loss_items = torch.ones(len(calibration_names))
        observed = {}

        def parent_validate(_trainer):
            observed["names"] = _trainer.loss_names
            observed["shape"] = tuple(_trainer.loss_items.shape)
            observed["values"] = _trainer.loss_items.clone()
            return "validated"

        with patch.object(IFDRDetectionTrainer, "validate", parent_validate):
            result = trainer.validate()

        self.assertEqual(result, "validated")
        self.assertEqual(observed["names"], IFDRDetectionTrainer.IFDR_LOSS_NAMES)
        self.assertEqual(observed["shape"], (5,))
        self.assertTrue(torch.equal(observed["values"], torch.zeros(5)))
        self.assertEqual(trainer.loss_names, calibration_names)
        self.assertTrue(torch.equal(trainer.loss_items, torch.ones(3)))

    def test_factor_calibration_validate_restores_state_on_parent_failure(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            FactorCalibrationTrainer,
            IFDRDetectionTrainer,
        )

        trainer = object.__new__(FactorCalibrationTrainer)
        calibration_names = FactorCalibrationTrainer.IFDR_LOSS_NAMES
        trainer.loss_names = calibration_names
        trainer.loss_items = torch.arange(3, dtype=torch.float32)

        def parent_validate(_trainer):
            self.assertEqual(_trainer.loss_names, IFDRDetectionTrainer.IFDR_LOSS_NAMES)
            self.assertEqual(tuple(_trainer.loss_items.shape), (5,))
            raise RuntimeError("validation failed")

        with patch.object(IFDRDetectionTrainer, "validate", parent_validate):
            with self.assertRaisesRegex(RuntimeError, "validation failed"):
                trainer.validate()

        self.assertEqual(trainer.loss_names, calibration_names)
        self.assertTrue(torch.equal(trainer.loss_items, torch.arange(3, dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
