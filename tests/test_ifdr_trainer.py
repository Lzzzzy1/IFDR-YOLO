from pathlib import Path
from types import SimpleNamespace
import unittest
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


class IFDRDetectionTrainerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
