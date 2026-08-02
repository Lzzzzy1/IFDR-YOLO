from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import tempfile
import unittest

import torch

from ifdr_yolo.experiments.config import load_ifdr_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_ifdr_config(
    ROOT / "configs/experiments/kitti_ifdr_yolov8m_s17.yaml",
    repository_root=ROOT,
)


class FakeReport:
    def to_payload(self) -> dict[str, object]:
        return {"strategy": "semantic_prefix", "transferred_items": 306}


class FakeTrainer:
    instances: list["FakeTrainer"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.model = None
        self.__class__.instances.append(self)

    def train(self) -> None:
        overrides = self.kwargs["overrides"]
        weights = (
            Path(overrides["project"])
            / overrides["name"]
            / "weights"
        )
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"ifdr")


class IFDRRuntimeAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTrainer.instances.clear()

    def test_prepare_builds_configured_model_and_semantic_initialization(self) -> None:
        from ifdr_yolo.experiments.ifdr_runtime import IFDRRuntimeAdapter

        model_calls = []
        seed_calls = []
        initialization_calls = []
        target = torch.nn.Linear(2, 2)
        source = torch.nn.Linear(2, 2)

        def model_factory(**kwargs):
            model_calls.append(kwargs)
            return target

        def yolo_factory(path: str):
            self.assertEqual(path, str(CONFIG.initialization.pretrained))
            return SimpleNamespace(model=source)

        def initializer(target_model, source_model, **kwargs):
            initialization_calls.append(
                (target_model, source_model, kwargs)
            )
            return FakeReport()

        adapter = IFDRRuntimeAdapter(
            CONFIG,
            model_factory=model_factory,
            yolo_factory=yolo_factory,
            seed_initializer=lambda seed, deterministic: seed_calls.append(
                (seed, deterministic)
            ),
            model_initializer=initializer,
            trainer_factory=FakeTrainer,
        )

        prepared = adapter.prepare_model(
            model_path=CONFIG.paths.model,
            model_sha256=CONFIG.paths.model_sha256,
            initialization=CONFIG.initialization,
            seed=17,
            deterministic=True,
        )

        self.assertIs(prepared.handle.model, target)
        self.assertEqual(seed_calls, [(17, True)])
        self.assertEqual(model_calls[0]["reliability_channels"], 32)
        self.assertAlmostEqual(model_calls[0]["dcli_beta"], 0.5)
        self.assertEqual(initialization_calls[0][0:2], (target, source))
        self.assertEqual(
            prepared.initialization["transferred_items"],
            306,
        )
        self.assertEqual(
            prepared.initialization["components"],
            {
                "fusion_gate": True,
                "dcli": True,
                "factor_supervision": True,
                "interventions": True,
                "semantic_protection": False,
                "counterfactual_consistency": False,
            },
        )

    def test_prepare_forwards_gradient_diagnostic_interval(self) -> None:
        from ifdr_yolo.experiments.ifdr_runtime import IFDRRuntimeAdapter

        config = replace(
            CONFIG,
            method=replace(
                CONFIG.method,
                gradient_diagnostic_interval=50,
            ),
        )
        model_calls = []
        target = torch.nn.Linear(1, 1)
        adapter = IFDRRuntimeAdapter(
            config,
            model_factory=lambda **kwargs: (
                model_calls.append(kwargs) or target
            ),
            yolo_factory=lambda _: SimpleNamespace(
                model=torch.nn.Linear(1, 1)
            ),
            model_initializer=lambda *_args, **_kwargs: FakeReport(),
            trainer_factory=FakeTrainer,
        )

        adapter.prepare_model(
            model_path=config.paths.model,
            model_sha256=config.paths.model_sha256,
            initialization=config.initialization,
            seed=17,
            deterministic=True,
        )

        self.assertEqual(
            model_calls[0]["gradient_diagnostic_interval"],
            50,
        )

    def test_train_attaches_prepared_model_and_locked_method_controls(self) -> None:
        from ifdr_yolo.experiments.ifdr_runtime import (
            IFDRPreparedHandle,
            IFDRRuntimeAdapter,
        )
        from ifdr_yolo.experiments.ultralytics_runtime import PreparedModel

        adapter = IFDRRuntimeAdapter(
            CONFIG,
            model_factory=lambda **_: torch.nn.Linear(1, 1),
            yolo_factory=lambda _: None,
            trainer_factory=FakeTrainer,
        )
        model = torch.nn.Linear(1, 1)
        prepared = PreparedModel(
            handle=IFDRPreparedHandle(
                model=model,
                model_path=CONFIG.paths.model,
            ),
            initialization=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            best = adapter.train(
                prepared_model=prepared,
                data_path=CONFIG.paths.data,
                run_dir=run_dir,
                args={"epochs": 1, "seed": 17},
            )

            self.assertEqual(best, run_dir / "weights/best.pt")
            trainer = FakeTrainer.instances[-1]
            self.assertIs(trainer.model, model)
            self.assertEqual(
                trainer.kwargs["fusion_schedule"].frozen_epochs,
                5,
            )
            self.assertAlmostEqual(
                trainer.kwargs[
                    "intervention_policy"
                ].visibility_probability,
                0.4,
            )
            switches = trainer.kwargs["component_switches"]
            self.assertTrue(switches.fusion_gate)
            self.assertTrue(switches.dcli)
            self.assertTrue(switches.factor_supervision)
            self.assertTrue(switches.interventions)
            self.assertFalse(switches.semantic_protection)
            self.assertFalse(switches.counterfactual_consistency)


if __name__ == "__main__":
    unittest.main()
