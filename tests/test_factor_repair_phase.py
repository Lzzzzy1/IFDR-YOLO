from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kitti-p2-m.yaml"
bootstrap_ultralytics_config(ROOT)


class FactorRepairPhaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        torch.manual_seed(17)
        cls.model = IFDRDetectionModel(str(MODEL_PATH), verbose=False)

    def test_semantic_calibration_trainable_names_are_exact(self) -> None:
        from ifdr_yolo.experiments.factor_repair import (
            factor_calibration_parameter_groups,
            semantic_calibration_phase,
        )

        phase = semantic_calibration_phase(self.model, variant="F3", epochs=30)
        actual = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(actual, set(phase.trainable_parameter_names))

        groups = factor_calibration_parameter_groups(self.model)
        self.assertEqual(
            tuple(groups),
            tuple(f"projection_{index:02d}" for index in range(12))
            + ("shared_core", "factor_head"),
        )
        self.assertEqual(
            tuple(sorted(phase.trainable_parameter_names)),
            tuple(
                sorted(
                    name
                    for parameters in groups.values()
                    for name, parameter in self.model.named_parameters()
                    if any(parameter is item for item in parameters)
                )
            ),
        )
        self.assertEqual(len(groups) - 2, 12)
        self.assertEqual(
            set(phase.diagnostic_group_names),
            set(groups),
        )
        self.assertTrue(all(phase.diagnostic_group_provenance.values()))

    def test_shared_semantic_modules_are_counted_once(self) -> None:
        from ifdr_yolo.experiments.factor_repair import (
            factor_calibration_parameter_groups,
        )

        groups = factor_calibration_parameter_groups(self.model)
        all_parameters = [
            parameter for parameters in groups.values() for parameter in parameters
        ]
        self.assertEqual(len(all_parameters), len({id(parameter) for parameter in all_parameters}))
        shared = tuple(
            parameter
            for parameter in self.model.model[self.model.fusion_node_indices[0]]
            .reliability_estimator.shared_core.parameters()
        )
        head = tuple(
            parameter
            for parameter in self.model.model[self.model.fusion_node_indices[0]]
            .reliability_estimator.factor_head.parameters()
        )
        self.assertEqual(
            {id(parameter) for parameter in groups["shared_core"]},
            {id(parameter) for parameter in shared},
        )
        self.assertEqual(
            {id(parameter) for parameter in groups["factor_head"]},
            {id(parameter) for parameter in head},
        )

    def test_calibration_excludes_detection_and_task_adapters(self) -> None:
        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase

        phase = semantic_calibration_phase(self.model, variant="F3", epochs=30)
        forbidden = (
            "detect",
            "router",
            "fusion_adapter",
            "localization_adapter",
            "gate_logit",
        )
        self.assertFalse(
            any(
                token in name
                for name in phase.trainable_parameter_names
                for token in forbidden
            )
        )
        frozen = set(phase.frozen_parameter_names)
        self.assertTrue(any("gate_logit" in name for name in frozen))

    def test_calibration_loss_masks_are_registered(self) -> None:
        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase

        expected = {
            "F0": {"synthetic": 1.0, "natural": 0.0, "specificity": 0.0},
            "F1": {"synthetic": 1.0, "natural": 1.0, "specificity": 0.0},
            "F2": {"synthetic": 1.0, "natural": 0.0, "specificity": 1.0},
            "F3": {"synthetic": 1.0, "natural": 1.0, "specificity": 1.0},
        }
        for variant, mask in expected.items():
            phase = semantic_calibration_phase(self.model, variant=variant, epochs=30)
            self.assertEqual(phase.loss_mask, mask)
            self.assertEqual(phase.fusion_schedule, 0.0)
            self.assertEqual(phase.dcli_schedule, 0.0)
            self.assertEqual(phase.factor_supervision_schedule, 1.0)
            self.assertFalse(phase.early_stopping)

    def test_calibration_rejects_unregistered_variants_and_budgets(self) -> None:
        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase

        for variant in ("f4", "", 3, None):
            with self.assertRaises(ValueError):
                semantic_calibration_phase(self.model, variant=variant, epochs=30)
        for epochs in (29, 31, 0, -1, True, 30.0):
            with self.assertRaises(ValueError):
                semantic_calibration_phase(self.model, variant="F0", epochs=epochs)

    def test_three_view_wrapper_splits_every_node(self) -> None:
        from ifdr_yolo.experiments.factor_repair import split_three_view_contexts

        clean = torch.zeros(1, 3, 128, 128)
        target = torch.ones(1, 3, 128, 128)
        background = torch.full((1, 3, 128, 128), 2.0)
        self.model.set_component_schedules(
            fusion=0.0,
            dcli=0.0,
            factor_supervision=1.0,
        )
        with torch.no_grad():
            self.model(torch.cat((clean, target, background), dim=0))
        raw_contexts = self.model.consume_reliability_context()
        split = split_three_view_contexts(raw_contexts, batch_size=1)
        self.assertEqual(set(split), {"clean", "target", "background"})
        for view in split.values():
            self.assertEqual(tuple(view), self.model.fusion_node_indices)
            self.assertTrue(
                all(context.factors.shape[0] == 1 for context in view.values())
            )

    def test_optimizer_state_is_cleared_when_phase_is_applied(self) -> None:
        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        first = next(self.model.parameters())
        optimizer.state[first]["step"] = torch.tensor(7.0)
        optimizer.state[first]["exp_avg"] = torch.ones_like(first)
        self.assertTrue(optimizer.state)
        semantic_calibration_phase(
            self.model,
            variant="F0",
            epochs=30,
            optimizer=optimizer,
        )
        self.assertEqual(len(optimizer.state), 0)

    def test_invalid_optimizer_is_rejected_atomically(self) -> None:
        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase

        before_flags = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
        before_schedule = (
            self.model.fusion_schedule,
            self.model.dcli_schedule,
            self.model.factor_supervision_schedule,
        )
        with self.assertRaises(TypeError):
            semantic_calibration_phase(
                self.model,
                variant="F0",
                epochs=30,
                optimizer=object(),
            )
        self.assertEqual(
            before_flags,
            {
                name: parameter.requires_grad
                for name, parameter in self.model.named_parameters()
            },
        )
        self.assertEqual(
            before_schedule,
            (
                self.model.fusion_schedule,
                self.model.dcli_schedule,
                self.model.factor_supervision_schedule,
            ),
        )

    def test_optimizer_clear_must_be_callable_and_safe_before_mutation(self) -> None:
        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase

        self.model.set_component_schedules(
            fusion=0.37,
            dcli=0.41,
            factor_supervision=0.53,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = True

        class NonCallableClear:
            state = type("State", (), {"clear": 1})()

        class RaisingClear:
            class State:
                def clear(self):
                    raise RuntimeError("clear failed")

            state = State()

        for optimizer in (NonCallableClear(), RaisingClear()):
            before_flags = {
                name: parameter.requires_grad
                for name, parameter in self.model.named_parameters()
            }
            before_schedule = (
                self.model.fusion_schedule,
                self.model.dcli_schedule,
                self.model.factor_supervision_schedule,
            )
            with self.assertRaises((TypeError, RuntimeError)):
                semantic_calibration_phase(
                    self.model,
                    variant="F2",
                    epochs=30,
                    optimizer=optimizer,
                )
            self.assertEqual(
                before_flags,
                {
                    name: parameter.requires_grad
                    for name, parameter in self.model.named_parameters()
                },
            )
            self.assertEqual(
                before_schedule,
                (
                    self.model.fusion_schedule,
                    self.model.dcli_schedule,
                    self.model.factor_supervision_schedule,
                ),
            )

    def test_invalid_model_structure_does_not_clear_optimizer_state(self) -> None:
        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase

        state = {"sentinel": 1}
        optimizer = SimpleNamespace(state=state)
        broken = SimpleNamespace(
            factor_semantic_named_parameters=lambda: (_ for _ in ()).throw(
                ValueError("broken semantic API")
            ),
            named_parameters=self.model.named_parameters,
            fusion_node_indices=self.model.fusion_node_indices,
            model=self.model.model,
            set_component_schedules=lambda **_: None,
        )
        with self.assertRaisesRegex(ValueError, "broken semantic API"):
            semantic_calibration_phase(
                broken,
                variant="F0",
                epochs=30,
                optimizer=optimizer,
            )
        self.assertEqual(state, {"sentinel": 1})

    def test_semantic_phase_freezes_audit_inputs_and_validates_mask(self) -> None:
        from ifdr_yolo.experiments.factor_repair import SemanticCalibrationPhase

        trainable = ["model.projection.weight"]
        frozen = ["model.detect.weight"]
        loss_mask = {"synthetic": 1.0, "natural": 0.0, "specificity": 0.0}
        group_names = ["projection_00"]
        group_provenance = {"projection_00": ["model.projection.weight"]}
        provenance = {"nested": {"items": ["model.projection.weight"]}}
        phase = SemanticCalibrationPhase(
            variant="F0",
            epochs=30,
            trainable_parameter_names=trainable,
            frozen_parameter_names=frozen,
            loss_mask=loss_mask,
            diagnostic_group_names=group_names,
            diagnostic_group_provenance=group_provenance,
            provenance=provenance,
        )

        trainable.append("mutated")
        frozen.append("mutated")
        loss_mask["natural"] = 1.0
        group_names.append("mutated")
        group_provenance["projection_00"].append("mutated")
        provenance["nested"]["items"].append("mutated")

        self.assertEqual(phase.trainable_parameter_names, ("model.projection.weight",))
        self.assertEqual(phase.frozen_parameter_names, ("model.detect.weight",))
        self.assertEqual(phase.diagnostic_group_names, ("projection_00",))
        self.assertEqual(
            phase.diagnostic_group_provenance["projection_00"],
            ("model.projection.weight",),
        )
        self.assertEqual(
            phase.provenance["nested"]["items"],
            ("model.projection.weight",),
        )
        with self.assertRaises(TypeError):
            phase.loss_mask["synthetic"] = 0.0
        with self.assertRaises(TypeError):
            phase.diagnostic_group_provenance["projection_00"] = ()
        with self.assertRaises(TypeError):
            phase.provenance["nested"]["items"] = ()

        with self.assertRaisesRegex(ValueError, "loss mask"):
            SemanticCalibrationPhase(
                variant="F0",
                epochs=30,
                trainable_parameter_names=(),
                frozen_parameter_names=(),
                loss_mask={"synthetic": 1.0, "natural": 1.0, "specificity": 0.0},
            )

    def test_validation_restores_training_mode_when_eval_or_forward_raises(self) -> None:
        from ifdr_yolo.experiments.factor_repair import run_calibration_validation

        class EvalFailure:
            training = True

            def eval(self):
                self.training = False
                raise RuntimeError("eval failed")

            def train(self, mode=True):
                self.training = mode

            def __call__(self, _batch):
                raise AssertionError("forward must not run")

        class ForwardFailure:
            training = True

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

            def __call__(self, _batch):
                raise RuntimeError("forward failed")

        eval_model = EvalFailure()
        with self.assertRaisesRegex(RuntimeError, "eval failed"):
            run_calibration_validation(eval_model, torch.zeros(1))
        self.assertTrue(eval_model.training)
        model = ForwardFailure()
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            run_calibration_validation(model, torch.zeros(1))
        self.assertTrue(model.training)

    def test_validation_forward_does_not_step_or_change_parameter_bytes(self) -> None:
        from ifdr_yolo.experiments.factor_repair import (
            run_calibration_validation,
            semantic_calibration_phase,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        phase = semantic_calibration_phase(
            self.model,
            variant="F3",
            epochs=30,
            optimizer=optimizer,
        )
        parameter_bytes = {
            name: parameter.detach().cpu().numpy().tobytes()
            for name, parameter in self.model.named_parameters()
        }
        state_before = {
            id(parameter): {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in state.items()
            }
            for parameter, state in optimizer.state.items()
        }
        step_before = getattr(optimizer, "_step_count", 0)
        with torch.no_grad():
            output = run_calibration_validation(
                self.model,
                torch.zeros(1, 3, 128, 128),
                optimizer=optimizer,
            )
        self.assertIsNotNone(output)
        self.assertEqual(
            parameter_bytes,
            {
                name: parameter.detach().cpu().numpy().tobytes()
                for name, parameter in self.model.named_parameters()
            },
        )
        self.assertEqual(getattr(optimizer, "_step_count", 0), step_before)
        self.assertEqual(
            set(state_before),
            {id(parameter) for parameter in optimizer.state},
        )
        self.assertEqual(phase.epochs, 30)

    def test_legacy_gradient_diagnostic_groups_remain_unchanged(self) -> None:
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        unprotected = self.model.gradient_diagnostic_parameter_groups()
        self.assertEqual(set(unprotected), {"semantic_anchor"})
        protected = IFDRDetectionModel(
            str(MODEL_PATH),
            verbose=False,
            semantic_protection=True,
        )
        groups = protected.gradient_diagnostic_parameter_groups()
        self.assertEqual(
            set(groups),
            {"semantic_anchor", "fusion_adapters", "localization_adapter"},
        )

    def test_trainer_phase_entry_applies_phase_without_optimizer_step(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            apply_semantic_calibration_phase,
            run_validation_without_optimizer_step,
        )

        trainer = type(
            "Recorder",
            (),
            {"model": self.model, "optimizer": None},
        )()
        phase = apply_semantic_calibration_phase(
            trainer,
            variant="F1",
            epochs=30,
        )
        self.assertEqual(trainer.semantic_calibration_phase, phase)
        self.assertEqual(phase.variant, "F1")
        output = run_validation_without_optimizer_step(
            trainer,
            torch.zeros(1, 3, 128, 128),
        )
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
