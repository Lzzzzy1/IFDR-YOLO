from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import torch

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kitti-p2-m.yaml"
bootstrap_ultralytics_config(ROOT)


class _TaskToyModel(torch.nn.Module):
    def __init__(self, *, rogue: bool = False) -> None:
        super().__init__()
        self.model = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False) for _ in range(15)]
        )
        self.semantic_modules = tuple(self.model[:14])
        if rogue:
            self.model[14].register_parameter(
                "unregistered",
                torch.nn.Parameter(torch.ones(1)),
            )

    def factor_semantic_named_parameters(self):
        return tuple(
            (f"model.{index}.weight", self.model[index].weight)
            for index in range(14)
        )

    def factor_semantic_modules(self):
        return self.semantic_modules

    def task_adaptation_named_parameters(self):
        return (("model.14.weight", self.model[14].weight),)


def _task_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, object]:
    torch.save({"state_dict": model.state_dict()}, path)
    return {
        "condition": "F0",
        "checkpoint_path": path.resolve().as_posix(),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _build_task_phase(
    directory: str,
    *,
    condition: str = "F0",
    model: torch.nn.Module | None = None,
    eta_schedule: tuple[float, ...] = (1.0,) * 60,
):
    from ifdr_yolo.experiments.factor_repair import task_adaptation_phase

    model = model or _TaskToyModel()
    checkpoint = Path(directory) / f"{condition}-calibration_last.pt"
    provenance = _task_checkpoint(model, checkpoint)
    provenance["condition"] = condition
    return model, task_adaptation_phase(
        model,
        condition=condition,
        calibration_checkpoint_path=checkpoint,
        calibration_provenance=provenance,
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.0},
        eta_schedule=eta_schedule,
    )


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

    def test_semantic_phase_model_can_be_ema_deepcopied(self) -> None:
        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model = IFDRDetectionModel(str(MODEL_PATH), verbose=False)
        semantic_calibration_phase(model, variant="F0", epochs=30)
        clone = copy.deepcopy(model)
        self.assertIsNot(clone, model)
        self.assertIsNot(clone._semantic_calibration_phase, model._semantic_calibration_phase)

    def test_semantic_phase_checkpoint_roundtrip_preserves_immutable_audit_data(self) -> None:
        from ifdr_yolo.experiments.factor_repair import SemanticCalibrationPhase

        phase = SemanticCalibrationPhase(
            variant="F0",
            epochs=30,
            trainable_parameter_names=("model.projection.weight",),
            frozen_parameter_names=("model.detect.weight",),
            loss_mask={"synthetic": 1.0, "natural": 0.0, "specificity": 0.0},
            diagnostic_group_names=("projection_00",),
            diagnostic_group_provenance={
                "projection_00": ("model.projection.weight",),
            },
            provenance={
                "nested": {
                    "items": ["model.projection.weight"],
                    "tuple": ("F0", 30),
                }
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "semantic-phase.pt"
            torch.save({"phase": phase}, checkpoint)
            restored = torch.load(checkpoint, map_location="cpu", weights_only=False)[
                "phase"
            ]

        self.assertEqual(restored, phase)
        self.assertEqual(type(restored.loss_mask), type(phase.loss_mask))
        self.assertEqual(type(restored.provenance), type(phase.provenance))
        self.assertEqual(
            type(restored.provenance["nested"]),
            type(phase.provenance["nested"]),
        )
        with self.assertRaises(TypeError):
            restored.loss_mask["synthetic"] = 0.0
        with self.assertRaises(TypeError):
            restored.provenance["nested"]["items"] = ()

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

    def test_task_adaptation_phase_matches_f0_and_candidate(self) -> None:
        """F0 and a selected repair adapt independently from own calibration bytes."""

        from ifdr_yolo.experiments.factor_repair import (
            TaskAdaptationPhase,
            enforce_semantic_eval_mode,
            semantic_module_ids,
            semantic_state_sha256,
            task_adaptation_phase,
        )

        class ToyTaskModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Twelve independent projections plus shared semantic modules.
                self.model = torch.nn.ModuleList(
                    [torch.nn.Linear(1, 1, bias=False) for _ in range(15)]
                )
                self.semantic_modules = tuple(self.model[:14])

            def factor_semantic_named_parameters(self):
                return tuple(
                    (f"model.{index}.weight", self.model[index].weight)
                    for index in range(14)
                )

            def factor_semantic_modules(self):
                return self.semantic_modules

            def task_adaptation_named_parameters(self):
                return (("model.14.weight", self.model[14].weight),)

        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(17)
            initial = ToyTaskModel()
            f0 = ToyTaskModel()
            candidate = ToyTaskModel()
            f0.load_state_dict(initial.state_dict())
            candidate.load_state_dict(initial.state_dict())
            with torch.no_grad():
                candidate.model[0].weight.add_(1.0)

            f0_checkpoint = Path(directory) / "f0-calibration_last.pt"
            candidate_checkpoint = Path(directory) / "candidate-calibration_last.pt"
            f0_provenance = _task_checkpoint(f0, f0_checkpoint)
            candidate_provenance = _task_checkpoint(candidate, candidate_checkpoint)
            candidate_provenance["condition"] = "F3"

            kwargs = {
                "optimizer_name": "SGD",
                "optimizer_hparams": {"lr": 0.1, "momentum": 0.0},
                "eta_schedule": (0.1,) * 60,
            }
            f0_phase = task_adaptation_phase(
                f0,
                condition="F0",
                calibration_checkpoint_path=f0_checkpoint,
                calibration_provenance=f0_provenance,
                **kwargs,
            )
            candidate_phase = task_adaptation_phase(
                candidate,
                condition="F3",
                calibration_checkpoint_path=candidate_checkpoint,
                calibration_provenance=candidate_provenance,
                **kwargs,
            )

            self.assertIsInstance(f0_phase, TaskAdaptationPhase)
            self.assertEqual(f0_phase.frozen_parameter_names, candidate_phase.frozen_parameter_names)
            self.assertEqual(len(f0_phase.frozen_parameter_names), 14)
            self.assertEqual(f0_phase.trainable_parameter_names, ("model.14.weight",))
            self.assertEqual(f0_phase.trainable_parameter_names, candidate_phase.trainable_parameter_names)
            self.assertNotEqual(f0_phase.semantic_state_sha256, candidate_phase.semantic_state_sha256)
            self.assertEqual(f0_phase.epochs, candidate_phase.epochs, 60)
            self.assertEqual(f0_phase.update_count, candidate_phase.update_count)
            self.assertEqual(f0_phase.eta_schedule, candidate_phase.eta_schedule)
            self.assertFalse(f0_phase.early_stopping)
            self.assertEqual(f0_phase.primary_checkpoint, "last.pt")
            self.assertEqual(f0_phase.optimizer.__class__, candidate_phase.optimizer.__class__)
            self.assertEqual(f0_phase.optimizer.defaults, candidate_phase.optimizer.defaults)
            self.assertFalse(f0_phase.optimizer.state)
            self.assertFalse(candidate_phase.optimizer.state)
            with self.assertRaises(AttributeError):
                f0_phase.epochs = 3
            with self.assertRaises(TypeError):
                f0_phase.optimizer_hparams["lr"] = 0.2
            self.assertEqual(
                f0_phase.calibration_checkpoint_path,
                f0_checkpoint.resolve(),
            )
            self.assertEqual(
                f0_phase.calibration_checkpoint_sha256,
                hashlib.sha256(f0_checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                f0_phase.provenance["condition"],
                "F0",
            )

            # Task-path updates must leave each condition's own semantic bytes frozen.
            for model, phase in ((f0, f0_phase), (candidate, candidate_phase)):
                semantic_ids = semantic_module_ids(model)
                for _ in range(2):
                    model.train()
                    enforce_semantic_eval_mode(model, semantic_ids)
                    phase.optimizer.zero_grad()
                    model.model[14].weight.sum().backward()
                    phase.optimizer.step()
                    self.assertEqual(
                        semantic_state_sha256(model, semantic_ids),
                        phase.semantic_state_sha256,
                    )
                self.assertTrue(all(not model.model[index].training for index in range(14)))

            # The trainer boundary owns the journal and resumes only its own
            # condition/provenance checkpoint.
            from ifdr_yolo.experiments.ifdr_trainer import (
                apply_task_adaptation_phase,
                resume_task_adaptation_phase,
                task_adaptation_epoch_commit,
                task_adaptation_epoch_start,
                task_adaptation_final_checkpoint,
            )

            trainer = SimpleNamespace(
                model=f0,
                epoch=0,
                task_adaptation_optimizer_steps=0,
            )
            apply_task_adaptation_phase(trainer, phase=f0_phase)
            task_adaptation_epoch_start(trainer)
            trainer.task_adaptation_optimizer_steps = f0_phase.updates_per_epoch
            task_adaptation_epoch_commit(trainer)
            task_checkpoint = Path(directory) / "last.pt"
            task_adaptation_final_checkpoint(trainer, task_checkpoint, epoch=0)
            payload = torch.load(task_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(
                payload["task_adaptation_provenance"]["task_checkpoint_path"],
                task_checkpoint.resolve().as_posix(),
            )
            with torch.no_grad():
                f0.model[14].weight.add_(99.0)
            resume_task_adaptation_phase(trainer, task_checkpoint)
            self.assertEqual(
                semantic_state_sha256(f0, semantic_module_ids(f0)),
                f0_phase.semantic_state_sha256,
            )
            self.assertFalse(
                any(
                    entry["event"] == "final_checkpoint"
                    for entry in f0_phase.semantic_state_journal
                )
            )
            candidate_trainer = SimpleNamespace(model=candidate)
            apply_task_adaptation_phase(candidate_trainer, phase=candidate_phase)
            with self.assertRaises((RuntimeError, ValueError)):
                resume_task_adaptation_phase(candidate_trainer, task_checkpoint)

            # Hashing must preserve raw bytes for dtypes NumPy cannot convert
            # directly (bfloat16 is the regression case).
            bfloat16_model = ToyTaskModel().to(dtype=torch.bfloat16)
            bfloat16_ids = semantic_module_ids(bfloat16_model)
            before_bfloat16 = semantic_state_sha256(bfloat16_model, bfloat16_ids)
            with torch.no_grad():
                bfloat16_model.model[0].weight.add_(1)
            self.assertNotEqual(
                before_bfloat16,
                semantic_state_sha256(bfloat16_model, bfloat16_ids),
            )

    def test_setup_train_rebuilds_phase_from_final_parent_model(self) -> None:
        """The parent setup may replace lifecycle state; adaptation must rebind it."""

        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        with tempfile.TemporaryDirectory() as directory:
            original_model, phase = _build_task_phase(directory)
            final_model = _TaskToyModel()
            parent_optimizer = torch.optim.SGD(final_model.parameters(), lr=0.9)
            parent_scheduler = object()

            def fake_parent_setup(trainer):
                trainer.model = final_model
                trainer.optimizer = parent_optimizer
                trainer.scheduler = parent_scheduler
                trainer.epochs = 3
                trainer.args = SimpleNamespace(epochs=3, time=2.0, patience=4)
                trainer.stopper = SimpleNamespace(patience=4)

            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = original_model
            trainer.task_adaptation_phase = phase
            with patch(
                "ultralytics.engine.trainer.BaseTrainer._setup_train",
                new=fake_parent_setup,
            ):
                trainer._setup_train()

            self.assertIs(trainer.model, final_model)
            rebuilt = trainer.task_adaptation_phase
            self.assertIsNot(rebuilt, phase)
            self.assertEqual(rebuilt.condition, phase.condition)
            self.assertEqual(
                rebuilt.calibration_checkpoint_path,
                phase.calibration_checkpoint_path,
            )
            self.assertEqual(
                rebuilt.calibration_checkpoint_sha256,
                phase.calibration_checkpoint_sha256,
            )
            expected_ids = {
                id(dict(final_model.named_parameters())[name])
                for name in rebuilt.trainable_parameter_names
            }
            optimizer_ids = {
                id(parameter)
                for group in trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            self.assertEqual(optimizer_ids, expected_ids)
            self.assertIsNot(trainer.optimizer, parent_optimizer)
            self.assertIsNot(trainer.scheduler, parent_scheduler)
            self.assertEqual(trainer.epochs, 60)
            self.assertEqual(trainer.args.epochs, 60)
            self.assertIsNone(trainer.args.time)
            self.assertTrue(torch.isinf(torch.tensor(trainer.stopper.patience)))

    def test_setup_train_resumes_task_checkpoint_without_pre_attached_phase(self) -> None:
        """A real resume path must restore task progress after parent setup."""

        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRDetectionTrainer,
            _lossless_state_sha256,
        )

        with tempfile.TemporaryDirectory() as directory:
            task_model, phase = _build_task_phase(directory)
            phase.optimizer.zero_grad()
            task_model.model[14].weight.sum().backward()
            phase.optimizer.step()
            phase.semantic_state_journal.extend(
                record
                for epoch in range(12)
                for record in (
                    {
                        "event": "resume_check",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": epoch,
                        "optimizer_steps": epoch,
                    },
                    {
                        "event": "epoch_commit",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": epoch,
                        "optimizer_steps": epoch + 1,
                    },
                )
            )
            task_checkpoint = Path(directory) / "last.pt"
            task_state = {
                name: value.detach().cpu().clone()
                for name, value in task_model.state_dict().items()
            }
            provenance = {
                "condition": phase.condition,
                "checkpoint_role": phase.calibration_checkpoint_role,
                "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
                "checkpoint_sha256": phase.calibration_checkpoint_sha256,
                "semantic_state_sha256": phase.semantic_state_sha256,
                "semantic_module_names": tuple(phase.semantic_module_names),
                "task_parameter_categories": dict(phase.task_parameter_categories),
                "updates_per_epoch": phase.updates_per_epoch,
                "expected_optimizer_steps": phase.update_count,
                "optimizer_steps": 12,
                "eta_schedule": tuple(phase.eta_schedule),
                "primary_checkpoint": phase.primary_checkpoint,
                "task_checkpoint_path": task_checkpoint.resolve().as_posix(),
                "semantic_state_journal": tuple(phase.semantic_state_journal),
                "optimizer_name": phase.optimizer_name,
                "optimizer_hparams": dict(phase.optimizer_hparams),
                "optimizer_defaults": dict(phase.optimizer.defaults),
                "task_state_key": "task_adaptation_state_dict",
                "task_state_source": "live_model_state_dict",
                "task_state_sha256": _lossless_state_sha256(task_state),
                "epoch": 11,
            }
            torch.save(
                {
                    "state_dict": task_model.state_dict(),
                    "task_adaptation_state_dict": task_state,
                    "optimizer": phase.optimizer.state_dict(),
                    "task_adaptation_provenance": provenance,
                },
                task_checkpoint,
            )
            expected_task_weight = task_model.model[14].weight.detach().clone()

            final_model = _TaskToyModel()
            parent_optimizer = torch.optim.SGD(final_model.parameters(), lr=0.9)
            parent_scheduler = object()

            def fake_parent_setup(trainer):
                # Simulate BaseTrainer.setup_model/resume_training: the parent has
                # already loaded the task checkpoint before the child hook runs.
                final_model.load_state_dict(task_model.state_dict())
                trainer.model = final_model
                trainer.optimizer = parent_optimizer
                trainer.scheduler = parent_scheduler
                trainer.args = SimpleNamespace(
                    epochs=60,
                    time=2.0,
                    patience=4,
                    resume=task_checkpoint.as_posix(),
                )
                trainer.resume = True
                trainer.start_epoch = 12
                trainer.stopper = SimpleNamespace(patience=4)

            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = _TaskToyModel()
            trainer.task_adaptation_phase = None
            trainer.args = SimpleNamespace(resume=task_checkpoint.as_posix())
            trainer.resume = True
            with patch(
                "ultralytics.engine.trainer.BaseTrainer._setup_train",
                new=fake_parent_setup,
            ):
                trainer._setup_train()

            resumed = trainer.task_adaptation_phase
            self.assertIsNotNone(resumed)
            self.assertEqual(resumed.condition, "F0")
            self.assertIsNot(trainer.optimizer, parent_optimizer)
            self.assertEqual(trainer.task_adaptation_optimizer_steps, 12)
            self.assertEqual(trainer.task_adaptation_start_epoch, 12)
            self.assertEqual(trainer.scheduler.last_epoch, 12)
            self.assertTrue(
                torch.equal(final_model.model[14].weight.detach(), expected_task_weight)
            )
            self.assertEqual(
                resumed.semantic_state_sha256,
                phase.semantic_state_sha256,
            )

    def test_resume_phase_rebuild_rolls_back_model_on_post_load_failure(self) -> None:
        """Resume phase reconstruction is transactional across lossless model loading."""

        from ifdr_yolo.experiments.ifdr_trainer import (
            _task_phase_from_resume_provenance,
            task_adaptation_epoch_commit,
            task_adaptation_epoch_start,
            task_adaptation_final_checkpoint,
        )

        with tempfile.TemporaryDirectory() as directory:
            source_model, phase = _build_task_phase(directory)
            writer = SimpleNamespace(
                model=source_model,
                task_adaptation_phase=phase,
                task_adaptation_optimizer_steps=0,
                epoch=0,
            )
            task_adaptation_epoch_start(writer)
            writer.task_adaptation_optimizer_steps = phase.updates_per_epoch
            task_adaptation_epoch_commit(writer)
            checkpoint = Path(directory) / "last.pt"
            task_adaptation_final_checkpoint(writer, checkpoint, epoch=0)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            provenance = payload["task_adaptation_provenance"]

            resumed_model = _TaskToyModel()
            with torch.no_grad():
                resumed_model.model[14].weight.add_(7.0)
            for index, (_, parameter) in enumerate(resumed_model.named_parameters()):
                parameter.requires_grad = index % 2 == 0
            before_state = {
                name: value.detach().clone()
                for name, value in resumed_model.state_dict().items()
            }
            before_requires_grad = {
                name: parameter.requires_grad
                for name, parameter in resumed_model.named_parameters()
            }

            with patch(
                "ifdr_yolo.experiments.factor_repair.task_adaptation_phase",
                side_effect=RuntimeError("post-load phase failure"),
            ):
                with self.assertRaises(ValueError):
                    _task_phase_from_resume_provenance(
                        SimpleNamespace(model=resumed_model),
                        checkpoint.resolve(),
                        provenance,
                        payload,
                    )

            for name, value in resumed_model.state_dict().items():
                self.assertTrue(torch.equal(value, before_state[name]))
            self.assertEqual(
                before_requires_grad,
                {
                    name: parameter.requires_grad
                    for name, parameter in resumed_model.named_parameters()
                },
            )

    def test_setup_train_keeps_ordinary_resume_without_task_provenance(self) -> None:
        """A normal Ultralytics checkpoint is not forced into Task6A."""

        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        with tempfile.TemporaryDirectory() as directory:
            ordinary_checkpoint = Path(directory) / "ordinary-last.pt"
            torch.save({"epoch": 3, "model": "ordinary"}, ordinary_checkpoint)
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = _TaskToyModel()
            trainer.task_adaptation_phase = None
            trainer.args = SimpleNamespace(resume=ordinary_checkpoint.as_posix())
            trainer.resume = True

            def fake_parent_setup(_trainer):
                _trainer.args = SimpleNamespace(
                    resume=ordinary_checkpoint.as_posix(),
                )

            with patch(
                "ultralytics.engine.trainer.BaseTrainer._setup_train",
                new=fake_parent_setup,
            ):
                result = trainer._setup_train()

            self.assertIsNone(result)
            self.assertIsNone(trainer.task_adaptation_phase)

    def test_real_ultralytics_setup_train_resumes_task_checkpoint(self) -> None:
        """The actual 8.4.98 parent lifecycle must not load a task optimizer into a full optimizer."""

        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRDetectionTrainer,
            _lossless_state_sha256,
        )

        class Dataset:
            def __len__(self):
                return 1

        class Loader:
            dataset = Dataset()

        class Validator:
            metrics = SimpleNamespace(keys=[])

        class FakeEMA:
            def __init__(self, model):
                self.ema = copy.deepcopy(model)
                with torch.no_grad():
                    self.ema.model[14].weight.zero_()
                self.updates = 0

        with tempfile.TemporaryDirectory() as directory:
            task_model, phase = _build_task_phase(directory)
            task_state = {
                name: value.detach().cpu().clone()
                for name, value in task_model.state_dict().items()
            }
            task_checkpoint = Path(directory) / "last.pt"
            provenance = {
                "condition": phase.condition,
                "checkpoint_role": phase.calibration_checkpoint_role,
                "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
                "checkpoint_sha256": phase.calibration_checkpoint_sha256,
                "semantic_state_sha256": phase.semantic_state_sha256,
                "semantic_module_names": tuple(phase.semantic_module_names),
                "task_parameter_categories": dict(phase.task_parameter_categories),
                "updates_per_epoch": phase.updates_per_epoch,
                "expected_optimizer_steps": phase.update_count,
                "optimizer_steps": 12,
                "primary_checkpoint": phase.primary_checkpoint,
                "task_checkpoint_path": task_checkpoint.resolve().as_posix(),
                "semantic_state_journal": tuple(
                    record
                    for epoch in range(12)
                    for record in (
                        {
                            "event": "resume_check",
                            "semantic_state_sha256": phase.semantic_state_sha256,
                            "epoch": epoch,
                            "optimizer_steps": epoch,
                        },
                        {
                            "event": "epoch_commit",
                            "semantic_state_sha256": phase.semantic_state_sha256,
                            "epoch": epoch,
                            "optimizer_steps": epoch + 1,
                        },
                    )
                ),
                "optimizer_name": phase.optimizer_name,
                "optimizer_hparams": dict(phase.optimizer_hparams),
                "optimizer_defaults": dict(phase.optimizer.defaults),
                "task_state_key": "task_adaptation_state_dict",
                "task_state_source": "live_model_state_dict",
                "task_state_sha256": _lossless_state_sha256(task_state),
                "eta_schedule": tuple(phase.eta_schedule),
                "epoch": 11,
            }
            checkpoint_payload = {
                "epoch": 11,
                "optimizer": phase.optimizer.state_dict(),
                "scaler": None,
                "ema": None,
                "best_fitness": 0.0,
                "state_dict": task_model.state_dict(),
                "task_adaptation_state_dict": task_state,
                "task_adaptation_provenance": provenance,
            }
            torch.save(checkpoint_payload, task_checkpoint)
            final_model = _TaskToyModel()
            final_model.load_state_dict(task_model.state_dict())
            final_model.stride = torch.tensor([32])
            parent_optimizer = torch.optim.SGD(final_model.parameters(), lr=0.9)
            args = SimpleNamespace(
                resume=task_checkpoint.as_posix(),
                model=task_checkpoint.as_posix(),
                data={},
                task="detect",
                distill_model=None,
                compile=False,
                freeze=[],
                amp=False,
                imgsz=32,
                batch=1,
                nbs=1,
                weight_decay=0.0,
                optimizer="SGD",
                lr0=0.9,
                momentum=0.0,
                workers=0,
                close_mosaic=0,
                plots=False,
                patience=4,
                time=None,
                save_period=-1,
                cos_lr=False,
                lrf=0.01,
            )
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = final_model
            trainer.task_adaptation_phase = None
            trainer.args = args
            trainer.resume = True
            trainer.device = torch.device("cpu")
            trainer.world_size = 1
            trainer.batch_size = 1
            trainer.epochs = 60
            trainer.data = {
                "train": "train",
                "val": "val",
                "nc": 1,
                "names": {0: "object"},
                "channels": 3,
            }
            trainer.loss_names = ("box_loss",)
            trainer.save_dir = Path(directory)
            trainer.setup_model = lambda: checkpoint_payload
            trainer.set_model_attributes = lambda: None
            trainer.get_dataloader = lambda *args, **kwargs: Loader()
            trainer.build_optimizer = lambda **kwargs: parent_optimizer
            trainer.get_validator = lambda: Validator()
            trainer.set_class_weights = lambda: None
            trainer.run_callbacks = lambda *args, **kwargs: None

            with patch("ultralytics.engine.trainer.ModelEMA", new=FakeEMA):
                trainer._setup_train()

            self.assertIsNotNone(trainer.task_adaptation_phase)
            self.assertEqual(trainer.start_epoch, 12)
            self.assertEqual(trainer.task_adaptation_optimizer_steps, 12)
            self.assertIsNot(trainer.optimizer, parent_optimizer)

    def test_real_ultralytics_save_model_preserves_last_on_publish_failure(self) -> None:
        """The actual BaseTrainer.save_model write is transactional for Task6A last.pt."""

        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        class FakeEMA:
            def __init__(self, model):
                self.ema = copy.deepcopy(model)
                self.updates = 0

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.task_adaptation_phase = phase
            trainer.last = Path(directory) / "last.pt"
            trainer.best = Path(directory) / "best.pt"
            trainer.wdir = Path(directory)
            trainer.optimizer = phase.optimizer
            trainer.ema = FakeEMA(model)
            trainer.scaler = SimpleNamespace(state_dict=lambda: {})
            trainer.args = SimpleNamespace()
            trainer.metrics = {}
            trainer.best_fitness = 0.0
            trainer.fitness = 0.0
            trainer.epoch = 4
            trainer.save_period = -1
            trainer.task_adaptation_optimizer_steps = 0
            phase.semantic_state_journal.extend(
                record
                for epoch in range(5)
                for record in (
                    {
                        "event": "resume_check",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": epoch,
                        "optimizer_steps": epoch,
                    },
                    {
                        "event": "epoch_commit",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": epoch,
                        "optimizer_steps": epoch + 1,
                    },
                )
            )
            trainer.read_results_csv = lambda: ""
            sentinel = b"previous-validated-last"
            trainer.last.write_bytes(sentinel)

            with patch(
                "ifdr_yolo.experiments.ifdr_trainer._atomic_torch_save",
                side_effect=OSError("injected publish failure"),
            ):
                with self.assertRaises(OSError):
                    trainer.save_model()

            self.assertEqual(trainer.last.read_bytes(), sentinel)
            self.assertEqual(tuple(trainer.last.parent.glob(".last.pt.*.tmp")), ())

    def test_real_save_rejects_journal_gap_before_publishing_last(self) -> None:
        """A malformed final journal cannot replace the prior validated last.pt."""

        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        class FakeEMA:
            def __init__(self, model):
                self.ema = copy.deepcopy(model)
                self.updates = 0

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.task_adaptation_phase = phase
            trainer.last = Path(directory) / "last.pt"
            trainer.best = Path(directory) / "best.pt"
            trainer.wdir = Path(directory)
            trainer.optimizer = phase.optimizer
            trainer.ema = FakeEMA(model)
            trainer.scaler = SimpleNamespace(state_dict=lambda: {})
            trainer.args = SimpleNamespace()
            trainer.metrics = {}
            trainer.best_fitness = -1.0
            trainer.fitness = 0.0
            trainer.epoch = phase.epochs - 1
            trainer.save_period = -1
            trainer.task_adaptation_optimizer_steps = phase.update_count
            trainer.read_results_csv = lambda: ""
            phase.semantic_state_journal.extend(
                [
                    {
                        "event": "resume_check",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": 0,
                        "optimizer_steps": 0,
                    },
                    {
                        "event": "epoch_commit",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": 0,
                        "optimizer_steps": 1,
                    },
                ]
            )
            sentinel = b"previous-validated-last"
            trainer.last.write_bytes(sentinel)

            with self.assertRaises(ValueError):
                trainer.save_model()

            self.assertEqual(trainer.last.read_bytes(), sentinel)
            self.assertEqual(tuple(trainer.last.parent.glob(".last.pt.*.tmp")), ())

    def test_real_save_and_resume_roundtrip_keeps_full_precision_task_state(self) -> None:
        """A real EMA-half save must resume from the exact live Task6A state."""

        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        class FakeEMA:
            def __init__(self, model):
                self.ema = copy.deepcopy(model)
                self.updates = 0

        class Dataset:
            def __len__(self):
                return 1

        class Loader:
            dataset = Dataset()

        class Validator:
            metrics = SimpleNamespace(keys=[])

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            with torch.no_grad():
                model.model[14].weight.fill_(0.1234567)
            phase.semantic_state_journal.extend(
                record
                for epoch in range(5)
                for record in (
                    {
                        "event": "resume_check",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": epoch,
                        "optimizer_steps": epoch,
                    },
                    {
                        "event": "epoch_commit",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": epoch,
                        "optimizer_steps": epoch + 1,
                    },
                )
            )
            saver = object.__new__(IFDRDetectionTrainer)
            saver.model = model
            saver.task_adaptation_phase = phase
            saver.last = Path(directory) / "last.pt"
            saver.best = Path(directory) / "best.pt"
            saver.wdir = Path(directory)
            saver.optimizer = phase.optimizer
            saver.ema = FakeEMA(model)
            saver.scaler = SimpleNamespace(state_dict=lambda: {})
            saver.args = SimpleNamespace()
            saver.metrics = {}
            saver.best_fitness = -1.0
            saver.fitness = 0.0
            saver.epoch = 4
            saver.save_period = -1
            saver.task_adaptation_optimizer_steps = 5
            saver.read_results_csv = lambda: ""
            saver.save_model()
            checkpoint = torch.load(
                saver.last,
                map_location="cpu",
                weights_only=False,
            )
            resumed_model = _TaskToyModel()
            resumed_model.load_state_dict(checkpoint["ema"].float().state_dict())
            resumed_model.stride = torch.tensor([32])
            parent_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.9)
            args = SimpleNamespace(
                resume=saver.last.as_posix(),
                model=saver.last.as_posix(),
                data={},
                task="detect",
                distill_model=None,
                compile=False,
                freeze=[],
                amp=False,
                imgsz=32,
                batch=1,
                nbs=1,
                weight_decay=0.0,
                optimizer="SGD",
                lr0=0.9,
                momentum=0.0,
                workers=0,
                close_mosaic=0,
                plots=False,
                patience=4,
                time=None,
                save_period=-1,
                cos_lr=False,
                lrf=0.01,
            )
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = resumed_model
            trainer.task_adaptation_phase = None
            trainer.args = args
            trainer.resume = True
            trainer.device = torch.device("cpu")
            trainer.world_size = 1
            trainer.batch_size = 1
            trainer.epochs = 60
            trainer.data = {
                "train": "train",
                "val": "val",
                "nc": 1,
                "names": {0: "object"},
                "channels": 3,
            }
            trainer.loss_names = ("box_loss",)
            trainer.save_dir = Path(directory)
            trainer.setup_model = lambda: checkpoint
            trainer.set_model_attributes = lambda: None
            trainer.get_dataloader = lambda *args, **kwargs: Loader()
            trainer.build_optimizer = lambda **kwargs: parent_optimizer
            trainer.get_validator = lambda: Validator()
            trainer.set_class_weights = lambda: None
            trainer.run_callbacks = lambda *args, **kwargs: None

            with patch("ultralytics.engine.trainer.ModelEMA", new=FakeEMA):
                trainer._setup_train()

            self.assertTrue(
                torch.equal(
                    resumed_model.model[14].weight,
                    model.model[14].weight,
                )
            )

    def test_setup_train_syncs_ema_after_fresh_calibration_rebuild(self) -> None:
        """Fresh calibration must not leave BaseTrainer's EMA at the old weights."""

        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        with tempfile.TemporaryDirectory() as directory:
            original_model, phase = _build_task_phase(directory)
            final_model = _TaskToyModel()
            stale_ema = _TaskToyModel()

            def fake_parent_setup(trainer):
                trainer.model = final_model
                trainer.optimizer = torch.optim.SGD(final_model.parameters(), lr=0.9)
                trainer.scheduler = object()
                trainer.ema = SimpleNamespace(ema=stale_ema, updates=17)
                trainer.args = SimpleNamespace(epochs=3, time=2.0, patience=4)
                trainer.stopper = SimpleNamespace(patience=4)

            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = original_model
            trainer.task_adaptation_phase = phase
            with patch(
                "ultralytics.engine.trainer.BaseTrainer._setup_train",
                new=fake_parent_setup,
            ):
                trainer._setup_train()

            self.assertEqual(
                stale_ema.state_dict(),
                final_model.state_dict(),
            )

    def test_final_eval_rejects_intermediate_task_checkpoint(self) -> None:
        """A non-final or under-budget last.pt cannot be reported as Task6A final."""

        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            phase.semantic_state_journal.extend(
                [
                    {
                        "event": "resume_check",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                    },
                    {
                        "event": "epoch_commit",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                    },
                ]
            )
            last = Path(directory) / "last.pt"
            provenance = {
                "condition": phase.condition,
                "checkpoint_role": phase.calibration_checkpoint_role,
                "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
                "checkpoint_sha256": phase.calibration_checkpoint_sha256,
                "semantic_state_sha256": phase.semantic_state_sha256,
                "primary_checkpoint": "last.pt",
                "task_checkpoint_path": last.resolve().as_posix(),
                "updates_per_epoch": phase.updates_per_epoch,
                "expected_optimizer_steps": phase.update_count,
                "optimizer_steps": 1,
                "epoch": 4,
                "semantic_state_journal": tuple(phase.semantic_state_journal),
            }
            torch.save({"task_adaptation_provenance": provenance}, last)
            class Validator:
                args = SimpleNamespace(plots=False, compile=False)

                def __call__(self, *, model):
                    return {"fitness": 0.0}

            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.task_adaptation_phase = phase
            trainer.last = last
            trainer.validator = Validator()
            trainer.args = SimpleNamespace(plots=False)
            trainer.metrics = {}
            trainer.epoch = 4
            trainer.run_callbacks = lambda *args, **kwargs: None
            with self.assertRaises(ValueError):
                trainer.final_eval()

    def test_resume_rejects_bad_progress_before_mutating_task_state(self) -> None:
        """Malformed progress metadata must not leave model or optimizer half-restored."""

        from ifdr_yolo.experiments.factor_repair import task_adaptation_phase
        from ifdr_yolo.experiments.ifdr_trainer import (
            apply_task_adaptation_phase,
            resume_task_adaptation_phase,
        )

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = SimpleNamespace(model=model)
            apply_task_adaptation_phase(trainer, phase=phase)
            phase.semantic_state_journal.append(
                {
                    "event": "resume_check",
                    "semantic_state_sha256": phase.semantic_state_sha256,
                }
            )
            checkpoint = Path(directory) / "last.pt"
            state_dict = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            state_dict["model.14.weight"] = state_dict["model.14.weight"] + 5.0
            provenance = {
                "condition": phase.condition,
                "checkpoint_role": phase.calibration_checkpoint_role,
                "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
                "checkpoint_sha256": phase.calibration_checkpoint_sha256,
                "semantic_state_sha256": phase.semantic_state_sha256,
                "semantic_module_names": tuple(phase.semantic_module_names),
                "task_parameter_categories": dict(phase.task_parameter_categories),
                "updates_per_epoch": phase.updates_per_epoch,
                "expected_optimizer_steps": phase.update_count,
                "optimizer_steps": 1,
                "primary_checkpoint": "last.pt",
                "task_checkpoint_path": checkpoint.resolve().as_posix(),
                "semantic_state_journal": tuple(phase.semantic_state_journal),
                "optimizer_name": phase.optimizer_name,
                "optimizer_hparams": dict(phase.optimizer_hparams),
                "optimizer_defaults": dict(phase.optimizer.defaults),
                "eta_schedule": tuple(phase.eta_schedule),
                "epoch": "not-an-epoch",
            }
            torch.save(
                {
                    "state_dict": state_dict,
                    "optimizer": phase.optimizer.state_dict(),
                    "task_adaptation_provenance": provenance,
                },
                checkpoint,
            )
            before = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            before_optimizer = copy.deepcopy(phase.optimizer.state_dict())
            with self.assertRaises(ValueError):
                resume_task_adaptation_phase(trainer, checkpoint)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))
            self.assertEqual(phase.optimizer.state_dict(), before_optimizer)

    def test_adaptation_final_eval_uses_validated_last_even_when_best_exists(self) -> None:
        """Fixed-budget adaptation evaluates immutable last.pt, never best.pt."""

        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRDetectionTrainer,
            task_adaptation_epoch_commit,
            task_adaptation_epoch_start,
            task_adaptation_final_checkpoint,
        )

        class Validator:
            def __init__(self):
                self.args = SimpleNamespace(plots=True, compile=True)
                self.calls = []

            def __call__(self, *, model):
                self.calls.append(model)
                return {"fitness": 0.4, "metrics/mAP50(B)": 0.2}

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            writer = SimpleNamespace(
                model=model,
                task_adaptation_phase=phase,
                task_adaptation_optimizer_steps=0,
                epoch=0,
            )
            for epoch in range(phase.epochs):
                writer.epoch = epoch
                writer.task_adaptation_optimizer_steps = epoch * phase.updates_per_epoch
                task_adaptation_epoch_start(writer)
                writer.task_adaptation_optimizer_steps = (epoch + 1) * phase.updates_per_epoch
                task_adaptation_epoch_commit(writer)
            last = Path(directory) / "last.pt"
            best = Path(directory) / "best.pt"
            writer.epoch = phase.epochs - 1
            writer.task_adaptation_optimizer_steps = phase.update_count
            task_adaptation_final_checkpoint(writer, last, epoch=phase.epochs - 1)
            best.write_bytes(b"diagnostic-best")
            last_before = last.read_bytes()
            best_before = best.read_bytes()
            validator = Validator()
            events = []
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.task_adaptation_phase = phase
            trainer.last = last
            trainer.best = best
            trainer.validator = validator
            trainer.args = SimpleNamespace(plots=False)
            trainer.metrics = {}
            trainer.epoch = 6
            trainer.run_callbacks = lambda event: events.append(event)

            trainer.final_eval()

            self.assertEqual(len(validator.calls), 1)
            self.assertIsNot(validator.calls[0], model)
            self.assertTrue(
                all(
                    torch.equal(value, model.state_dict()[name])
                    for name, value in validator.calls[0].state_dict().items()
                )
            )
            self.assertEqual(last.read_bytes(), last_before)
            self.assertEqual(best.read_bytes(), best_before)
            self.assertEqual(trainer.metrics["metrics/mAP50(B)"], 0.2)
            self.assertEqual(events, ["on_fit_epoch_end"])
            self.assertEqual(trainer.epoch, 6)

    def test_final_eval_loads_one_immutable_payload_for_validation_and_model(self) -> None:
        """Final evaluation cannot switch payloads between validation and materialization."""

        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRDetectionTrainer,
            task_adaptation_epoch_commit,
            task_adaptation_epoch_start,
            task_adaptation_final_checkpoint,
        )

        class Validator:
            def __init__(self):
                self.args = SimpleNamespace(plots=False, compile=False)
                self.models = []

            def __call__(self, *, model):
                self.models.append(model)
                return {"fitness": 0.1}

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            writer = SimpleNamespace(
                model=model,
                task_adaptation_phase=phase,
                task_adaptation_optimizer_steps=0,
                epoch=0,
            )
            for epoch in range(phase.epochs):
                writer.epoch = epoch
                writer.task_adaptation_optimizer_steps = epoch * phase.updates_per_epoch
                task_adaptation_epoch_start(writer)
                writer.task_adaptation_optimizer_steps = (epoch + 1) * phase.updates_per_epoch
                task_adaptation_epoch_commit(writer)
            last = Path(directory) / "last.pt"
            writer.epoch = phase.epochs - 1
            writer.task_adaptation_optimizer_steps = phase.update_count
            task_adaptation_final_checkpoint(writer, last, epoch=phase.epochs - 1)
            valid_payload = torch.load(last, map_location="cpu", weights_only=False)
            calls = []

            def load_once(*args, **kwargs):
                calls.append(args[0])
                if len(calls) == 1:
                    return valid_payload
                corrupted = copy.deepcopy(valid_payload)
                semantic_name = phase.frozen_parameter_names[0]
                corrupted["task_adaptation_state_dict"][semantic_name].add_(1)
                return corrupted

            validator = Validator()
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.task_adaptation_phase = phase
            trainer.last = last
            trainer.validator = validator
            trainer.args = SimpleNamespace(plots=False)
            trainer.metrics = {}
            trainer.epoch = 6
            trainer.run_callbacks = lambda *args, **kwargs: None
            with patch(
                "ifdr_yolo.experiments.ifdr_trainer.torch.load",
                side_effect=load_once,
            ):
                trainer.final_eval()

            self.assertEqual(len(calls), 1)
            self.assertEqual(len(validator.models), 1)

    def test_validated_last_rejects_lossless_state_and_recipe_tampering(self) -> None:
        """Final validation rejects incomplete state or any recipe drift."""

        from ifdr_yolo.experiments.ifdr_trainer import (
            _validated_adaptation_last,
            task_adaptation_epoch_commit,
            task_adaptation_epoch_start,
            task_adaptation_final_checkpoint,
        )

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            writer = SimpleNamespace(
                model=model,
                task_adaptation_phase=phase,
                task_adaptation_optimizer_steps=0,
                epoch=0,
            )
            for epoch in range(phase.epochs):
                writer.epoch = epoch
                writer.task_adaptation_optimizer_steps = epoch * phase.updates_per_epoch
                task_adaptation_epoch_start(writer)
                writer.task_adaptation_optimizer_steps = (epoch + 1) * phase.updates_per_epoch
                task_adaptation_epoch_commit(writer)
            last = Path(directory) / "last.pt"
            writer.epoch = phase.epochs - 1
            writer.task_adaptation_optimizer_steps = phase.update_count
            task_adaptation_final_checkpoint(writer, last, epoch=phase.epochs - 1)
            payload = torch.load(last, map_location="cpu", weights_only=False)
            trainer = SimpleNamespace(
                model=model,
                task_adaptation_phase=phase,
                last=last,
            )
            self.assertEqual(_validated_adaptation_last(trainer, phase), last)

            state_key = "task_adaptation_state_dict"
            state_names = tuple(payload[state_key])
            semantic_name = phase.frozen_parameter_names[0]
            mutations = {
                "empty state": lambda item: item[state_key].clear(),
                "missing key": lambda item: item[state_key].pop(state_names[0]),
                "shape": lambda item: item[state_key].__setitem__(
                    state_names[0],
                    item[state_key][state_names[0]].reshape(-1),
                ),
                "dtype": lambda item: item[state_key].__setitem__(
                    state_names[0],
                    item[state_key][state_names[0]].half(),
                ),
                "semantic bytes": lambda item: item[state_key].__setitem__(
                    semantic_name,
                    item[state_key][semantic_name] + 1,
                ),
                "recipe": lambda item: item["task_adaptation_provenance"].update(
                    {"optimizer_hparams": {"lr": 0.2}}
                ),
            }
            for label, mutate in mutations.items():
                corrupted = copy.deepcopy(payload)
                mutate(corrupted)
                torch.save(corrupted, last)
                with self.assertRaises(ValueError, msg=label):
                    _validated_adaptation_last(trainer, phase)

    def test_real_model_train_order_reapplies_semantic_eval_and_hash(self) -> None:
        from ifdr_yolo.experiments.factor_repair import (
            semantic_module_ids,
            semantic_state_sha256,
        )
        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.freeze_layer_names = []
            trainer.task_adaptation_phase = phase
            before = semantic_state_sha256(model, semantic_module_ids(model))

            # BaseTrainer._model_train() calls model.train() after the epoch
            # callback; the override must reapply semantic eval afterwards.
            trainer._model_train()

            self.assertTrue(all(not module.training for module in model.semantic_modules))
            self.assertEqual(
                semantic_state_sha256(model, semantic_module_ids(model)),
                before,
            )

    def test_task_budget_controls_real_trainer_and_optimizer_step_count(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRDetectionTrainer,
            apply_task_adaptation_phase,
        )

        class Scaler:
            def unscale_(self, _optimizer):
                return None

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = SimpleNamespace(
                model=model,
                args=SimpleNamespace(epochs=3, time=2.0, patience=5),
                epochs=3,
                optimizer=phase.optimizer,
                scaler=Scaler(),
                ema=None,
            )
            trainer.stopper = SimpleNamespace(patience=5)
            apply_task_adaptation_phase(trainer, phase=phase)

            self.assertEqual(trainer.epochs, 60)
            self.assertEqual(trainer.args.epochs, 60)
            self.assertIsNone(trainer.args.time)
            self.assertEqual(trainer.args.patience, 0)
            self.assertTrue(torch.isinf(torch.tensor(trainer.stopper.patience)))
            self.assertEqual(len(phase.eta_schedule), 60)

            IFDRDetectionTrainer.optimizer_step(trainer)
            self.assertEqual(trainer.task_adaptation_optimizer_steps, 1)

    def test_task_eta_schedule_is_consumed_by_scheduler_step(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import apply_task_adaptation_phase

        schedule = (0.5, 0.25) + (0.25,) * 58
        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory, eta_schedule=schedule)
            trainer = SimpleNamespace(model=model)
            apply_task_adaptation_phase(trainer, phase=phase)
            self.assertAlmostEqual(
                trainer.optimizer.param_groups[0]["lr"],
                0.1 * schedule[0],
            )
            trainer.optimizer.step()
            trainer.scheduler.step()
            self.assertAlmostEqual(
                trainer.optimizer.param_groups[0]["lr"],
                0.1 * schedule[0],
            )

    def test_task_eta_schedule_uses_absolute_epoch_without_first_step_offset(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import apply_task_adaptation_phase

        schedule = tuple((index + 1) / 100.0 for index in range(60))
        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory, eta_schedule=schedule)
            trainer = SimpleNamespace(model=model, task_adaptation_start_epoch=0)
            apply_task_adaptation_phase(trainer, phase=phase)
            for epoch, multiplier in enumerate(schedule):
                trainer.optimizer.step()
                trainer.scheduler.step()
                self.assertAlmostEqual(
                    trainer.optimizer.param_groups[0]["lr"],
                    0.1 * multiplier,
                    msg=f"scheduler epoch {epoch}",
                )

    def test_calibration_provenance_is_required_and_compared_before_load(self) -> None:
        from ifdr_yolo.experiments.factor_repair import task_adaptation_phase

        with tempfile.TemporaryDirectory() as directory:
            model = _TaskToyModel()
            checkpoint = Path(directory) / "calibration_last.pt"
            provenance = _task_checkpoint(model, checkpoint)
            kwargs = {
                "condition": "F0",
                "calibration_checkpoint_path": checkpoint,
                "optimizer_name": "SGD",
                "optimizer_hparams": {"lr": 0.1},
                "eta_schedule": (1.0,) * 60,
            }
            with self.assertRaises(ValueError):
                task_adaptation_phase(model, **kwargs)
            with self.assertRaises(ValueError):
                task_adaptation_phase(
                    model,
                    calibration_provenance={
                        **provenance,
                        "checkpoint_sha256": "0" * 64,
                    },
                    **kwargs,
                )
            with self.assertRaises(ValueError):
                task_adaptation_phase(
                    model,
                    calibration_provenance={
                        **provenance,
                        "condition": "F3",
                    },
                    **kwargs,
                )

    def test_ultralytics_save_lifecycle_preserves_standard_checkpoint_fields(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.task_adaptation_phase = phase
            trainer.last = Path(directory) / "last.pt"
            trainer.wdir = Path(directory)
            trainer.optimizer = phase.optimizer
            trainer.task_adaptation_optimizer_steps = 0
            trainer.epoch = 4
            phase.semantic_state_journal.extend(
                [
                    {
                        "event": "resume_check",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": 4,
                        "optimizer_steps": 4,
                    },
                    {
                        "event": "epoch_commit",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": 4,
                        "optimizer_steps": 5,
                    },
                ]
            )

            def fake_parent_save(_trainer):
                torch.save(
                    {
                        "epoch": 4,
                        "model": "standard-model",
                        "ema": "standard-ema",
                        "optimizer": {"state": "standard"},
                    },
                    trainer.last,
                )
                return True

            with patch(
                "ultralytics.engine.trainer.BaseTrainer.save_model",
                new=fake_parent_save,
            ):
                result = trainer.save_model()

            self.assertTrue(result)
            payload = torch.load(trainer.last, map_location="cpu", weights_only=False)
            self.assertEqual(payload["model"], "standard-model")
            self.assertEqual(payload["ema"], "standard-ema")
            self.assertEqual(payload["task_adaptation_provenance"]["primary_checkpoint"], "last.pt")
            self.assertEqual(
                payload["task_adaptation_provenance"]["semantic_state_sha256"],
                phase.semantic_state_sha256,
            )
            self.assertNotIn(
                "final_checkpoint",
                [
                    record["event"]
                    for record in payload["task_adaptation_provenance"][
                        "semantic_state_journal"
                    ]
                ],
            )

    def test_standard_ema_checkpoint_roundtrip_resumes_into_fresh_model(self) -> None:
        from ifdr_yolo.experiments.factor_repair import task_adaptation_phase
        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRDetectionTrainer,
            apply_task_adaptation_phase,
            resume_task_adaptation_phase,
        )

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.task_adaptation_phase = phase
            trainer.last = Path(directory) / "last.pt"
            trainer.wdir = Path(directory)
            trainer.optimizer = phase.optimizer
            trainer.epoch = 4
            trainer.task_adaptation_optimizer_steps = 5
            phase.semantic_state_journal.extend(
                record
                for epoch in range(5)
                for record in (
                    {
                        "event": "resume_check",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": epoch,
                        "optimizer_steps": epoch,
                    },
                    {
                        "event": "epoch_commit",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": epoch,
                        "optimizer_steps": epoch + 1,
                    },
                )
            )

            def fake_parent_save(_trainer):
                torch.save(
                    {
                        "epoch": trainer.epoch,
                        "model": None,
                        "ema": copy.deepcopy(model),
                        "optimizer": phase.optimizer.state_dict(),
                    },
                    trainer.last,
                )
                return True

            with patch(
                "ultralytics.engine.trainer.BaseTrainer.save_model",
                new=fake_parent_save,
            ):
                trainer.save_model()
            expected_weight = model.model[14].weight.detach().clone()

            checkpoint = Path(directory) / "F0-calibration_last.pt"
            provenance = {
                "condition": "F0",
                "checkpoint_path": checkpoint.resolve().as_posix(),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
            fresh_model = _TaskToyModel()
            fresh_phase = task_adaptation_phase(
                fresh_model,
                condition="F0",
                calibration_checkpoint_path=checkpoint,
                calibration_provenance=provenance,
                optimizer_name="SGD",
                optimizer_hparams={"lr": 0.1, "momentum": 0.0},
                eta_schedule=(1.0,) * 60,
            )
            fresh_trainer = SimpleNamespace(model=fresh_model)
            apply_task_adaptation_phase(fresh_trainer, phase=fresh_phase)
            resume_task_adaptation_phase(fresh_trainer, trainer.last)
            self.assertTrue(
                torch.equal(
                    fresh_model.model[14].weight.detach(),
                    expected_weight,
                )
            )

    def test_final_save_guard_runs_before_parent_and_preserves_existing_last(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.task_adaptation_phase = phase
            trainer.last = Path(directory) / "last.pt"
            trainer.wdir = Path(directory)
            trainer.optimizer = phase.optimizer
            trainer.epoch = phase.epochs - 1
            trainer.task_adaptation_optimizer_steps = 0
            torch.save(
                {
                    "epoch": trainer.epoch,
                    "model": None,
                    "ema": copy.deepcopy(model),
                    "optimizer": phase.optimizer.state_dict(),
                },
                trainer.last,
            )
            sentinel = trainer.last.read_bytes()
            parent_called = False

            def fake_parent_save(_trainer):
                nonlocal parent_called
                parent_called = True
                torch.save(
                    {
                        "epoch": trainer.epoch,
                        "model": None,
                        "ema": copy.deepcopy(model),
                        "optimizer": phase.optimizer.state_dict(),
                    },
                    trainer.last,
                )
                return True

            with patch(
                "ultralytics.engine.trainer.BaseTrainer.save_model",
                new=fake_parent_save,
            ):
                with self.assertRaises(RuntimeError):
                    trainer.save_model()
            self.assertFalse(parent_called)
            self.assertEqual(trainer.last.read_bytes(), sentinel)

    def test_final_checkpoint_rejects_incomplete_registered_optimizer_budget(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            apply_task_adaptation_phase,
            task_adaptation_epoch_commit,
            task_adaptation_epoch_start,
            task_adaptation_final_checkpoint,
        )

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = SimpleNamespace(
                model=model,
                epoch=0,
                task_adaptation_optimizer_steps=0,
            )
            apply_task_adaptation_phase(trainer, phase=phase)
            task_adaptation_epoch_start(trainer)
            trainer.task_adaptation_optimizer_steps = phase.updates_per_epoch
            task_adaptation_epoch_commit(trainer)
            checkpoint = Path(directory) / "last.pt"
            sentinel = b"existing-checkpoint"
            checkpoint.write_bytes(sentinel)
            with self.assertRaises(RuntimeError):
                task_adaptation_final_checkpoint(
                    trainer,
                    checkpoint,
                    epoch=phase.epochs - 1,
                )
            self.assertEqual(checkpoint.read_bytes(), sentinel)

    def test_standalone_checkpoint_publication_is_atomic_on_serialization_failure(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import (
            apply_task_adaptation_phase,
            task_adaptation_epoch_commit,
            task_adaptation_epoch_start,
            task_adaptation_final_checkpoint,
        )

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = SimpleNamespace(
                model=model,
                epoch=0,
                task_adaptation_optimizer_steps=0,
            )
            apply_task_adaptation_phase(trainer, phase=phase)
            task_adaptation_epoch_start(trainer)
            trainer.task_adaptation_optimizer_steps = phase.updates_per_epoch
            task_adaptation_epoch_commit(trainer)
            checkpoint = Path(directory) / "last.pt"
            sentinel = b"valid-primary-checkpoint"
            checkpoint.write_bytes(sentinel)

            def corrupt_save(_payload, destination, *args, **kwargs):
                del args, kwargs
                if hasattr(destination, "write"):
                    destination.write(b"partially-serialized")
                    destination.flush()
                else:
                    Path(destination).write_bytes(b"partially-serialized")
                raise OSError("injected serialization failure")

            with patch(
                "ifdr_yolo.experiments.ifdr_trainer.torch.save",
                new=corrupt_save,
            ):
                with self.assertRaises(OSError):
                    task_adaptation_final_checkpoint(trainer, checkpoint, epoch=0)

            self.assertEqual(checkpoint.read_bytes(), sentinel)
            self.assertEqual(tuple(checkpoint.parent.glob(".last.pt.*.tmp")), ())

    def test_save_model_provenance_injection_is_atomic_on_serialization_failure(self) -> None:
        from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

        with tempfile.TemporaryDirectory() as directory:
            model, phase = _build_task_phase(directory)
            trainer = object.__new__(IFDRDetectionTrainer)
            trainer.model = model
            trainer.task_adaptation_phase = phase
            trainer.last = Path(directory) / "last.pt"
            trainer.wdir = Path(directory)
            trainer.optimizer = phase.optimizer
            trainer.task_adaptation_optimizer_steps = 0
            trainer.epoch = 4
            old_payload = {
                "epoch": 3,
                "model": "previous-model",
                "ema": "previous-ema",
                "optimizer": {"state": "previous"},
            }
            trainer.last.write_bytes(b"")
            torch.save(old_payload, trainer.last)
            previous_last = trainer.last.read_bytes()
            phase.semantic_state_journal.extend(
                [
                    {
                        "event": "resume_check",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": 4,
                        "optimizer_steps": 4,
                    },
                    {
                        "event": "epoch_commit",
                        "semantic_state_sha256": phase.semantic_state_sha256,
                        "epoch": 4,
                        "optimizer_steps": 5,
                    },
                ]
            )
            original_save = torch.save
            parent_finished = False

            def fake_parent_save(_trainer):
                nonlocal parent_finished
                original_save(
                    {
                        "epoch": 4,
                        "model": "standard-model",
                        "ema": "standard-ema",
                        "optimizer": {"state": "standard"},
                    },
                    trainer.last,
                )
                parent_finished = True
                return True

            def corrupt_injection(payload, destination, *args, **kwargs):
                del payload, args, kwargs
                if parent_finished:
                    if hasattr(destination, "write"):
                        destination.write(b"partially-serialized")
                        destination.flush()
                    else:
                        Path(destination).write_bytes(b"partially-serialized")
                    raise OSError("injected provenance serialization failure")
                return original_save({}, destination)

            with patch(
                "ultralytics.engine.trainer.BaseTrainer.save_model",
                new=fake_parent_save,
            ), patch(
                "ifdr_yolo.experiments.ifdr_trainer.torch.save",
                new=corrupt_injection,
            ):
                with self.assertRaises(OSError):
                    trainer.save_model()

            self.assertEqual(trainer.last.read_bytes(), previous_last)
            payload = torch.load(trainer.last, map_location="cpu", weights_only=False)
            self.assertEqual(payload["model"], "previous-model")
            self.assertEqual(
                tuple(trainer.last.parent.glob(".last.pt.*.tmp")),
                (),
            )

    def test_registered_graph_allowlist_rejects_extra_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _TaskToyModel(rogue=True)
            checkpoint = Path(directory) / "calibration_last.pt"
            provenance = _task_checkpoint(model, checkpoint)
            provenance["condition"] = "F0"
            from ifdr_yolo.experiments.factor_repair import task_adaptation_phase

            with self.assertRaises(ValueError):
                task_adaptation_phase(
                    model,
                    condition="F0",
                    calibration_checkpoint_path=checkpoint,
                    calibration_provenance=provenance,
                    optimizer_name="SGD",
                    optimizer_hparams={"lr": 0.1},
                    eta_schedule=(1.0,) * 60,
                )

    def test_actual_ifdr_graph_rejects_parameter_on_upsample_layer(self) -> None:
        from ifdr_yolo.experiments.factor_repair import task_adaptation_phase
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        with tempfile.TemporaryDirectory() as directory:
            model = IFDRDetectionModel(str(MODEL_PATH), verbose=False)
            model.model[10].register_parameter(
                "rogue",
                torch.nn.Parameter(torch.ones(1)),
            )
            checkpoint = Path(directory) / "calibration_last.pt"
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            provenance = {
                "condition": "F0",
                "checkpoint_path": checkpoint.resolve().as_posix(),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
            with self.assertRaises(ValueError):
                task_adaptation_phase(
                    model,
                    condition="F0",
                    calibration_checkpoint_path=checkpoint,
                    calibration_provenance=provenance,
                    optimizer_name="SGD",
                    optimizer_hparams={"lr": 0.1},
                    eta_schedule=(1.0,) * 60,
                )


if __name__ == "__main__":
    unittest.main()
