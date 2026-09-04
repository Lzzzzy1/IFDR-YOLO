from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, RANK

from ifdr_yolo.data.ifdr_dataset import (
    BACKGROUND_IMAGE_KEY,
    CLEAN_IMAGE_KEY,
    COUNTERFACTUAL_IMAGE_KEY,
    TARGET_IMAGE_KEY,
    SpecificityRejectionCounter,
    build_ifdr_dataset,
)
from ifdr_yolo.data.interventions.sampler import SamplingPolicy
from ifdr_yolo.experiments.factor_repair_runtime import CALIBRATION_GEOMETRY_OVERRIDES
from ifdr_yolo.models.ifdr_model import IFDRDetectionModel


TASK_ADAPTATION_STATE_KEY = "task_adaptation_state_dict"
TASK_ADAPTATION_STATE_SOURCE = "live_model_state_dict"
_AMP_PREFLIGHT_ATOL = 0.5
_AMP_PREFLIGHT_RTOL = 0.1


def _lossless_model_state_dict(model: object) -> dict[str, object]:
    """Capture a detached CPU copy of the live model without EMA casting."""

    model = _unwrap_training_model(model)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("task adaptation model must be a torch module")
    return {
        name: value.detach().cpu().clone()
        if isinstance(value, torch.Tensor)
        else copy.deepcopy(value)
        for name, value in model.state_dict().items()
    }


def _lossless_task_state_from_payload(
    payload: Mapping[str, object],
    provenance: Mapping[str, object],
) -> Mapping[str, object]:
    """Require the complete-precision live state for a Task6A checkpoint."""

    if provenance.get("task_state_key") != TASK_ADAPTATION_STATE_KEY:
        raise ValueError("task adaptation lossless state provenance is missing")
    if provenance.get("task_state_source") != TASK_ADAPTATION_STATE_SOURCE:
        raise ValueError("task adaptation lossless state source is invalid")
    state = payload.get(TASK_ADAPTATION_STATE_KEY)
    if not isinstance(state, Mapping):
        raise ValueError("task adaptation lossless state is missing")
    return state


def _lossless_state_sha256(state: Mapping[str, object]) -> str:
    """Hash a complete state mapping canonically, including tensor metadata."""

    if not state:
        raise ValueError("task adaptation lossless state is empty")
    digest = hashlib.sha256(b"ifdr-task-state-v1\0")
    for name in sorted(state):
        if not isinstance(name, str):
            raise ValueError("task adaptation state key is invalid")
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError("task adaptation state contains a non-tensor")
        value = value.detach().cpu().contiguous()
        metadata = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validated_lossless_task_model(
    trainer: object,
    phase: object,
    payload: Mapping[str, object],
    provenance: Mapping[str, object],
) -> torch.nn.Module:
    """Validate and materialize an independent model from the live state."""

    from ifdr_yolo.experiments.factor_repair import (
        semantic_module_ids,
        semantic_state_sha256,
    )

    state = _lossless_task_state_from_payload(payload, provenance)
    model = _unwrap_training_model(getattr(trainer, "model", None))
    if not isinstance(model, torch.nn.Module):
        raise ValueError("task adaptation validation model is invalid")
    reference = model.state_dict()
    if set(state) != set(reference):
        raise ValueError("task adaptation lossless state keys mismatch")
    for name, expected in reference.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise ValueError("task adaptation lossless state tensor mismatch")
    state_digest = _lossless_state_sha256(state)
    if provenance.get("task_state_sha256") != state_digest:
        raise ValueError("task adaptation lossless state digest mismatch")
    try:
        validated = copy.deepcopy(model)
        validated.load_state_dict(state, strict=True)
    except Exception as error:
        raise ValueError("task adaptation lossless state is not loadable") from error
    if (
        semantic_state_sha256(validated, semantic_module_ids(validated))
        != phase.semantic_state_sha256
    ):
        raise ValueError("task adaptation semantic hash provenance mismatch")
    validated.eval()
    return validated


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _atomic_torch_save(payload: object, path: str | Path) -> Path:
    """Serialize a checkpoint to a same-directory temp file and publish atomically."""

    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = -1
        if directory_descriptor != -1:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
        return target
    finally:
        if descriptor != -1:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class FusionSchedule:
    """Epoch schedule that preserves the pretrained graph before gating."""

    frozen_epochs: int = 5
    ramp_epochs: int = 10

    def __post_init__(self) -> None:
        _non_negative_integer(self.frozen_epochs, "frozen_epochs")
        ramp_epochs = _non_negative_integer(self.ramp_epochs, "ramp_epochs")
        if ramp_epochs == 0:
            raise ValueError("ramp_epochs must be positive")

    def value_at(self, epoch: int) -> float:
        epoch = _non_negative_integer(epoch, "epoch")
        if epoch < self.frozen_epochs:
            return 0.0
        progress = (epoch - self.frozen_epochs + 1) / self.ramp_epochs
        return min(1.0, progress)


@dataclass(frozen=True)
class IFDRComponentSwitches:
    fusion_gate: bool = True
    dcli: bool = True
    factor_supervision: bool = True
    interventions: bool = True
    semantic_protection: bool = False
    counterfactual_consistency: bool = False

    def __post_init__(self) -> None:
        for field in (
            "fusion_gate",
            "dcli",
            "factor_supervision",
            "interventions",
            "semantic_protection",
            "counterfactual_consistency",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be a boolean")


def _unwrap_training_model(model: object) -> object:
    while hasattr(model, "module"):
        model = getattr(model, "module")
    return model


def _collect_amp_output_tensors(value: object, output: list[torch.Tensor]) -> None:
    if isinstance(value, torch.Tensor):
        output.append(value)
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _collect_amp_output_tensors(value[key], output)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _collect_amp_output_tensors(item, output)


def validate_amp_outputs(fp32_outputs: object, amp_outputs: object) -> None:
    """Validate deterministic FP32/autocast output parity for AMP preflight."""

    fp32: list[torch.Tensor] = []
    amp: list[torch.Tensor] = []
    _collect_amp_output_tensors(fp32_outputs, fp32)
    _collect_amp_output_tensors(amp_outputs, amp)
    if not fp32 or not amp:
        raise RuntimeError("local AMP preflight produced no tensor outputs")
    if len(fp32) != len(amp):
        raise RuntimeError("local AMP preflight output count mismatch")
    for index, (full, mixed) in enumerate(zip(fp32, amp)):
        if full.shape != mixed.shape:
            raise RuntimeError(f"local AMP preflight output shape mismatch at index {index}")
        if not bool(torch.isfinite(full).all().item()) or not bool(torch.isfinite(mixed).all().item()):
            raise RuntimeError(f"local AMP preflight output is not finite at index {index}")
        if not torch.allclose(
            full.detach().float().cpu(),
            mixed.detach().float().cpu(),
            atol=_AMP_PREFLIGHT_ATOL,
            rtol=_AMP_PREFLIGHT_RTOL,
        ):
            raise RuntimeError(f"local AMP preflight output value mismatch at index {index}")


def _consume_amp_reliability_context(model: object) -> None:
    consume = getattr(model, "consume_reliability_context", None)
    if not callable(consume):
        return
    try:
        consume()
    except RuntimeError as error:
        if "no reliability context" not in str(error):
            raise


def factor_amp_preflight(model: object) -> bool:
    """Run project-local FP32/autocast parity checks on the active CUDA model."""

    model = _unwrap_training_model(model)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("local AMP preflight requires a torch module")
    parameters = tuple(model.parameters())
    if not parameters:
        raise RuntimeError("local AMP preflight model has no parameters")
    device = parameters[0].device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("local AMP preflight requires a CUDA model")
    if any(parameter.device != device for parameter in parameters):
        raise RuntimeError("local AMP preflight model parameters span devices")
    stride_value = getattr(model, "stride", 32)
    try:
        stride = max(int(stride_value.max()), 1)
    except (AttributeError, TypeError, ValueError):
        stride = 32
    image_size = max(64, stride * 2)
    image_size = ((image_size + stride - 1) // stride) * stride
    image = torch.zeros((1, 3, image_size, image_size), device=device, dtype=torch.float32)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            fp32_outputs = model(image)
        _consume_amp_reliability_context(model)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_outputs = model(image)
        _consume_amp_reliability_context(model)
        validate_amp_outputs(fp32_outputs, amp_outputs)
    except RuntimeError:
        _consume_amp_reliability_context(model)
        raise
    finally:
        model.train(was_training)
        _consume_amp_reliability_context(model)
    return True


run_local_amp_preflight = factor_amp_preflight


def apply_fusion_schedule(trainer: object) -> float:
    """Set the schedule from the absolute epoch, including resumed runs."""

    epoch = getattr(trainer, "epoch", None)
    schedule = getattr(trainer, "fusion_schedule", None)
    if not isinstance(schedule, FusionSchedule):
        raise RuntimeError("trainer does not define a valid fusion schedule")
    value = schedule.value_at(epoch)
    model = _unwrap_training_model(getattr(trainer, "model", None))
    switches = getattr(
        trainer,
        "component_switches",
        IFDRComponentSwitches(),
    )
    if not isinstance(switches, IFDRComponentSwitches):
        raise RuntimeError("trainer component switches are invalid")
    component_setter = getattr(model, "set_component_schedules", None)
    legacy_setter = getattr(model, "set_reliability_schedule", None)
    if callable(component_setter):
        component_setter(
            fusion=value if switches.fusion_gate else 0.0,
            dcli=value if switches.dcli else 0.0,
            factor_supervision=(
                value if switches.factor_supervision else 0.0
            ),
        )
    elif callable(legacy_setter):
        legacy_setter(value)
    else:
        raise RuntimeError("training model does not support reliability gating")
    train_loader = getattr(trainer, "train_loader", None)
    dataset = getattr(train_loader, "dataset", None)
    epoch_setter = getattr(dataset, "set_epoch", None)
    if callable(epoch_setter):
        epoch_setter(epoch)
    setattr(trainer, "fusion_schedule_value", value)
    return value


def flush_gradient_diagnostics(trainer: object) -> None:
    raw_epoch = getattr(trainer, "epoch", None)
    if (
        isinstance(raw_epoch, bool)
        or not isinstance(raw_epoch, int)
        or raw_epoch < 0
    ):
        raise ValueError("gradient diagnostic trainer epoch must be non-negative")
    epoch = int(raw_epoch) + 1
    model = _unwrap_training_model(getattr(trainer, "model", None))
    drain = getattr(model, "drain_gradient_diagnostics", None)
    if not callable(drain):
        return
    records = drain()
    if not records:
        return
    process_id = os.getpid()
    save_dir = Path(getattr(trainer, "save_dir"))
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "gradient_diagnostics.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            payload = dict(record)
            payload.update({"epoch": epoch, "process_id": process_id})
            file.write(json.dumps(payload, sort_keys=True) + "\n")


def apply_semantic_calibration_phase(
    trainer: object,
    phase: object | None = None,
    *,
    variant: str | None = None,
    epochs: int = 30,
    optimizer: object | None = None,
) -> object:
    from ifdr_yolo.experiments.factor_repair import (
        SemanticCalibrationPhase,
        semantic_calibration_phase,
    )

    if phase is not None:
        if not isinstance(phase, SemanticCalibrationPhase):
            raise TypeError("phase must be a SemanticCalibrationPhase")
        variant = phase.variant
        epochs = phase.epochs
    elif variant is None:
        raise ValueError("variant is required when phase is not provided")
    model = _unwrap_training_model(getattr(trainer, "model", None))
    if optimizer is None:
        optimizer = getattr(trainer, "optimizer", None)
    applied = semantic_calibration_phase(
        model,
        variant=variant,
        epochs=epochs,
        optimizer=optimizer,
    )
    setattr(trainer, "semantic_calibration_phase", applied)
    return applied


def run_validation_without_optimizer_step(trainer: object, batch: object) -> object:
    from ifdr_yolo.experiments.factor_repair import run_calibration_validation

    model = _unwrap_training_model(getattr(trainer, "model", None))
    return run_calibration_validation(
        model,
        batch,
        optimizer=getattr(trainer, "optimizer", None),
    )


def apply_task_adaptation_phase(
    trainer: object,
    phase: object | None = None,
    *,
    condition: str | None = None,
    calibration_checkpoint_path: str | Path | None = None,
    calibration_provenance: dict[str, object] | None = None,
    calibration_checkpoint_role: str = "calibration_last",
    optimizer_name: str = "AdamW",
    optimizer_hparams: dict[str, object] | None = None,
    eta_schedule: tuple[object, ...] = (1.0,) * 60,
) -> object:
    """Attach a registered, condition-local task adaptation phase."""

    from ifdr_yolo.experiments.factor_repair import (
        TaskAdaptationPhase,
        enforce_semantic_eval_mode,
        semantic_module_ids,
        task_adaptation_phase,
        verify_semantic_state,
    )

    model = _unwrap_training_model(getattr(trainer, "model", None))
    if phase is None:
        if condition is None or calibration_checkpoint_path is None:
            raise ValueError("condition and calibration_checkpoint_path are required")
        phase = task_adaptation_phase(
            model,
            condition=condition,
            calibration_checkpoint_path=calibration_checkpoint_path,
            calibration_provenance=calibration_provenance,
            calibration_checkpoint_role=calibration_checkpoint_role,
            optimizer_name=optimizer_name,
            optimizer_hparams=(
                {} if optimizer_hparams is None else dict(optimizer_hparams)
            ),
            eta_schedule=eta_schedule,
        )
    elif not isinstance(phase, TaskAdaptationPhase):
        raise TypeError("phase must be a TaskAdaptationPhase")
    ids = semantic_module_ids(model)
    verify_semantic_state(model, phase)
    enforce_semantic_eval_mode(model, ids)
    trainer.task_adaptation_phase = phase
    trainer.task_adaptation_semantic_module_ids = ids
    trainer.task_adaptation_epochs = phase.epochs
    trainer.task_adaptation_primary_checkpoint = phase.primary_checkpoint
    trainer.task_adaptation_epoch_journal = phase.semantic_state_journal
    trainer.optimizer = phase.optimizer
    trainer.early_stopping = False
    trainer.task_adaptation_optimizer_steps = 0
    # The fixed-budget contract takes precedence over command-line limits.
    trainer.epochs = phase.epochs
    args = getattr(trainer, "args", None)
    if args is not None:
        args.epochs = phase.epochs
        args.time = None
        args.patience = 0
    stopper = getattr(trainer, "stopper", None)
    if stopper is not None and hasattr(stopper, "patience"):
        stopper.patience = float("inf")
    _bind_task_adaptation_scheduler(trainer, phase)
    return phase


def _bind_task_adaptation_scheduler(trainer: object, phase: object) -> None:
    """Install the immutable per-epoch eta schedule on the active optimizer."""

    optimizer = getattr(trainer, "optimizer", None)
    schedule = tuple(getattr(phase, "eta_schedule", ()))
    if optimizer is None or len(schedule) != phase.epochs:
        return
    start_epoch = getattr(trainer, "task_adaptation_start_epoch", 0)
    if (
        isinstance(start_epoch, bool)
        or not isinstance(start_epoch, int)
        or start_epoch < 0
    ):
        raise ValueError("task adaptation start epoch must be a non-negative integer")
    # LambdaLR consumes epoch 0 during construction.  Indexing by max(step-1,
    # 0) makes the first external scheduler step consume schedule[0], while a
    # resumed trainer can set last_epoch to its absolute start epoch.
    trainer.lf = lambda step: float(
        schedule[min(max(int(step) - 1, 0), len(schedule) - 1)]
    )
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=trainer.lf,
    )
    trainer.scheduler.last_epoch = start_epoch
    current_index = min(start_epoch, len(schedule) - 1)
    current_lrs = []
    for group in optimizer.param_groups:
        base_lr = group.get("initial_lr", group["lr"])
        group["lr"] = base_lr * float(schedule[current_index])
        current_lrs.append(group["lr"])
    trainer.scheduler._last_lr = current_lrs


def task_adaptation_epoch_start(trainer: object) -> None:
    """Prepare one task epoch and record a condition-local resume check."""

    from ifdr_yolo.experiments.factor_repair import (
        TaskAdaptationPhase,
        enforce_semantic_eval_mode,
        semantic_module_ids,
        verify_semantic_state,
    )

    phase = getattr(trainer, "task_adaptation_phase", None)
    if phase is None:
        return
    if not isinstance(phase, TaskAdaptationPhase):
        raise TypeError("trainer task adaptation phase is invalid")
    model = _unwrap_training_model(getattr(trainer, "model", None))
    enforce_semantic_eval_mode(model, semantic_module_ids(model))
    epoch = getattr(trainer, "epoch", None)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        committed = [
            record.get("epoch")
            for record in phase.semantic_state_journal
            if isinstance(record, Mapping) and record.get("event") == "epoch_commit"
        ]
        epoch = max(committed, default=-1) + 1
    steps = getattr(trainer, "task_adaptation_optimizer_steps", epoch * phase.updates_per_epoch)
    existing = any(
        isinstance(record, Mapping)
        and record.get("event") == "resume_check"
        and record.get("epoch") == epoch
        for record in phase.semantic_state_journal
    )
    verify_semantic_state(
        model,
        phase,
        event=None if existing else "resume_check",
        epoch=epoch,
        optimizer_steps=steps,
    )


def task_adaptation_epoch_commit(trainer: object) -> str | None:
    """Record an epoch boundary and fail closed on semantic drift."""

    from ifdr_yolo.experiments.factor_repair import (
        TaskAdaptationPhase,
        verify_semantic_state,
    )

    phase = getattr(trainer, "task_adaptation_phase", None)
    if phase is None:
        return None
    if not isinstance(phase, TaskAdaptationPhase):
        raise TypeError("trainer task adaptation phase is invalid")
    model = _unwrap_training_model(getattr(trainer, "model", None))
    epoch = getattr(trainer, "epoch", None)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        resumes = [
            record.get("epoch")
            for record in phase.semantic_state_journal
            if isinstance(record, Mapping) and record.get("event") == "resume_check"
        ]
        epoch = max(resumes, default=0)
    steps = getattr(
        trainer,
        "task_adaptation_optimizer_steps",
        (epoch + 1) * phase.updates_per_epoch,
    )
    return verify_semantic_state(
        model,
        phase,
        event="epoch_commit",
        epoch=epoch,
        optimizer_steps=steps,
    )


def task_adaptation_final_checkpoint(
    trainer: object,
    checkpoint_path: str | Path,
    *,
    epoch: int | None = None,
) -> Path | None:
    """Write the fixed-budget ``last.pt`` checkpoint with bound provenance."""

    from ifdr_yolo.data.learned_factor_manifest import resolve_provenance_path
    from ifdr_yolo.experiments.factor_repair import (
        TaskAdaptationPhase,
        verify_semantic_state,
    )

    phase = getattr(trainer, "task_adaptation_phase", None)
    if phase is None:
        return None
    if not isinstance(phase, TaskAdaptationPhase):
        raise TypeError("trainer task adaptation phase is invalid")
    path = Path(checkpoint_path).expanduser().resolve(strict=False)
    if path.name != phase.primary_checkpoint or path.name != "last.pt":
        raise ValueError("task adaptation primary checkpoint must be last.pt")
    actual_optimizer_steps = getattr(trainer, "task_adaptation_optimizer_steps", None)
    if epoch is None:
        epoch = getattr(trainer, "epoch", None)
    if epoch == phase.epochs - 1 and actual_optimizer_steps != phase.update_count:
        raise RuntimeError("task adaptation optimizer update budget is incomplete")
    recorded_events = {
        record.get("event")
        for record in phase.semantic_state_journal
        if isinstance(record, dict)
    }
    if not {"resume_check", "epoch_commit"} <= recorded_events:
        raise RuntimeError("task adaptation semantic journal is incomplete")
    model = _unwrap_training_model(getattr(trainer, "model", None))
    steps = getattr(trainer, "task_adaptation_optimizer_steps", None)
    if epoch is None or steps is None:
        raise ValueError("task adaptation final checkpoint progress is required")
    checkpoint_event = "final_checkpoint" if epoch == phase.epochs - 1 else None
    semantic_hash = verify_semantic_state(
        model,
        phase,
        event=checkpoint_event,
        epoch=epoch,
        optimizer_steps=steps,
    )
    _validate_task_adaptation_journal(
        phase,
        phase.semantic_state_journal,
        completed_epoch=epoch,
        require_final=checkpoint_event == "final_checkpoint",
    )
    lossless_state = _lossless_model_state_dict(model)
    lossless_digest = _lossless_state_sha256(lossless_state)
    _atomic_torch_save(
        {
            "state_dict": model.state_dict(),
            TASK_ADAPTATION_STATE_KEY: lossless_state,
            "optimizer": phase.optimizer.state_dict(),
            "task_adaptation_provenance": {
                "condition": phase.condition,
                "checkpoint_role": phase.calibration_checkpoint_role,
                "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
                "checkpoint_sha256": phase.calibration_checkpoint_sha256,
                "semantic_state_sha256": semantic_hash,
                "semantic_module_names": tuple(phase.semantic_module_names),
                "task_parameter_categories": dict(phase.task_parameter_categories),
                "updates_per_epoch": phase.updates_per_epoch,
                "expected_optimizer_steps": phase.update_count,
                "optimizer_steps": int(actual_optimizer_steps),
                "primary_checkpoint": phase.primary_checkpoint,
                "task_checkpoint_path": path.as_posix(),
                "semantic_state_journal": tuple(phase.semantic_state_journal),
                "optimizer_name": phase.optimizer_name,
                "optimizer_hparams": dict(phase.optimizer_hparams),
                "optimizer_defaults": dict(phase.optimizer.defaults),
                "eta_schedule": tuple(phase.eta_schedule),
                "task_state_key": TASK_ADAPTATION_STATE_KEY,
                "task_state_source": TASK_ADAPTATION_STATE_SOURCE,
                "task_state_sha256": lossless_digest,
                "epoch": epoch,
            },
        },
        path,
    )
    return resolve_provenance_path(path)


def resume_task_adaptation_phase(
    trainer: object,
    checkpoint_path: str | Path,
) -> object:
    """Resume only from the trainer's own condition/provenance checkpoint."""

    from ifdr_yolo.data.learned_factor_manifest import resolve_provenance_path
    from ifdr_yolo.data.splits import sha256_file
    from ifdr_yolo.experiments.factor_repair import (
        TaskAdaptationPhase,
        enforce_semantic_eval_mode,
        semantic_module_ids,
        verify_semantic_state,
    )

    phase = getattr(trainer, "task_adaptation_phase", None)
    if not isinstance(phase, TaskAdaptationPhase):
        raise ValueError("trainer has no registered task adaptation phase")
    path = resolve_provenance_path(checkpoint_path)
    if path.name != phase.primary_checkpoint or path.name != "last.pt":
        raise ValueError("task adaptation resume requires last.pt")
    model = _unwrap_training_model(getattr(trainer, "model", None))
    calibration_path = resolve_provenance_path(phase.calibration_checkpoint_path)
    if (
        calibration_path != phase.calibration_checkpoint_path
        or sha256_file(calibration_path) != phase.calibration_checkpoint_sha256
    ):
        raise ValueError("calibration checkpoint provenance is no longer valid")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("unable to load task adaptation checkpoint") from error
    if not isinstance(payload, dict):
        raise ValueError("task adaptation checkpoint has no provenance")
    provenance = payload.get("task_adaptation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("task adaptation checkpoint has no provenance")
    expected = {
        "condition": phase.condition,
        "checkpoint_role": phase.calibration_checkpoint_role,
        "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
        "checkpoint_sha256": phase.calibration_checkpoint_sha256,
        "primary_checkpoint": phase.primary_checkpoint,
        "task_checkpoint_path": path.as_posix(),
        "task_state_key": TASK_ADAPTATION_STATE_KEY,
        "task_state_source": TASK_ADAPTATION_STATE_SOURCE,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("task adaptation checkpoint provenance mismatch")
    if provenance.get("semantic_state_sha256") != phase.semantic_state_sha256:
        raise ValueError("task adaptation semantic hash provenance mismatch")
    if tuple(provenance.get("semantic_module_names", ())) != tuple(
        phase.semantic_module_names
    ):
        raise ValueError("task adaptation semantic module provenance mismatch")
    if dict(provenance.get("task_parameter_categories", {})) != dict(
        phase.task_parameter_categories
    ):
        raise ValueError("task adaptation task parameter provenance mismatch")
    if provenance.get("updates_per_epoch") != phase.updates_per_epoch:
        raise ValueError("task adaptation update schedule provenance mismatch")
    if provenance.get("expected_optimizer_steps") != phase.update_count:
        raise ValueError("task adaptation optimizer budget provenance mismatch")
    saved_schedule = provenance.get("eta_schedule")
    if saved_schedule is not None and tuple(saved_schedule) != tuple(phase.eta_schedule):
        raise ValueError("task adaptation eta schedule provenance mismatch")
    if provenance.get("optimizer_name") != phase.optimizer_name:
        raise ValueError("task adaptation optimizer provenance mismatch")
    if provenance.get("optimizer_hparams") != dict(phase.optimizer_hparams):
        raise ValueError("task adaptation optimizer hyperparameters mismatch")
    if provenance.get("optimizer_defaults") != dict(phase.optimizer.defaults):
        raise ValueError("task adaptation optimizer hyperparameters mismatch")
    saved_journal = provenance.get("semantic_state_journal")
    saved_epoch = provenance.get("epoch")
    if (
        isinstance(saved_epoch, bool)
        or not isinstance(saved_epoch, int)
        or saved_epoch < 0
        or saved_epoch >= phase.epochs
    ):
        raise ValueError("task adaptation checkpoint epoch is invalid")
    saved_steps = provenance.get("optimizer_steps")
    if (
        isinstance(saved_steps, bool)
        or not isinstance(saved_steps, int)
        or saved_steps < 0
        or saved_steps > phase.update_count
        or saved_steps != (saved_epoch + 1) * phase.updates_per_epoch
    ):
        raise ValueError("task adaptation optimizer step provenance is invalid")
    # Task6A must restore from the detached CPU live state.  Standard
    # Ultralytics EMA/model fields may be FP16 and are not lossless.
    state_dict = _lossless_task_state_from_payload(payload, provenance)
    if _lossless_state_sha256(state_dict) != provenance.get("task_state_sha256"):
        raise ValueError("task adaptation lossless state digest mismatch")
    optimizer_state = payload.get("optimizer")
    if not isinstance(state_dict, Mapping) or not isinstance(optimizer_state, Mapping):
        raise ValueError("task adaptation checkpoint is incomplete")
    _validate_task_adaptation_journal(
        phase,
        saved_journal,
        completed_epoch=saved_epoch,
    )
    before_model_state = {
        name: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for name, value in model.state_dict().items()
    }
    before_optimizer_state = copy.deepcopy(phase.optimizer.state_dict())
    before_journal = [dict(record) for record in phase.semantic_state_journal]
    try:
        model.load_state_dict(state_dict, strict=True)
        phase.optimizer.load_state_dict(optimizer_state)
        phase.semantic_state_journal[:] = [dict(record) for record in saved_journal]
        enforce_semantic_eval_mode(model, semantic_module_ids(model))
        verify_semantic_state(
            model,
            phase,
            event="resume_check",
            epoch=saved_epoch + 1,
            optimizer_steps=saved_steps,
        )
    except Exception as error:
        model.load_state_dict(before_model_state, strict=True)
        phase.optimizer.load_state_dict(before_optimizer_state)
        phase.semantic_state_journal[:] = before_journal
        raise ValueError("task adaptation checkpoint state does not match model") from error
    trainer.task_adaptation_resume_epoch = saved_epoch
    trainer.task_adaptation_optimizer_steps = saved_steps
    trainer.task_adaptation_start_epoch = saved_epoch + 1
    trainer.start_epoch = saved_epoch + 1
    _bind_task_adaptation_scheduler(trainer, phase)
    return phase


def _validate_task_adaptation_journal(
    phase: object,
    journal: object,
    *,
    completed_epoch: int | None = None,
    require_final: bool = False,
) -> None:
    """Require one continuous resume/commit pair for every completed epoch."""

    if not isinstance(journal, (list, tuple)) or not journal:
        raise ValueError("task adaptation semantic journal is incomplete")
    commits: dict[int, Mapping[str, object]] = {}
    resumes: dict[int, Mapping[str, object]] = {}
    finals: list[Mapping[str, object]] = []
    for record in journal:
        if not isinstance(record, Mapping):
            raise ValueError("task adaptation semantic journal record is invalid")
        event = record.get("event")
        if event not in {"epoch_commit", "resume_check", "final_checkpoint"}:
            raise ValueError("task adaptation semantic journal event is invalid")
        if record.get("semantic_state_sha256") != phase.semantic_state_sha256:
            raise ValueError("task adaptation semantic journal hash mismatch")
        epoch = record.get("epoch")
        steps = record.get("optimizer_steps")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or epoch >= phase.epochs
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 0
            or steps > phase.update_count
        ):
            raise ValueError("task adaptation semantic journal progress is invalid")
        if event == "epoch_commit":
            if epoch in commits:
                raise ValueError("task adaptation duplicate epoch commit")
            if steps != (epoch + 1) * phase.updates_per_epoch:
                raise ValueError("task adaptation epoch commit steps mismatch")
            commits[epoch] = record
        elif event == "resume_check":
            if epoch in resumes:
                raise ValueError("task adaptation duplicate resume check")
            if steps != epoch * phase.updates_per_epoch:
                raise ValueError("task adaptation resume steps mismatch")
            resumes[epoch] = record
        else:
            finals.append(record)
    if not commits:
        raise ValueError("task adaptation semantic journal has no committed epoch")
    last_epoch = max(commits)
    if completed_epoch is not None and last_epoch != completed_epoch:
        raise ValueError("task adaptation semantic journal completed epoch mismatch")
    if sorted(commits) != list(range(last_epoch + 1)):
        raise ValueError("task adaptation epoch commits are not continuous")
    if any(epoch > last_epoch + 1 for epoch in resumes):
        raise ValueError("task adaptation resume journal has a gap")
    if any(epoch not in resumes for epoch in range(last_epoch + 1)):
        raise ValueError("task adaptation resume journal has a gap")
    if require_final:
        if last_epoch != phase.epochs - 1 or sorted(resumes) != list(range(phase.epochs)):
            raise ValueError("task adaptation final journal is incomplete")
        if len(finals) != 1:
            raise ValueError("task adaptation final journal is incomplete")
        final = finals[0]
        if final.get("epoch") != phase.epochs - 1 or final.get("optimizer_steps") != phase.update_count:
            raise ValueError("task adaptation final journal progress mismatch")
    elif finals:
        raise ValueError("task adaptation final journal is not terminal")


def _validated_adaptation_last(
    trainer: object,
    phase: object,
    *,
    payload: Mapping[str, object] | None = None,
) -> Path:
    """Validate the condition-local provenance before final evaluation."""

    from ifdr_yolo.data.learned_factor_manifest import resolve_provenance_path
    from ifdr_yolo.data.splits import sha256_file

    path = resolve_provenance_path(getattr(trainer, "last", None))
    if path.name != "last.pt" or path.name != phase.primary_checkpoint:
        raise ValueError("task adaptation primary checkpoint must be last.pt")
    calibration_path = resolve_provenance_path(phase.calibration_checkpoint_path)
    if (
        calibration_path != phase.calibration_checkpoint_path
        or sha256_file(calibration_path) != phase.calibration_checkpoint_sha256
    ):
        raise ValueError("calibration checkpoint provenance is no longer valid")
    if payload is None:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise ValueError("unable to load task adaptation primary checkpoint") from error
    if not isinstance(payload, Mapping):
        raise ValueError("task adaptation primary checkpoint has no provenance")
    provenance = payload.get("task_adaptation_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("task adaptation primary checkpoint has no provenance")
    expected = {
        "condition": phase.condition,
        "checkpoint_role": phase.calibration_checkpoint_role,
        "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
        "checkpoint_sha256": phase.calibration_checkpoint_sha256,
        "semantic_state_sha256": phase.semantic_state_sha256,
        "semantic_module_names": tuple(phase.semantic_module_names),
        "task_parameter_categories": dict(phase.task_parameter_categories),
        "updates_per_epoch": phase.updates_per_epoch,
        "expected_optimizer_steps": phase.update_count,
        "optimizer_steps": phase.update_count,
        "primary_checkpoint": phase.primary_checkpoint,
        "task_checkpoint_path": path.as_posix(),
        "optimizer_name": phase.optimizer_name,
        "optimizer_hparams": dict(phase.optimizer_hparams),
        "optimizer_defaults": dict(phase.optimizer.defaults),
        "eta_schedule": tuple(phase.eta_schedule),
        "task_state_key": TASK_ADAPTATION_STATE_KEY,
        "task_state_source": TASK_ADAPTATION_STATE_SOURCE,
        "epoch": phase.epochs - 1,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("task adaptation primary checkpoint provenance mismatch")
    _validated_lossless_task_model(trainer, phase, payload, provenance)
    journal = provenance.get("semantic_state_journal")
    _validate_task_adaptation_journal(
        phase,
        journal,
        completed_epoch=phase.epochs - 1,
        require_final=True,
    )
    return path


def _validated_adaptation_model(trainer: object, phase: object) -> torch.nn.Module:
    """Materialize the validated lossless Task6A state for evaluation."""

    path = Path(getattr(trainer, "last")).expanduser().resolve(strict=False)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("unable to load task adaptation evaluation checkpoint") from error
    provenance = payload.get("task_adaptation_provenance") if isinstance(payload, Mapping) else None
    if not isinstance(payload, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("task adaptation evaluation checkpoint has no provenance")
    _validated_adaptation_last(trainer, phase, payload=payload)
    return _validated_lossless_task_model(trainer, phase, payload, provenance)


def _resume_checkpoint_path(trainer: object, hint: object | None = None) -> Path | None:
    """Resolve the task checkpoint selected by Ultralytics' resume arguments."""

    candidates = [
        hint,
        getattr(getattr(trainer, "args", None), "resume", None),
        getattr(trainer, "model", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, (str, Path)) and str(candidate).lower().endswith(".pt"):
            return Path(candidate).expanduser().resolve(strict=False)
    return None


def _task_phase_from_resume_provenance(
    trainer: object,
    checkpoint_path: Path,
    provenance: Mapping[str, object],
    payload: Mapping[str, object],
) -> object:
    """Rebuild a phase around the parent-loaded task model without calibration reload."""

    from ifdr_yolo.experiments.factor_repair import task_adaptation_phase

    required = (
        "condition",
        "checkpoint_role",
        "checkpoint_path",
        "checkpoint_sha256",
        "semantic_state_sha256",
        "semantic_module_names",
        "task_parameter_categories",
        "updates_per_epoch",
        "expected_optimizer_steps",
        "optimizer_steps",
        "optimizer_name",
        "eta_schedule",
        "semantic_state_journal",
        "epoch",
        "primary_checkpoint",
        "task_checkpoint_path",
        "task_state_key",
        "task_state_source",
        "task_state_sha256",
    )
    if any(key not in provenance for key in required):
        raise ValueError("task adaptation resume provenance is incomplete")
    if provenance["checkpoint_role"] != "calibration_last":
        raise ValueError("task adaptation resume requires calibration_last")
    if provenance["primary_checkpoint"] != "last.pt":
        raise ValueError("task adaptation resume requires last.pt")
    if provenance["task_checkpoint_path"] != checkpoint_path.as_posix():
        raise ValueError("task adaptation task checkpoint path mismatch")
    updates_per_epoch = provenance["updates_per_epoch"]
    expected_steps = provenance["expected_optimizer_steps"]
    if (
        isinstance(updates_per_epoch, bool)
        or not isinstance(updates_per_epoch, int)
        or updates_per_epoch <= 0
        or expected_steps != 60 * updates_per_epoch
    ):
        raise ValueError("task adaptation resume budget provenance mismatch")
    if (
        isinstance(provenance["epoch"], bool)
        or not isinstance(provenance["epoch"], int)
        or provenance["epoch"] < 0
        or provenance["epoch"] >= 60
        or provenance["optimizer_steps"] != (provenance["epoch"] + 1) * updates_per_epoch
    ):
        raise ValueError("task adaptation resume progress provenance mismatch")
    eta_schedule = provenance["eta_schedule"]
    if isinstance(eta_schedule, (str, bytes)):
        raise ValueError("task adaptation resume eta schedule is invalid")
    try:
        eta_schedule = tuple(eta_schedule)
    except TypeError as error:
        raise ValueError("task adaptation resume eta schedule is invalid") from error
    if len(eta_schedule) != 60:
        raise ValueError("task adaptation resume requires a 60-epoch eta schedule")

    optimizer_name = provenance["optimizer_name"]
    if not isinstance(optimizer_name, str):
        raise ValueError("task adaptation optimizer provenance is invalid")
    optimizer_hparams = provenance.get("optimizer_hparams")
    if not isinstance(optimizer_hparams, Mapping):
        # Older checkpoints only stored constructor defaults.  Restrict those
        # values to the current optimizer signature instead of silently
        # accepting an unknown or unsafe option.
        defaults = provenance.get("optimizer_defaults")
        if not isinstance(defaults, Mapping):
            raise ValueError("task adaptation optimizer hyperparameters are missing")
        optimizer_cls = getattr(torch.optim, optimizer_name, None)
        if optimizer_cls is None:
            raise ValueError("task adaptation optimizer is not registered")
        accepted = set(inspect.signature(optimizer_cls).parameters)
        optimizer_hparams = {
            key: value for key, value in defaults.items() if key in accepted
        }
    optimizer_hparams = dict(optimizer_hparams)
    model = _unwrap_training_model(getattr(trainer, "model", None))
    if not isinstance(model, torch.nn.Module):
        raise ValueError("task adaptation resume model is invalid")
    before_state = _lossless_model_state_dict(model)
    before_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    before_training = {
        id(module): module.training
        for module in model.modules()
    }
    try:
        task_state = _lossless_task_state_from_payload(payload, provenance)
        if _lossless_state_sha256(task_state) != provenance["task_state_sha256"]:
            raise ValueError("task adaptation lossless state digest mismatch")
        model.load_state_dict(task_state, strict=True)
        calibration_path = Path(provenance["checkpoint_path"]).expanduser().resolve(strict=False)
        calibration_provenance = {
            "condition": provenance["condition"],
            "checkpoint_path": calibration_path.as_posix(),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
        }
        phase = task_adaptation_phase(
            model,
            condition=provenance["condition"],
            calibration_checkpoint_path=calibration_path,
            calibration_provenance=calibration_provenance,
            calibration_checkpoint_role="calibration_last",
            optimizer_name=optimizer_name,
            optimizer_hparams=optimizer_hparams,
            eta_schedule=eta_schedule,
            updates_per_epoch=updates_per_epoch,
            load_calibration=False,
        )
        if phase.semantic_state_sha256 != provenance["semantic_state_sha256"]:
            raise ValueError("task adaptation semantic hash provenance mismatch")
        if tuple(phase.semantic_module_names) != tuple(provenance["semantic_module_names"]):
            raise ValueError("task adaptation semantic module provenance mismatch")
        if dict(phase.task_parameter_categories) != dict(provenance["task_parameter_categories"]):
            raise ValueError("task adaptation task parameter provenance mismatch")
        _validate_task_adaptation_journal(
            phase,
            provenance["semantic_state_journal"],
            completed_epoch=provenance["epoch"],
        )
        return phase
    except Exception as error:
        try:
            model.load_state_dict(before_state, strict=True)
            for name, parameter in model.named_parameters():
                parameter.requires_grad = before_requires_grad[name]
            for module in model.modules():
                module.training = before_training[id(module)]
        except Exception as rollback_error:
            raise RuntimeError("task adaptation resume rollback failed") from rollback_error
        if isinstance(error, ValueError):
            raise
        raise ValueError("task adaptation resume phase reconstruction failed") from error


def _synchronize_ema_with_model(trainer: object) -> None:
    """Make an existing Ultralytics EMA snapshot exactly match live task weights."""

    ema = getattr(trainer, "ema", None)
    ema_model = getattr(ema, "ema", None)
    model = _unwrap_training_model(getattr(trainer, "model", None))
    if not isinstance(ema_model, torch.nn.Module) or not isinstance(model, torch.nn.Module):
        return
    ema_model.load_state_dict(model.state_dict(), strict=True)
    if hasattr(ema, "updates"):
        ema.updates = 0


class IFDRDetectionTrainer(DetectionTrainer):
    """Ultralytics-compatible trainer that owns the IFDR model lifecycle."""

    IFDR_LOSS_NAMES = (
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "factor_loss",
        "counterfactual_loss",
    )

    def __init__(
        self,
        cfg: dict | str = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        *,
        fusion_schedule: FusionSchedule | None = None,
        component_switches: IFDRComponentSwitches | None = None,
        intervention_seed: int | None = None,
        intervention_policy: SamplingPolicy | None = None,
        gradient_diagnostic_interval: int = 0,
    ) -> None:
        if (
            isinstance(gradient_diagnostic_interval, bool)
            or not isinstance(gradient_diagnostic_interval, int)
            or gradient_diagnostic_interval < 0
        ):
            raise ValueError(
                "gradient_diagnostic_interval must be a non-negative integer"
            )
        self.fusion_schedule = fusion_schedule or FusionSchedule()
        self.component_switches = (
            component_switches or IFDRComponentSwitches()
        )
        self.fusion_schedule_value = 0.0
        self.intervention_seed = intervention_seed
        self.intervention_policy = intervention_policy or SamplingPolicy()
        self.gradient_diagnostic_interval = gradient_diagnostic_interval
        self.task_adaptation_phase = None
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.loss_names = self.IFDR_LOSS_NAMES
        if self.intervention_seed is None:
            self.intervention_seed = int(self.args.seed)
        if (
            isinstance(self.intervention_seed, bool)
            or not isinstance(self.intervention_seed, int)
            or self.intervention_seed < 0
        ):
            raise ValueError("intervention_seed must be a non-negative integer")
        self.add_callback("on_train_epoch_start", apply_fusion_schedule)
        self.add_callback("on_train_epoch_start", task_adaptation_epoch_start)
        self.add_callback("on_train_epoch_end", task_adaptation_epoch_commit)
        self.add_callback(
            "on_train_batch_end",
            flush_gradient_diagnostics,
        )

    def resume_training(self, ckpt):
        """Resume task checkpoints without loading their narrow optimizer into the parent optimizer."""

        provenance = ckpt.get("task_adaptation_provenance") if isinstance(ckpt, Mapping) else None
        if not isinstance(provenance, Mapping):
            return super().resume_training(ckpt)
        start_epoch = ckpt.get("epoch", -1) + 1
        if (
            isinstance(start_epoch, bool)
            or not isinstance(start_epoch, int)
            or not 0 < start_epoch < self.epochs
        ):
            raise AssertionError(
                f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume."
            )
        self.best_fitness = ckpt.get("best_fitness")
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _setup_train(self):
        """Run Ultralytics setup, then rebind any adaptation phase to its final model."""

        pre_setup_args = getattr(self, "args", None)
        resume_hint = getattr(pre_setup_args, "resume", None)
        resume_requested = bool(getattr(self, "resume", False) or resume_hint)
        result = super()._setup_train()
        phase = getattr(self, "task_adaptation_phase", None)

        if resume_requested:
            checkpoint_path = _resume_checkpoint_path(self, resume_hint)
            if checkpoint_path is None:
                raise ValueError("task adaptation resume checkpoint is not specified")
            try:
                payload = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception as error:
                raise ValueError("unable to load task adaptation resume checkpoint") from error
            if not isinstance(payload, Mapping):
                if phase is None:
                    return result
                raise ValueError("task adaptation resume checkpoint has no provenance")
            provenance = payload.get("task_adaptation_provenance")
            if not isinstance(provenance, Mapping):
                if phase is None:
                    return result
                raise ValueError("task adaptation resume checkpoint has no provenance")
            if phase is not None and provenance.get("condition") != phase.condition:
                raise ValueError("task adaptation resume condition mismatch")
            rebuilt = _task_phase_from_resume_provenance(
                self,
                checkpoint_path,
                provenance,
                payload,
            )
            apply_task_adaptation_phase(self, phase=rebuilt)
            resume_task_adaptation_phase(self, checkpoint_path)
            _synchronize_ema_with_model(self)
            return result

        if phase is None:
            return result

        # BaseTrainer._setup_train is allowed to rebuild the model and replace
        # optimizer/scheduler state.  Recreate the condition-local phase from
        # its immutable calibration provenance after that lifecycle boundary;
        # never reuse the optimizer that the parent just installed.
        from ifdr_yolo.experiments.factor_repair import task_adaptation_phase

        calibration_provenance = {
            "condition": phase.condition,
            "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
            "checkpoint_sha256": phase.calibration_checkpoint_sha256,
        }
        rebuilt = task_adaptation_phase(
            _unwrap_training_model(getattr(self, "model", None)),
            condition=phase.condition,
            calibration_checkpoint_path=phase.calibration_checkpoint_path,
            calibration_provenance=calibration_provenance,
            calibration_checkpoint_role=phase.calibration_checkpoint_role,
            optimizer_name=phase.optimizer_name,
            optimizer_hparams=dict(phase.optimizer_hparams),
            eta_schedule=phase.eta_schedule,
            primary_checkpoint=phase.primary_checkpoint,
            updates_per_epoch=phase.updates_per_epoch,
        )
        apply_task_adaptation_phase(self, phase=rebuilt)
        _synchronize_ema_with_model(self)
        return result

    def _model_train(self):
        """Run Ultralytics' train transition, then restore semantic eval mode."""

        result = super()._model_train()
        phase = getattr(self, "task_adaptation_phase", None)
        if phase is None:
            return result
        from ifdr_yolo.experiments.factor_repair import (
            enforce_semantic_eval_mode,
            semantic_module_ids,
            verify_semantic_state,
        )

        model = _unwrap_training_model(getattr(self, "model", None))
        ids = semantic_module_ids(model)
        enforce_semantic_eval_mode(model, ids)
        verify_semantic_state(model, phase)
        return result

    def _setup_scheduler(self):
        result = super()._setup_scheduler()
        phase = getattr(self, "task_adaptation_phase", None)
        if phase is not None:
            _bind_task_adaptation_scheduler(self, phase)
        return result

    def final_eval(self):
        """Evaluate only the provenance-validated fixed-budget ``last.pt``."""

        phase = getattr(self, "task_adaptation_phase", None)
        if phase is None:
            return super().final_eval()
        primary = _validated_adaptation_model(self, phase)
        validator = self.validator
        validator.args.plots = self.args.plots
        validator.args.compile = False
        self.metrics = validator(model=primary)
        self.metrics.pop("fitness", None)
        self.epoch += 1
        try:
            self.run_callbacks("on_fit_epoch_end")
        finally:
            self.epoch -= 1

    def optimizer_step(self):
        # Use the concrete parent function so this boundary remains directly
        # callable in small trainer doubles used by phase-level tests.
        result = DetectionTrainer.optimizer_step(self)
        phase = getattr(self, "task_adaptation_phase", None)
        if phase is not None:
            steps = int(getattr(self, "task_adaptation_optimizer_steps", 0)) + 1
            if steps > phase.update_count:
                raise RuntimeError("task adaptation exceeded registered optimizer updates")
            self.task_adaptation_optimizer_steps = steps
        return result

    def save_model(self):
        """Preserve Ultralytics checkpoint fields and append adaptation provenance."""

        phase = getattr(self, "task_adaptation_phase", None)
        if phase is None:
            return super().save_model()
        path = Path(getattr(self, "last")).expanduser().resolve(strict=False)
        if path.name != phase.primary_checkpoint or path.name != "last.pt":
            raise ValueError("task adaptation primary checkpoint must be last.pt")
        epoch = getattr(self, "epoch", None)
        final_round = isinstance(epoch, int) and not isinstance(epoch, bool) and (
            epoch + 1 >= phase.epochs
        )
        if final_round and getattr(self, "task_adaptation_optimizer_steps", None) != phase.update_count:
            raise RuntimeError(
                "task adaptation optimizer update budget is incomplete"
            )

        # BaseTrainer writes ``self.last`` directly.  Redirect only that path
        # to a same-directory temporary file so any later provenance failure
        # leaves the previously validated real last.pt untouched.
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        original_last = self.last
        self.last = temporary
        try:
            result = super().save_model()
            if result is False:
                return result
            from ifdr_yolo.experiments.factor_repair import verify_semantic_state

            model = _unwrap_training_model(getattr(self, "model", None))
            checkpoint_epoch = getattr(self, "epoch", None)
            checkpoint_steps = getattr(self, "task_adaptation_optimizer_steps", None)
            semantic_hash = verify_semantic_state(
                model,
                phase,
                event="final_checkpoint" if final_round else None,
                epoch=checkpoint_epoch if final_round else None,
                optimizer_steps=checkpoint_steps if final_round else None,
            )
            if final_round:
                _validate_task_adaptation_journal(
                    phase,
                    phase.semantic_state_journal,
                    completed_epoch=checkpoint_epoch,
                    require_final=True,
                )
            lossless_state = _lossless_model_state_dict(model)
            lossless_digest = _lossless_state_sha256(lossless_state)
            payload = torch.load(temporary, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError("Ultralytics checkpoint payload must be a mapping")
            payload["task_adaptation_provenance"] = {
                "condition": phase.condition,
                "checkpoint_role": phase.calibration_checkpoint_role,
                "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
                "checkpoint_sha256": phase.calibration_checkpoint_sha256,
                "semantic_state_sha256": semantic_hash,
                "semantic_module_names": tuple(phase.semantic_module_names),
                "task_parameter_categories": dict(phase.task_parameter_categories),
                "updates_per_epoch": phase.updates_per_epoch,
                "expected_optimizer_steps": phase.update_count,
                "optimizer_steps": int(getattr(self, "task_adaptation_optimizer_steps", 0)),
                "primary_checkpoint": phase.primary_checkpoint,
                "task_checkpoint_path": path.as_posix(),
                "semantic_state_journal": tuple(
                    dict(record) for record in phase.semantic_state_journal
                ),
                "optimizer_name": phase.optimizer_name,
                "optimizer_hparams": dict(phase.optimizer_hparams),
                "optimizer_defaults": dict(phase.optimizer.defaults),
                "eta_schedule": tuple(phase.eta_schedule),
                "task_state_key": TASK_ADAPTATION_STATE_KEY,
                "task_state_source": TASK_ADAPTATION_STATE_SOURCE,
                "task_state_sha256": lossless_digest,
                "epoch": getattr(self, "epoch", None),
            }
            payload[TASK_ADAPTATION_STATE_KEY] = lossless_state
            _atomic_torch_save(payload, temporary)
            verified_payload = torch.load(
                temporary,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(verified_payload, Mapping):
                raise ValueError("published task checkpoint payload is invalid")
            verified_provenance = verified_payload.get("task_adaptation_provenance")
            if not isinstance(verified_provenance, Mapping):
                raise ValueError("published task checkpoint provenance is missing")
            _lossless_task_state_from_payload(verified_payload, verified_provenance)
            if verified_provenance.get("task_state_sha256") != lossless_digest:
                raise ValueError("published task checkpoint state digest mismatch")
            expected_provenance = {
                "condition": phase.condition,
                "checkpoint_role": phase.calibration_checkpoint_role,
                "checkpoint_path": phase.calibration_checkpoint_path.as_posix(),
                "checkpoint_sha256": phase.calibration_checkpoint_sha256,
                "semantic_state_sha256": semantic_hash,
                "primary_checkpoint": phase.primary_checkpoint,
                "task_checkpoint_path": path.as_posix(),
                "updates_per_epoch": phase.updates_per_epoch,
                "expected_optimizer_steps": phase.update_count,
                "optimizer_steps": int(getattr(self, "task_adaptation_optimizer_steps", 0)),
                "optimizer_name": phase.optimizer_name,
                "eta_schedule": tuple(phase.eta_schedule),
                "task_state_key": TASK_ADAPTATION_STATE_KEY,
                "task_state_source": TASK_ADAPTATION_STATE_SOURCE,
                "task_state_sha256": lossless_digest,
                "epoch": getattr(self, "epoch", None),
            }
            if any(
                verified_provenance.get(key) != value
                for key, value in expected_provenance.items()
            ):
                raise ValueError("published task checkpoint provenance mismatch")
            os.replace(temporary, path)
            return result
        finally:
            self.last = original_last
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> IFDRDetectionModel:
        model = self.set_model_names_for_load(
            IFDRDetectionModel(
                cfg=cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                gradient_diagnostic_interval=getattr(
                    self, "gradient_diagnostic_interval", 0
                ),
            )
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = self.IFDR_LOSS_NAMES
        return validator

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ):
        model = _unwrap_training_model(self.model)
        stride = max(int(model.stride.max()), 32)
        component_switches = getattr(
            self,
            "component_switches",
            IFDRComponentSwitches(),
        )
        return build_ifdr_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
            intervention_seed=self.intervention_seed,
            interventions_enabled=(
                mode == "train"
                and component_switches.interventions
            ),
            counterfactual_enabled=(
                mode == "train"
                and component_switches.counterfactual_consistency
            ),
            intervention_policy=getattr(
                self,
                "intervention_policy",
                None,
            ),
        )

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        counterfactual = batch.get(COUNTERFACTUAL_IMAGE_KEY)
        if counterfactual is None:
            return batch
        if not isinstance(counterfactual, torch.Tensor):
            raise RuntimeError("counterfactual images must be a tensor")
        counterfactual = counterfactual.float() / 255
        if counterfactual.shape[-2:] != batch["img"].shape[-2:]:
            counterfactual = F.interpolate(
                counterfactual,
                size=batch["img"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        batch[COUNTERFACTUAL_IMAGE_KEY] = counterfactual
        return batch


class FactorCalibrationTrainer(IFDRDetectionTrainer):
    """Ultralytics trainer bound to one immutable F0--F3 runtime."""

    IFDR_LOSS_NAMES = (
        "synthetic_factor_loss",
        "natural_factor_loss",
        "specificity_loss",
    )

    @staticmethod
    def ultralytics_overrides(runtime: object) -> dict[str, object]:
        """Return the complete registered Ultralytics override mapping."""

        def value(name: str) -> object:
            result = getattr(runtime, name, None)
            if result is None:
                raise ValueError(f"factor runtime is missing {name}")
            return result

        geometry_overrides = getattr(runtime, "geometry_overrides", CALIBRATION_GEOMETRY_OVERRIDES)
        if not isinstance(geometry_overrides, Mapping):
            raise ValueError("factor runtime geometry overrides are invalid")
        expected_geometry = dict(CALIBRATION_GEOMETRY_OVERRIDES)
        if dict(geometry_overrides) != expected_geometry:
            raise ValueError("factor calibration geometry overrides must all be zero")
        return {
            "model": str(Path(value("model_yaml")).resolve()),
            "data": str(Path(value("resolved_data_yaml")).resolve()),
            "pretrained": str(Path(value("initialization_checkpoint")).resolve()),
            "epochs": int(value("epochs")),
            "imgsz": int(value("imgsz")),
            "batch": int(value("batch")),
            "workers": int(value("workers")),
            "device": str(value("device")),
            "optimizer": str(value("optimizer")),
            "lr0": float(value("lr0")),
            "lrf": float(value("lrf")),
            "momentum": float(value("momentum")),
            "weight_decay": float(value("weight_decay")),
            "warmup_epochs": float(value("warmup_epochs")),
            "seed": int(value("seed")),
            "amp": bool(value("amp")),
            "deterministic": bool(value("deterministic")),
            "cache": bool(value("cache")),
            **expected_geometry,
            "patience": 0,
            "save_dir": str(Path(value("run_dir")).resolve()),
        }

    def __init__(
        self,
        runtime: object,
        *,
        config: object | None = None,
        condition: str | None = None,
        run_dir: str | Path | None = None,
        metadata_index: object | None = None,
        draw_callback: object | None = None,
        draw_journal: object | None = None,
    ) -> None:
        del draw_journal  # the callback owns the durable journal binding
        runtime_condition = getattr(runtime, "condition", None)
        self.runtime = runtime
        self.config = config if config is not None else getattr(runtime, "config", None)
        self.condition = condition if condition is not None else runtime_condition
        if self.condition not in {"F0", "F1", "F2", "F3"}:
            raise ValueError("factor calibration condition must be F0, F1, F2, or F3")
        self.run_dir = Path(run_dir if run_dir is not None else getattr(runtime, "run_dir")).resolve()
        self.metadata_index = (
            metadata_index if metadata_index is not None else getattr(runtime, "metadata_index", None)
        )
        if self.metadata_index is None:
            raise ValueError("factor calibration requires a metadata index")
        self.specificity_rejection_counter = SpecificityRejectionCounter()
        self.seed = int(getattr(runtime, "seed"))
        self.draw_callback = draw_callback if callable(draw_callback) else None
        self.calibration_phase = None
        self._calibration_optimizer_bound = False
        overrides = self.ultralytics_overrides(runtime)
        super().__init__(
            overrides=overrides,
            component_switches=IFDRComponentSwitches(
                fusion_gate=False,
                dcli=False,
                factor_supervision=True,
                interventions=True,
            ),
            intervention_seed=self.seed,
        )
        self.add_callback("on_train_epoch_start", self._record_epoch_draw)

    def _record_epoch_draw(self, trainer: object | None = None) -> None:
        owner = trainer if trainer is not None and hasattr(trainer, "epoch") else self
        epoch = getattr(owner, "epoch", None)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("factor calibration epoch is invalid")
        key = f"{self.condition}:seed={self.seed}:epoch={epoch}"
        if self.draw_callback is not None:
            self.draw_callback(epoch, key)

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ):
        model = _unwrap_training_model(self.model)
        stride = max(int(model.stride.max()), 32)
        component_switches = getattr(self, "component_switches", IFDRComponentSwitches())
        train_mode = mode == "train"
        return build_ifdr_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
            intervention_seed=self.intervention_seed,
            interventions_enabled=train_mode and component_switches.interventions,
            counterfactual_enabled=train_mode and component_switches.counterfactual_consistency,
            intervention_policy=getattr(self, "intervention_policy", None),
            calibration_enabled=train_mode,
            metadata_index=self.metadata_index if train_mode else None,
            specificity_rejection_counter=self.specificity_rejection_counter,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        base = batch.get("img")
        if not isinstance(base, torch.Tensor):
            raise RuntimeError("calibration batch image is missing")
        device = getattr(self, "device", base.device)
        target_shape = tuple(base.shape[-2:])
        for key in (CLEAN_IMAGE_KEY, TARGET_IMAGE_KEY, BACKGROUND_IMAGE_KEY):
            view = batch.get(key)
            if view is None:
                raise RuntimeError(f"calibration batch view is missing: {key}")
            if not isinstance(view, torch.Tensor):
                raise RuntimeError(f"calibration batch view must be a tensor: {key}")
            view = view.to(device=device).float() / 255.0
            if tuple(view.shape[-2:]) != target_shape:
                view = F.interpolate(view, size=target_shape, mode="bilinear", align_corners=False)
            if view.shape != base.shape:
                raise RuntimeError(f"calibration batch view shape mismatch: {key}")
            batch[key] = view
        return batch

    def _setup_train(self):
        """Use the project-local AMP preflight only during parent setup."""

        import ultralytics.engine.trainer as ultralytics_trainer

        original_check_amp = ultralytics_trainer.check_amp
        ultralytics_trainer.check_amp = factor_amp_preflight
        try:
            return super()._setup_train()
        finally:
            ultralytics_trainer.check_amp = original_check_amp

    @staticmethod
    def _filter_optimizer_for_phase(optimizer: object, phase: object, model: object) -> None:
        named = tuple(model.named_parameters())
        expected_names = tuple(getattr(phase, "trainable_parameter_names", ()))
        expected_ids = {id(parameter) for name, parameter in named if name in expected_names}
        if not expected_ids or len(expected_ids) != len(expected_names):
            raise ValueError("semantic calibration phase trainable parameters are invalid")
        for name, parameter in named:
            parameter.requires_grad = name in expected_names
        groups = getattr(optimizer, "param_groups", None)
        if not isinstance(groups, list):
            raise TypeError("optimizer must expose param_groups")
        for group in groups:
            parameters = group.get("params")
            if not isinstance(parameters, list):
                parameters = list(parameters or ())
            group["params"] = [parameter for parameter in parameters if parameter.requires_grad]
        actual_ids = {
            id(parameter)
            for group in groups
            for parameter in group.get("params", ())
        }
        if actual_ids != expected_ids:
            raise AssertionError("optimizer parameters do not exactly match semantic calibration phase")

    def build_optimizer(
        self,
        model: object,
        name: str = "auto",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ):
        """Bind semantic calibration at the post-model optimizer boundary."""

        from ifdr_yolo.experiments.factor_repair import semantic_calibration_phase

        model = _unwrap_training_model(model)
        if self.calibration_phase is None:
            self.calibration_phase = semantic_calibration_phase(
                model,
                variant=self.condition,
                epochs=30,
            )
        optimizer = super().build_optimizer(
            model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )
        self._filter_optimizer_for_phase(optimizer, self.calibration_phase, model)
        self._calibration_optimizer_bound = True
        runtime_writer = getattr(self.runtime, "write_provenance", None)
        if callable(runtime_writer):
            runtime_writer(
                trainable=tuple(self.calibration_phase.trainable_parameter_names),
                frozen=tuple(self.calibration_phase.frozen_parameter_names),
            )
        return optimizer

    def validate(self):
        """Run development-set detection validation with its five loss terms.

        Calibration training reports three factor objectives.  The validation
        dataloader intentionally remains the ordinary detection split, so its
        model loss reports the five IFDR detection terms.  Ultralytics sizes
        the validator accumulator from ``trainer.loss_items``; temporarily
        switching that shape and label set avoids a 3-vs-5 broadcast while
        preserving the calibration labels for subsequent training epochs.
        """

        calibration_loss_items = self.loss_items
        calibration_loss_names = self.loss_names
        try:
            self.loss_items = torch.zeros(
                len(IFDRDetectionTrainer.IFDR_LOSS_NAMES),
                device=calibration_loss_items.device,
                dtype=calibration_loss_items.dtype,
            )
            self.loss_names = IFDRDetectionTrainer.IFDR_LOSS_NAMES
            return super().validate()
        finally:
            self.loss_items = calibration_loss_items
            self.loss_names = calibration_loss_names

    def evaluate_primary_last(self, path: str | Path):
        """Evaluate only a caller-supplied, provenance-bound ``last.pt`` path."""

        candidate = Path(path).expanduser().resolve()
        run_dir = Path(getattr(self.runtime, "run_dir")).resolve()
        allowed = {run_dir / "last.pt", run_dir / "weights" / "last.pt"}
        if candidate.name != "last.pt" or candidate not in allowed:
            raise ValueError("factor calibration evaluator requires the run's last.pt")
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise FileNotFoundError(f"primary checkpoint is missing or empty: {candidate}")
        validator = getattr(self, "validator", None)
        if not callable(validator):
            raise ValueError("factor calibration validator is not initialized")
        # The training validation call exhausts BaseValidator.dataloader's
        # generator.  Independent checkpoint evaluation must force
        # BaseValidator.__call__ to build a fresh loader rather than reusing
        # that exhausted iterator; assignment errors are intentionally
        # propagated instead of silently falling back to stale state.
        validator.dataloader = None
        # Keep evaluation path-bound: the validator receives the verified
        # checkpoint itself rather than the in-memory training model or a
        # zero-argument ``final_eval`` fallback.
        self.metrics = validator(model=candidate)
        return self.metrics


__all__ = [
    "FusionSchedule",
    "IFDRComponentSwitches",
    "IFDRDetectionTrainer",
    "FactorCalibrationTrainer",
    "factor_amp_preflight",
    "run_local_amp_preflight",
    "validate_amp_outputs",
    "apply_fusion_schedule",
    "apply_semantic_calibration_phase",
    "run_validation_without_optimizer_step",
    "apply_task_adaptation_phase",
]
