"""Lossless state boundary for the registered Stage11 single-GPU runs."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

from ifdr_yolo.experiments.kitti_seed0_training_benchmark import (
    BenchmarkEpochResumeHook,
    DATALOADER_EPOCH_SEED_BASE,
    _canonical,
    build_benchmark_state,
    restore_benchmark_training_state,
)


class Stage11RecoveryError(ValueError):
    """The live trainer cannot satisfy the formal Stage11 recovery contract."""


_STATE_FIELDS = frozenset(
    {
        "completed_epoch",
        "model",
        "ema",
        "ema_updates",
        "optimizer",
        "scaler",
        "scheduler",
        "rng_state",
        "sampler",
        "dataloader",
        "train_args",
    }
)
_COMPONENT_FIELDS = frozenset(
    {"model", "optimizer", "scheduler", "ema", "scaler", "rng", "sampler", "dataloader"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GENERATION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ARTIFACT_PATTERN = re.compile(r"epoch-([0-9]{3})\.recovery\.pt")
_FINAL_CHECKPOINT_PATTERN = re.compile(r"epoch-(030)\.checkpoint\.pt")


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage11RecoveryError(f"{name} must be a positive integer")
    return value


def _qualified_name(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _loader_boundary(trainer: Any, next_epoch: int) -> tuple[dict[str, object], dict[str, object]]:
    if isinstance(next_epoch, bool) or not isinstance(next_epoch, int) or next_epoch < 1:
        raise Stage11RecoveryError("next epoch must be a positive integer")
    if getattr(trainer, "world_size", 1) != 1:
        raise Stage11RecoveryError("Stage11 recovery supports only registered single-GPU training")
    loader = getattr(trainer, "train_loader", None)
    sampler = getattr(loader, "sampler", None)
    if loader is None or sampler is None:
        raise Stage11RecoveryError("trainer has no sampler-backed train loader")
    if hasattr(sampler, "num_replicas"):
        raise Stage11RecoveryError("Stage11 recovery supports only registered single-GPU sampling")
    generator = getattr(loader, "generator", None)
    if not callable(getattr(generator, "manual_seed", None)):
        raise Stage11RecoveryError("train loader has no seedable generator")
    if not callable(getattr(loader, "reset", None)):
        raise Stage11RecoveryError("train loader has no deterministic reset")

    sample_count = _positive_integer("sampler sample count", getattr(sampler, "num_samples", None))
    batch_count = _positive_integer("dataloader batch count", len(loader))
    dataset = getattr(loader, "dataset", None)
    dataset_count = _positive_integer("dataloader dataset count", len(dataset) if dataset is not None else None)
    batch_size = _positive_integer("dataloader batch size", getattr(loader, "batch_size", None))
    workers = getattr(loader, "num_workers", None)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise Stage11RecoveryError("dataloader workers must be a nonnegative integer")
    replacement = getattr(sampler, "replacement", None)
    if replacement is not False:
        raise Stage11RecoveryError("Stage11 sampler must use replacement=false")
    drop_last = getattr(loader, "drop_last", None)
    if not isinstance(drop_last, bool):
        raise Stage11RecoveryError("dataloader drop_last must be boolean")

    sampler_state: dict[str, object] = {
        "schema": "stage11-single-gpu-epoch-sampler-v1",
        "class": _qualified_name(sampler),
        "sample_count": sample_count,
        "replacement": False,
        "next_epoch": next_epoch,
        "next_epoch_seed": DATALOADER_EPOCH_SEED_BASE + next_epoch,
    }
    dataloader_state: dict[str, object] = {
        "schema": "stage11-single-gpu-epoch-loader-v1",
        "class": _qualified_name(loader),
        "dataset_class": _qualified_name(dataset),
        "dataset_count": dataset_count,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "workers": workers,
        "drop_last": drop_last,
        "world_size": 1,
        "next_epoch": next_epoch,
        "next_epoch_seed": DATALOADER_EPOCH_SEED_BASE + next_epoch,
        "reset_policy": "manual_seed_then_infinite_loader_reset",
    }
    return sampler_state, dataloader_state


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_identity(path: Path, digest: str) -> tuple[int, int, int, str]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Stage11RecoveryError("Stage11 recovery artifact must be a regular file")
    return metadata.st_dev, metadata.st_ino, metadata.st_size, digest


def _matches_identity(path: Path, identity: tuple[int, int, int, str]) -> bool:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            return False
        return (metadata.st_dev, metadata.st_ino, metadata.st_size, _sha_file(path)) == identity
    except OSError:
        return False


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_checkpoint_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise Stage11RecoveryError("observed checkpoint must be a nonempty regular file")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    left = before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    right = after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    if left != right:
        raise Stage11RecoveryError("observed checkpoint changed while hashing")
    return before.st_size, digest.hexdigest()


def _stable_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise Stage11RecoveryError("Stage11 recovery artifact must be a nonempty regular file")
            content = stream.read()
            after = os.fstat(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    observed = path.lstat()
    left = before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    right = after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    current = observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns
    if left != right or left != current or len(content) != before.st_size:
        raise Stage11RecoveryError("Stage11 recovery artifact changed while reading")
    return content


def _recovery_root(path: Path) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute():
        raise Stage11RecoveryError("Stage11 recovery root must be absolute")
    current = supplied
    while True:
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Stage11RecoveryError("Stage11 recovery root ancestors must not be symlinks")
        if current == current.parent:
            break
        current = current.parent
    root = supplied.resolve(strict=True)
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise Stage11RecoveryError("Stage11 recovery root must be a directory")
    return root


def _rollback_created(path: Path, identity: tuple[int, int, int, str], temporary: Path) -> None:
    failures: list[BaseException] = []
    if path.exists() or path.is_symlink():
        if not _matches_identity(path, identity):
            failures.append(Stage11RecoveryError("Stage11 artifact rollback identity differs"))
        else:
            try:
                path.unlink()
            except BaseException as error:
                failures.append(error)
    if temporary.exists() or temporary.is_symlink():
        try:
            temporary.unlink()
        except BaseException as error:
            failures.append(error)
    try:
        _fsync_directory(path.parent)
    except BaseException as error:
        failures.append(error)
    if failures:
        raise Stage11RecoveryError("Stage11 artifact rollback is incomplete") from failures[0]


def _write_create_only(path: Path, content: bytes) -> tuple[int, int, int, str]:
    if path.exists() or path.is_symlink():
        raise FileExistsError("Stage11 epoch recovery artifact must be fresh")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    expected: tuple[int, int, int, str] | None = None
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(content).hexdigest()
        expected = _regular_identity(temporary, digest)
        os.link(temporary, path)
        linked = True
        _fsync_directory(path.parent)
        if not _matches_identity(path, expected):
            raise Stage11RecoveryError("Stage11 artifact readback differs")
        temporary.unlink()
        _fsync_directory(path.parent)
        return expected
    except BaseException:
        if linked and expected is not None:
            _rollback_created(path, expected, temporary)
        elif temporary.exists() or temporary.is_symlink():
            temporary.unlink()
            _fsync_directory(path.parent)
        raise


def _serialize_artifact(artifact: Mapping[str, Any]) -> bytes:
    try:
        import torch
    except ImportError as error:
        raise Stage11RecoveryError("Stage11 recovery artifact requires PyTorch") from error
    buffer = BytesIO()
    torch.save(dict(artifact), buffer)
    return buffer.getvalue()


def _deserialize_artifact(content: bytes) -> Mapping[str, Any]:
    try:
        import torch
        artifact = torch.load(BytesIO(content), map_location="cpu", weights_only=False)
    except Exception as error:
        raise Stage11RecoveryError("Stage11 recovery artifact is not CPU-loadable") from error
    if not isinstance(artifact, Mapping):
        raise Stage11RecoveryError("Stage11 recovery artifact is not a mapping")
    return artifact


def _component_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    return {component: component_sha256(state, component) for component in sorted(_COMPONENT_FIELDS)}


def publish_stage11_local_artifact(
    recovery_root: Path,
    checkpoint: Path,
    state: Mapping[str, Any],
    *,
    execution_identity_sha256: str,
    boundary_monotonic_seconds: float,
) -> Path:
    """Publish one immutable local recovery envelope for the external watcher."""
    root = _recovery_root(recovery_root)
    if set(state) != _STATE_FIELDS:
        raise Stage11RecoveryError("Stage11 recovery state schema is not exact")
    epoch = state.get("completed_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 30:
        raise Stage11RecoveryError("completed epoch must be in the registered 1..30 range")
    if (
        isinstance(boundary_monotonic_seconds, bool)
        or not isinstance(boundary_monotonic_seconds, (int, float))
        or not math.isfinite(float(boundary_monotonic_seconds))
        or float(boundary_monotonic_seconds) < 0.0
    ):
        raise Stage11RecoveryError("boundary monotonic time must be finite and nonnegative")
    observed_checkpoint = Path(checkpoint)
    frozen_checkpoint: Path | None = None
    frozen_identity: tuple[int, int, int, str] | None = None
    if epoch == 30:
        checkpoint_content = _stable_regular_bytes(observed_checkpoint)
        frozen_checkpoint = root / "epoch-030.checkpoint.pt"
        frozen_identity = _write_create_only(frozen_checkpoint, checkpoint_content)
        observed_checkpoint = frozen_checkpoint
    try:
        checkpoint_size, checkpoint_sha256 = _stable_checkpoint_identity(observed_checkpoint)
        artifact: dict[str, Any] = {
            "schema": "stage11-local-recovery-artifact-v1",
            "boundary_monotonic_seconds": float(boundary_monotonic_seconds),
            "execution_identity_sha256": _sha256("execution_identity_sha256", execution_identity_sha256),
            "epoch_completed": epoch,
            "checkpoint_size": checkpoint_size,
            "checkpoint_sha256": checkpoint_sha256,
            "component_sha256": _component_hashes(state),
            "state": dict(state),
        }
        target = root / f"epoch-{epoch:03d}.recovery.pt"
        _write_create_only(target, _serialize_artifact(artifact))
        return target
    except BaseException:
        if frozen_checkpoint is not None and frozen_identity is not None:
            try:
                if not _matches_identity(frozen_checkpoint, frozen_identity):
                    raise Stage11RecoveryError("Stage11 final checkpoint rollback identity differs")
                frozen_checkpoint.unlink()
                _fsync_directory(root)
            except BaseException as rollback_error:
                raise Stage11RecoveryError("Stage11 final checkpoint rollback is incomplete") from rollback_error
        raise


def _validated_artifact(
    artifact: Mapping[str, Any],
    checkpoint: Path,
    execution_identity_sha256: str,
) -> Mapping[str, Any]:
    required = {
        "schema", "boundary_monotonic_seconds", "execution_identity_sha256", "epoch_completed", "checkpoint_size",
        "checkpoint_sha256", "component_sha256", "state",
    }
    if set(artifact) != required or artifact.get("schema") != "stage11-local-recovery-artifact-v1":
        raise Stage11RecoveryError("Stage11 recovery artifact schema is not exact")
    if artifact.get("execution_identity_sha256") != _sha256("execution_identity_sha256", execution_identity_sha256):
        raise Stage11RecoveryError("Stage11 recovery artifact execution identity differs")
    boundary = artifact.get("boundary_monotonic_seconds")
    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, (int, float))
        or not math.isfinite(float(boundary))
        or float(boundary) < 0.0
    ):
        raise Stage11RecoveryError("Stage11 recovery artifact boundary time differs")
    state = artifact.get("state")
    if not isinstance(state, Mapping) or set(state) != _STATE_FIELDS:
        raise Stage11RecoveryError("Stage11 recovery artifact state schema is not exact")
    if artifact.get("epoch_completed") != state.get("completed_epoch"):
        raise Stage11RecoveryError("Stage11 recovery artifact epoch differs")
    if artifact.get("component_sha256") != _component_hashes(state):
        raise Stage11RecoveryError("Stage11 recovery artifact component identity differs")
    size, digest = _stable_checkpoint_identity(Path(checkpoint))
    if artifact.get("checkpoint_size") != size or artifact.get("checkpoint_sha256") != digest:
        raise Stage11RecoveryError("Stage11 recovery artifact checkpoint identity differs")
    return state


def load_latest_stage11_local_state(
    recovery_root: Path,
    checkpoint: Path,
    *,
    execution_identity_sha256: str,
) -> Mapping[str, Any]:
    """Load only an exact strict prefix ending at the current checkpoint."""
    root = _recovery_root(recovery_root)
    epochs: list[tuple[int, Path]] = []
    final_checkpoint: Path | None = None
    for path in root.iterdir():
        match = _ARTIFACT_PATTERN.fullmatch(path.name)
        if match is None:
            final_match = _FINAL_CHECKPOINT_PATTERN.fullmatch(path.name)
            if final_match is None or final_checkpoint is not None or not path.is_file() or path.is_symlink():
                raise Stage11RecoveryError("Stage11 recovery root contains an unregistered entry")
            final_checkpoint = path
            continue
        if not path.is_file() or path.is_symlink():
            raise Stage11RecoveryError("Stage11 recovery root contains an unregistered entry")
        epochs.append((int(match.group(1)), path))
    epochs.sort()
    if not epochs or [epoch for epoch, _ in epochs] != list(range(1, epochs[-1][0] + 1)):
        raise Stage11RecoveryError("Stage11 recovery artifacts are not a strict epoch prefix")
    if (epochs[-1][0] == 30) != (final_checkpoint is not None):
        raise Stage11RecoveryError("Stage11 final checkpoint snapshot differs")
    content = _stable_regular_bytes(epochs[-1][1])
    observed_checkpoint = final_checkpoint if final_checkpoint is not None else Path(checkpoint)
    return _validated_artifact(_deserialize_artifact(content), observed_checkpoint, execution_identity_sha256)


def validate_stage11_local_recovery_root(recovery_root: Path, *, resume: bool) -> Path:
    """Validate the runner-owned root before any trainer or GPU initialization."""
    if not isinstance(resume, bool):
        raise Stage11RecoveryError("Stage11 resume flag must be boolean")
    root = _recovery_root(recovery_root)
    entries = tuple(root.iterdir())
    if not resume and entries:
        raise Stage11RecoveryError("fresh Stage11 recovery root must be empty")
    if resume:
        epochs: list[int] = []
        final_checkpoint = False
        for path in entries:
            match = _ARTIFACT_PATTERN.fullmatch(path.name)
            if match is None:
                final_match = _FINAL_CHECKPOINT_PATTERN.fullmatch(path.name)
                if final_match is None or final_checkpoint or not path.is_file() or path.is_symlink():
                    raise Stage11RecoveryError("Stage11 recovery root contains an unregistered entry")
                final_checkpoint = True
                continue
            if not path.is_file() or path.is_symlink():
                raise Stage11RecoveryError("Stage11 recovery root contains an unregistered entry")
            epochs.append(int(match.group(1)))
        epochs.sort()
        if not epochs or epochs != list(range(1, epochs[-1] + 1)):
            raise Stage11RecoveryError("Stage11 resume requires a strict local recovery prefix")
        if (epochs[-1] == 30) != final_checkpoint:
            raise Stage11RecoveryError("Stage11 final checkpoint snapshot differs")
    return root


def configure_stage11_local_recovery(
    target: Any,
    output_dir: Path,
    recovery_root: Path,
    *,
    execution_identity_sha256: str,
    resume: bool,
) -> None:
    """Bind callbacks to an externally provisioned, independent recovery root."""
    if not isinstance(resume, bool):
        raise Stage11RecoveryError("Stage11 resume flag must be boolean")
    output = Path(output_dir).resolve(strict=True)
    if not output.is_dir() or output.is_symlink():
        raise Stage11RecoveryError("Stage11 output must be a real existing directory")
    root = validate_stage11_local_recovery_root(recovery_root, resume=resume)
    if root == output or root in output.parents or output in root.parents:
        raise Stage11RecoveryError("Stage11 recovery root must be independent from the training output")
    checkpoint = output / "weights" / "last.pt"

    def publisher(path: Path, state: dict[str, Any], boundary: float) -> None:
        if path != checkpoint:
            raise Stage11RecoveryError("Stage11 callback checkpoint path differs from the fixed output")
        publish_stage11_local_artifact(
            root,
            checkpoint,
            state,
            execution_identity_sha256=execution_identity_sha256,
            boundary_monotonic_seconds=boundary,
        )

    state_loader: Callable[[], Mapping[str, Any]] | None = None
    if resume:
        state_loader = lambda: load_latest_stage11_local_state(
            root,
            checkpoint,
            execution_identity_sha256=execution_identity_sha256,
        )
    configure_stage11_recovery_callbacks(target, state_loader=state_loader, publisher=publisher)


def component_sha256(state: Mapping[str, Any], component: str) -> str:
    """Hash one independently named component of the formal recovery state."""
    if component not in _COMPONENT_FIELDS:
        raise Stage11RecoveryError(f"unknown Stage11 recovery component: {component}")
    field = "rng_state" if component == "rng" else component
    if field not in state:
        raise Stage11RecoveryError(f"Stage11 recovery state has no {component} component")
    value: object = state[field]
    if component == "ema":
        if "ema_updates" not in state:
            raise Stage11RecoveryError("Stage11 recovery state has no ema update counter")
        value = {"state": value, "updates": state["ema_updates"]}
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Stage11RecoveryError(f"{name} must be a lowercase SHA256")
    return value


def build_stage11_checkpoint_receipt(
    state: Mapping[str, Any],
    *,
    task_sha256: str,
    generation: str,
    checkpoint_sha256: str,
    interval_seconds: float,
    created_at_utc: str,
) -> dict[str, object]:
    """Build the exact checkpoint receipt required by the corrected plan."""
    if set(state) != _STATE_FIELDS:
        raise Stage11RecoveryError("Stage11 recovery state schema is not exact")
    completed_epoch = state.get("completed_epoch")
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int) or not 1 <= completed_epoch <= 30:
        raise Stage11RecoveryError("completed epoch must be in the registered 1..30 range")
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        raise Stage11RecoveryError("generation must be a canonical token")
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, (int, float))
        or not math.isfinite(float(interval_seconds))
        or not 0.0 <= float(interval_seconds) <= 300.0
    ):
        raise Stage11RecoveryError("checkpoint interval must be finite and at most 300 seconds")
    if not isinstance(created_at_utc, str) or not created_at_utc.endswith("Z"):
        raise Stage11RecoveryError("created_at_utc must be an explicit UTC timestamp")

    return {
        "schema": "stage11-checkpoint-receipt-v1",
        "state": "DURABLE",
        "task_sha256": _sha256("task_sha256", task_sha256),
        "generation": generation,
        "epoch_completed": completed_epoch,
        "epoch_prefix": list(range(1, completed_epoch + 1)),
        "checkpoint_sha256": _sha256("checkpoint_sha256", checkpoint_sha256),
        "optimizer_sha256": component_sha256(state, "optimizer"),
        "scheduler_sha256": component_sha256(state, "scheduler"),
        "ema_sha256": component_sha256(state, "ema"),
        "scaler_sha256": component_sha256(state, "scaler"),
        "rng_sha256": component_sha256(state, "rng"),
        "sampler_sha256": component_sha256(state, "sampler"),
        "dataloader_sha256": component_sha256(state, "dataloader"),
        "interval_seconds": float(interval_seconds),
        "created_at_utc": created_at_utc,
    }


def build_stage11_recovery_state(trainer: Any) -> dict[str, Any]:
    """Capture every formal state component at an end-of-epoch boundary."""
    base = build_benchmark_state(trainer)
    completed_epoch = base.get("completed_epoch")
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int) or completed_epoch < 1:
        raise Stage11RecoveryError("trainer has no completed epoch boundary")
    sampler, dataloader = _loader_boundary(trainer, completed_epoch)
    state = {**base, "sampler": sampler, "dataloader": dataloader}
    if set(state) != _STATE_FIELDS:
        raise Stage11RecoveryError("Stage11 recovery state schema is not exact")
    for component in sorted(_COMPONENT_FIELDS):
        component_sha256(state, component)
    return state


def restore_stage11_recovery_state(trainer: Any, state: Mapping[str, Any]) -> None:
    """Validate the reconstructible loader boundary, then restore live state."""
    if set(state) != _STATE_FIELDS:
        raise Stage11RecoveryError("Stage11 recovery state schema is not exact")
    completed_epoch = state.get("completed_epoch")
    if completed_epoch != getattr(trainer, "start_epoch", None):
        raise Stage11RecoveryError("resume start epoch differs from the durable Stage11 boundary")
    expected_sampler, expected_dataloader = _loader_boundary(trainer, completed_epoch)
    if state.get("sampler") != expected_sampler:
        raise Stage11RecoveryError("resume sampler identity differs from the durable boundary")
    if state.get("dataloader") != expected_dataloader:
        raise Stage11RecoveryError("resume dataloader identity differs from the durable boundary")
    for component in sorted(_COMPONENT_FIELDS):
        component_sha256(state, component)

    restore_benchmark_training_state(trainer, state)
    loader = trainer.train_loader
    loader.generator.manual_seed(expected_dataloader["next_epoch_seed"])
    loader.reset()


class Stage11EpochStateHook:
    """Capture a complete state only after the ordinary checkpoint is visible."""

    def __init__(self, publisher: Callable[[Path, dict[str, Any], float], None]) -> None:
        if not callable(publisher):
            raise Stage11RecoveryError("Stage11 recovery publisher must be callable")
        self._publisher = publisher

    def on_model_save(self, trainer: Any) -> None:
        boundary = time.monotonic()
        checkpoint = Path(trainer.save_dir) / "weights" / "last.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise Stage11RecoveryError("on_model_save ran before ordinary last.pt publication")
        state = build_stage11_recovery_state(trainer)
        if state["completed_epoch"] != int(trainer.epoch) + 1:
            raise Stage11RecoveryError("live state epoch differs from the checkpoint callback boundary")
        self._publisher(checkpoint, state, boundary)


def configure_stage11_recovery_callbacks(
    target: Any,
    *,
    state_loader: Callable[[], Mapping[str, Any]] | None,
    publisher: Callable[[Path, dict[str, Any], float], None],
) -> None:
    """Attach deterministic loader, optional restore, and full-state capture callbacks."""
    add_callback = getattr(target, "add_callback", None)
    if not callable(add_callback):
        raise Stage11RecoveryError("Stage11 callback target has no add_callback")
    if state_loader is not None:
        if not callable(state_loader):
            raise Stage11RecoveryError("Stage11 recovery state loader must be callable")

        def restore_after_setup(trainer: Any) -> None:
            state = state_loader()
            if not isinstance(state, Mapping):
                raise Stage11RecoveryError("Stage11 state loader returned no mapping")
            restore_stage11_recovery_state(trainer, state)

        add_callback("on_train_start", restore_after_setup)
    epoch_hook = BenchmarkEpochResumeHook()
    add_callback("on_train_epoch_start", epoch_hook.on_train_epoch_start)
    add_callback("on_train_batch_start", epoch_hook.on_train_batch_start)
    add_callback("on_train_batch_end", epoch_hook.on_train_batch_end)
    state_hook = Stage11EpochStateHook(publisher)
    add_callback("on_model_save", state_hook.on_model_save)


__all__ = [
    "Stage11RecoveryError",
    "Stage11EpochStateHook",
    "build_stage11_checkpoint_receipt",
    "build_stage11_recovery_state",
    "component_sha256",
    "configure_stage11_local_recovery",
    "configure_stage11_recovery_callbacks",
    "load_latest_stage11_local_state",
    "publish_stage11_local_artifact",
    "validate_stage11_local_recovery_root",
    "restore_stage11_recovery_state",
]
