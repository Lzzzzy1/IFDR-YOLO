"""Local durable-publication contract for the approved seed-0 benchmark.

This module deliberately supplies no GPU launcher.  It proves the publication
and recovery invariants with a deterministic synthetic epoch state; the thin
CLI may prepare or inspect real-run stages but cannot silently start training.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import inspect
import shutil
import subprocess
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class BenchmarkIdentityError(ValueError):
    """A frozen benchmark or recovery contract was violated."""


class BenchmarkInterrupted(RuntimeError):
    """Intentional post-publication interruption; callers must not mark complete."""


@dataclass(frozen=True)
class BenchmarkRunContext:
    identity: Mapping[str, Any]
    primary_root: Path
    mirror_root: Path
    stop_after_epoch: int | None = None

    def __post_init__(self) -> None:
        role = self.identity.get("execution_role")
        effective_epochs = self.identity.get("effective_epochs")
        if role not in _ROLES or effective_epochs not in {1, 2}:
            raise BenchmarkIdentityError("benchmark context has an unregistered role or effective epoch count")
        if self.stop_after_epoch is not None and not (role == "recovery_interrupted_two_epoch" and self.stop_after_epoch == 1):
            raise BenchmarkIdentityError("stop-after is allowed only for recovery interrupted epoch 1")


class BenchmarkEpochHook:
    """Last `on_model_save` callback: publish first, then request a safe stop."""

    def __init__(self, primary_root: Path, mirror_root: Path, identity: Mapping[str, Any], *, stop_after_epoch: int | None = None, publisher: Any = None) -> None:
        self.primary_root, self.mirror_root = Path(primary_root), Path(mirror_root)
        self.identity, self.stop_after_epoch = dict(identity), stop_after_epoch
        self.publisher = publish_generation if publisher is None else publisher

    def on_model_save(self, trainer: Any) -> None:
        epoch = int(trainer.epoch) + 1
        checkpoint_path = Path(trainer.save_dir) / "weights" / "last.pt"
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
            raise BenchmarkIdentityError("on_model_save ran before ordinary last.pt publication")
        try:
            import torch
        except ImportError as error:
            raise BenchmarkIdentityError("benchmark hook requires PyTorch") from error
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise BenchmarkIdentityError("last.pt is not a checkpoint mapping")
        if checkpoint.get("epoch") != epoch - 1:
            raise BenchmarkIdentityError("ordinary last.pt epoch is not the expected zero-based epoch")
        results = (Path(trainer.save_dir) / "results.csv").read_text(encoding="utf-8")
        self.publisher(self.primary_root, self.mirror_root, self.identity, epoch=epoch,
                       checkpoint_archive=checkpoint_path.read_bytes(), benchmark_state=build_benchmark_state(trainer),
                       results_csv=results, diagnostics={"epoch": epoch})
        if self.stop_after_epoch == epoch:
            primary_manifest = self.primary_root / "generations" / str(epoch) / "manifest.json"
            mirror_manifest = self.mirror_root / "generations" / str(epoch) / "manifest.json"
            if not primary_manifest.is_file() or primary_manifest.read_bytes() != mirror_manifest.read_bytes():
                raise BenchmarkIdentityError("stop-after requires matching durable manifests")
            raise BenchmarkInterrupted(f"intentional interruption after durable epoch {epoch}")


def configure_benchmark_callbacks(target: Any, context: BenchmarkRunContext, *, resume: bool) -> None:
    """Append callbacks in documented Ultralytics order; caller supplies a real trainer/YOLO handle."""
    add_callback = getattr(target, "add_callback", None)
    if not callable(add_callback):
        raise BenchmarkIdentityError("benchmark callback target has no add_callback")
    if resume:
        def restore_after_setup(trainer: Any) -> None:
            _, state, _ = _load_committed_benchmark_state(context.primary_root, context.mirror_root, context.identity)
            restore_benchmark_training_state(trainer, state)
        add_callback("on_train_start", restore_after_setup)
    epoch_resume_hook = BenchmarkEpochResumeHook()
    add_callback("on_train_epoch_start", epoch_resume_hook.on_train_epoch_start)
    add_callback("on_train_batch_start", epoch_resume_hook.on_train_batch_start)
    add_callback("on_train_batch_end", epoch_resume_hook.on_train_batch_end)
    hook = BenchmarkEpochHook(context.primary_root, context.mirror_root, context.identity, stop_after_epoch=context.stop_after_epoch)
    add_callback("on_model_save", hook.on_model_save)


_ARMS = {"P3P5_CONTROL", "DCLI"}
_ROLES = {"timing_one_epoch", "recovery_uninterrupted_two_epoch", "recovery_interrupted_two_epoch"}
_HEX = set("0123456789abcdef")
_REGISTERED_CONFIG_ARMS = {
    "kitti_p3p5_control_s0.yaml": "P3P5_CONTROL",
    "kitti_dcli_s0.yaml": "DCLI",
}
DATALOADER_EPOCH_SEED_BASE = 6148914691236517204  # Ultralytics build_dataloader() base + registered RANK=-1.


def reset_train_loader_for_epoch(trainer: Any) -> None:
    """Rebuild the benchmark loader from its registered, epoch-only seed."""
    epoch = getattr(trainer, "epoch", None)
    if not isinstance(epoch, int) or epoch < 0:
        raise BenchmarkIdentityError("train loader epoch must be a nonnegative integer")
    loader = getattr(trainer, "train_loader", None)
    if getattr(trainer, "world_size", 1) > 1 or hasattr(getattr(loader, "sampler", None), "num_replicas"):
        raise BenchmarkIdentityError("benchmark loader reset supports only registered single-GPU rank -1")
    generator = getattr(loader, "generator", None)
    seed = getattr(generator, "manual_seed", None)
    reset = getattr(loader, "reset", None)
    if not callable(seed):
        raise BenchmarkIdentityError("train loader has no seedable generator")
    if not callable(reset):
        raise BenchmarkIdentityError("train loader has no reset")
    seed(DATALOADER_EPOCH_SEED_BASE + epoch)
    reset()


class BenchmarkEpochResumeHook:
    """Reset loader state and preserve the otherwise-local warmup accumulation phase."""

    def __init__(self) -> None:
        self._suppress_remaining = 0
        self._saved: tuple[str, int | float] | None = None

    def on_train_epoch_start(self, trainer: Any) -> None:
        if self._suppress_remaining or self._saved is not None:
            raise BenchmarkIdentityError("benchmark accumulation callback has an unpaired batch")
        epoch = getattr(trainer, "epoch", None)
        if not isinstance(epoch, int) or epoch < 0:
            raise BenchmarkIdentityError("train loader epoch must be a nonnegative integer")
        reset_train_loader_for_epoch(trainer)
        loader = trainer.train_loader
        nb = len(loader)
        args = getattr(trainer, "args", None)
        warmup_epochs = getattr(args, "warmup_epochs", None)
        batch_size = getattr(trainer, "batch_size", None)
        nbs = getattr(args, "nbs", None)
        if isinstance(nb, bool) or not isinstance(nb, int) or nb <= 0:
            raise BenchmarkIdentityError("train loader has no positive batch count")
        if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, (int, float)) or warmup_epochs < 0:
            raise BenchmarkIdentityError("trainer has an invalid warmup setting")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise BenchmarkIdentityError("trainer has no positive batch size")
        if isinstance(nbs, bool) or not isinstance(nbs, (int, float)) or nbs <= 0:
            raise BenchmarkIdentityError("trainer has no positive nominal batch size")
        self._suppress_remaining = optimizer_step_offset_for_epoch(
            epoch=epoch, batches_per_epoch=nb, warmup_epochs=warmup_epochs, nominal_batch_size=nbs, batch_size=batch_size,
        )

    def on_train_batch_start(self, trainer: Any) -> None:
        if self._saved is not None:
            raise BenchmarkIdentityError("benchmark accumulation callback has an unpaired batch")
        if not self._suppress_remaining:
            return
        args = getattr(trainer, "args", None)
        nbs = getattr(args, "nbs", None)
        batch_size = getattr(trainer, "batch_size", None)
        nb = len(trainer.train_loader)
        epoch = int(trainer.epoch)
        warmup_steps = max(round(args.warmup_epochs * nb), 100) if args.warmup_epochs > 0 else -1
        ni = epoch * nb + (optimizer_step_offset_for_epoch(
            epoch=epoch, batches_per_epoch=nb, warmup_epochs=args.warmup_epochs, nominal_batch_size=nbs, batch_size=batch_size,
        ) - self._suppress_remaining)
        if not isinstance(nbs, (int, float)) or not isinstance(batch_size, int) or ni < 0:
            raise BenchmarkIdentityError("benchmark accumulation phase is malformed")
        # BaseTrainer's local last_opt_step restarts at -1 after resume. Suppress only the
        # initial batches preceding the uninterrupted path's first optimizer step.
        if ni <= warmup_steps:
            self._saved = ("nbs", nbs)
            args.nbs = batch_size * (2 + (ni + 1) * warmup_steps / max(ni, 1))
        else:
            accumulate = getattr(trainer, "accumulate", None)
            if not isinstance(accumulate, int) or accumulate < 1:
                raise BenchmarkIdentityError("trainer has no positive accumulation state outside warmup")
            self._saved = ("accumulate", accumulate)
            trainer.accumulate = ni + 2

    def on_train_batch_end(self, trainer: Any) -> None:
        if self._saved is None:
            return
        field, value = self._saved
        if field == "nbs":
            trainer.args.nbs = value
        else:
            trainer.accumulate = value
        self._saved = None
        self._suppress_remaining -= 1


def optimizer_step_offset_for_epoch(*, epoch: int, batches_per_epoch: int, warmup_epochs: float,
                                    nominal_batch_size: int | float, batch_size: int) -> int:
    """Replay the frozen Ultralytics accumulation predicate to the epoch's first optimizer step."""
    if min(epoch, batches_per_epoch, batch_size) < 0 or batches_per_epoch == 0 or batch_size == 0 or warmup_epochs < 0 or nominal_batch_size <= 0:
        raise BenchmarkIdentityError("optimizer accumulation replay inputs are invalid")
    try:
        import numpy as np
    except ImportError as error:
        raise BenchmarkIdentityError("optimizer accumulation replay requires NumPy") from error
    warmup_steps = max(round(warmup_epochs * batches_per_epoch), 100) if warmup_epochs > 0 else -1
    accumulate = max(round(nominal_batch_size / batch_size), 1)
    last_step = -1
    first = epoch * batches_per_epoch
    for ni in range(first + batches_per_epoch):
        if ni <= warmup_steps:
            accumulate = max(1, int(np.interp(ni, [0, warmup_steps], [1, nominal_batch_size / batch_size]).round()))
        if ni - last_step >= accumulate:
            if ni >= first:
                return ni - first
            last_step = ni
    raise BenchmarkIdentityError("optimizer accumulation replay found no epoch optimizer step")


def clean_git_identity(root: Path, code_files: Sequence[str], config_path: Path) -> tuple[str, str, str]:
    """Return the clean source identity required by the formal trainer preflight."""
    head_run = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    if head_run.returncode != 0 or len(head_run.stdout.strip()) != 40:
        raise BenchmarkIdentityError("NO_GO git HEAD is unavailable")
    dirty_run = subprocess.run(("git", "-C", str(root), "diff", "--", *code_files, str(Path(config_path).resolve())), capture_output=True, text=True, check=False)
    status_run = subprocess.run(("git", "-C", str(root), "status", "--porcelain", "--", *code_files, str(Path(config_path).resolve())), capture_output=True, text=True, check=False)
    if dirty_run.returncode != 0 or status_run.returncode != 0:
        raise BenchmarkIdentityError("NO_GO relevant git identity is unavailable")
    if dirty_run.stdout or status_run.stdout:
        raise BenchmarkIdentityError("NO_GO preflight requires clean tracked Git state")
    return head_run.stdout.strip(), _sha_bytes(b""), _sha_bytes(b"")
_ULTRALYTICS_TRAINER_SHA256 = "d98009b8d9acfc61fde8941e8b029990da53757dfe7ac2a946d771860c754c1d"
_FIT_IMAGE_MANIFEST_SHA256 = "15d326c539153c2a54c78f9af196038639e82be0de0af600808d25e67de23df3"
_RAW_LABEL_DIR_SHA256 = "72e50ec65d019a8da17393c9e6d3e592c8eea52561bbb136173831b7325259d9"
_FIT_IDS_SHA256 = "50a1f8d72b747d1e2f460a4e0a355469484d33dbf1e9928b64ba6594c1873362"
_DEVELOPMENT_IDS_SHA256 = "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"


def verify_ultralytics_callback_contract() -> dict[str, str]:
    """Bind the callback-order assumption to the approved installed runtime."""
    import ultralytics
    from ultralytics.engine.trainer import BaseTrainer
    source_path = Path(inspect.getfile(BaseTrainer))
    digest = _sha_bytes(source_path.read_bytes())
    if ultralytics.__version__ != "8.4.98" or digest != _ULTRALYTICS_TRAINER_SHA256:
        raise BenchmarkIdentityError("Ultralytics callback source identity does not match the frozen contract")
    source = inspect.getsource(BaseTrainer._do_train)
    required_order = ("self._setup_train()", 'self.run_callbacks("on_train_start")', "self.save_model()", 'self.run_callbacks("on_model_save")')
    if any(part not in source for part in required_order) or not (source.index(required_order[0]) < source.index(required_order[1]) < source.index(required_order[2]) < source.index(required_order[3])):
        raise BenchmarkIdentityError("Ultralytics callback ordering differs from the frozen contract")
    return {"version": ultralytics.__version__, "trainer_sha256": digest}


def _sha_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def registered_arm_for_config_path(config_path: Path) -> str:
    """Resolve the frozen arm externally, without extending training config schema."""
    try:
        return _REGISTERED_CONFIG_ARMS[Path(config_path).name]
    except KeyError as error:
        raise BenchmarkIdentityError("unregistered benchmark config path") from error


def _sha_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise BenchmarkIdentityError(f"required regular file is missing: {path}")
    return _sha_bytes(path.read_bytes())


def _only_allowed_parent_difference(parent: Any, candidate: Any, allowed: Mapping[str, tuple[Any, Any]]) -> bool:
    from dataclasses import asdict
    def leaves(value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, dict):
            return {path: leaf for key, item in value.items() for path, leaf in leaves(item, f"{prefix}.{key}" if prefix else key).items()}
        return {prefix: value}
    parent_data, candidate_data = asdict(parent), asdict(candidate)
    parent_data.pop("source_path", None); candidate_data.pop("source_path", None)
    left, right = leaves(parent_data), leaves(candidate_data)
    actual = {key: (left.get(key), right.get(key)) for key in set(left) | set(right) if left.get(key) != right.get(key)}
    return actual == dict(allowed)


def run_preflight(*, arm: str, execution_role: str, config_path: Path, fit_ids: Path, development_ids: Path,
                  resolved_data: Path, raw_label_dir: Path, repository_root: Path, output_dir: Path, mirror_dir: Path, device: str) -> dict[str, Any]:
    """Read-only gate: validate inputs then atomically publish only preflight identity evidence."""
    output, mirror, root = Path(output_dir).resolve(), Path(mirror_dir).resolve(), Path(repository_root).resolve()
    if output == mirror or output.is_relative_to(mirror) or mirror.is_relative_to(output):
        raise BenchmarkIdentityError("preflight output and mirror roots must be disjoint")
    if output.exists() and any(output.iterdir()):
        raise BenchmarkIdentityError("preflight requires an empty fresh output root")
    if mirror.exists() and any(mirror.iterdir()):
        raise BenchmarkIdentityError("preflight requires an empty fresh mirror root")
    expected_arm = registered_arm_for_config_path(config_path)
    if arm != expected_arm:
        raise BenchmarkIdentityError("preflight arm does not match frozen config path")
    if arm == "P3P5_CONTROL" and execution_role != "timing_one_epoch":
        raise BenchmarkIdentityError("P3P5_CONTROL is registered only for timing_one_epoch")
    from ifdr_yolo.experiments.config import load_baseline_config, load_ifdr_config
    from ifdr_yolo.data.splits import load_ids
    from ifdr_yolo.experiments.p2_candidate_survival_audit import _directory_sha256, _fit_image_manifest_sha256
    from ifdr_yolo.experiments.provenance import verify_dataset
    config = load_baseline_config(config_path, repository_root=root) if arm == "P3P5_CONTROL" else load_ifdr_config(config_path, repository_root=root)
    parent_path = root / ("configs/experiments/kitti_yolov8m_baseline_s17.yaml" if arm == "P3P5_CONTROL" else "configs/experiments/diagnostics/kitti_ifdr_p2_interaction_b_s17.yaml")
    parent = load_baseline_config(parent_path, repository_root=root) if arm == "P3P5_CONTROL" else load_ifdr_config(parent_path, repository_root=root)
    allowed = ({"experiment.seed": (17, 0), "training.epochs": (300, 30)} if arm == "P3P5_CONTROL" else {"experiment.seed": (17, 0), "method.intervention.base_seed": (17, 0)})
    if not _only_allowed_parent_difference(parent, config, allowed):
        raise BenchmarkIdentityError("preflight config semantics differ from the frozen parent")
    fit, development = load_ids(fit_ids), load_ids(development_ids)
    if len(fit) != 3341 or len(development) != 371 or set(fit) & set(development):
        raise BenchmarkIdentityError("preflight requires registered 3341/371 disjoint split manifests")
    fit_ids_sha, development_ids_sha = _sha_file(Path(fit_ids)), _sha_file(Path(development_ids))
    if fit_ids_sha != _FIT_IDS_SHA256 or development_ids_sha != _DEVELOPMENT_IDS_SHA256:
        raise BenchmarkIdentityError("preflight split manifests differ from the registered hashes")
    if config.experiment.seed != 0 or config.training.epochs != 30 or (config.training.imgsz, config.training.batch, config.training.workers, config.training.amp, config.training.deterministic) != (640, 16, 8, True, True):
        raise BenchmarkIdentityError("preflight training protocol differs from the frozen seed-0 contract")
    if (config.prediction.conf, config.prediction.iou, config.prediction.max_det, config.prediction.half) != (0.001, 0.7, 300, False):
        raise BenchmarkIdentityError("preflight prediction protocol differs from the frozen contract")
    try:
        fit_image_sha = _fit_image_manifest_sha256(Path(resolved_data), fit)
        raw_label_sha = _directory_sha256(Path(raw_label_dir))
    except (FileNotFoundError, ValueError, OSError) as error:
        raise BenchmarkIdentityError("NO_GO preflight content manifest is missing or unreadable") from error
    if fit_image_sha != _FIT_IMAGE_MANIFEST_SHA256 or raw_label_sha != _RAW_LABEL_DIR_SHA256:
        raise BenchmarkIdentityError("NO_GO preflight content manifest differs from the frozen registered hash")
    try:
        generated_dataset = verify_dataset(config, verify_all_hashes=False)
        generated_manifest_sha = _sha_file(config.paths.generated_data / "manifest.json")
        split_source_sha = _sha_file(config.paths.train_ids.parent / "source.json")
    except (FileNotFoundError, ValueError, OSError) as error:
        raise BenchmarkIdentityError("NO_GO preflight generated dataset is missing or invalid") from error
    model_sha = _sha_file(config.paths.model)
    pretrained_sha = _sha_file(config.initialization.pretrained) if config.initialization is not None else model_sha
    effective_epochs = 1 if execution_role == "timing_one_epoch" else 2
    code_files = ("ifdr_yolo/experiments/kitti_seed0_training_benchmark.py", "scripts/run_kitti_seed0_training_benchmark.py", "ifdr_yolo/experiments/ifdr_runtime.py", "ifdr_yolo/experiments/p2_fit_reference.py", "ifdr_yolo/experiments/p2_candidate_survival_audit.py", "ifdr_yolo/experiments/provenance.py", "ifdr_yolo/experiments/gradient_diagnostics.py", "scripts/run_p2_interaction_s0.py", "ifdr_yolo/experiments/config.py", "ifdr_yolo/experiments/ultralytics_runtime.py", "ifdr_yolo/experiments/ifdr_trainer.py", "ifdr_yolo/models/ifdr_model.py", "ifdr_yolo/models/gated_fusion.py", "ifdr_yolo/losses/ifdr_detection.py", "ifdr_yolo/eval/evaluate.py", "ifdr_yolo/eval/kitti_ap40.py")
    code_manifest = {name: _sha_file(root / name) for name in code_files}
    runtime = verify_ultralytics_callback_contract()
    try:
        import torch
    except ImportError as error:
        raise BenchmarkIdentityError("NO_GO PyTorch runtime is unavailable") from error
    if device != "0" or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise BenchmarkIdentityError("NO_GO preflight requires visible CUDA device 0")
    device_props = torch.cuda.get_device_properties(0)
    git_head, tracked_diff_sha, relevant_status_sha = clean_git_identity(root, code_files, Path(config_path))
    preflight_facts = {"fit_image_manifest_sha256": fit_image_sha, "raw_label_dir_sha256": raw_label_sha, "resolved_data_sha256": _sha_file(Path(resolved_data)), "generated_dataset_manifest_sha256": generated_manifest_sha, "split_source_sha256": split_source_sha, "generated_dataset": generated_dataset, "git_head": git_head, "tracked_diff_sha256": tracked_diff_sha, "relevant_status_sha256": relevant_status_sha, "relevant_dirty_paths": [], "runtime": {**runtime, "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": True, "cuda_device_count": torch.cuda.device_count(), "device": device, "device_name": device_props.name, "total_memory": device_props.total_memory}}
    identity = build_benchmark_identity(arm=arm, execution_role=execution_role, config_sha256=_sha_file(Path(config_path)), code_sha256=_sha_bytes(_json_bytes(code_manifest)), fit_ids_sha256=fit_ids_sha, development_ids_sha256=development_ids_sha, model_sha256=model_sha, pretrained_sha256=pretrained_sha, effective_epochs=effective_epochs, preflight_facts=preflight_facts)
    output.parent.mkdir(parents=True, exist_ok=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    disk = {"primary_free": shutil.disk_usage(output.parent).free, "mirror_free": shutil.disk_usage(mirror.parent).free}
    payload = {"preflight_state": "PASS", "benchmark_launch_authorized": True, "identity": identity, "code_files": code_manifest, "code_manifest_sha256": _sha_bytes(_json_bytes(code_manifest)), "preflight_facts": preflight_facts, "disk": disk, "training_authorized": False}
    _atomic_write(output / "preflight_identity.json", _json_bytes(payload))
    _atomic_write(mirror / "preflight_identity.json", _json_bytes(payload))
    manifest = {"schema_version": 1, "identity_sha256": identity["identity_sha256"], "files": [{"name": "preflight_identity.json", "sha256": _sha_file(output / "preflight_identity.json")}]} 
    _atomic_write(output / "manifest.json", _json_bytes(manifest)); _atomic_write(mirror / "manifest.json", _json_bytes(manifest))
    if (output / "preflight_identity.json").read_bytes() != (mirror / "preflight_identity.json").read_bytes() or (output / "manifest.json").read_bytes() != (mirror / "manifest.json").read_bytes():
        raise BenchmarkIdentityError("NO_GO preflight mirror publication is not byte-identical")
    return payload


def load_preflight_pair(primary_dir: Path, mirror_dir: Path, *, arm: str, execution_role: str) -> dict[str, Any]:
    """Load a byte-identical, manifest-bound real preflight pair."""
    primary, mirror = Path(primary_dir).resolve(), Path(mirror_dir).resolve()
    if primary == mirror:
        raise BenchmarkIdentityError("preflight primary and mirror must be disjoint")
    names = ("preflight_identity.json", "manifest.json")
    contents: dict[str, bytes] = {}
    for name in names:
        left, right = primary / name, mirror / name
        if not left.is_file() or not right.is_file() or left.read_bytes() != right.read_bytes():
            raise BenchmarkIdentityError("preflight primary/mirror publication differs or is incomplete")
        contents[name] = left.read_bytes()
    payload = json.loads(contents["preflight_identity.json"])
    manifest = json.loads(contents["manifest.json"])
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise BenchmarkIdentityError("preflight identity payload is malformed")
    expected_manifest = {
        "schema_version": 1,
        "identity_sha256": identity.get("identity_sha256"),
        "files": [{"name": "preflight_identity.json", "sha256": _sha_bytes(contents["preflight_identity.json"])}],
    }
    if manifest != expected_manifest:
        raise BenchmarkIdentityError("preflight manifest does not bind the exact identity payload")
    unhashed = dict(identity)
    claimed_sha = unhashed.pop("identity_sha256", None)
    if claimed_sha != _sha_bytes(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()):
        raise BenchmarkIdentityError("preflight embedded identity SHA256 is invalid")
    if (payload.get("preflight_state"), payload.get("benchmark_launch_authorized"), payload.get("training_authorized")) != ("PASS", True, False):
        raise BenchmarkIdentityError("preflight did not authorize the bounded benchmark launch")
    if identity.get("arm") != arm or identity.get("execution_role") != execution_role:
        raise BenchmarkIdentityError("preflight arm or execution role differs from the requested benchmark")
    return identity


def run_registered_benchmark_stage(*, arm: str, execution_role: str, config_path: Path,
                                   fit_ids: Path, development_ids: Path, repository_root: Path,
                                   output_dir: Path, mirror_dir: Path, preflight_dir: Path,
                                   preflight_mirror_dir: Path, device: str, resume: bool,
                                   stop_after_epoch: int | None = None) -> Any:
    """Run exactly one registered real benchmark leg through the existing trainer path."""
    identity = load_preflight_pair(preflight_dir, preflight_mirror_dir, arm=arm, execution_role=execution_role)
    output, mirror = Path(output_dir).resolve(), Path(mirror_dir).resolve()
    if output == mirror or output.is_relative_to(mirror) or mirror.is_relative_to(output):
        raise BenchmarkIdentityError("benchmark output and mirror roots must be disjoint")
    state_primary = output.with_name(f"{output.name}.benchmark-state")
    state_mirror = mirror.with_name(f"{mirror.name}.benchmark-state")
    if not resume:
        for path in (output, mirror, state_primary, state_mirror):
            if path.exists() and any(path.iterdir()):
                raise BenchmarkIdentityError("fresh benchmark leg requires empty output and state roots")
    context = BenchmarkRunContext(
        identity=identity, primary_root=state_primary, mirror_root=state_mirror,
        stop_after_epoch=stop_after_epoch,
    )
    if arm == "P3P5_CONTROL":
        if execution_role != "timing_one_epoch" or resume or stop_after_epoch is not None:
            raise BenchmarkIdentityError("P3P5_CONTROL is registered only as a fresh one-epoch timing leg")
        from ifdr_yolo.experiments.config import load_baseline_config
        from ifdr_yolo.experiments.p2_fit_reference import run_p3p5_fit_reference
        config = load_baseline_config(Path(config_path), repository_root=Path(repository_root).resolve())
        return run_p3p5_fit_reference(
            config, repository_root=repository_root, output_dir=output, mirror_dir=mirror,
            fit_ids=fit_ids, development_ids=development_ids, mode="full", device=device,
            resume=False, benchmark_context=context,
        )
    if arm != "DCLI":
        raise BenchmarkIdentityError("unregistered real benchmark arm")
    from scripts.run_p2_interaction_s0 import run_screen
    return run_screen(
        config_path=Path(config_path), fit_ids=Path(fit_ids), development_ids=Path(development_ids),
        output_dir=output, mirror_dir=mirror, mode="full", device=device, resume=resume,
        benchmark_context=context,
    )


def _canonical(value: Any) -> Any:
    try:
        import torch
        is_tensor = isinstance(value, torch.Tensor)
    except ImportError:
        is_tensor = False
    if is_tensor:
        if not bool(torch.isfinite(value).all()):
            raise BenchmarkIdentityError("checkpoint tensor contains nonfinite values")
        cpu = value.detach().cpu().contiguous()
        return {"__tensor__": True, "dtype": str(cpu.dtype), "shape": list(cpu.shape),
                "bytes_sha256": _sha_bytes(cpu.numpy().tobytes())}
    try:
        import torch
        is_module = isinstance(value, torch.nn.Module)
    except ImportError:
        is_module = False
    if is_module:
        return {"__module_state_dict__": _canonical(value.state_dict())}
    try:
        import numpy as np
        is_array = isinstance(value, np.ndarray)
    except ImportError:
        is_array = False
    if is_array:
        if np.issubdtype(value.dtype, np.inexact) and not bool(np.isfinite(value).all()):
            raise BenchmarkIdentityError("checkpoint array contains nonfinite values")
        array = np.ascontiguousarray(value)
        return {"__ndarray__": True, "dtype": str(array.dtype), "shape": list(array.shape),
                "bytes_sha256": _sha_bytes(array.tobytes())}
    if isinstance(value, bytes):
        return {"__bytes_sha256__": _sha_bytes(value), "size": len(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise BenchmarkIdentityError("checkpoint scalar contains nonfinite values")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise BenchmarkIdentityError(f"unsupported canonical checkpoint value: {type(value).__name__}")


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG source restored by the frozen recovery contract."""
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    try:
        import torch
        state["torch_cpu"] = torch.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    except ImportError:
        state["torch_cpu"] = None
        state["torch_cuda"] = []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state.get("python"), tuple):
        raise BenchmarkIdentityError("RNG snapshot has no Python state")
    random.setstate(state["python"])
    try:
        import numpy as np
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass
    try:
        import torch
        if state.get("torch_cpu") is not None:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_sha(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value.lower()) - _HEX:
        raise BenchmarkIdentityError(f"{name} must be a SHA256")
    return value.lower()


def build_benchmark_identity(*, arm: str, execution_role: str, config_sha256: str, code_sha256: str,
                             fit_ids_sha256: str, development_ids_sha256: str, model_sha256: str,
                             pretrained_sha256: str, effective_epochs: int, seed: int = 0, preflight_facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if arm not in _ARMS:
        raise BenchmarkIdentityError("benchmark arm must be P3P5_CONTROL or DCLI")
    if execution_role not in _ROLES:
        raise BenchmarkIdentityError("unregistered benchmark execution role")
    if seed != 0:
        raise BenchmarkIdentityError("benchmark seed must be 0")
    expected_epochs = 1 if execution_role == "timing_one_epoch" else 2
    if effective_epochs != expected_epochs:
        raise BenchmarkIdentityError("effective epochs do not match frozen execution role")
    hashes = {name: _require_sha(name, value) for name, value in {
        "config_sha256": config_sha256, "code_sha256": code_sha256,
        "fit_ids_sha256": fit_ids_sha256, "development_ids_sha256": development_ids_sha256,
        "model_sha256": model_sha256, "pretrained_sha256": pretrained_sha256,
    }.items()}
    payload = {"schema_version": 1, "arm": arm, "execution_role": execution_role, "seed": 0,
               "effective_epochs": effective_epochs, "formal_epochs": 30,
               "imgsz": 640, "batch": 16, "workers": 8, "amp": True,
               "deterministic": True, "checkpoint_role": "last.pt", **hashes}
    if preflight_facts is not None:
        payload["preflight_facts"] = _canonical(dict(preflight_facts))
    payload["identity_sha256"] = _sha_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return payload


def canonical_checkpoint_digest(checkpoint: Mapping[str, Any]) -> str:
    required = {"completed_epoch", "model", "ema", "ema_updates", "optimizer", "scaler", "scheduler", "rng_state", "train_args"}
    missing = required - set(checkpoint)
    if missing:
        raise BenchmarkIdentityError(f"checkpoint missing required state: {sorted(missing)}")
    ema_updates = checkpoint["ema_updates"]
    if isinstance(ema_updates, bool) or not isinstance(ema_updates, int) or ema_updates < 0:
        raise BenchmarkIdentityError("checkpoint has no valid EMA update count")
    return _sha_bytes(json.dumps(_canonical({key: checkpoint[key] for key in sorted(required)}), sort_keys=True, separators=(",", ":")).encode())


def _torch_load_bytes(content: bytes) -> Mapping[str, Any]:
    try:
        import torch
        checkpoint = torch.load(BytesIO(content), map_location="cpu", weights_only=False)
    except Exception as error:
        raise BenchmarkIdentityError("checkpoint is not CPU-loadable") from error
    if not isinstance(checkpoint, Mapping):
        raise BenchmarkIdentityError("checkpoint is not a mapping")
    return checkpoint


def _torch_save_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        import torch
    except ImportError as error:
        raise BenchmarkIdentityError("benchmark durable state requires PyTorch") from error
    buffer = BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def build_benchmark_state(trainer: Any) -> dict[str, Any]:
    """Snapshot live state separately from Ultralytics' ordinary checkpoint archive."""
    completed_epoch = int(trainer.epoch) + 1
    ema_owner = getattr(trainer, "ema", None)
    ema_updates = getattr(ema_owner, "updates", None)
    if isinstance(ema_updates, bool) or not isinstance(ema_updates, int) or ema_updates < 0:
        raise BenchmarkIdentityError("trainer EMA has no valid update count")
    fields = {
        "completed_epoch": completed_epoch,
        "model": getattr(trainer, "model", None),
        "ema": ema_owner,
        "ema_updates": ema_updates,
        "optimizer": getattr(trainer, "optimizer", None),
        "scaler": getattr(trainer, "scaler", None),
        "scheduler": getattr(trainer, "scheduler", None),
        "rng_state": capture_rng_state(),
        "train_args": getattr(trainer, "args", None),
    }
    for name in ("model", "ema", "optimizer", "scaler", "scheduler"):
        value = fields[name]
        if name == "ema" and not callable(getattr(value, "state_dict", None)):
            value = getattr(value, "ema", None)
        state_dict = getattr(value, "state_dict", None)
        if value is None or not callable(state_dict):
            raise BenchmarkIdentityError(f"trainer has no serializable {name} state")
        fields[name] = state_dict()
    train_args = fields["train_args"]
    if hasattr(train_args, "__dict__"):
        train_args = vars(train_args)
    if not isinstance(train_args, Mapping):
        raise BenchmarkIdentityError("trainer has no mapping train args")
    fields["train_args"] = dict(train_args)
    canonical_checkpoint_digest(fields)
    return fields


def restore_benchmark_training_state(trainer: Any, state: Mapping[str, Any]) -> None:
    """Restore the lossless sidecar after trainer setup, replacing ordinary FP16 resume state."""
    canonical_checkpoint_digest(state)
    completed_epoch = state.get("completed_epoch")
    if completed_epoch != getattr(trainer, "start_epoch", None):
        raise BenchmarkIdentityError("resume trainer start epoch differs from benchmark state")
    ema_updates = state.get("ema_updates")
    if isinstance(ema_updates, bool) or not isinstance(ema_updates, int) or ema_updates < 0:
        raise BenchmarkIdentityError("benchmark state has no valid EMA update count")
    ema_owner = getattr(trainer, "ema", None)
    if isinstance(getattr(ema_owner, "updates", None), bool) or not isinstance(getattr(ema_owner, "updates", None), int):
        raise BenchmarkIdentityError("resume trainer has no loadable EMA update count")
    targets = {
        "model": getattr(trainer, "model", None),
        "ema": getattr(ema_owner, "ema", None),
        "optimizer": getattr(trainer, "optimizer", None),
        "scaler": getattr(trainer, "scaler", None),
        "scheduler": getattr(trainer, "scheduler", None),
    }
    for name, target in targets.items():
        load_state_dict = getattr(target, "load_state_dict", None)
        if not callable(load_state_dict):
            raise BenchmarkIdentityError(f"resume trainer has no loadable {name} state")
        try:
            load_state_dict(state[name])
        except Exception as error:
            raise BenchmarkIdentityError(f"resume {name} state is not loadable") from error
    ema_owner.updates = ema_updates
    restore_rng_state(state["rng_state"])


def _generation_files(generation: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(generation.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        content = path.read_bytes()
        records.append({"name": path.name, "size": len(content), "sha256": _sha_bytes(content)})
    return records


def publish_generation(primary_root: Path, mirror_root: Path, identity: Mapping[str, Any], *, epoch: int,
                       checkpoint_archive: bytes, benchmark_state: Mapping[str, Any], results_csv: str, diagnostics: Mapping[str, Any],
                       fail_mirror: bool = False) -> dict[str, Any]:
    """Commit one closed generation, publishing both manifests only as the last step."""
    if epoch < 1 or identity.get("identity_sha256") is None:
        raise BenchmarkIdentityError("generation requires a frozen identity and positive epoch")
    primary = Path(primary_root) / "generations" / str(epoch)
    mirror = Path(mirror_root) / "generations" / str(epoch)
    if primary_root == mirror_root:
        raise BenchmarkIdentityError("primary and mirror roots must be disjoint")
    existing_manifest = primary / "manifest.json"
    if existing_manifest.is_file():
        try:
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkIdentityError("existing generation manifest is unreadable") from error
        if existing.get("identity_sha256") != identity["identity_sha256"]:
            raise BenchmarkIdentityError("output reuse has a different frozen identity")
        raise BenchmarkIdentityError("closed generation cannot be republished")
    ordinary = _torch_load_bytes(checkpoint_archive)
    if ordinary.get("epoch") != epoch - 1:
        raise BenchmarkIdentityError("ordinary last.pt epoch is not the expected zero-based epoch")
    if benchmark_state.get("completed_epoch") != epoch:
        raise BenchmarkIdentityError("benchmark state completed epoch does not match generation")
    digest = canonical_checkpoint_digest(benchmark_state)
    _atomic_write(primary / "last.pt", checkpoint_archive)
    _atomic_write(primary / "benchmark_state.pt", _torch_save_bytes(benchmark_state))
    _atomic_write(primary / "results.csv", results_csv.encode("utf-8"))
    _atomic_write(primary / "diagnostics.json", _json_bytes(dict(diagnostics)))
    _atomic_write(primary / "benchmark_state.digest", f"{digest}\n".encode())
    _atomic_write(primary / "identity.json", _json_bytes(dict(identity)))
    if fail_mirror:
        raise OSError("mirror write failure injected before manifest publication")
    mirror.mkdir(parents=True, exist_ok=True)
    for source in primary.iterdir():
        if source.is_file() and source.name != "manifest.json":
            _atomic_write(mirror / source.name, source.read_bytes())
    if _generation_files(primary) != _generation_files(mirror):
        raise BenchmarkIdentityError("primary/mirror generation mismatch")
    manifest = {"schema_version": 1, "epoch": epoch, "identity_sha256": identity["identity_sha256"],
                "checkpoint_digest": digest, "files": _generation_files(primary)}
    # Both sides are closed only when their byte-identical manifests exist.
    _atomic_write(primary / "manifest.json", _json_bytes(manifest))
    _atomic_write(mirror / "manifest.json", _json_bytes(manifest))
    return manifest


def _latest_common_generation(primary_root: Path, mirror_root: Path, identity: Mapping[str, Any]) -> int:
    reconcile_common_generation(primary_root, mirror_root, identity)
    primary_generations = Path(primary_root) / "generations"
    mirror_generations = Path(mirror_root) / "generations"
    common: list[int] = []
    for path in primary_generations.glob("*/manifest.json") if primary_generations.exists() else ():
        mirror_manifest = mirror_generations / path.parent.name / "manifest.json"
        if not mirror_manifest.is_file() or path.read_bytes() != mirror_manifest.read_bytes():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("identity_sha256") == identity["identity_sha256"]:
            common.append(int(path.parent.name))
    if not common:
        raise BenchmarkIdentityError("resume requires a common committed generation")
    return max(common)


def reconcile_common_generation(primary_root: Path, mirror_root: Path, identity: Mapping[str, Any]) -> None:
    """Complete a manifest-last crash only when both payloads are byte-identical."""
    for root_a, root_b in ((Path(primary_root), Path(mirror_root)), (Path(mirror_root), Path(primary_root))):
        for manifest in (root_a / "generations").glob("*/manifest.json") if (root_a / "generations").exists() else ():
            peer = root_b / "generations" / manifest.parent.name
            peer_manifest = peer / "manifest.json"
            if peer_manifest.exists():
                continue
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("identity_sha256") != identity.get("identity_sha256"):
                raise BenchmarkIdentityError("single-sided manifest identity mismatch")
            for record in payload.get("files", []):
                name = record["name"]
                left, right = manifest.parent / name, peer / name
                if not left.is_file() or not right.is_file() or left.read_bytes() != right.read_bytes() or _sha_bytes(left.read_bytes()) != record["sha256"]:
                    raise BenchmarkIdentityError("single-sided manifest payload is not safely reconcilable")
            _atomic_write(peer_manifest, manifest.read_bytes())


def _load_committed_benchmark_state(primary_root: Path, mirror_root: Path, identity: Mapping[str, Any]) -> tuple[int, Mapping[str, Any], bytes]:
    generation = _latest_common_generation(primary_root, mirror_root, identity)
    primary = Path(primary_root) / "generations" / str(generation)
    mirror = Path(mirror_root) / "generations" / str(generation)
    try:
        manifest = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkIdentityError("resume committed manifest is unreadable") from error
    expected_files = {record.get("name"): record for record in manifest.get("files", []) if isinstance(record, Mapping)}
    committed: dict[str, bytes] = {}
    for name in ("last.pt", "benchmark_state.pt", "benchmark_state.digest"):
        record = expected_files.get(name)
        primary_path, mirror_path = primary / name, mirror / name
        if not isinstance(record, Mapping) or not primary_path.is_file() or not mirror_path.is_file():
            raise BenchmarkIdentityError("resume payload differs from committed manifest")
        primary_content, mirror_content = primary_path.read_bytes(), mirror_path.read_bytes()
        if (primary_content != mirror_content or record.get("size") != len(primary_content)
                or record.get("sha256") != _sha_bytes(primary_content)):
            raise BenchmarkIdentityError("resume payload differs from committed manifest")
        committed[name] = primary_content
    checkpoint = _torch_load_bytes(committed["benchmark_state.pt"])
    if checkpoint.get("completed_epoch") != generation:
        raise BenchmarkIdentityError("common benchmark state completed epoch differs")
    checkpoint_digest = canonical_checkpoint_digest(checkpoint)
    digest_file = committed["benchmark_state.digest"].decode("utf-8").strip()
    if checkpoint_digest != digest_file or checkpoint_digest != manifest.get("checkpoint_digest"):
        raise BenchmarkIdentityError("resume benchmark state differs from committed manifest")
    return generation, checkpoint, committed["last.pt"]


def prepare_resume_checkpoint(primary_root: Path, mirror_root: Path, identity: Mapping[str, Any], target: Path, *, ambient_seed: int) -> Path:
    """Select a byte-identical committed checkpoint, never a primary-only tail."""
    if ambient_seed != 999:
        raise BenchmarkIdentityError("resumed process must begin under ambient seed 999")
    generation, checkpoint, checkpoint_archive = _load_committed_benchmark_state(primary_root, mirror_root, identity)
    primary = Path(primary_root) / "generations" / str(generation)
    mirror = Path(mirror_root) / "generations" / str(generation)
    rng = checkpoint.get("rng_state")
    if not isinstance(rng, Mapping):
        raise BenchmarkIdentityError("common generation has no RNG snapshot")
    _atomic_write(Path(target), checkpoint_archive)
    return Path(target)


def _outcome(root: Path) -> dict[str, Any]:
    generation = root / "generations" / "2"
    checkpoint = _torch_load_bytes((generation / "benchmark_state.pt").read_bytes())
    prediction = (root / "predictions" / "000001.txt").read_bytes()
    return {"completed_epochs": [1, 2], "results_csv": (generation / "results.csv").read_text(encoding="utf-8"),
            "checkpoint_digest": canonical_checkpoint_digest(checkpoint),
            "diagnostics": json.loads((generation / "diagnostics.json").read_text(encoding="utf-8")),
            "prediction_bytes": prediction, "evaluator": {"moderate_macro_ap_r40": 0.0}}


def run_synthetic_recovery_probe(primary_root: Path, mirror_root: Path, *, stop_after_epoch: int | None,
                                 resume: bool = False, ambient_seed: int = 0) -> dict[str, Any]:
    """Synthetic RED/GREEN fixture that models stop only after epoch-1 durability."""
    identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_uninterrupted_two_epoch",
        config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64,
        development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=2)
    start = 1
    if resume:
        if ambient_seed != 999:
            raise BenchmarkIdentityError("resumed process must begin under ambient seed 999")
        start = _latest_common_generation(primary_root, mirror_root, identity) + 1
    rows = "epoch,loss,time\n" if start == 1 else (Path(primary_root) / "generations" / str(start - 1) / "results.csv").read_text(encoding="utf-8")
    for epoch in range(start, 3):
        rows += f"{epoch},1.0,{epoch * 10}\n"
        state = {"completed_epoch": epoch, "model": {"w": b"fixed-model"}, "ema": {"w": b"fixed-ema"}, "ema_updates": epoch,
                 "optimizer": {"step": epoch}, "scaler": {"scale": 1.0}, "scheduler": {"last_epoch": epoch},
                 "rng_state": b"seed0-restored", "train_args": {"seed": 0, "epochs": 2}}
        archive = _torch_save_bytes({"epoch": epoch - 1, "model": {}, "ema": None, "optimizer": {}, "scaler": {}, "train_args": {}})
        publish_generation(primary_root, mirror_root, identity, epoch=epoch, checkpoint_archive=archive, benchmark_state=state, results_csv=rows,
                           diagnostics={"epoch": epoch, "stable": True})
        if stop_after_epoch == epoch:
            raise RuntimeError(f"intentional interruption after durable epoch {epoch}")
    predictions = Path(primary_root) / "predictions" / "000001.txt"
    _atomic_write(predictions, b"Pedestrian 0 0 0 0 0 1 1 1\n")
    _atomic_write(Path(mirror_root) / "predictions" / "000001.txt", predictions.read_bytes())
    return _outcome(Path(primary_root))


def compare_recovery(uninterrupted: Mapping[str, Any], resumed: Mapping[str, Any]) -> dict[str, str]:
    if uninterrupted.get("completed_epochs") != [1, 2] or resumed.get("completed_epochs") != [1, 2]:
        raise BenchmarkIdentityError("recovery has duplicate or missing completed epochs")
    def non_time_csv(value: str) -> str:
        lines = value.strip().splitlines()
        return "\n".join(",".join(item for index, item in enumerate(line.split(",")) if index != 2) for line in lines)
    for label, left, right in (("results", non_time_csv(str(uninterrupted.get("results_csv", ""))), non_time_csv(str(resumed.get("results_csv", "")))),
                               ("checkpoint", uninterrupted.get("checkpoint_digest"), resumed.get("checkpoint_digest")),
                               ("diagnostics", uninterrupted.get("diagnostics"), resumed.get("diagnostics")),
                               ("predictions", uninterrupted.get("prediction_bytes"), resumed.get("prediction_bytes")),
                               ("evaluator", uninterrupted.get("evaluator"), resumed.get("evaluator"))):
        if left != right:
            raise BenchmarkIdentityError(f"recovery {label} differs outside frozen exclusions")
    return {"decision": "GO_SEED0_30_EPOCH_PREFLIGHT"}
