"""Run the registered, diagnostic-only 2x2 P2 interaction screen.

This wrapper owns split/provenance/audit artifacts and delegates optimization to
the existing IFDR runtime/trainer.  It deliberately has no S1--S4 logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.experiments.config import load_ifdr_config


FIT_COUNT = 3341
DEVELOPMENT_COUNT = 371
FIT_SHA256 = "50a1f8d72b747d1e2f460a4e0a355469484d33dbf1e9928b64ba6594c1873362"
DEVELOPMENT_SHA256 = "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
EPOCHS = 30
FROZEN_EPOCHS = 5
RAMP_EPOCHS = 10
DIAGNOSTIC_NODES = (17, 20, 23, 26)
MIRROR_INTERVAL_SECONDS = 300
REGISTERED_SEEDS = (17, 29, 41)
DIAGNOSTIC_EXECUTION_PURPOSE = "diagnostic_screen"
STAGE9_EXECUTION_PURPOSE = "stage9_seed0_candidate_selection"
STAGE11_DCLI_EXECUTION_PURPOSE = "stage11_seeds1_4_matched_dcli"
STAGE11_DCLI_SEEDS = (1, 2, 3, 4)
FORMAL_RERUN_DCLI_EXECUTION_PURPOSE = "stage11_formal_rerun_dcli"
FORMAL_RERUN_DCLI_SEEDS = (0, 1, 2, 3, 4)
VARIANT_COMPONENTS = {
    "ifdr-p2-interaction-c": (False, False),
    "ifdr-p2-interaction-a": (True, False),
    "ifdr-p2-interaction-b": (False, True),
    "ifdr-p2-interaction-ab": (True, True),
    # Historical S0 name; it is the registered AB compatibility alias.
    "ifdr-p2-interaction-s0": (True, True),
}


def variant_components(variant: str) -> tuple[bool, bool]:
    if not isinstance(variant, str) or variant not in VARIANT_COMPONENTS:
        raise ValueError(f"unknown registered 2x2 variant: {variant!r}")
    return VARIANT_COMPONENTS[variant]


def _execution_contract(
    *,
    variant: str,
    seed: int,
    mode: str,
    benchmark_context: Any | None,
    execution_purpose: str,
) -> dict[str, object]:
    if execution_purpose == FORMAL_RERUN_DCLI_EXECUTION_PURPOSE:
        if (
            variant != "ifdr-p2-interaction-b"
            or seed not in FORMAL_RERUN_DCLI_SEEDS
            or mode != "full"
            or benchmark_context is not None
        ):
            raise ValueError(
                "Stage11 formal rerun DCLI requires variant B, a seed in 0-4, full mode, and no benchmark context"
            )
        return {
            "diagnostic_only": False,
            "execution_purpose": FORMAL_RERUN_DCLI_EXECUTION_PURPOSE,
        }
    if execution_purpose == STAGE11_DCLI_EXECUTION_PURPOSE:
        if (
            variant != "ifdr-p2-interaction-b"
            or seed not in STAGE11_DCLI_SEEDS
            or mode != "full"
            or benchmark_context is not None
        ):
            raise ValueError(
                "Stage11 matched DCLI requires variant B, a seed in 1-4, full mode, and no benchmark context"
            )
        return {
            "diagnostic_only": False,
            "execution_purpose": STAGE11_DCLI_EXECUTION_PURPOSE,
        }
    if execution_purpose == STAGE9_EXECUTION_PURPOSE:
        if (
            variant != "ifdr-p2-interaction-b"
            or seed != 0
            or mode != "full"
            or benchmark_context is not None
        ):
            raise ValueError(
                "Stage9 candidate selection requires DCLI variant B, seed 0, full mode, and no benchmark context"
            )
        return {
            "diagnostic_only": False,
            "execution_purpose": STAGE9_EXECUTION_PURPOSE,
        }
    if execution_purpose != DIAGNOSTIC_EXECUTION_PURPOSE:
        raise ValueError("unknown execution purpose")
    return {
        "diagnostic_only": True,
        "execution_purpose": DIAGNOSTIC_EXECUTION_PURPOSE,
    }


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_screen_identity(
    *,
    variant: str,
    seed: int,
    config_sha256: str,
    code_sha256: str,
    model_sha256: str,
    pretrained_sha256: str,
    fit_ids_sha256: str,
    development_ids_sha256: str,
    run_mode: str = "full",
    git_commit: str = "unknown",
    fusion_gate: bool | None = None,
    dcli: bool | None = None,
    execution_purpose: str = DIAGNOSTIC_EXECUTION_PURPOSE,
    diagnostic_only: bool = True,
    benchmark_identity_sha256: str | None = None,
    benchmark_execution_role: str | None = None,
    benchmark_effective_epochs: int | None = None,
) -> dict[str, object]:
    """Return the immutable S0 identity, including its self digest."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(variant, str) or not variant:
        raise ValueError("variant must be non-empty")
    if run_mode not in {"dry-run", "smoke", "full"}:
        raise ValueError("run_mode must be dry-run, smoke, or full")
    expected_fusion_gate, expected_dcli = variant_components(variant)
    if fusion_gate is None:
        fusion_gate = expected_fusion_gate
    if dcli is None:
        dcli = expected_dcli
    if type(fusion_gate) is not bool or type(dcli) is not bool:
        raise ValueError("component flags must be booleans")
    if fusion_gate != expected_fusion_gate or dcli != expected_dcli:
        raise ValueError("variant and component flags do not match")
    if type(diagnostic_only) is not bool:
        raise ValueError("diagnostic_only must be a boolean")
    benchmark_bound = any(
        value is not None
        for value in (
            benchmark_identity_sha256,
            benchmark_execution_role,
            benchmark_effective_epochs,
        )
    )
    contract = _execution_contract(
        variant=variant,
        seed=seed,
        mode=run_mode,
        benchmark_context=object() if benchmark_bound else None,
        execution_purpose=execution_purpose,
    )
    if diagnostic_only != contract["diagnostic_only"]:
        raise ValueError("execution purpose and diagnostic_only do not match")
    hashes = {
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "model_sha256": model_sha256,
        "pretrained_sha256": pretrained_sha256,
        "fit_ids_sha256": fit_ids_sha256,
        "development_ids_sha256": development_ids_sha256,
    }
    for name, value in hashes.items():
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError(f"{name} must be a 64-hex SHA256")
        hashes[name] = value.lower()
    identity: dict[str, object] = {
        "schema_version": 1,
        "variant": variant,
        "seed": seed,
        "run_mode": run_mode,
        "git_commit": git_commit,
        "epochs": EPOCHS,
        "frozen_epochs": FROZEN_EPOCHS,
        "ramp_epochs": RAMP_EPOCHS,
        "diagnostic_interval": 50,
        "diagnostic_nodes": list(DIAGNOSTIC_NODES),
        "fusion_gate": fusion_gate,
        "dcli": dcli,
        "factor_supervision": True,
        "interventions": True,
        "p2_fusion_modulation": True,
        "p2_dcli_factor_conditioning": True,
        "primary_checkpoint_role": "last.pt",
        "checkpoint_interval_seconds": MIRROR_INTERVAL_SECONDS,
        "metric": "Pedestrian/Cyclist Moderate macro AP_R40",
        "diagnostic_only": diagnostic_only,
        "execution_purpose": execution_purpose,
        "eta_seconds": None,
        "eta_source": "first remote smoke",
        **hashes,
        "fit_count": FIT_COUNT,
        "development_count": DEVELOPMENT_COUNT,
    }
    if benchmark_bound:
        if not (isinstance(benchmark_identity_sha256, str) and len(benchmark_identity_sha256) == 64 and benchmark_execution_role in {"timing_one_epoch", "recovery_uninterrupted_two_epoch", "recovery_interrupted_two_epoch"} and benchmark_effective_epochs in {1, 2}):
            raise ValueError("invalid benchmark identity binding")
        identity.update({"benchmark_identity_sha256": benchmark_identity_sha256, "benchmark_execution_role": benchmark_execution_role, "benchmark_effective_epochs": benchmark_effective_epochs})
    identity["identity_sha256"] = _payload_sha256(identity)
    return identity


def resume_epoch_from_results(path: Path) -> int:
    """Return the number of contiguous epochs in a trainer results CSV."""

    if not path.is_file():
        return 0
    rows = path.read_text(encoding="utf-8").splitlines()
    epochs: list[int] = []
    for row in rows[1:]:
        if not row.strip():
            continue
        try:
            epoch = int(float(row.split(",", 1)[0]))
        except (ValueError, IndexError) as error:
            raise ValueError("results.csv contains an invalid epoch") from error
        epochs.append(epoch)
    if not epochs:
        return 0
    start = epochs[0]
    if start not in (0, 1):
        raise ValueError("results.csv epochs must start at zero or one")
    if epochs != list(range(start, start + len(epochs))):
        raise ValueError("results.csv epochs are not contiguous")
    return len(epochs)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _validate_disjoint(primary: Path, mirror: Path) -> None:
    primary = primary.resolve()
    mirror = mirror.resolve()
    if primary == mirror or primary.is_relative_to(mirror) or mirror.is_relative_to(primary):
        raise ValueError("mirror directory must be disjoint from output directory")


def sync_mirror(primary: Path, mirror: Path) -> None:
    """Atomically mirror the small resumability/provenance set."""

    _validate_disjoint(primary, mirror)
    mirror.mkdir(parents=True, exist_ok=True)
    names = (
        "screen_manifest.json",
        "status.json",
        "results.csv",
        "gradient_diagnostics.jsonl",
        "assignment_diagnostics.jsonl",
        "post_training_leakage_audit.json",
        "checkpoint_provenance.json",
        "metrics_ap40.json",
    )
    records: list[dict[str, object]] = []

    def record(path: str, target: Path) -> None:
        records.append({
            "name": path,
            "path": path,
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
        })

    def copy_atomically(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)

    for name in names:
        source = primary / name
        if not source.is_file():
            continue
        target = mirror / name
        copy_atomically(source, target)
        record(name, target)
    checkpoint = primary / "weights" / "last.pt"
    if checkpoint.is_file() and checkpoint.stat().st_size:
        target = mirror / "weights" / "last.pt.sha256"
        _atomic_bytes(target, f"{sha256_file(checkpoint)}  {checkpoint.name}\n".encode("utf-8"))
        record("weights/last.pt", checkpoint)

    prediction_root = primary / "predictions" / "labels"
    expected_predictions: set[Path] = set()
    if prediction_root.exists():
        if prediction_root.is_symlink() or not prediction_root.is_dir():
            raise ValueError("prediction labels must be a regular directory")
        for source in sorted(prediction_root.rglob("*")):
            if source.is_symlink():
                raise ValueError("prediction labels must not contain symlinks")
            if source.is_dir():
                continue
            if not source.is_file() or source.suffix != ".txt":
                raise ValueError("prediction labels must contain only regular .txt files")
            relative = source.relative_to(prediction_root)
            expected_predictions.add(relative)
            path = Path("predictions") / "labels" / relative
            target = mirror / path
            copy_atomically(source, target)
            record(path.as_posix(), target)

    mirrored_prediction_root = mirror / "predictions" / "labels"
    if mirrored_prediction_root.exists():
        if mirrored_prediction_root.is_symlink() or not mirrored_prediction_root.is_dir():
            raise ValueError("mirrored prediction labels must be a regular directory")
        for target in sorted(mirrored_prediction_root.rglob("*"), reverse=True):
            if target.is_symlink():
                raise ValueError("mirrored prediction labels must not contain symlinks")
            if target.is_file():
                if target.relative_to(mirrored_prediction_root) not in expected_predictions:
                    target.unlink()
            elif target.is_dir():
                try:
                    target.rmdir()
                except OSError:
                    pass
    manifest = {
        "schema_version": 1,
        "primary_output": str(primary.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    _atomic_json(mirror / "manifest.json", {**manifest, "generation": _payload_sha256(records)})


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ("git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _code_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_git_commit(root).encode("utf-8"))
    for relative in (
        "scripts/run_p2_interaction_s0.py",
        "ifdr_yolo/experiments/config.py",
        "ifdr_yolo/experiments/ifdr_runtime.py",
        "ifdr_yolo/experiments/ifdr_trainer.py",
        "ifdr_yolo/experiments/gradient_diagnostics.py",
        "ifdr_yolo/models/gated_fusion.py",
        "ifdr_yolo/models/ifdr_model.py",
        "ifdr_yolo/losses/ifdr_detection.py",
    ):
        path = root / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_split(config: Any, fit_path: Path, development_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if fit_path.is_symlink() or development_path.is_symlink():
        raise ValueError("S0 split manifests must be regular files")
    fit = load_ids(fit_path)
    development = load_ids(development_path)
    if len(fit) != FIT_COUNT or len(development) != DEVELOPMENT_COUNT:
        raise ValueError(f"S0 requires {FIT_COUNT} fit and {DEVELOPMENT_COUNT} development IDs")
    overlap = sorted(set(fit) & set(development))
    if overlap:
        raise ValueError(f"fit/development overlap (leakage): {overlap[:5]}")
    full = load_ids(config.paths.train_ids)
    if len(full) != FIT_COUNT + DEVELOPMENT_COUNT or set(full) != set(fit) | set(development):
        raise ValueError("fit/development manifests do not exactly cover the registered training split")
    if sha256_file(fit_path) != FIT_SHA256:
        raise ValueError("registered fit split hash mismatch")
    if sha256_file(development_path) != DEVELOPMENT_SHA256:
        raise ValueError("registered development split hash mismatch")
    return fit, development


def _materialize_views(config: Any, output: Path, fit: Sequence[str], development: Sequence[str]) -> None:
    generated = config.paths.generated_data.resolve()
    for split_name, ids, view_split in (("fit", fit, "train"), ("development", development, "val")):
        for image_id in ids:
            for kind, extension in (("images", ".png"), ("labels", ".txt")):
                source = generated / kind / "train" / f"{image_id}{extension}"
                if not source.is_file() or source.is_symlink():
                    raise FileNotFoundError(f"missing generated {kind} for {image_id}: {source}")
                target = output / "view" / kind / view_split / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if not target.is_file() or not os.path.samefile(source, target):
                        raise ValueError(f"existing S0 view differs: {target}")
                else:
                    os.link(source, target)


def _write_resolved_data(config: Any, output: Path) -> Path:
    payload = yaml.safe_load(config.paths.data.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("data config must be a mapping")
    payload.update({"path": str((output / "view").resolve()), "train": "images/train", "val": "images/val"})
    path = output / "resolved_data.yaml"
    _atomic_bytes(path, yaml.safe_dump(payload, sort_keys=False).encode("utf-8"))
    return path


def _decode_cache(path: Path) -> object:
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    try:
        import numpy as np

        loaded = np.load(path, allow_pickle=True)
        if hasattr(loaded, "files"):
            return {name: loaded[name] for name in loaded.files}
        return loaded.item() if hasattr(loaded, "item") else loaded
    except (ImportError, OSError, ValueError, EOFError, AttributeError, pickle.UnpicklingError, TypeError):
        return pickle.loads(raw)


def _cache_paths(value: object) -> list[str]:
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_cache_paths(item))
        return result
    tolist = getattr(value, "tolist", None)
    return _cache_paths(tolist()) if callable(tolist) else []


def _cache_ids(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError("train label cache must contain a mapping")
    records = payload.get("labels")
    if isinstance(records, Mapping):
        records = list(records.values())
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("train label cache has no labels records")
    ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("train label cache contains a malformed record")
        values = record.get("im_file", record.get("im_files"))
        paths = _cache_paths(values)
        if not paths:
            raise ValueError("train label cache record has no im_file")
        ids.extend(Path(value).stem for value in paths)
    if len(set(ids)) != len(ids):
        raise ValueError("train label cache contains duplicate image IDs")
    return tuple(ids)


def _audit_cache(output: Path, fit: Sequence[str], development: Sequence[str]) -> Path:
    candidates = (
        output / "train.cache",
        output / "labels.cache",
        output / "view" / "labels" / "train.cache",
        output / "view" / "labels" / "train.cache.npy",
        output / "view" / "labels" / "train.cache.npz",
    )
    cache = next((path for path in candidates if path.is_file()), None)
    if cache is None:
        raise FileNotFoundError("S0 trainer did not produce a train label cache")
    observed = _cache_ids(_decode_cache(cache))
    overlap = sorted(set(observed) & set(development))
    if set(observed) != set(fit) or len(observed) != len(fit):
        raise ValueError("S0 train cache IDs do not match the fit manifest")
    if overlap:
        raise ValueError(f"S0 train cache contains development IDs: {overlap[:5]}")
    audit = {
        "schema_version": 1,
        "train_cache_path": str(cache.resolve()),
        "train_cache_sha256": sha256_file(cache),
        "observed_train_count": len(observed),
        "fit_count": len(fit),
        "development_count": len(development),
        "fit_ids_sha256": FIT_SHA256,
        "development_ids_sha256": DEVELOPMENT_SHA256,
        "intersection_count": len(overlap),
        "intersection_ids": overlap,
        "observed_train_ids_sha256": _payload_sha256(list(observed)),
    }
    path = output / "post_training_leakage_audit.json"
    _atomic_json(path, audit)
    return path


def _validate_gradient_diagnostics(output: Path) -> None:
    path = output / "gradient_diagnostics.jsonl"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("S0 trainer did not produce gradient_diagnostics.jsonl")
    found = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        node_payload = payload.get("node_diagnostics") if isinstance(payload, Mapping) else None
        nodes = node_payload.get("nodes") if isinstance(node_payload, Mapping) else None
        if isinstance(nodes, Mapping):
            keys = {int(key) for key in nodes if str(key).isdigit()}
            if set(DIAGNOSTIC_NODES).issubset(keys):
                found = True
                break
    if not found:
        raise ValueError("gradient diagnostics are missing nodes 17/20/23/26")


def _status(output: Path, identity: Mapping[str, object], state: str, *, epoch: int, next_action: str, error: BaseException | None = None) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "state": state,
        "pid": os.getpid(),
        "epoch": epoch,
        "next_action": next_action,
        "identity_sha256": identity["identity_sha256"],
        "variant": identity["variant"],
        "fusion_gate": identity["fusion_gate"],
        "dcli": identity["dcli"],
        "execution_purpose": identity["execution_purpose"],
        "diagnostic_only": identity["diagnostic_only"],
        "checkpoint_role": "last.pt",
        "eta_seconds": identity.get("eta_seconds"),
        "eta_source": identity.get("eta_source"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint = output / "weights" / "last.pt"
    if checkpoint.is_file() and checkpoint.stat().st_size:
        payload.update({"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint)})
    if error is not None:
        payload.update({"error_type": type(error).__name__, "error_message": str(error)})
    _atomic_json(output / "status.json", payload)


class _ProgressMirror:
    def __init__(self, output: Path, mirror: Path, identity: Mapping[str, object]) -> None:
        self.output = output
        self.mirror = mirror
        self.identity = identity
        self.stop = threading.Event()
        self.errors: list[BaseException] = []
        self.thread = threading.Thread(target=self._loop, name="s0-progress-mirror", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=15.0)
        sync_mirror(self.output, self.mirror)

    def _loop(self) -> None:
        seen = 0
        last_sync = time.monotonic()
        while not self.stop.wait(5.0):
            try:
                epoch = resume_epoch_from_results(self.output / "results.csv")
                checkpoint = self.output / "weights" / "last.pt"
                if epoch > seen and checkpoint.is_file() and checkpoint.stat().st_size:
                    seen = epoch
                    _status(self.output, self.identity, "running", epoch=epoch, next_action="continue S0 training")
                    sync_mirror(self.output, self.mirror)
                    last_sync = time.monotonic()
                elif time.monotonic() - last_sync >= MIRROR_INTERVAL_SECONDS - 60:
                    sync_mirror(self.output, self.mirror)
                    last_sync = time.monotonic()
            except ValueError:
                # Ultralytics can hold results.csv between header/row writes.
                continue
            except BaseException as error:  # fail closed after trainer returns
                self.errors.append(error)
                return


def _training_args(config: Any, *, data_path: Path, output: Path, device: str, mode: str, benchmark_context: Any | None = None) -> dict[str, object]:
    args = asdict(config.training)
    args.update({
        "data": str(data_path),
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
        "seed": config.experiment.seed,
        "device": device,
        "val": True,
        "save": True,
        "plots": True,
        "pretrained": False,
        "save_period": -1,
    })
    if benchmark_context is not None:
        args["epochs"] = benchmark_context.identity["effective_epochs"]
        # The registered 1/2-epoch benchmark truncates the formal 30-epoch
        # schedule. Keeping Ultralytics' default close_mosaic=10 makes a
        # resumed epoch-2 run close Mosaic during resume setup while the
        # uninterrupted epoch-2 run never reaches the negative close epoch.
        # Disable the end-of-training transition for the shortened benchmark
        # so both trajectories retain the same augmentation policy.
        args["close_mosaic"] = 0
    elif mode == "smoke":
        # Remote ETA calibration must use the real 640 geometry and formal
        # batch/AMP settings; only the epoch budget is shortened.
        args["epochs"] = 1
    return args


def _train(
    config: Any,
    output: Path,
    data_path: Path,
    *,
    device: str,
    mode: str,
    resume: bool,
    benchmark_context: Any | None,
    stage11_recovery_root: Path | None,
    stage11_execution_identity_sha256: str | None,
) -> Path:
    if (stage11_recovery_root is None) != (stage11_execution_identity_sha256 is None):
        raise ValueError("Stage11 recovery root and execution identity must be supplied together")
    if benchmark_context is not None and stage11_recovery_root is not None:
        raise ValueError("benchmark and Stage11 recovery contexts are mutually exclusive")
    from ifdr_yolo.experiments.ifdr_runtime import IFDRRuntimeAdapter
    from ifdr_yolo.experiments.ifdr_trainer import IFDRComponentSwitches, IFDRDetectionTrainer, FusionSchedule
    from ifdr_yolo.data.interventions.sampler import SamplingPolicy

    method = config.method
    switches = IFDRComponentSwitches(
        fusion_gate=method.components.fusion_gate,
        dcli=method.components.dcli,
        factor_supervision=method.components.factor_supervision,
        interventions=method.components.interventions,
        semantic_protection=method.components.semantic_protection,
        counterfactual_consistency=method.components.counterfactual_consistency,
    )
    intervention = method.intervention
    if resume:
        if benchmark_context is not None:
            from ifdr_yolo.experiments.kitti_seed0_training_benchmark import prepare_resume_checkpoint
            prepare_resume_checkpoint(
                benchmark_context.primary_root, benchmark_context.mirror_root, benchmark_context.identity,
                output / "weights" / "last.pt", ambient_seed=999,
            )
        trainer = IFDRDetectionTrainer(
            overrides={
                "resume": str(output / "weights" / "last.pt"),
                "data": str(data_path),
                "save_dir": str(output),
                "project": str(output.parent),
                "name": output.name,
                "exist_ok": True,
                "device": device,
                "workers": config.training.workers,
                "val": True,
                "plots": True,
                "save_period": -1,
            },
            fusion_schedule=FusionSchedule(
                frozen_epochs=method.schedule.frozen_epochs,
                ramp_epochs=method.schedule.ramp_epochs,
            ),
            component_switches=switches,
            intervention_seed=intervention.base_seed,
            intervention_policy=SamplingPolicy(
                identity_probability=intervention.identity_probability,
                sampling_probability=intervention.sampling_probability,
                visibility_probability=intervention.visibility_probability,
                minimum_strength=intervention.minimum_strength,
                maximum_strength=intervention.maximum_strength,
            ),
            gradient_diagnostic_interval=method.gradient_diagnostic_interval,
        )
        if benchmark_context is not None:
            from ifdr_yolo.experiments.kitti_seed0_training_benchmark import configure_benchmark_callbacks
            configure_benchmark_callbacks(trainer, benchmark_context, resume=True)
        elif stage11_recovery_root is not None and stage11_execution_identity_sha256 is not None:
            from ifdr_yolo.experiments.stage11_full_recovery import configure_stage11_local_recovery
            configure_stage11_local_recovery(
                trainer,
                output,
                stage11_recovery_root,
                execution_identity_sha256=stage11_execution_identity_sha256,
                resume=True,
            )
        trainer.train()
    else:
        runtime = IFDRRuntimeAdapter(config)
        prepared = runtime.prepare_model(
            model_path=config.paths.model,
            model_sha256=config.paths.model_sha256,
            initialization=config.initialization,
            seed=config.experiment.seed,
            deterministic=config.training.deterministic,
        )
        if benchmark_context is not None:
            from ifdr_yolo.experiments.kitti_seed0_training_benchmark import configure_benchmark_callbacks
            configure_benchmark_callbacks(prepared.handle, benchmark_context, resume=False)
        elif stage11_recovery_root is not None and stage11_execution_identity_sha256 is not None:
            from ifdr_yolo.experiments.stage11_full_recovery import configure_stage11_local_recovery
            configure_stage11_local_recovery(
                prepared.handle,
                output,
                stage11_recovery_root,
                execution_identity_sha256=stage11_execution_identity_sha256,
                resume=False,
            )
        runtime.train(
            prepared_model=prepared,
            data_path=data_path,
            run_dir=output,
            args=_training_args(config, data_path=data_path, output=output, device=device, mode=mode, benchmark_context=benchmark_context),
        )
    best = output / "weights" / "best.pt"
    last = output / "weights" / "last.pt"
    if not best.is_file() or not last.is_file() or best.stat().st_size <= 0 or last.stat().st_size <= 0:
        raise FileNotFoundError("S0 trainer did not produce best.pt and last.pt")
    return best


def primary_checkpoint(output: Path) -> Path:
    """Return the checkpoint role registered as the S0 primary metric weight."""
    checkpoint = Path(output) / "weights" / "last.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError("S0 primary checkpoint last.pt is missing or empty")
    return checkpoint


def _completed_status_epoch(*, mode: str, benchmark_context: Any | None) -> int:
    if benchmark_context is not None:
        return int(benchmark_context.identity["effective_epochs"])
    return EPOCHS if mode == "full" else 1


def _completion_next_action(execution_purpose: str) -> str:
    if execution_purpose == FORMAL_RERUN_DCLI_EXECUTION_PURPOSE:
        return "freeze Stage11 formal rerun DCLI result"
    if execution_purpose == STAGE11_DCLI_EXECUTION_PURPOSE:
        return "freeze Stage11 matched DCLI result"
    if execution_purpose == STAGE9_EXECUTION_PURPOSE:
        return "freeze Stage9 seed-0 candidate-selection result"
    if execution_purpose == DIAGNOSTIC_EXECUTION_PURPOSE:
        return "do not use S0 for final paper AP"
    raise ValueError("unknown execution purpose")


def _macro_ap(metrics: Mapping[str, object]) -> float:
    classes = metrics.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("AP40 payload has no classes")
    values: list[float] = []
    for name in ("Pedestrian", "Cyclist"):
        payload = classes.get(name)
        if not isinstance(payload, Mapping):
            raise ValueError(f"AP40 payload has no {name}")
        moderate = payload.get("moderate")
        if not isinstance(moderate, Mapping) or not isinstance(moderate.get("ap40"), (int, float)):
            raise ValueError(f"AP40 payload has no {name} Moderate ap40")
        values.append(float(moderate["ap40"]))
    return sum(values) / len(values)


def _moderate_ap_values(metrics: Mapping[str, object]) -> tuple[float, float]:
    classes = metrics.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("AP40 payload has no classes")
    values: list[float] = []
    for name in ("Pedestrian", "Cyclist"):
        payload = classes.get(name)
        moderate = payload.get("moderate") if isinstance(payload, Mapping) else None
        value = moderate.get("ap40") if isinstance(moderate, Mapping) else None
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"AP40 payload has no {name} Moderate ap40")
        values.append(float(value))
    return values[0], values[1]


def run_screen(*, config_path: Path, fit_ids: Path, development_ids: Path, output_dir: Path, mirror_dir: Path, mode: str, device: str | None = None, resume: bool = False, benchmark_context: Any | None = None, execution_purpose: str = DIAGNOSTIC_EXECUTION_PURPOSE) -> Path | None:
    if mode not in {"dry-run", "smoke", "full"}:
        raise ValueError(f"unknown S0 mode: {mode}")
    root = Path(__file__).resolve().parents[1]
    if mode != "dry-run":
        from ifdr_yolo.experiments.ultralytics_runtime import bootstrap_ultralytics_config

        bootstrap_ultralytics_config(root)
    config_path = (config_path if config_path.is_absolute() else root / config_path).resolve()
    fit_ids = fit_ids.resolve()
    development_ids = development_ids.resolve()
    output = output_dir.resolve()
    mirror = mirror_dir.resolve()
    _validate_disjoint(output, mirror)
    config = load_ifdr_config(config_path, repository_root=root)
    expected_fusion_gate, expected_dcli = variant_components(config.experiment.variant)
    if benchmark_context is not None:
        from ifdr_yolo.experiments.kitti_seed0_training_benchmark import BenchmarkRunContext
        if not isinstance(benchmark_context, BenchmarkRunContext):
            raise ValueError("benchmark_context must be a BenchmarkRunContext")
        if config.experiment.seed != 0 or config.experiment.variant != "ifdr-p2-interaction-b":
            raise ValueError("benchmark DCLI requires frozen variant B and seed 0")
    elif execution_purpose == DIAGNOSTIC_EXECUTION_PURPOSE and config.experiment.seed not in REGISTERED_SEEDS:
        raise ValueError(f"2x2 screen requires one of seeds {REGISTERED_SEEDS}")
    execution_contract = _execution_contract(
        variant=config.experiment.variant,
        seed=config.experiment.seed,
        mode=mode,
        benchmark_context=benchmark_context,
        execution_purpose=execution_purpose,
    )
    method = config.method
    if (
        method.components.fusion_gate != expected_fusion_gate
        or method.components.dcli != expected_dcli
    ):
        raise ValueError("variant and component flags do not match")
    if not method.components.factor_supervision or not method.components.interventions:
        raise ValueError("2x2 screen requires factor supervision and interventions")
    if method.components.semantic_protection or method.components.counterfactual_consistency or method.loss.counterfactual_gain != 0.0:
        raise ValueError("2x2 screen must use the unprotected, non-counterfactual path")
    if not method.p2_path_switches.fusion_modulation or not method.p2_path_switches.dcli_factor_conditioning:
        raise ValueError("2x2 screen requires the registered P2 switches")
    if method.gradient_diagnostic_interval != 50:
        raise ValueError("2x2 screen requires gradient_diagnostic_interval=50")
    try:
        diagnostic_nodes = tuple(method.p2_path_switches.nodes)
    except TypeError as error:
        raise ValueError("2x2 screen diagnostic nodes are invalid") from error
    if diagnostic_nodes != (17,):
        raise ValueError("2x2 screen requires diagnostic node 17")
    if method.schedule.frozen_epochs != FROZEN_EPOCHS or method.schedule.ramp_epochs != RAMP_EPOCHS or config.training.epochs != EPOCHS:
        raise ValueError("S0 schedule and 30-epoch budget are immutable")
    fit, development = _validate_split(config, fit_ids, development_ids)
    model_sha = sha256_file(config.paths.model)
    pretrained_sha = sha256_file(config.initialization.pretrained)
    if model_sha != config.paths.model_sha256.lower() or pretrained_sha != config.initialization.pretrained_sha256.lower():
        raise ValueError("model/pretrained SHA256 does not match config identity")
    identity = build_screen_identity(
        variant=config.experiment.variant,
        seed=config.experiment.seed,
        config_sha256=sha256_file(config_path),
        code_sha256=_code_sha256(root),
        model_sha256=model_sha,
        pretrained_sha256=pretrained_sha,
        fit_ids_sha256=sha256_file(fit_ids),
        development_ids_sha256=sha256_file(development_ids),
        run_mode=mode,
        git_commit=_git_commit(root),
        fusion_gate=method.components.fusion_gate,
        dcli=method.components.dcli,
        execution_purpose=str(execution_contract["execution_purpose"]),
        diagnostic_only=bool(execution_contract["diagnostic_only"]),
        benchmark_identity_sha256=(str(benchmark_context.identity["identity_sha256"]) if benchmark_context is not None else None),
        benchmark_execution_role=(str(benchmark_context.identity["execution_role"]) if benchmark_context is not None else None),
        benchmark_effective_epochs=(int(benchmark_context.identity["effective_epochs"]) if benchmark_context is not None else None),
    )
    stage11_recovery_root: Path | None = None
    stage11_execution_identity_sha256: str | None = None
    if execution_purpose in {
        STAGE11_DCLI_EXECUTION_PURPOSE,
        FORMAL_RERUN_DCLI_EXECUTION_PURPOSE,
    }:
        from ifdr_yolo.experiments.stage11_full_recovery import validate_stage11_local_recovery_root
        stage11_recovery_root = output.with_name(f"{output.name}.stage11-recovery")
        validate_stage11_local_recovery_root(stage11_recovery_root, resume=resume)
        stage11_execution_identity_sha256 = str(identity["identity_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "screen_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("existing S0 output identity does not match")
    elif any(output.iterdir()):
        raise ValueError("existing S0 output has no identity manifest")
    _atomic_json(manifest_path, identity)
    _materialize_views(config, output, fit, development)
    data_path = _write_resolved_data(config, output)
    sync_mirror(output, mirror)
    status_path = output / "status.json"
    state = "prepared"
    if status_path.is_file():
        state_payload = json.loads(status_path.read_text(encoding="utf-8"))
        if state_payload.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("existing S0 status identity does not match")
        state = str(state_payload.get("state", "prepared"))
    if mode == "dry-run":
        _status(output, identity, "prepared", epoch=0, next_action="run S0 with --mode full")
        sync_mirror(output, mirror)
        return None
    if state == "complete":
        raise ValueError("S0 run is already complete")
    if state == "running":
        try:
            owner_pid = int(state_payload.get("pid", 0))
        except (TypeError, ValueError):
            owner_pid = 0
        if owner_pid and owner_pid != os.getpid():
            try:
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                pass
            except PermissionError as error:
                raise RuntimeError(f"S0 run owner PID {owner_pid} cannot be inspected") from error
            else:
                raise RuntimeError(f"S0 run is still active as PID {owner_pid}")
    if state in {"failed", "interrupted", "running"} and not resume:
        raise ValueError("existing S0 run requires --resume")
    if resume and not (output / "weights" / "last.pt").is_file():
        raise ValueError("S0 resume requires weights/last.pt")
    _status(output, identity, "running", epoch=resume_epoch_from_results(output / "results.csv"), next_action="resume S0 training" if resume else "train S0")
    monitor = _ProgressMirror(output, mirror, identity)
    monitor.start()
    best: Path | None = None
    try:
        best = _train(
            config,
            output,
            data_path,
            device=device or config.training.device,
            mode=mode,
            resume=resume,
            benchmark_context=benchmark_context,
            stage11_recovery_root=stage11_recovery_root,
            stage11_execution_identity_sha256=stage11_execution_identity_sha256,
        )
        if monitor.errors:
            raise RuntimeError("S0 mirror monitor failed") from monitor.errors[0]
        _validate_gradient_diagnostics(output)
        _audit_cache(output, fit, development)
        last = primary_checkpoint(output)
        _atomic_json(
            output / "checkpoint_provenance.json",
            {
                "schema_version": 1,
                "checkpoint_role": "last.pt",
                "checkpoint_path": str(last.resolve()),
                "checkpoint_sha256": sha256_file(last),
                "best_checkpoint_sha256": sha256_file(best),
                "identity_sha256": identity["identity_sha256"],
            },
        )
        from ifdr_yolo.eval.evaluate import evaluate_prediction_directory, write_evaluation_json
        from ifdr_yolo.experiments.ifdr_runtime import IFDRRuntimeAdapter

        runtime = IFDRRuntimeAdapter(config)
        labels = runtime.predict(
            weights=last,
            image_paths=tuple(output / "view" / "images" / "val" / f"{image_id}.png" for image_id in development),
            output_dir=output / "predictions",
            args={"device": device or config.training.device, "imgsz": config.training.imgsz, "conf": config.prediction.conf, "iou": config.prediction.iou, "max_det": config.prediction.max_det, "augment": False, "verbose": False},
        )
        metrics = evaluate_prediction_directory(prediction_dir=labels, label_dir=config.paths.raw_labels, image_dir=config.paths.raw_images, split_path=development_ids)
        pedestrian_ap, cyclist_ap = _moderate_ap_values(metrics)
        metrics = {
            **metrics,
            "pedestrian_moderate_ap_r40": pedestrian_ap,
            "cyclist_moderate_ap_r40": cyclist_ap,
            "moderate_macro_ap_r40": (pedestrian_ap + cyclist_ap) / 2.0,
            "diagnostic_only": identity["diagnostic_only"],
            "execution_purpose": identity["execution_purpose"],
            "identity_sha256": identity["identity_sha256"],
        }
        write_evaluation_json(output / "metrics_ap40.json", metrics)
        next_action = _completion_next_action(execution_purpose)
        _status(output, identity, "complete", epoch=_completed_status_epoch(mode=mode, benchmark_context=benchmark_context), next_action=next_action)
        return output / "metrics_ap40.json"
    except BaseException as error:
        try:
            current_epoch = resume_epoch_from_results(output / "results.csv")
        except ValueError:
            current_epoch = 0
        _status(output, identity, "failed", epoch=current_epoch, next_action="resume S0 with the same identity", error=error)
        raise
    finally:
        monitor.close()
        sync_mirror(output, mirror)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the registered 30-epoch 2x2 P2 interaction screen.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-ids", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "smoke", "full"), default="dry-run")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution-purpose",
        choices=(
            DIAGNOSTIC_EXECUTION_PURPOSE,
            STAGE9_EXECUTION_PURPOSE,
            STAGE11_DCLI_EXECUTION_PURPOSE,
            FORMAL_RERUN_DCLI_EXECUTION_PURPOSE,
        ),
        default=DIAGNOSTIC_EXECUTION_PURPOSE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = run_screen(
        config_path=args.config,
        fit_ids=args.fit_ids,
        development_ids=args.development_ids,
        output_dir=args.output_dir,
        mirror_dir=args.mirror_dir,
        mode=args.mode,
        device=args.device,
        resume=args.resume,
        execution_purpose=args.execution_purpose,
    )
    print(f"S0 {args.mode.upper()} {'READY' if metrics is None else 'COMPLETE'}")
    if metrics is not None:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        print(f"metrics_ap40={metrics}")
        print(f"moderate_macro_ap_r40={payload['moderate_macro_ap_r40']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
