"""Registered factor-repair runner.

This module deliberately keeps orchestration small.  Configuration parsing,
semantic calibration and task adaptation remain owned by the experiment
modules; the runner only binds their immutable identities to a recoverable run
directory and refuses to publish ambiguous checkpoints or metrics.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import mkstemp
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.run_store import RunStore, atomic_write_json


REGISTERED_CONDITIONS = ("M1", "M2", "M3", "F0", "F1", "F2", "F3")
CALIBRATION_CONDITIONS = ("F0", "F1", "F2", "F3")
PRIMARY_CHECKPOINT = "last.pt"
DIAGNOSTIC_CHECKPOINT = "best.pt"
CHECKPOINT_ROLE = "calibration_last"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _get(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nested(value: object, *names: str, default: object = None) -> object:
    current: object = value
    for name in names:
        current = _get(current, name, default)
        if current is default:
            return default
    return current


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.lower()) is None:
        raise ValueError(f"{field} must be a 64-hex SHA256")
    return value.lower()


def file_sha256(path: Path) -> str:
    """Hash bytes without relying on a trusted metadata value."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file_hash(path: Path, expected: object, field: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{field} file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{field} file is empty: {path}")
    expected_hash = _sha256(expected, f"{field}.sha256")
    actual = file_sha256(path)
    if actual != expected_hash:
        raise ValueError(
            f"{field} SHA256 mismatch: expected={expected_hash}, actual={actual}"
        )
    return actual


def _identity_mapping(config: object) -> dict[str, object]:
    identity = _get(config, "identity")
    if identity is None:
        identity = _get(config, "scientific_identity")
    if identity is None:
        raise ValueError("scientific identity is required")
    if isinstance(identity, Mapping):
        values = dict(identity)
    else:
        values = dict(vars(identity)) if hasattr(identity, "__dict__") else {}
        if not values:
            # Frozen dataclasses expose their fields through the dataclasses API.
            try:
                from dataclasses import fields

                values = {field.name: getattr(identity, field.name) for field in fields(identity)}
            except (TypeError, ValueError):
                values = {}
    if not values:
        raise ValueError("scientific identity is required")
    for field, value in values.items():
        _sha256(value, f"identity.{field}")
    return values


def _condition_name(config: object, condition: object | None) -> str:
    value = condition if condition is not None else _get(config, "condition")
    if not isinstance(value, str) or value not in REGISTERED_CONDITIONS:
        raise ValueError(
            "condition must be one of registered M1, M2, M3, F0, F1, F2, or F3"
        )
    return value


def _development_ids(config: object) -> frozenset[str]:
    values: object = _get(config, "development_ids")
    if values is None:
        values = _get(config, "dev_ids")
    if values is None:
        values = _nested(config, "development", "ids")
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes)):
        raise ValueError("development IDs must be a sequence")
    return frozenset(str(item) for item in values)


def validate_fit_loader_ids(config: object, loader_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Return sorted unique fit IDs and reject every development overlap."""

    if loader_ids is None:
        candidate = _get(config, "fit_ids")
        if candidate is None:
            return ()
        loader_ids = candidate  # type: ignore[assignment]
    if isinstance(loader_ids, (str, bytes)):
        raise ValueError("fit loader IDs must be a sequence")
    normalized = tuple(str(item) for item in loader_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError("fit loader IDs must be unique")
    development = _development_ids(config)
    overlap = sorted(set(normalized) & development)
    if overlap:
        raise ValueError(f"development leakage in fit loader: {overlap}")
    expected_fit = _get(config, "fit_ids")
    if expected_fit is not None and set(normalized) != {str(item) for item in expected_fit}:
        raise ValueError("fit loader IDs do not match registered fit IDs")
    return tuple(sorted(normalized))


def collect_git_provenance(root: Path) -> dict[str, object]:
    """Collect the minimal clean-checkout identity needed by a run."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": commit,
        "branch": run("branch", "--show-current"),
        "tracked_clean": not bool(status),
        "tracked_changes": tuple(status.splitlines()) if status else (),
        "untracked_files": (),
    }


def require_clean_commit(root: Path, provenance: Mapping[str, object] | None = None) -> dict[str, object]:
    record = dict(provenance or collect_git_provenance(root))
    commit = record.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("clean Git commit identity is required")
    tracked_clean = bool(record.get("tracked_clean", False))
    tracked = record.get("tracked_changes", ())
    untracked = record.get("untracked_files", ())
    if not tracked_clean or tuple(tracked or ()) or tuple(untracked or ()):
        raise ValueError("working tree must be a clean Git checkout")
    return record


def validate_initialization_checkpoint(config: object) -> tuple[Path, str]:
    paths = _get(config, "paths")
    path = _get(paths, "initialization_checkpoint")
    if path is None:
        path = _get(config, "initialization_checkpoint")
    identity = _get(config, "identity")
    expected = _get(identity, "initialization_checkpoint_sha256")
    if expected is None:
        expected = _get(config, "initialization_checkpoint_sha256")
    if path is None or expected is None:
        raise ValueError("initialization checkpoint identity is required")
    resolved = Path(path).expanduser().resolve()
    return resolved, _require_file_hash(resolved, expected, "initialization checkpoint")


class ProcessLock:
    """Small O_EXCL lock used for both runner and queue process ownership."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"duplicate factor-repair process lock: {self.path}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(f"pid={os.getpid()}\n")
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.acquired = False

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _finite(value: object, field: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")
        return
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            tensor = detach()
            try:
                import torch

                if isinstance(tensor, torch.Tensor):
                    if not bool(torch.isfinite(tensor).all().item()):
                        raise ValueError(f"{field} must be finite")
                    return
            except ImportError:
                pass
            finite = getattr(tensor, "isfinite", None)
            if callable(finite):
                result = finite()
                all_method = getattr(result, "all", None)
                if callable(all_method) and not bool(all_method()):
                    raise ValueError(f"{field} must be finite")
                return
            scalar = tensor.item()
            if isinstance(scalar, (int, float)) and not math.isfinite(float(scalar)):
                raise ValueError(f"{field} must be finite")
            return
        except ValueError:
            raise
        except Exception:
            # Unknown tensor-like objects are handled by the structural cases
            # below; they are not silently accepted as scalar loss values.
            pass
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{field}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite(item, f"{field}[{index}]")


def validate_finite_loss(loss: object) -> float | object:
    """Fail closed on NaN/Inf loss values while preserving the input shape."""

    _finite(loss, "loss")
    return loss


def _checkpoint_path(run_dir: Path, name: str) -> Path:
    candidates = (run_dir / "weights" / name, run_dir / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def verify_checkpoint_artifacts(
    run_dir: Path,
    *,
    expected: Mapping[str, object] | None = None,
) -> dict[str, dict[str, str]]:
    """Verify non-empty fixed-budget primary and diagnostic checkpoints."""

    expected = expected or {}
    primary = _checkpoint_path(run_dir, PRIMARY_CHECKPOINT)
    diagnostic = _checkpoint_path(run_dir, DIAGNOSTIC_CHECKPOINT)
    primary_spec = expected.get("primary_checkpoint")
    diagnostic_spec = expected.get("diagnostic_checkpoint")
    primary_expected = expected.get("primary_sha256", expected.get("last_sha256"))
    diagnostic_expected = expected.get("diagnostic_sha256", expected.get("best_sha256"))
    if primary_expected is None and isinstance(primary_spec, Mapping):
        primary_expected = primary_spec.get("sha256", primary_spec.get("hash"))
    if diagnostic_expected is None and isinstance(diagnostic_spec, Mapping):
        diagnostic_expected = diagnostic_spec.get("sha256", diagnostic_spec.get("hash"))
    if primary_expected is None:
        if not primary.is_file() or primary.stat().st_size <= 0:
            raise ValueError(f"primary checkpoint file is missing or empty: {primary}")
        primary_hash = file_sha256(primary)
    else:
        primary_hash = _require_file_hash(primary, primary_expected, "primary checkpoint")
    if diagnostic_expected is None:
        if not diagnostic.is_file() or diagnostic.stat().st_size <= 0:
            raise ValueError(f"diagnostic checkpoint file is missing or empty: {diagnostic}")
        diagnostic_hash = file_sha256(diagnostic)
    else:
        diagnostic_hash = _require_file_hash(diagnostic, diagnostic_expected, "diagnostic checkpoint")
    return {
        "primary_checkpoint": {"path": PRIMARY_CHECKPOINT, "sha256": primary_hash, "role": "primary"},
        "diagnostic_checkpoint": {"path": DIAGNOSTIC_CHECKPOINT, "sha256": diagnostic_hash, "role": "diagnostic"},
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_json(path, payload)


@dataclass
class EpochDrawJournal:
    """Exactly-once epoch draw journal with crash-safe atomic writes."""

    path: Path
    records: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path) -> "EpochDrawJournal":
        if not path.is_file():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("records", []) if isinstance(payload, Mapping) else []
        if not isinstance(raw, list):
            raise ValueError("epoch draw journal records must be a list")
        return cls(path=path, records=[dict(item) for item in raw])

    def _write(self) -> None:
        _atomic_json(self.path, {"schema_version": 1, "records": self.records})

    def resume(self, epoch: int, draw_key: str) -> bool:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not isinstance(draw_key, str) or not draw_key:
            raise ValueError("draw_key must be non-empty text")
        committed = [
            item for item in self.records
            if item.get("epoch") == epoch and item.get("draw_key") == draw_key and item.get("state") == "committed"
        ]
        if committed:
            return False
        # Drop a crash-left inflight record before committing the same draw.
        self.records = [
            item for item in self.records
            if not (item.get("epoch") == epoch and item.get("draw_key") == draw_key)
        ]
        self.records.append({"epoch": epoch, "draw_key": draw_key, "state": "committed"})
        self._write()
        return True

    def append(self, epoch: int, draw_key: str) -> bool:
        return self.resume(epoch, draw_key)


def resume_epoch_draw_journal(path: Path, epoch: int, draw_key: str) -> bool:
    return EpochDrawJournal.open(Path(path)).resume(epoch, draw_key)


def evaluate_primary_last(
    run_dir: Path,
    *,
    checkpoint_hash: str,
    evaluator: Callable[[Path], object],
) -> Path:
    """Evaluate only ``last.pt`` and publish the primary metric filename."""

    last = _checkpoint_path(run_dir, PRIMARY_CHECKPOINT)
    actual = _require_file_hash(last, checkpoint_hash, "primary checkpoint")
    if actual != checkpoint_hash.lower():
        raise ValueError("primary checkpoint hash mismatch")
    metrics = evaluator(last)
    if isinstance(metrics, Mapping):
        payload: object = {
            **dict(metrics),
            "primary_checkpoint": PRIMARY_CHECKPOINT,
            "primary_checkpoint_sha256": actual,
        }
    else:
        payload = metrics
    output = run_dir / "metrics_ap40_primary_last.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = mkstemp(prefix=".metrics.", suffix=".tmp", dir=output.parent)
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


@dataclass
class FactorRepairRun:
    config: object
    condition: str
    run_dir: Path
    store: RunStore
    lock: ProcessLock
    identity: dict[str, object]
    git_provenance: dict[str, object]
    initialization_checkpoint: Path
    initialization_checkpoint_sha256: str
    fit_ids: tuple[str, ...]
    journal: EpochDrawJournal

    @property
    def status_path(self) -> Path:
        return self.store.status_path

    def record_epoch_draw(self, epoch: int, draw_key: str) -> bool:
        return self.journal.append(epoch, draw_key)

    def release(self) -> None:
        self.lock.release()


def build_factor_repair_run(
    config: object,
    loader_ids: Sequence[str] | None = None,
    *,
    condition: str | None = None,
    repository_root: Path | None = None,
    run_dir: Path | None = None,
    git_provenance: Mapping[str, object] | None = None,
) -> FactorRepairRun:
    """Validate immutable scientific identity and create a prepared run."""

    condition_name = _condition_name(config, condition)
    fit_ids = validate_fit_loader_ids(config, loader_ids)
    identity = _identity_mapping(config)
    root = (repository_root or Path.cwd()).resolve()
    git = require_clean_commit(root, git_provenance)
    initialization_path, initialization_hash = validate_initialization_checkpoint(config)
    if run_dir is None:
        output_root = _nested(config, "paths", "output_root")
        if output_root is None:
            output_root = root / "runs" / "factor-repair"
        run_dir = Path(output_root) / condition_name
    run_dir = Path(run_dir).expanduser().resolve()
    lock = ProcessLock(run_dir.parent / f".{run_dir.name}.factor_repair.lock")
    lock.acquire()
    try:
        store = RunStore.create(run_dir)
        journal = EpochDrawJournal.open(run_dir / "epoch_draw_journal.json")
        payload = {
            "condition": condition_name,
            "identity": identity,
            "git": git,
            "initialization_checkpoint": {
                "path": initialization_path.as_posix(),
                "sha256": initialization_hash,
            },
            "fit_ids": list(fit_ids),
        }
        _atomic_json(run_dir / "provenance.json", payload)
        return FactorRepairRun(
            config=config,
            condition=condition_name,
            run_dir=run_dir,
            store=store,
            lock=lock,
            identity=identity,
            git_provenance=git,
            initialization_checkpoint=initialization_path,
            initialization_checkpoint_sha256=initialization_hash,
            fit_ids=fit_ids,
            journal=journal,
        )
    except Exception:
        lock.release()
        raise


def run_registered_condition(
    run: FactorRepairRun,
    *,
    trainer_factory: Callable[..., object] | None = None,
    evaluator: Callable[[Path], object] | None = None,
    checkpoint_hashes: Mapping[str, object] | None = None,
) -> FactorRepairRun:
    """Run a registered condition when a caller supplies the GPU services."""

    run.store.transition("running")
    try:
        if trainer_factory is None:
            from ifdr_yolo.experiments.ifdr_trainer import IFDRDetectionTrainer

            trainer_factory = IFDRDetectionTrainer
        trainer_kwargs = {
            "config": run.config,
            "condition": run.condition,
            "run_dir": run.run_dir,
            "draw_callback": run.record_epoch_draw,
            "draw_journal": run.journal,
        }
        try:
            trainer = trainer_factory(**trainer_kwargs)
        except TypeError:
            try:
                trainer = trainer_factory(condition=run.condition, run_dir=run.run_dir)
            except TypeError:
                try:
                    trainer = trainer_factory(overrides={"save_dir": str(run.run_dir)})
                except TypeError:
                    trainer = trainer_factory()
        train = getattr(trainer, "train", None)
        if not callable(train):
            raise TypeError("trainer must expose train()")
        if evaluator is None:
            evaluator = getattr(trainer, "evaluate_primary_last", None)
            if evaluator is None:
                evaluator = getattr(trainer, "evaluate", None)
            if evaluator is None:
                evaluator = getattr(trainer, "final_eval", None)
            if evaluator is None:
                raise ValueError("formal factor-repair evaluator is required")
        result = train()
        if isinstance(result, Mapping):
            validate_finite_loss(result.get("losses", result))
        elif result is not None:
            validate_finite_loss(result)
        trainer_loss = getattr(trainer, "loss", None)
        if trainer_loss is not None:
            validate_finite_loss(trainer_loss)
        trainer_losses = getattr(trainer, "losses", None)
        if trainer_losses is not None:
            validate_finite_loss(trainer_losses)
        run.store.transition("trained")
        roles = verify_checkpoint_artifacts(run.run_dir, expected=checkpoint_hashes)
        _atomic_json(run.run_dir / "checkpoint_roles.json", roles)
        run.store.transition("evaluating")
        def _evaluate(path: Path) -> object:
            # The formal evaluator is a path-bound API.  Do not retry a
            # TypeError with a no-argument call: an evaluator can raise
            # TypeError internally, and retrying would hide that failure and
            # potentially evaluate an unverified checkpoint.
            return evaluator(path)  # type: ignore[misc]

        evaluate_primary_last(
            run.run_dir,
            checkpoint_hash=roles["primary_checkpoint"]["sha256"],
            evaluator=_evaluate,
        )
        run.store.transition("complete")
        return run
    except Exception as error:
        if run.store.state not in {"failed", "complete"}:
            run.store.fail(stage="training", error=error)
        raise
    finally:
        run.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a registered factor-repair condition.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=REGISTERED_CONDITIONS, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    from ifdr_yolo.experiments.config import load_factor_repair_config

    config_path = args.config if args.config.is_absolute() else repository_root / args.config
    config = load_factor_repair_config(config_path.resolve(), repository_root=repository_root)
    run = build_factor_repair_run(config, condition=args.condition, repository_root=repository_root)
    run_registered_condition(run)
    print(f"FACTOR REPAIR {args.condition} COMPLETE")
    print(f"run_dir={run.run_dir}")
    return 0


__all__ = [
    "REGISTERED_CONDITIONS",
    "CALIBRATION_CONDITIONS",
    "FactorRepairRun",
    "EpochDrawJournal",
    "ProcessLock",
    "build_factor_repair_run",
    "build_parser",
    "collect_git_provenance",
    "evaluate_primary_last",
    "file_sha256",
    "require_clean_commit",
    "resume_epoch_draw_journal",
    "run_registered_condition",
    "validate_fit_loader_ids",
    "validate_finite_loss",
    "validate_initialization_checkpoint",
    "verify_checkpoint_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
