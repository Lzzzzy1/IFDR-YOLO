from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ifdr_yolo.data.splits import load_ids
from ifdr_yolo.eval.evaluate import (
    evaluate_prediction_directory,
    write_evaluation_json,
)
from ifdr_yolo.experiments.baseline import ensure_prediction_files
from ifdr_yolo.experiments.config import BaselineConfig
from ifdr_yolo.experiments.provenance import collect_git_provenance
from ifdr_yolo.experiments.recovery import _completed_epochs
from ifdr_yolo.experiments.run_store import atomic_write_json


@dataclass(frozen=True)
class BaselineRecoveryServices:
    resume_training: Callable[[Path, Path, str, int], None]
    prediction_adapter: Any
    evaluate: Callable[..., dict[str, object]]
    collect_git: Callable[[Path], dict[str, object]]
    now: Callable[[], datetime]


@dataclass(frozen=True)
class BaselineRecoveryResult:
    run_dir: Path
    metrics_path: Path
    completed_epochs: int


def _resume_training(
    checkpoint: Path,
    run_dir: Path,
    device: str,
    workers: int,
) -> None:
    from ultralytics import YOLO

    del run_dir
    YOLO(str(checkpoint)).train(
        resume=True,
        device=device,
        workers=workers,
    )


def _default_services() -> BaselineRecoveryServices:
    from ifdr_yolo.experiments.ultralytics_runtime import UltralyticsAdapter

    return BaselineRecoveryServices(
        resume_training=_resume_training,
        prediction_adapter=UltralyticsAdapter(),
        evaluate=evaluate_prediction_directory,
        collect_git=collect_git_provenance,
        now=lambda: datetime.now(timezone.utc),
    )


def _prediction_args(
    config: BaselineConfig,
    device: str,
) -> dict[str, object]:
    result = asdict(config.prediction)
    if result.get("half") is False:
        result.pop("half")
    result.update(
        {
            "device": device,
            "imgsz": config.training.imgsz,
            "augment": False,
            "verbose": False,
        }
    )
    return result


def recover_baseline_run(
    config: BaselineConfig,
    *,
    run_dir: Path,
    repository_root: Path,
    device: str,
    services: BaselineRecoveryServices | None = None,
) -> BaselineRecoveryResult:
    if not isinstance(config, BaselineConfig):
        raise ValueError("config must be a BaselineConfig")
    run_dir = run_dir.resolve()
    repository_root = repository_root.resolve()
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"run status does not exist: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "failed" or status.get("stage") != "training":
        raise ValueError("only a run failed during training can be recovered")

    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    for checkpoint in (last, best):
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise FileNotFoundError(
                f"recovery checkpoint does not exist: {checkpoint}"
            )
    completed_before = _completed_epochs(run_dir / "results.csv")
    if completed_before >= config.training.epochs:
        raise ValueError("training is already complete; recovery is unnecessary")

    dependencies = services or _default_services()
    provenance = dependencies.collect_git(repository_root)
    if not bool(provenance.get("tracked_clean")):
        raise RuntimeError("recovery requires a clean tracked repository")
    commit = provenance.get("commit")
    if not isinstance(commit, str):
        raise RuntimeError("recovery Git provenance has no commit")

    before_status = run_dir / "status.before-recovery.json"
    if not before_status.exists():
        before_status.write_bytes(status_path.read_bytes())
    recovery_status = run_dir / "recovery_status.json"
    started_at = dependencies.now().astimezone(timezone.utc).isoformat()
    atomic_write_json(
        recovery_status,
        {
            "state": "resuming",
            "completed_epochs_before": completed_before,
            "checkpoint": str(last),
            "recovery_commit": commit,
            "started_at_utc": started_at,
        },
    )

    try:
        dependencies.resume_training(
            last,
            run_dir,
            device,
            config.training.workers,
        )
        completed_epochs = _completed_epochs(run_dir / "results.csv")
        if completed_epochs != config.training.epochs:
            raise RuntimeError(
                "recovered training did not reach configured epoch count: "
                f"{completed_epochs}/{config.training.epochs}"
            )
        if not best.is_file() or best.stat().st_size <= 0:
            raise FileNotFoundError("recovered run has no best checkpoint")

        val_ids = load_ids(config.paths.val_ids)
        image_dir = config.paths.generated_data / "images" / "val"
        labels_dir = dependencies.prediction_adapter.predict(
            weights=best,
            image_paths=tuple(
                image_dir / f"{image_id}.png" for image_id in val_ids
            ),
            output_dir=run_dir / "predictions",
            args=_prediction_args(config, device),
        )
        ensure_prediction_files(labels_dir, val_ids)
        metrics = dependencies.evaluate(
            prediction_dir=labels_dir,
            label_dir=config.paths.raw_labels,
            image_dir=config.paths.raw_images,
            split_path=config.paths.val_ids,
        )
        metrics_path = run_dir / "metrics_ap40.json"
        write_evaluation_json(metrics_path, metrics)
        finished_at = dependencies.now().astimezone(timezone.utc).isoformat()
        atomic_write_json(
            recovery_status,
            {
                "state": "complete",
                "completed_epochs_before": completed_before,
                "completed_epochs_after": completed_epochs,
                "checkpoint": str(last),
                "recovery_commit": commit,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "metrics_path": str(metrics_path),
            },
        )
        atomic_write_json(
            status_path,
            {
                "state": "complete",
                "recovered": True,
                "recovery_commit": commit,
                "completed_epochs": completed_epochs,
                "metrics_path": str(metrics_path),
                "updated_at_utc": finished_at,
            },
        )
        return BaselineRecoveryResult(
            run_dir=run_dir,
            metrics_path=metrics_path,
            completed_epochs=completed_epochs,
        )
    except BaseException as error:
        atomic_write_json(
            recovery_status,
            {
                "state": "failed",
                "completed_epochs_before": completed_before,
                "checkpoint": str(last),
                "recovery_commit": commit,
                "started_at_utc": started_at,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
