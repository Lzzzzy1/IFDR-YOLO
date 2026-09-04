from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

from ifdr_yolo.data.splits import load_ids
from ifdr_yolo.eval.evaluate import (
    evaluate_prediction_directory,
    write_evaluation_json,
)
from ifdr_yolo.experiments.config import BaselineConfig, InitializationConfig
from ifdr_yolo.experiments.provenance import (
    collect_git_provenance,
    verify_dataset,
    verify_file_sha256,
)
from ifdr_yolo.experiments.run_store import (
    RunStore,
    atomic_write_json,
    build_run_id,
)
from ifdr_yolo.experiments.smoke_data import build_smoke_view
from ifdr_yolo.experiments.ultralytics_runtime import (
    EXPECTED_ULTRALYTICS_VERSION,
    PreparedModel,
    UltralyticsAdapter,
    bootstrap_ultralytics_config,
    validate_runtime,
)


Mode = Literal["dry-run", "smoke", "full"]


class RuntimeAdapter(Protocol):
    def runtime_info(self) -> dict[str, object]: ...

    def prepare_model(
        self,
        *,
        model_path: Path,
        model_sha256: str,
        initialization: InitializationConfig | None,
        seed: int,
        deterministic: bool,
    ) -> PreparedModel: ...

    def train(
        self,
        *,
        prepared_model: PreparedModel,
        data_path: Path,
        run_dir: Path,
        args: Mapping[str, object],
    ) -> Path: ...

    def predict(
        self,
        *,
        weights: Path,
        image_paths: tuple[Path, ...],
        output_dir: Path,
        args: Mapping[str, object],
    ) -> Path: ...


@dataclass(frozen=True)
class BaselineServices:
    verify_dataset: Callable[..., dict[str, object]]
    collect_git: Callable[[Path], dict[str, object]]
    verify_file_sha256: Callable[..., str]
    evaluate: Callable[..., dict[str, object]]
    now: Callable[[], datetime]


@dataclass(frozen=True)
class BaselineResult:
    mode: str
    run_dir: Path | None
    metrics_path: Path | None


@dataclass(frozen=True)
class _Preflight:
    dataset: dict[str, object]
    git: dict[str, object]
    environment: dict[str, object]
    device: str


def _default_services() -> BaselineServices:
    return BaselineServices(
        verify_dataset=verify_dataset,
        collect_git=collect_git_provenance,
        verify_file_sha256=verify_file_sha256,
        evaluate=evaluate_prediction_directory,
        now=lambda: datetime.now(timezone.utc),
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _preflight(
    config: BaselineConfig,
    *,
    mode: Mode,
    adapter: RuntimeAdapter,
    repository_root: Path,
    services: BaselineServices,
    device_override: str | None,
) -> _Preflight:
    bootstrap_ultralytics_config(repository_root)
    _require_file(config.paths.data, "data config")
    _require_file(config.paths.model, "model weight")
    services.verify_file_sha256(
        config.paths.model,
        config.paths.model_sha256,
        label="model",
    )
    if config.initialization is not None:
        _require_file(
            config.initialization.pretrained,
            "pretrained model",
        )
        services.verify_file_sha256(
            config.initialization.pretrained,
            config.initialization.pretrained_sha256,
            label="pretrained model",
        )
    dataset = services.verify_dataset(
        config,
        verify_all_hashes=True,
    )
    git = services.collect_git(repository_root)
    if mode == "full" and not bool(git.get("tracked_clean")):
        raise RuntimeError(
            "formal full mode requires no tracked Git changes: "
            f"{git.get('tracked_changes')}"
        )
    environment = adapter.runtime_info()
    actual_version_value = environment.get("ultralytics")
    actual_version = (
        actual_version_value if isinstance(actual_version_value, str) else None
    )
    cuda_available = bool(environment.get("cuda_available"))
    device_count_value = environment.get("cuda_device_count", 0)
    if isinstance(device_count_value, bool) or not isinstance(
        device_count_value,
        int,
    ):
        raise RuntimeError("runtime cuda_device_count must be an integer")
    device = device_override or config.training.device
    validate_runtime(
        actual_ultralytics=actual_version,
        expected_ultralytics=EXPECTED_ULTRALYTICS_VERSION,
        cuda_available=cuda_available,
        device_count=device_count_value,
        requested_device=device,
        require_cuda=mode == "full",
    )
    return _Preflight(
        dataset=dataset,
        git=git,
        environment=environment,
        device=device,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_run_metadata(
    *,
    store: RunStore,
    config: BaselineConfig,
    preflight: _Preflight,
    mode: Mode,
    data_path: Path,
    training_args: Mapping[str, object],
    prediction_args: Mapping[str, object],
) -> None:
    resolved = _jsonable(asdict(config))
    assert isinstance(resolved, dict)
    resolved["mode"] = mode
    paths = resolved["paths"]
    assert isinstance(paths, dict)
    paths["data"] = str(data_path)
    resolved["training"] = _jsonable(dict(training_args))
    resolved["prediction"] = _jsonable(dict(prediction_args))
    (store.root / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    if config.source_path is not None:
        input_text = config.source_path.read_text(encoding="utf-8")
    else:
        input_text = yaml.safe_dump(resolved, sort_keys=False)
    (store.root / "config.input.yaml").write_text(
        input_text,
        encoding="utf-8",
        newline="\n",
    )
    atomic_write_json(store.root / "data_manifest.json", preflight.dataset)
    atomic_write_json(store.root / "environment.json", preflight.environment)
    atomic_write_json(store.root / "git_status.json", preflight.git)
    commit = preflight.git.get("commit")
    if not isinstance(commit, str):
        raise RuntimeError("Git provenance does not contain a commit SHA")
    (store.root / "git_commit.txt").write_text(
        commit + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _training_args(
    config: BaselineConfig,
    *,
    mode: Mode,
    device: str,
) -> dict[str, object]:
    result = asdict(config.training)
    result["device"] = device
    result["seed"] = config.experiment.seed
    result["val"] = True
    result["save"] = True
    result["plots"] = True
    result["pretrained"] = False
    result["save_period"] = 10 if mode == "full" else -1
    if mode == "smoke":
        result.update(
            {
                "epochs": 1,
                "imgsz": 320,
                "batch": 2,
                "workers": 0,
                "amp": False,
            }
        )
    return result


def _prediction_args(
    config: BaselineConfig,
    *,
    mode: Mode,
    device: str,
) -> dict[str, object]:
    result = asdict(config.prediction)
    if result.get("half") is False:
        result.pop("half")
    result["device"] = device
    result["imgsz"] = 320 if mode == "smoke" else config.training.imgsz
    result["augment"] = False
    result["verbose"] = False
    return result


def ensure_prediction_files(
    labels_dir: Path,
    image_ids: tuple[str, ...],
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    expected = set(image_ids)
    actual = {path.stem for path in labels_dir.glob("*.txt")}
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(f"unexpected prediction IDs: {extra[:5]}")
    for image_id in image_ids:
        path = labels_dir / f"{image_id}.txt"
        if not path.exists():
            path.write_text("", encoding="utf-8", newline="\n")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{path}:{line_number}: prediction must contain 6 fields"
                )
            try:
                class_id = int(fields[0])
                values = tuple(float(value) for value in fields[1:])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid prediction number"
                ) from error
            if class_id not in (0, 1, 2):
                raise ValueError(
                    f"{path}:{line_number}: unknown class ID {class_id}"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{path}:{line_number}: prediction values must be finite"
                )


def _write_evaluation_split(path: Path, image_ids: tuple[str, ...]) -> None:
    path.write_text(
        "".join(f"{image_id}\n" for image_id in image_ids),
        encoding="utf-8",
        newline="\n",
    )


def run_baseline(
    config: BaselineConfig,
    *,
    mode: Mode,
    repository_root: Path,
    adapter: RuntimeAdapter | None = None,
    device_override: str | None = None,
    services: BaselineServices | None = None,
) -> BaselineResult:
    if mode not in ("dry-run", "smoke", "full"):
        raise ValueError(f"unknown baseline mode: {mode}")
    repository_root = repository_root.resolve()
    runtime = adapter or UltralyticsAdapter()
    dependencies = services or _default_services()
    preflight = _preflight(
        config,
        mode=mode,
        adapter=runtime,
        repository_root=repository_root,
        services=dependencies,
        device_override=device_override,
    )
    if mode == "dry-run":
        runtime.prepare_model(
            model_path=config.paths.model,
            model_sha256=config.paths.model_sha256,
            initialization=config.initialization,
            seed=config.experiment.seed,
            deterministic=config.training.deterministic,
        )
        return BaselineResult(
            mode="dry-run",
            run_dir=None,
            metrics_path=None,
        )

    commit = preflight.git.get("commit")
    if not isinstance(commit, str):
        raise RuntimeError("Git provenance does not contain a commit SHA")
    run_id = build_run_id(
        timestamp=dependencies.now(),
        dataset=config.experiment.dataset,
        model=config.experiment.model,
        variant=config.experiment.variant,
        seed=config.experiment.seed,
        git_sha=commit,
    )
    store = RunStore.create(repository_root / "runs" / run_id)
    stage = "initialization"
    try:
        prepared = runtime.prepare_model(
            model_path=config.paths.model,
            model_sha256=config.paths.model_sha256,
            initialization=config.initialization,
            seed=config.experiment.seed,
            deterministic=config.training.deterministic,
        )
        if prepared.initialization is not None:
            atomic_write_json(
                store.root / "initialization.json",
                prepared.initialization,
            )

        stage = "preparation"
        train_ids = load_ids(config.paths.train_ids)
        val_ids = load_ids(config.paths.val_ids)
        data_path = config.paths.data
        evaluation_split = config.paths.val_ids
        selected_val_ids = val_ids
        prediction_image_dir = config.paths.generated_data / "images" / "val"
        if mode == "smoke":
            smoke_view = build_smoke_view(
                output_dir=repository_root / "tmp" / "smoke-kitti",
                generated_dir=config.paths.generated_data,
                train_ids=train_ids,
                val_ids=val_ids,
                train_source_sha256=str(
                    preflight.dataset["train_file_sha256"]
                ),
                val_source_sha256=str(
                    preflight.dataset["val_file_sha256"]
                ),
            )
            data_path = smoke_view.data_yaml
            selected_val_ids = smoke_view.val_ids
            prediction_image_dir = smoke_view.root / "images" / "val"
            evaluation_split = store.root / "evaluation_split.txt"
            _write_evaluation_split(evaluation_split, selected_val_ids)
        training_args = _training_args(
            config,
            mode=mode,
            device=preflight.device,
        )
        prediction_args = _prediction_args(
            config,
            mode=mode,
            device=preflight.device,
        )
        _write_run_metadata(
            store=store,
            config=config,
            preflight=preflight,
            mode=mode,
            data_path=data_path,
            training_args=training_args,
            prediction_args=prediction_args,
        )

        stage = "training"
        store.transition("running")
        best = runtime.train(
            prepared_model=prepared,
            data_path=data_path,
            run_dir=store.root,
            args=training_args,
        )
        store.transition("trained")

        stage = "evaluation"
        store.transition("evaluating")
        image_paths = tuple(
            prediction_image_dir / f"{image_id}.png"
            for image_id in selected_val_ids
        )
        labels_dir = runtime.predict(
            weights=best,
            image_paths=image_paths,
            output_dir=store.root / "predictions",
            args=prediction_args,
        )
        ensure_prediction_files(labels_dir, selected_val_ids)
        metrics = dependencies.evaluate(
            prediction_dir=labels_dir,
            label_dir=config.paths.raw_labels,
            image_dir=config.paths.raw_images,
            split_path=evaluation_split,
        )
        metrics_path = store.root / "metrics_ap40.json"
        write_evaluation_json(metrics_path, metrics)
        store.transition("complete")
        return BaselineResult(
            mode=mode,
            run_dir=store.root,
            metrics_path=metrics_path,
        )
    except BaseException as error:
        store.fail(stage=stage, error=error)
        raise
