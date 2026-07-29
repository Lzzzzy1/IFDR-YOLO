from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
from typing import Any

from ifdr_yolo.experiments.provenance import collect_environment


EXPECTED_ULTRALYTICS_VERSION = "8.4.98"


def bootstrap_ultralytics_config(repository_root: Path) -> Path:
    config_dir = (repository_root.resolve() / "tmp" / "yolo-config").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    return config_dir


def validate_runtime(
    *,
    actual_ultralytics: str | None,
    expected_ultralytics: str,
    cuda_available: bool,
    device_count: int,
    requested_device: str,
    require_cuda: bool,
) -> None:
    if actual_ultralytics != expected_ultralytics:
        raise RuntimeError(
            "Ultralytics version mismatch: "
            f"expected={expected_ultralytics}, actual={actual_ultralytics}"
        )
    if requested_device.lower() == "cpu":
        if require_cuda:
            raise RuntimeError("CUDA is required for formal baseline training")
        return
    if not cuda_available:
        raise RuntimeError(
            f"CUDA is required for requested device {requested_device}"
        )
    try:
        requested_indexes = tuple(
            int(value.strip()) for value in requested_device.split(",")
        )
    except ValueError as error:
        raise RuntimeError(
            f"invalid CUDA device specification: {requested_device}"
        ) from error
    if not requested_indexes or any(index < 0 for index in requested_indexes):
        raise RuntimeError(f"invalid CUDA device specification: {requested_device}")
    for index in requested_indexes:
        if index >= device_count:
            raise RuntimeError(
                f"CUDA device {index} is unavailable; device_count={device_count}"
            )


def _load_yolo_factory() -> Callable[[str], Any]:
    from ultralytics import YOLO

    return YOLO


class UltralyticsAdapter:
    def __init__(
        self,
        *,
        yolo_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._yolo_factory = yolo_factory

    def _factory(self) -> Callable[[str], Any]:
        if self._yolo_factory is None:
            self._yolo_factory = _load_yolo_factory()
        return self._yolo_factory

    def runtime_info(self) -> dict[str, object]:
        return collect_environment()

    def train(
        self,
        *,
        model_path: Path,
        data_path: Path,
        run_dir: Path,
        args: Mapping[str, object],
    ) -> Path:
        model = self._factory()(str(model_path))
        model.train(
            data=str(data_path),
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            **dict(args),
        )
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"training did not create best weight: {best}")
        return best

    def predict(
        self,
        *,
        weights: Path,
        image_paths: tuple[Path, ...],
        output_dir: Path,
        args: Mapping[str, object],
    ) -> Path:
        if not image_paths:
            raise ValueError("prediction requires at least one image path")
        source_dir = image_paths[0].resolve().parent
        expected_paths = {path.resolve() for path in image_paths}
        if any(path.resolve().parent != source_dir for path in image_paths):
            raise ValueError("prediction image paths must share one directory")
        directory_paths = {path.resolve() for path in source_dir.glob("*.png")}
        if directory_paths != expected_paths:
            raise ValueError(
                "prediction source directory must contain exactly the requested "
                "PNG images"
            )
        model = self._factory()(str(weights))
        model.predict(
            source=str(source_dir),
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            save_txt=True,
            save_conf=True,
            **dict(args),
        )
        labels = output_dir / "labels"
        if not labels.is_dir():
            raise FileNotFoundError(
                f"prediction did not create labels directory: {labels}"
            )
        return labels
