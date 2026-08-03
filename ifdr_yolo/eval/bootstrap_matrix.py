from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ifdr_yolo.data.kitti_types import EVAL_CLASSES
from ifdr_yolo.eval.stratified_ap40 import KITTI_RESEARCH_SLICES


VALID_SLICE_NAMES = {
    target_slice.name for target_slice in KITTI_RESEARCH_SLICES
}


@dataclass(frozen=True)
class BootstrapTask:
    reference: str
    candidate: str
    seed: int
    class_name: str
    slice_name: str
    reference_dir: Path
    candidate_dir: Path

    @property
    def task_id(self) -> str:
        return (
            f"{self.reference}_vs_{self.candidate}__s{self.seed}__"
            f"{self.class_name}__{self.slice_name}"
        )


def parse_run_spec(spec: str) -> tuple[str, int, Path]:
    try:
        method_seed, raw_path = spec.split("=", 1)
        method, raw_seed = method_seed.rsplit(":", 1)
        seed = int(raw_seed)
    except (ValueError, TypeError) as error:
        raise ValueError(
            "run must use METHOD:SEED=PREDICTION_DIR"
        ) from error
    if not method or seed < 0 or not raw_path:
        raise ValueError("run contains an empty or invalid field")
    return method, seed, Path(raw_path)


def parse_comparison_spec(spec: str) -> tuple[str, str]:
    try:
        reference, candidate = spec.split("=", 1)
    except (ValueError, TypeError) as error:
        raise ValueError(
            "comparison must use REFERENCE=CANDIDATE"
        ) from error
    if not reference or not candidate or reference == candidate:
        raise ValueError("comparison methods must be distinct and non-empty")
    return reference, candidate


def result_is_complete(
    payload: Any,
    *,
    task: BootstrapTask,
    iterations: int,
    bootstrap_seed: int,
) -> bool:
    if not isinstance(payload, dict):
        return False
    try:
        return (
            payload["schema_version"] == 1
            and payload["metric"]
            == "KITTI_2D_CONDITIONAL_AP40_PAIRED_IMAGE_BOOTSTRAP"
            and payload["reference"]["name"] == task.reference
            and payload["candidate"]["name"] == task.candidate
            and payload["class_name"] == task.class_name
            and payload["target_slice"]["name"] == task.slice_name
            and payload["comparison"]["iterations"] == iterations
            and payload["comparison"]["seed"] == bootstrap_seed
        )
    except (KeyError, TypeError):
        return False


def build_bootstrap_tasks(
    *,
    run_dirs: dict[str, dict[int, Path]],
    comparisons: tuple[tuple[str, str], ...],
    class_names: tuple[str, ...],
    slice_names: tuple[str, ...],
) -> tuple[BootstrapTask, ...]:
    if not comparisons or not class_names or not slice_names:
        raise ValueError("bootstrap matrix dimensions must not be empty")
    if any(class_name not in EVAL_CLASSES for class_name in class_names):
        raise ValueError("bootstrap matrix contains an unknown class")
    if any(name not in VALID_SLICE_NAMES for name in slice_names):
        raise ValueError("bootstrap matrix contains an unknown target slice")

    tasks: list[BootstrapTask] = []
    for reference, candidate in comparisons:
        if reference not in run_dirs or candidate not in run_dirs:
            raise ValueError("bootstrap comparison references an unknown method")
        reference_runs = run_dirs[reference]
        candidate_runs = run_dirs[candidate]
        if set(reference_runs) != set(candidate_runs):
            raise ValueError(
                "bootstrap comparison methods must use the same seeds"
            )
        for seed in sorted(reference_runs):
            for class_name in class_names:
                for slice_name in slice_names:
                    tasks.append(
                        BootstrapTask(
                            reference=reference,
                            candidate=candidate,
                            seed=seed,
                            class_name=class_name,
                            slice_name=slice_name,
                            reference_dir=reference_runs[seed],
                            candidate_dir=candidate_runs[seed],
                        )
                    )
    return tuple(tasks)
