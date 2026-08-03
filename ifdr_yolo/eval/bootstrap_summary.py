from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import fmean, stdev

from ifdr_yolo.data.kitti_types import EVAL_CLASSES
from ifdr_yolo.eval.bootstrap_matrix import VALID_SLICE_NAMES


@dataclass(frozen=True)
class BootstrapTaskIdentity:
    reference: str
    candidate: str
    seed: int
    class_name: str
    slice_name: str


@dataclass(frozen=True)
class SeedBootstrapResult:
    seed: int
    reference_ap40: float
    candidate_ap40: float
    difference_ap40: float
    ci_lower: float
    ci_upper: float
    probability_improvement: float


@dataclass(frozen=True)
class SeedBootstrapSummary:
    seed_count: int
    mean_reference_ap40: float
    mean_candidate_ap40: float
    mean_difference_ap40: float
    sample_std_difference_ap40: float | None
    positive_seed_count: int
    negative_seed_count: int
    positive_ci_seed_count: int
    negative_ci_seed_count: int
    direction_consistency: str
    seed_results: tuple[SeedBootstrapResult, ...]


@dataclass(frozen=True)
class BootstrapGroupSummary:
    reference: str
    candidate: str
    class_name: str
    slice_name: str
    iterations: int
    bootstrap_seed: int
    seed_summary: SeedBootstrapSummary


def parse_task_id(task_id: str) -> BootstrapTaskIdentity:
    parts = task_id.split("__")
    if len(parts) != 4 or not parts[1].startswith("s"):
        raise ValueError(f"invalid bootstrap task ID: {task_id}")
    try:
        reference, candidate = parts[0].split("_vs_", 1)
        seed = int(parts[1][1:])
    except ValueError as error:
        raise ValueError(f"invalid bootstrap task ID: {task_id}") from error
    class_name, slice_name = parts[2], parts[3]
    if (
        not reference
        or not candidate
        or seed < 0
        or class_name not in EVAL_CLASSES
        or slice_name not in VALID_SLICE_NAMES
    ):
        raise ValueError(f"invalid bootstrap task ID: {task_id}")
    return BootstrapTaskIdentity(
        reference=reference,
        candidate=candidate,
        seed=seed,
        class_name=class_name,
        slice_name=slice_name,
    )


def summarize_seed_results(
    results: tuple[SeedBootstrapResult, ...],
    *,
    expected_seeds: tuple[int, ...],
) -> SeedBootstrapSummary:
    ordered = tuple(sorted(results, key=lambda result: result.seed))
    seeds = tuple(result.seed for result in ordered)
    if len(set(seeds)) != len(seeds) or seeds != tuple(sorted(expected_seeds)):
        raise ValueError(
            f"bootstrap results must contain expected seeds {sorted(expected_seeds)}"
        )
    differences = tuple(result.difference_ap40 for result in ordered)
    if all(value > 0.0 for value in differences):
        direction = "positive"
    elif all(value < 0.0 for value in differences):
        direction = "negative"
    elif all(value == 0.0 for value in differences):
        direction = "zero"
    else:
        direction = "mixed"
    return SeedBootstrapSummary(
        seed_count=len(ordered),
        mean_reference_ap40=fmean(
            result.reference_ap40 for result in ordered
        ),
        mean_candidate_ap40=fmean(
            result.candidate_ap40 for result in ordered
        ),
        mean_difference_ap40=fmean(differences),
        sample_std_difference_ap40=(
            stdev(differences) if len(differences) >= 2 else None
        ),
        positive_seed_count=sum(value > 0.0 for value in differences),
        negative_seed_count=sum(value < 0.0 for value in differences),
        positive_ci_seed_count=sum(result.ci_lower > 0.0 for result in ordered),
        negative_ci_seed_count=sum(result.ci_upper < 0.0 for result in ordered),
        direction_consistency=direction,
        seed_results=ordered,
    )


def summarize_bootstrap_directory(
    input_dir: Path,
    *,
    expected_seeds: tuple[int, ...],
) -> tuple[BootstrapGroupSummary, ...]:
    input_dir = input_dir.resolve()
    status_path = input_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("bootstrap matrix status is missing or invalid") from error
    if status.get("state") != "complete":
        raise ValueError("bootstrap matrix is not complete")

    grouped: dict[
        tuple[str, str, str, str],
        list[SeedBootstrapResult],
    ] = {}
    settings: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    result_paths = tuple(
        path for path in sorted(input_dir.glob("*.json")) if path != status_path
    )
    if not result_paths:
        raise ValueError("bootstrap matrix contains no result files")
    for path in result_paths:
        identity = parse_task_id(path.stem)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            comparison = payload["comparison"]
            valid_identity = (
                payload["schema_version"] == 1
                and payload["metric"]
                == "KITTI_2D_CONDITIONAL_AP40_PAIRED_IMAGE_BOOTSTRAP"
                and payload["reference"]["name"] == identity.reference
                and payload["candidate"]["name"] == identity.candidate
                and payload["class_name"] == identity.class_name
                and payload["target_slice"]["name"] == identity.slice_name
            )
            if not valid_identity:
                raise ValueError("result identity does not match its filename")
            result = SeedBootstrapResult(
                seed=identity.seed,
                reference_ap40=float(comparison["reference_ap40"]),
                candidate_ap40=float(comparison["candidate_ap40"]),
                difference_ap40=float(comparison["difference_ap40"]),
                ci_lower=float(comparison["ci_lower"]),
                ci_upper=float(comparison["ci_upper"]),
                probability_improvement=float(
                    comparison["probability_improvement"]
                ),
            )
            current_settings = (
                int(comparison["iterations"]),
                int(comparison["seed"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid bootstrap result: {path}") from error
        key = (
            identity.reference,
            identity.candidate,
            identity.class_name,
            identity.slice_name,
        )
        if key in settings and settings[key] != current_settings:
            raise ValueError("bootstrap settings differ within one comparison")
        settings[key] = current_settings
        grouped.setdefault(key, []).append(result)

    summaries: list[BootstrapGroupSummary] = []
    for key in sorted(grouped):
        iterations, bootstrap_seed = settings[key]
        summaries.append(
            BootstrapGroupSummary(
                reference=key[0],
                candidate=key[1],
                class_name=key[2],
                slice_name=key[3],
                iterations=iterations,
                bootstrap_seed=bootstrap_seed,
                seed_summary=summarize_seed_results(
                    tuple(grouped[key]), expected_seeds=expected_seeds
                ),
            )
        )
    return tuple(summaries)
