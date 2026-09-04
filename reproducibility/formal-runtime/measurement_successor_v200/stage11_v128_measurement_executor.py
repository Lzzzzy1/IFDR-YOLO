"""Fixed helpers for the Stage11 v128 seed0 measurement executor."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Mapping, Sequence
import uuid

from stage11_v124_measurement_contract import (
    canonical_sha256,
    cost_measurement_contract,
    validate_cost_measurement,
)


ROLE_ORDER: tuple[str, ...] = ("P3P5_CONTROL", "PLAIN_P2")
OUTPUT_ORDER: tuple[str, ...] = (
    "slice-measurement.json",
    "cost-measurement.json",
    "runner-identity.json",
    "execution-receipt.json",
    "publication-receipt.json",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def build_cost_measurement(
    identity: Mapping[str, object],
    parameters: int,
    flops: float,
    per_pass_latency_ms: Sequence[Sequence[float]],
    peak_vram_bytes: int,
    training_runtime_seconds: float,
) -> dict[str, object]:
    """Build one cost measurement under the frozen v124 sampling contract."""

    if len(per_pass_latency_ms) != 5 or any(
        len(values) != 371 for values in per_pass_latency_ms
    ):
        raise ValueError("latency samples must contain five passes of 371 images")
    samples = [
        _positive_finite(value, "latency sample")
        for values in per_pass_latency_ms
        for value in values
    ]
    parameter_count = _positive_finite(parameters, "parameters")
    flop_count = _positive_finite(flops, "flops")
    peak_memory = _positive_finite(peak_vram_bytes, "peak VRAM")
    training_runtime = _positive_finite(
        training_runtime_seconds,
        "training runtime",
    )
    total_seconds = sum(samples) / 1000.0
    sorted_samples = sorted(samples)
    p95_index = math.ceil(0.95 * len(sorted_samples)) - 1
    measurement: dict[str, object] = {
        "schema": "stage11-v124-cost-measurement-v1",
        "state": "PASS",
        "measurement_contract_sha256": canonical_sha256(
            cost_measurement_contract()
        ),
        "identity": dict(identity),
        "metrics": {
            "parameters": int(parameter_count),
            "flops": flop_count,
            "median_latency_ms": statistics.median(samples),
            "fps": len(samples) / total_seconds,
            "peak_vram_bytes": int(peak_memory),
            "training_runtime_seconds": training_runtime,
        },
        "diagnostics": {
            "warmup_images": 50,
            "measured_passes": 5,
            "latency_sample_count": len(samples),
            "per_pass_median_latency_ms": [
                statistics.median(values) for values in per_pass_latency_ms
            ],
            "p95_latency_ms": sorted_samples[p95_index],
            "total_synchronized_seconds": total_seconds,
        },
    }
    return validate_cost_measurement(measurement)


def format_yolo_detections(
    detections: Sequence[tuple[int, float, float, float, float, float]],
) -> bytes:
    """Render class, normalized xywh, and confidence as deterministic text."""

    lines: list[str] = []
    for class_id, confidence, x_center, y_center, width, height in detections:
        if isinstance(class_id, bool) or class_id not in (0, 1, 2):
            raise ValueError("prediction class ID differs")
        values = (x_center, y_center, width, height, confidence)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("prediction value must be finite")
        lines.append(
            " ".join(
                [str(class_id), *(format(float(value), ".12g") for value in values)]
            )
        )
    return ("\n".join(lines) + ("\n" if lines else "")).encode("ascii")


def build_role_documents(
    role: str,
    plan_sha256: str,
    slice_measurement: Mapping[str, object],
    cost_measurement: Mapping[str, object],
    runner_identity: Mapping[str, object],
) -> dict[str, bytes]:
    """Build the exact five-file role closure with publication last."""

    if role not in ROLE_ORDER:
        raise ValueError("measurement role is not registered")
    if len(plan_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in plan_sha256
    ):
        raise ValueError("plan SHA must be lowercase SHA256")
    documents: dict[str, bytes] = {
        "slice-measurement.json": _canonical_bytes(dict(slice_measurement)),
        "cost-measurement.json": _canonical_bytes(dict(cost_measurement)),
        "runner-identity.json": _canonical_bytes(dict(runner_identity)),
    }
    execution_payload: dict[str, object] = {
        "schema": "stage11-v128-execution-receipt-v1",
        "state": "PASS",
        "role": role,
        "plan_sha256": plan_sha256,
        "output_sha256": {
            name: _sha256_bytes(content) for name, content in documents.items()
        },
        "authorization": {
            "measurement_execution": True,
            "training": False,
            "five_seed": False,
        },
    }
    documents["execution-receipt.json"] = _canonical_bytes(execution_payload)
    publication_payload: dict[str, object] = {
        "schema": "stage11-v128-publication-receipt-v1",
        "state": "PASS",
        "role": role,
        "plan_sha256": plan_sha256,
        "published_output_sha256": {
            name: _sha256_bytes(content) for name, content in documents.items()
        },
        "paired_storage": {
            "sides": ["primary", "mirror"],
            "byte_identical": True,
            "independent_roots_required": True,
        },
    }
    documents["publication-receipt.json"] = _canonical_bytes(publication_payload)
    if tuple(documents) != OUTPUT_ORDER:
        raise RuntimeError("role output order differs")
    return documents


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_create_only(path: Path, content: bytes) -> dict[str, object]:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        state = os.lstat(path)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("published output is not a regular file")
        if _sha256_bytes(path.read_bytes()) != _sha256_bytes(content):
            raise RuntimeError("published output SHA differs")
        _fsync_directory(path.parent)
        return {
            "path": path.as_posix(),
            "device": state.st_dev,
            "inode": state.st_ino,
            "size": state.st_size,
            "sha256": _sha256_bytes(content),
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def _remove_owned_file(path: Path, identity: Mapping[str, object]) -> None:
    if not path.exists():
        return
    state = os.lstat(path)
    if (
        state.st_dev != identity["device"]
        or state.st_ino != identity["inode"]
        or state.st_size != identity["size"]
        or _sha256_bytes(path.read_bytes()) != identity["sha256"]
    ):
        raise RuntimeError(f"NO_GO output cleanup identity differs: {path}")
    path.unlink()
    _fsync_directory(path.parent)


def publish_role_documents(
    primary_root: Path,
    mirror_root: Path,
    role: str,
    documents: Mapping[str, bytes],
) -> dict[str, object]:
    """Create the exact role closure on two independent existing roots."""

    if role not in ROLE_ORDER:
        raise ValueError("measurement role is not registered")
    if tuple(documents) != OUTPUT_ORDER:
        raise ValueError("role output closure differs")
    primary = primary_root.resolve(strict=True)
    mirror = mirror_root.resolve(strict=True)
    if primary == mirror or primary in mirror.parents or mirror in primary.parents:
        raise ValueError("paired roots must be independent")
    for root in (primary, mirror):
        if root.is_symlink() or not root.is_dir():
            raise ValueError("paired root must be a regular directory")
    role_directories = (primary / role, mirror / role)
    if any(path.exists() for path in role_directories):
        raise FileExistsError("role output must be fresh")

    created_directories: list[Path] = []
    created_files: list[tuple[Path, dict[str, object]]] = []
    try:
        for directory in role_directories:
            directory.mkdir(mode=0o700)
            created_directories.append(directory)
            _fsync_directory(directory.parent)
        for name in OUTPUT_ORDER:
            content = documents[name]
            if not isinstance(content, bytes):
                raise TypeError("role output must contain bytes")
            for directory in role_directories:
                path = directory / name
                identity = _write_create_only(path, content)
                created_files.append((path, identity))
        for name in OUTPUT_ORDER:
            primary_content = (role_directories[0] / name).read_bytes()
            mirror_content = (role_directories[1] / name).read_bytes()
            if primary_content != mirror_content or primary_content != documents[name]:
                raise RuntimeError("paired output readback differs")
    except Exception:
        for path, identity in reversed(created_files):
            _remove_owned_file(path, identity)
        for directory in reversed(created_directories):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
                _fsync_directory(directory.parent)
        raise

    return {
        "schema": "stage11-v128-role-publication-result-v1",
        "state": "PASS",
        "role": role,
        "output_sha256": {
            name: _sha256_bytes(content) for name, content in documents.items()
        },
    }


def publish_paired_file(
    source_path: Path,
    primary_path: Path,
    mirror_path: Path,
) -> dict[str, object]:
    """Publish one registered evidence file to two existing directories."""

    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("paired publication source must be a regular file")
    if primary_path == mirror_path:
        raise ValueError("paired publication targets must differ")
    for target in (primary_path, mirror_path):
        if target.exists():
            raise FileExistsError("paired publication target must be fresh")
        if target.parent.is_symlink() or not target.parent.is_dir():
            raise ValueError("paired publication parent must be a real directory")
    content = source_path.read_bytes()
    created: list[tuple[Path, dict[str, object]]] = []
    try:
        for target in (primary_path, mirror_path):
            created.append((target, _write_create_only(target, content)))
        if primary_path.read_bytes() != mirror_path.read_bytes():
            raise RuntimeError("paired publication readback differs")
    except Exception:
        for target, identity in reversed(created):
            _remove_owned_file(target, identity)
        raise
    return {
        "schema": "stage11-v128-paired-file-publication-v1",
        "state": "PASS",
        "sha256": _sha256_bytes(content),
        "size": len(content),
        "primary_path": primary_path.as_posix(),
        "mirror_path": mirror_path.as_posix(),
    }


__all__ = [
    "build_cost_measurement",
    "build_role_documents",
    "format_yolo_detections",
    "publish_paired_file",
    "publish_role_documents",
]
