"""Run a deterministic, inference-only clean/target/background mechanism smoke.

The runner deliberately keeps the evidence path small: it reads local KITTI
PNG/YOLO files, performs one ordered 3B forward for one registered factor,
journals each completed image durably and seals a validated summary after all
selected images succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from ifdr_yolo.experiments.ultralytics_runtime import bootstrap_ultralytics_config

# Keep the standalone smoke independent of a user's global Ultralytics config.
bootstrap_ultralytics_config(Path(__file__).resolve().parents[1])

from ifdr_yolo.data.interventions import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
    apply_intervention,
    factor_target_for_spec,
)
from ifdr_yolo.eval.factor_observer import (
    DEFAULT_REQUIRED_NODES,
    LetterboxGeometry,
    letterbox_image,
    map_box_to_feature_roi,
)
from ifdr_yolo.eval.factor_observer_runtime import load_ifdr_checkpoint
from ifdr_yolo.models.ifdr_model import split_three_view_contexts


OBSERVATIONS_FILENAME = "three_view_observations.jsonl"
SUMMARY_FILENAME = "three_view_summary.json"
MANIFEST_FILENAME = "three_view_manifest.json"
MAIN_NODES = (17, 20, 23, 26)
FACTOR_SPECS = {
    "sampling": (0, InterventionKind.SAMPLING),
    "visibility": (1, InterventionKind.VISIBILITY),
}


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite numeric")
    return result


def _box(value: Sequence[float], field: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{field} must contain four coordinates")
    values = tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{field} must be ordered normalized coordinates")
    return values


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _stable_int(seed: int, *parts: object) -> int:
    material = "\x1f".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _parse_yolo_labels(path: Path, *, width: int, height: int) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unable to read label file: {path}") from exc
    parsed: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"malformed YOLO label at {path}:{line_number}")
        try:
            class_id = int(fields[0])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed class id at {path}:{line_number}") from exc
        if class_id < 0:
            raise ValueError(f"class id must be non-negative at {path}:{line_number}")
        try:
            coordinates = tuple(float(fields[index]) for index in range(1, 5))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed label coordinates at {path}:{line_number}") from exc
        center_x, center_y, box_width, box_height = tuple(
            _finite(value, f"label coordinate {index}")
            for index, value in enumerate(coordinates, 1)
        )
        if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0 and 0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            raise ValueError(f"label coordinates out of range at {path}:{line_number}")
        normalized = (
            center_x - box_width / 2.0,
            center_y - box_height / 2.0,
            center_x + box_width / 2.0,
            center_y + box_height / 2.0,
        )
        normalized = _box(normalized, f"label box {path}:{line_number}")
        parsed.append(
            {
                "class_id": class_id,
                "box": normalized,
                "pixel_box": (
                    normalized[0] * width,
                    normalized[1] * height,
                    normalized[2] * width,
                    normalized[3] * height,
                ),
            }
        )
    return tuple(parsed)


def _read_image(path: Path) -> np.ndarray:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read image: {path}") from exc
    if not raw:
        raise ValueError(f"image is empty: {path}")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image is not a readable PNG: {path}")
    return image


def _select_images(
    images: Path,
    labels: Path,
    *,
    max_images: int,
) -> tuple[tuple[tuple[Path, tuple[dict[str, Any], ...]], ...], int]:
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    if not images.is_dir() or not labels.is_dir():
        raise ValueError("images and labels must be directories")
    candidates: list[tuple[float, Path, tuple[dict[str, Any], ...]]] = []
    rejected = 0
    for image_path in sorted(images.glob("*.png"), key=lambda item: item.stem):
        label_path = labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            rejected += 1
            continue
        try:
            image = _read_image(image_path)
            objects = _parse_yolo_labels(label_path, width=image.shape[1], height=image.shape[0])
        except ValueError:
            rejected += 1
            continue
        small = [item for item in objects if (item["box"][3] - item["box"][1]) <= 0.25]
        if not small:
            rejected += 1
            continue
        minimum_height = min(float(item["box"][3] - item["box"][1]) for item in small)
        candidates.append((minimum_height, image_path, tuple(objects)))
    candidates.sort(key=lambda item: (item[0], item[1].stem))
    selected = tuple((path, objects) for _, path, objects in candidates[:max_images])
    if not selected:
        raise ValueError(f"no eligible small-target evidence (rejected={rejected})")
    return selected, rejected


def _choose_background(target: Sequence[float], boxes: Sequence[Sequence[float]], seed: int) -> tuple[float, float, float, float]:
    width = min(max(float(target[2] - target[0]), 0.05), 0.45)
    height = min(max(float(target[3] - target[1]), 0.05), 0.45)
    candidates: list[tuple[float, float, float, float]] = []
    for center_y in np.linspace(height / 2.0, 1.0 - height / 2.0, 11):
        for center_x in np.linspace(width / 2.0, 1.0 - width / 2.0, 11):
            candidates.append((float(center_x - width / 2.0), float(center_y - height / 2.0), float(center_x + width / 2.0), float(center_y + height / 2.0)))
    candidates.sort(key=lambda candidate: _stable_int(seed, candidate))
    for candidate in candidates:
        if all(_iou(candidate, box) == 0.0 for box in boxes):
            return candidate
    raise ValueError("no strict zero-IoU background region")


def _context_tensors(context: object, node: int) -> torch.Tensor:
    factors = context.get("factors") if isinstance(context, Mapping) else getattr(context, "factors", None)
    if not isinstance(factors, torch.Tensor) or factors.ndim != 4 or factors.shape[1] != 2 or not factors.is_floating_point():
        raise ValueError(f"node {node} context factors are incomplete")
    if not torch.isfinite(factors).all():
        raise ValueError(f"node {node} context factors are non-finite")
    return factors


def _pool_delta(
    clean: torch.Tensor,
    view: torch.Tensor,
    *,
    batch_index: int,
    pixel_box: Sequence[float],
    geometry: LetterboxGeometry,
    channel: int,
) -> tuple[float, tuple[int, int, int, int]]:
    _, _, feature_height, feature_width = clean.shape
    roi = map_box_to_feature_roi(pixel_box, geometry, (feature_height, feature_width))
    x1, y1, x2, y2 = roi
    delta = (view[batch_index, channel, y1:y2, x1:x2] - clean[batch_index, channel, y1:y2, x1:x2]).mean()
    value = float(delta.detach().cpu())
    if not math.isfinite(value):
        raise ValueError("factor delta is non-finite")
    return value, roi


def _source_commit() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository_root}",
                "rev-parse",
                "HEAD",
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None:
        commit = result.stdout.strip().lower()
        if len(commit) == 40 and all(character in "0123456789abcdef" for character in commit):
            return commit
    source_commit = repository_root / "SOURCE_COMMIT"
    try:
        fallback = source_commit.read_text(encoding="utf-8").strip().lower()
    except OSError:
        fallback = ""
    if len(fallback) == 40 and all(character in "0123456789abcdef" for character in fallback):
        return fallback
    raise ValueError("source commit provenance is unavailable or invalid")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid JSON provenance file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"provenance file must contain an object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return
    except OSError as exc:
        raise ValueError(f"unable to persist provenance file: {path}") from exc


def _read_observation_rows(
    path: Path,
    *,
    selected_ids: Sequence[str],
    checkpoint_sha256: str,
    source_commit: str,
    seed: int,
    device: str,
    max_images: int,
    input_size: int,
    factor_kind: str,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    allowed = set(selected_ids)
    rows: dict[str, dict[str, Any]] = {}
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ValueError(f"unable to read existing observations: {path}") from exc
    with handle:
        for raw_line in handle:
            if not raw_line.endswith(b"\n"):
                raise ValueError("observation JSONL contains an unterminated line")
            try:
                row = json.loads(
                    raw_line.decode("utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                )
            except (UnicodeDecodeError, TypeError, ValueError) as exc:
                raise ValueError("observation JSONL contains malformed JSON") from exc
            if not isinstance(row, dict):
                raise ValueError("observation row must be an object")
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or image_id not in allowed:
                raise ValueError("observation image_id is outside deterministic selection")
            if image_id in rows:
                raise ValueError("observation image_id is duplicated")
            if row.get("schema_version") != 1:
                raise ValueError("observation schema_version is invalid")
            if row.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError("observation checkpoint provenance mismatch")
            if row.get("source_commit") != source_commit:
                raise ValueError("observation source provenance mismatch")
            if row.get("seed") != seed or row.get("device") != device:
                raise ValueError("observation run configuration mismatch")
            config = row.get("config")
            expected_config = {
                "max_images": max_images,
                "input_size": input_size,
                "factor_kind": factor_kind,
            }
            if config != expected_config:
                raise ValueError("observation configuration mismatch")
            mechanisms = row.get("mechanisms")
            if not isinstance(mechanisms, list) or len(mechanisms) != 1:
                raise ValueError("observation mechanisms are incomplete")
            mechanism = mechanisms[0]
            if not isinstance(mechanism, dict) or mechanism.get("factor_kind") != factor_kind:
                raise ValueError("observation factor provenance mismatch")
            nodes = mechanism.get("nodes")
            if not isinstance(nodes, list) or tuple(item.get("node") for item in nodes if isinstance(item, dict)) != MAIN_NODES:
                raise ValueError("observation node evidence is incomplete")
            for node in nodes:
                if not isinstance(node, dict):
                    raise ValueError("observation node evidence is malformed")
                for field in ("target_delta", "background_delta", "gap"):
                    _finite(node.get(field), f"observation {field}")
            rows[image_id] = row
    return rows


def _aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    node_values: dict[int, dict[str, list[float]]] = {
        node: {field: [] for field in ("target_delta", "background_delta", "gap")}
        for node in MAIN_NODES
    }
    for row in rows:
        mechanisms = row.get("mechanisms")
        if not isinstance(mechanisms, list) or len(mechanisms) != 1:
            raise ValueError("observation mechanisms are incomplete")
        nodes = mechanisms[0].get("nodes") if isinstance(mechanisms[0], dict) else None
        if not isinstance(nodes, list):
            raise ValueError("observation node evidence is incomplete")
        for item in nodes:
            if not isinstance(item, dict) or item.get("node") not in node_values:
                raise ValueError("observation node evidence is malformed")
            node = int(item["node"])
            for field in node_values[node]:
                node_values[node][field].append(_finite(item.get(field), field))
    all_gaps = [value for values in node_values.values() for value in values["gap"]]
    mean_gap = sum(all_gaps) / len(all_gaps) if all_gaps else 0.0
    return {
        "mean_gap": mean_gap,
        "finite_count": len(all_gaps),
        "finite_counts": {
            "target_delta": sum(len(values["target_delta"]) for values in node_values.values()),
            "background_delta": sum(len(values["background_delta"]) for values in node_values.values()),
            "gap": len(all_gaps),
        },
        "nodes": {
            str(node): {
                "target_delta_mean": sum(values["target_delta"]) / len(values["target_delta"]),
                "background_delta_mean": sum(values["background_delta"]) / len(values["background_delta"]),
                "gap_mean": sum(values["gap"]) / len(values["gap"]),
                "finite_count": len(values["gap"]),
                "finite_counts": {
                    "target_delta": len(values["target_delta"]),
                    "background_delta": len(values["background_delta"]),
                    "gap": len(values["gap"]),
                },
            }
            for node, values in node_values.items()
        },
    }


def _summary_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
    source_commit: str,
    image_ids: Sequence[str],
    seed: int,
    device: str,
    max_images: int,
    input_size: int,
    factor_kind: str,
    rejection_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "image_ids_sha256": hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest(),
        "image_ids": list(image_ids),
        "seed": seed,
        "config": {
            "max_images": max_images,
            "input_size": input_size,
            "factor_kind": factor_kind,
        },
        "device": device,
        "processed_images": len(rows),
        "rejection_count": rejection_count,
        "aggregates": _aggregates(rows),
        "finite_checks": {"all_deltas_finite": True, "all_gaps_finite": True},
    }


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    for field in (
        "schema_version",
        "source_commit",
        "checkpoint_sha256",
        "image_ids_sha256",
        "image_ids",
        "seed",
        "config",
        "device",
        "processed_images",
        "rejection_count",
        "aggregates",
        "finite_checks",
    ):
        if summary.get(field) != expected.get(field):
            raise ValueError(f"existing summary {field} does not match current run")
    if tuple(summary["image_ids"]) != tuple(rows):
        raise ValueError("existing summary does not cover all observations")
    return dict(summary)


def _manifest_payload(
    *,
    checkpoint_sha256: str,
    source_commit: str,
    image_ids: Sequence[str],
    seed: int,
    device: str,
    max_images: int,
    input_size: int,
    factor_kind: str,
    rejection_count: int,
    images: Path,
    labels: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": source_commit,
        "image_ids": list(image_ids),
        "seed": seed,
        "device": device,
        "config": {
            "max_images": max_images,
            "input_size": input_size,
            "factor_kind": factor_kind,
        },
        "rejection_count": rejection_count,
        "images": str(images.resolve()),
        "labels": str(labels.resolve()),
    }


def _ensure_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(expected):
            raise ValueError("existing run manifest does not match current run")
        return
    _write_json_exclusive(path, expected)
    if _read_json(path) != dict(expected):
        raise ValueError("run manifest changed while starting")


def run_three_view_mechanism_smoke(
    *,
    checkpoint: str | Path,
    images: str | Path,
    labels: str | Path,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    max_images: int = 8,
    input_size: int = 640,
    seed: int = 17,
    factor_kind: str = "sampling",
) -> dict[str, Any]:
    """Run the registered mechanism smoke and persist JSON evidence."""

    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
        raise ValueError("input_size must be a positive integer")
    if not isinstance(factor_kind, str) or factor_kind not in FACTOR_SPECS:
        raise ValueError(f"factor_kind must be one of {tuple(FACTOR_SPECS)}")
    factor_channel, intervention_kind = FACTOR_SPECS[factor_kind]
    images_path = Path(images)
    labels_path = Path(labels)
    selected, rejection_count = _select_images(images_path, labels_path, max_images=max_images)
    selected_ids = tuple(path.stem for path, _ in selected)
    loaded = load_ifdr_checkpoint(Path(checkpoint), device=torch_device)
    model = getattr(loaded, "model", None)
    checkpoint_sha256 = getattr(loaded, "checkpoint_sha256", None)
    if model is None or not callable(model) or not callable(getattr(model, "consume_reliability_context", None)):
        raise ValueError("loaded checkpoint does not expose an IFDR model")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("loaded checkpoint hash is missing")
    source_commit = _source_commit()
    device_name = str(torch_device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    observations_path = destination / OBSERVATIONS_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    manifest = _manifest_payload(
        checkpoint_sha256=checkpoint_sha256,
        source_commit=source_commit,
        image_ids=selected_ids,
        seed=seed,
        device=device_name,
        max_images=max_images,
        input_size=input_size,
        factor_kind=factor_kind,
        rejection_count=rejection_count,
        images=images_path,
        labels=labels_path,
    )
    existing = _read_observation_rows(
        observations_path,
        selected_ids=selected_ids,
        checkpoint_sha256=checkpoint_sha256,
        source_commit=source_commit,
        seed=seed,
        device=device_name,
        max_images=max_images,
        input_size=input_size,
        factor_kind=factor_kind,
    )
    if summary_path.is_file():
        if manifest_path.is_file() and _read_json(manifest_path) != manifest:
            raise ValueError("existing run manifest does not match current run")
        if set(existing) != set(selected_ids):
            raise ValueError("existing summary is present but observations are incomplete")
        ordered_rows = tuple(existing[image_id] for image_id in selected_ids)
        expected_summary = _summary_payload(
            ordered_rows,
            checkpoint_sha256=checkpoint_sha256,
            source_commit=source_commit,
            image_ids=selected_ids,
            seed=seed,
            device=device_name,
            max_images=max_images,
            input_size=input_size,
            factor_kind=factor_kind,
            rejection_count=rejection_count,
        )
        return _validate_summary(_read_json(summary_path), expected=expected_summary, rows=existing)
    _ensure_manifest(manifest_path, manifest)
    model.eval()
    try:
        observation_handle = observations_path.open("a", encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ValueError(f"unable to open observation JSONL: {observations_path}") from exc
    with observation_handle, torch.inference_mode():
        for image_path, objects in selected:
            if image_path.stem in existing:
                continue
            image = _read_image(image_path)
            target_index = min(
                range(len(objects)),
                key=lambda index: (objects[index]["box"][3] - objects[index]["box"][1], index),
            )
            target = objects[target_index]
            background = _choose_background(
                target["box"],
                tuple(item["box"] for item in objects),
                _stable_int(seed, image_path.stem, "background"),
            )
            strength = 0.5
            transform_seed = _stable_int(seed, image_path.stem, factor_kind)
            target_spec = InterventionSpec(
                image_id=image_path.stem,
                kind=intervention_kind,
                role=InterventionRole.OBJECT,
                strength=strength,
                seed=transform_seed,
                object_id=target_index,
                region_xyxy=target["box"],
            )
            background_spec = InterventionSpec(
                image_id=image_path.stem,
                kind=intervention_kind,
                role=InterventionRole.BACKGROUND,
                strength=strength,
                seed=transform_seed,
                region_xyxy=background,
            )
            target_image = apply_intervention(image, target_spec, factor_target_for_spec(target_spec)).image
            background_image = apply_intervention(image, background_spec, factor_target_for_spec(background_spec)).image
            clean_tensor, geometry = letterbox_image(image, input_size)
            target_tensor, target_geometry = letterbox_image(target_image, input_size)
            background_tensor, background_geometry = letterbox_image(background_image, input_size)
            if geometry != target_geometry or geometry != background_geometry:
                raise ValueError("three-view letterbox geometries do not match")
            batch = torch.stack((clean_tensor, target_tensor, background_tensor), dim=0).to(torch_device)
            model(batch)
            contexts = model.consume_reliability_context()
            if not isinstance(contexts, dict) or set(contexts) != set(DEFAULT_REQUIRED_NODES):
                raise ValueError("three-view reliability contexts are incomplete")
            clean_context, target_context, background_context = split_three_view_contexts(
                contexts, 1, required_nodes=tuple(DEFAULT_REQUIRED_NODES)
            )
            mechanism_nodes: list[dict[str, Any]] = []
            background_pixel_box = (
                background[0] * image.shape[1],
                background[1] * image.shape[0],
                background[2] * image.shape[1],
                background[3] * image.shape[0],
            )
            for node in MAIN_NODES:
                clean_factors = _context_tensors(clean_context[node], node)
                target_factors = _context_tensors(target_context[node], node)
                background_factors = _context_tensors(background_context[node], node)
                if clean_factors.shape[0] != 1 or target_factors.shape != clean_factors.shape or background_factors.shape != clean_factors.shape:
                    raise ValueError(f"node {node} contexts have inconsistent three-view shape")
                target_delta, target_roi = _pool_delta(
                    clean_factors, target_factors, batch_index=0, pixel_box=target["pixel_box"], geometry=geometry, channel=factor_channel
                )
                background_delta, background_roi = _pool_delta(
                    clean_factors, background_factors, batch_index=0, pixel_box=background_pixel_box, geometry=geometry, channel=factor_channel
                )
                gap = target_delta - background_delta
                if not math.isfinite(gap):
                    raise ValueError("specificity gap is non-finite")
                mechanism_nodes.append({
                    "node": node,
                    "target_delta": target_delta,
                    "background_delta": background_delta,
                    "gap": gap,
                    "target_roi": target_roi,
                    "background_roi": background_roi,
                })
            row = {
                "schema_version": 1,
                "image_id": image_path.stem,
                "class_id": target["class_id"],
                "target_index": target_index,
                "checkpoint_sha256": checkpoint_sha256,
                "source_commit": source_commit,
                "seed": seed,
                "device": device_name,
                "config": {
                    "max_images": max_images,
                    "input_size": input_size,
                    "factor_kind": factor_kind,
                },
                "mechanisms": [{
                    "factor_kind": factor_kind,
                    "factor_channel": factor_channel,
                    "severity": strength,
                    "transform_seed": transform_seed,
                    "target_box": target["box"],
                    "background_box": background,
                    "background_max_iou": max(_iou(background, item["box"]) for item in objects),
                    "nodes": mechanism_nodes,
                }],
            }
            observation_handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            observation_handle.flush()
            os.fsync(observation_handle.fileno())
            existing[image_path.stem] = row
    if set(existing) != set(selected_ids):
        raise ValueError("no finite three-view evidence was produced")
    ordered_rows = tuple(existing[image_id] for image_id in selected_ids)
    summary = _summary_payload(
        ordered_rows,
        checkpoint_sha256=checkpoint_sha256,
        source_commit=source_commit,
        image_ids=selected_ids,
        seed=seed,
        device=device_name,
        max_images=max_images,
        input_size=input_size,
        factor_kind=factor_kind,
        rejection_count=rejection_count,
    )
    if summary_path.is_file():
        return _validate_summary(_read_json(summary_path), expected=summary, rows=existing)
    _write_json_exclusive(summary_path, summary)
    return _validate_summary(_read_json(summary_path), expected=summary, rows=existing)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--factor-kind", choices=tuple(FACTOR_SPECS), default="sampling")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_three_view_mechanism_smoke(
        checkpoint=args.checkpoint,
        images=args.images,
        labels=args.labels,
        output_dir=args.output_dir,
        device=args.device,
        max_images=args.max_images,
        input_size=args.input_size,
        seed=args.seed,
        factor_kind=args.factor_kind,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
