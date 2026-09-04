"""Provenance-bound learned factor calibration manifests.

This module is deliberately small and deterministic.  A manifest is the
immutable bridge between a calibration checkpoint, the fit image/object set,
and the factor-guided replay distribution.  No evaluation or development ID
is admitted while the manifest is being built.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
from types import MappingProxyType
from urllib.parse import urlsplit
from typing import Any
from uuid import uuid4

from ifdr_yolo.data.metadata_index import FactorMetadataIndex, FactorObjectRecord
from ifdr_yolo.data.replay_sampler import (
    ReplayDistribution,
    digest_distribution,
    mix_m3_probabilities,
    replay_eta,
    sha256_canonical,
    uniform_probabilities,
)
from ifdr_yolo.data.splits import sha256_file


PRIMARY_NODE_IDS = (17, 20, 23, 26)
SCHEMA_VERSION = "factor-manifest-v1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA256")
    return value


def _text(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{field} must be exact non-empty text")
    return value


def _factor_id(value: Any, field: str) -> str:
    # ``aggregate_primary_node_factors`` intentionally returns an unbound
    # record with blank IDs; object records written into a manifest are still
    # required to carry exact non-empty IDs by the manifest coverage gate.
    if value == "":
        return ""
    return _text(value, field)


def _finite_score(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and in [0, 1]") from exc
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return result


def _ids(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of image IDs")
    result = tuple(sorted(_text(item, field) for item in values))
    if not result:
        raise ValueError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate IDs")
    return result


def digest_ids(values: Sequence[Any]) -> str:
    """Hash a sorted immutable ID sequence for manifest identity."""

    if isinstance(values, (str, bytes)):
        raise ValueError("identity IDs must be a sequence")
    normalized = tuple(values)
    if not normalized:
        return sha256_canonical(())
    if all(isinstance(value, str) for value in normalized):
        result = tuple(sorted(_text(value, "identity ID") for value in normalized))
    elif all(
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(part, str) for part in value)
        for value in normalized
    ):
        result = tuple(
            sorted(
                (_text(value[0], "identity image_id"), _text(value[1], "identity object_id"))
                for value in normalized
            )
        )
    else:
        raise ValueError("identity IDs must be all strings or all (string, string) pairs")
    return sha256_canonical(result)


def resolve_provenance_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a local checkpoint path and reject credential-bearing URIs."""

    if isinstance(value, os.PathLike):
        text = os.fspath(value)
    else:
        text = value
    if not isinstance(text, str) or "\x00" in text:
        raise ValueError("checkpoint path must be a local path")
    # ``urlsplit`` treats a Windows drive letter as a URI scheme; preserve
    # ordinary ``C:\\...`` paths while still rejecting real URI schemes.
    windows_path = re.match(r"^[A-Za-z]:[\\/]", text) is not None
    parsed = urlsplit(text) if not windows_path else None
    if parsed is not None and (parsed.scheme or parsed.netloc):
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("credential-bearing checkpoint path is forbidden")
        raise ValueError("checkpoint path must be a local path")
    path = Path(text).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {path}")
    return path


@dataclass(frozen=True)
class LearnedObjectFactor:
    image_id: str
    object_id: str
    sampling: float
    visibility: float
    learned_joint: float
    eligible_cyclist: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _factor_id(self.image_id, "image_id"))
        object.__setattr__(self, "object_id", _factor_id(self.object_id, "object_id"))
        object.__setattr__(self, "sampling", _finite_score(self.sampling, "sampling"))
        object.__setattr__(self, "visibility", _finite_score(self.visibility, "visibility"))
        learned_joint = _finite_score(self.learned_joint, "learned_joint")
        expected_joint = 1.0 - (1.0 - self.sampling) * (1.0 - self.visibility)
        if abs(learned_joint - expected_joint) > 1e-12:
            raise ValueError("learned_joint must equal the canonical sampling/visibility joint")
        object.__setattr__(self, "learned_joint", expected_joint)
        if not isinstance(self.eligible_cyclist, bool):
            raise ValueError("eligible_cyclist must be boolean")


@dataclass(frozen=True)
class LearnedFactorManifest:
    schema_version: str
    condition: str
    checkpoint_path: str
    checkpoint_role: str
    checkpoint_sha256: str
    fit_ids_sha256: str
    fit_ids: tuple[str, ...]
    metadata_index_sha256: str
    primary_node_ids: tuple[int, ...]
    expected_object_ids_sha256: str
    objects: tuple[LearnedObjectFactor, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _text(self.schema_version, "schema_version"))
        object.__setattr__(self, "condition", _text(self.condition, "condition"))
        if self.condition != "F0":
            raise ValueError("learned factor manifest condition must be F0")
        object.__setattr__(self, "checkpoint_path", _text(self.checkpoint_path, "checkpoint_path"))
        object.__setattr__(self, "checkpoint_role", _text(self.checkpoint_role, "checkpoint_role"))
        for field_name in (
            "checkpoint_sha256",
            "fit_ids_sha256",
            "metadata_index_sha256",
            "expected_object_ids_sha256",
            "manifest_sha256",
        ):
            object.__setattr__(self, field_name, _hash(getattr(self, field_name), field_name))
        object.__setattr__(self, "fit_ids", _ids(self.fit_ids, "fit_ids"))
        nodes = tuple(self.primary_node_ids)
        if nodes != PRIMARY_NODE_IDS:
            raise ValueError("primary_node_ids must equal the registered nodes")
        object.__setattr__(self, "primary_node_ids", nodes)
        normalized_objects = tuple(self.objects)
        if any(not isinstance(item, LearnedObjectFactor) for item in normalized_objects):
            raise ValueError("objects must contain LearnedObjectFactor records")
        ordered = tuple(sorted(normalized_objects, key=lambda item: (item.image_id, item.object_id)))
        if ordered != normalized_objects:
            raise ValueError("objects must be sorted by image_id and object_id")
        object_ids = tuple((item.image_id, item.object_id) for item in ordered)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("objects contain duplicate identities")
        object.__setattr__(self, "objects", ordered)

    def payload(self, *, include_digest: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "condition": self.condition,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_role": self.checkpoint_role,
            "checkpoint_sha256": self.checkpoint_sha256,
            "fit_ids_sha256": self.fit_ids_sha256,
            "fit_ids": self.fit_ids,
            "metadata_index_sha256": self.metadata_index_sha256,
            "primary_node_ids": self.primary_node_ids,
            "expected_object_ids_sha256": self.expected_object_ids_sha256,
            "objects": tuple(
                {
                    "image_id": item.image_id,
                    "object_id": item.object_id,
                    "sampling": item.sampling,
                    "visibility": item.visibility,
                    "learned_joint": item.learned_joint,
                    "eligible_cyclist": item.eligible_cyclist,
                }
                for item in self.objects
            ),
        }
        if include_digest:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


@dataclass(frozen=True)
class ValidatedMetadataPriorities:
    metadata_index_sha256: str
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata_index_sha256", _hash(self.metadata_index_sha256, "metadata_index_sha256"))
        if not isinstance(self.values, Mapping) or not self.values:
            raise ValueError("metadata priorities must be a non-empty mapping")
        normalized = {
            _text(image_id, "metadata priority image_id"): _finite_score(value, "metadata priority")
            for image_id, value in self.values.items()
        }
        object.__setattr__(self, "values", MappingProxyType(dict(sorted(normalized.items()))))


def aggregate_primary_node_factors(node_values: Mapping[int, Sequence[float]]) -> LearnedObjectFactor:
    """Macro-average P2--P5 sampling/visibility outputs and combine them."""

    if not isinstance(node_values, Mapping) or tuple(sorted(node_values)) != PRIMARY_NODE_IDS:
        raise ValueError("primary node factors must contain exactly P2-P5 nodes")
    sampling: list[float] = []
    visibility: list[float] = []
    for node_id in PRIMARY_NODE_IDS:
        pair = node_values[node_id]
        if isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError("each primary node must provide sampling and visibility")
        sampling.append(_finite_score(pair[0], f"sampling[{node_id}]"))
        visibility.append(_finite_score(pair[1], f"visibility[{node_id}]"))
    mean_sampling = sum(sampling) / len(sampling)
    mean_visibility = sum(visibility) / len(visibility)
    return LearnedObjectFactor(
        image_id="",
        object_id="",
        sampling=mean_sampling,
        visibility=mean_visibility,
        learned_joint=1.0 - (1.0 - mean_sampling) * (1.0 - mean_visibility),
        eligible_cyclist=True,
    )


def average_tie_percentile_rank(scores: Mapping[str, float]) -> Mapping[str, float]:
    """Return an ascending percentile rank, assigning tied values one rank."""

    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("scores must be a non-empty mapping")
    if any(not isinstance(key, str) for key in scores):
        raise ValueError("score keys must be exact strings")
    normalized = {_text(key, "score key"): _finite_score(value, "score") for key, value in scores.items()}
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    denominator = max(len(ordered) - 1, 1)
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0 / denominator
        for key, _ in ordered[index:end]:
            result[key] = rank
        index = end
    return MappingProxyType(dict(sorted(result.items())))


def image_max_eligible_cyclist_joint(
    objects: Sequence[LearnedObjectFactor],
) -> Mapping[str, float]:
    """Collapse object factors to the maximum eligible Cyclist joint per image."""

    if isinstance(objects, (str, bytes)):
        raise ValueError("learned objects must be a sequence")
    result: dict[str, float] = {}
    for item in objects:
        if not isinstance(item, LearnedObjectFactor):
            raise TypeError("learned objects must contain LearnedObjectFactor records")
        if not item.eligible_cyclist:
            continue
        if not item.image_id:
            raise ValueError("eligible learned factors require an image ID")
        result[item.image_id] = max(result.get(item.image_id, 0.0), item.learned_joint)
    return MappingProxyType(dict(sorted(result.items())))


def _eligible_metadata_objects(
    metadata_index: FactorMetadataIndex, fit_ids: Sequence[str]
) -> tuple[FactorObjectRecord, ...]:
    result: list[FactorObjectRecord] = []
    for image_id in fit_ids:
        for record in metadata_index.by_image.get(image_id, ()):
            if (
                record.image_id == image_id
                and record.class_name == "Cyclist"
                and record.sampling_valid
                and record.visibility_valid
            ):
                result.append(record)
    return tuple(sorted(result, key=lambda item: (item.image_id, item.object_id)))


def expected_eligible_cyclist_object_ids(
    metadata_index: FactorMetadataIndex, fit_ids: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    return tuple((record.image_id, record.object_id) for record in _eligible_metadata_objects(metadata_index, fit_ids))


def _manifest_digest_payload(payload: Mapping[str, Any]) -> str:
    return sha256_canonical(payload)


def manifest_digest(manifest: LearnedFactorManifest) -> str:
    return _manifest_digest_payload(manifest.payload())


def _validate_manifest_digest(manifest: LearnedFactorManifest) -> None:
    if manifest_digest(manifest) != manifest.manifest_sha256:
        raise ValueError("manifest digest mismatch")


def verify_manifest_binding(
    manifest: LearnedFactorManifest, metadata_index: FactorMetadataIndex
) -> None:
    if not isinstance(metadata_index, FactorMetadataIndex):
        raise ValueError("metadata_index binding is required")
    _validate_manifest_digest(manifest)
    if manifest.checkpoint_role != "calibration_last":
        raise ValueError("factor manifest requires calibration_last checkpoint")
    if manifest.metadata_index_sha256 != metadata_index.sha256:
        raise ValueError("manifest metadata index hash mismatch")
    if manifest.fit_ids_sha256 != digest_ids(manifest.fit_ids):
        raise ValueError("manifest fit ID hash mismatch")
    expected = expected_eligible_cyclist_object_ids(metadata_index, manifest.fit_ids)
    if manifest.expected_object_ids_sha256 != digest_ids(expected):
        raise ValueError("manifest expected object hash mismatch")
    observed = tuple((item.image_id, item.object_id) for item in manifest.objects)
    if observed != expected:
        raise ValueError("manifest object identity coverage mismatch")


def _coerce_factor(value: Any) -> LearnedObjectFactor:
    if isinstance(value, LearnedObjectFactor):
        return value
    if isinstance(value, Mapping):
        return LearnedObjectFactor(
            image_id=value["image_id"],
            object_id=value["object_id"],
            sampling=value["sampling"],
            visibility=value["visibility"],
            learned_joint=value["learned_joint"],
            eligible_cyclist=value.get("eligible_cyclist", True),
        )
    raise TypeError("learned factor records must be mappings or LearnedObjectFactor")


def _flatten_records(records: Iterable[Any] | Mapping[Any, Any]) -> tuple[Any, ...]:
    if isinstance(records, Mapping):
        flattened: list[Any] = []
        for value in records.values():
            if isinstance(value, (str, bytes)):
                flattened.append(value)
            elif isinstance(value, Iterable) and not isinstance(value, Mapping):
                flattened.extend(value)
            else:
                flattened.append(value)
        return tuple(flattened)
    return tuple(records)


def _manifest_from_objects(
    *,
    condition: str,
    checkpoint_path: Path,
    checkpoint_role: str,
    checkpoint_sha256: str,
    fit_ids: tuple[str, ...],
    metadata_index: FactorMetadataIndex,
    objects: tuple[LearnedObjectFactor, ...],
    fit_ids_sha256: str | None = None,
    metadata_index_sha256: str | None = None,
    expected_object_ids_sha256: str | None = None,
    primary_node_ids: tuple[int, ...] = PRIMARY_NODE_IDS,
) -> LearnedFactorManifest:
    expected_ids = expected_eligible_cyclist_object_ids(metadata_index, fit_ids)
    ordered = tuple(sorted(objects, key=lambda item: (item.image_id, item.object_id)))
    if tuple((item.image_id, item.object_id) for item in ordered) != expected_ids:
        raise ValueError("object identity coverage mismatch")
    provisional = LearnedFactorManifest(
        schema_version=SCHEMA_VERSION,
        condition=condition,
        checkpoint_path=str(checkpoint_path),
        checkpoint_role=checkpoint_role,
        checkpoint_sha256=checkpoint_sha256,
        fit_ids_sha256=fit_ids_sha256 or digest_ids(fit_ids),
        fit_ids=fit_ids,
        metadata_index_sha256=metadata_index_sha256 or metadata_index.sha256,
        primary_node_ids=primary_node_ids,
        expected_object_ids_sha256=expected_object_ids_sha256 or digest_ids(expected_ids),
        objects=ordered,
        manifest_sha256="0" * 64,
    )
    digest = manifest_digest(provisional)
    return LearnedFactorManifest(
        schema_version=provisional.schema_version,
        condition=provisional.condition,
        checkpoint_path=provisional.checkpoint_path,
        checkpoint_role=provisional.checkpoint_role,
        checkpoint_sha256=provisional.checkpoint_sha256,
        fit_ids_sha256=provisional.fit_ids_sha256,
        fit_ids=provisional.fit_ids,
        metadata_index_sha256=provisional.metadata_index_sha256,
        primary_node_ids=provisional.primary_node_ids,
        expected_object_ids_sha256=provisional.expected_object_ids_sha256,
        objects=provisional.objects,
        manifest_sha256=digest,
    )


def _write_manifest(manifest: LearnedFactorManifest, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        manifest.payload(include_digest=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if output_path.exists():
        if output_path.read_bytes() == data:
            return
        raise ValueError("manifest output already exists with different content")
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError:
            if output_path.read_bytes() != data:
                raise ValueError("manifest output already exists with different content")
        except OSError:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                descriptor = os.open(output_path, flags)
            except FileExistsError:
                if output_path.read_bytes() != data:
                    raise ValueError("manifest output already exists with different content")
            else:
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        descriptor = -1
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                finally:
                    if descriptor != -1:
                        os.close(descriptor)
        try:
            directory_flags = os.O_RDONLY
            directory_fd = os.open(output_path.parent, directory_flags)
        except OSError:
            directory_fd = -1
        if directory_fd != -1:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_manifest_from_records(
    *,
    condition: str,
    checkpoint_path: str | os.PathLike[str],
    checkpoint_role: str,
    checkpoint_sha256: str,
    fit_ids: Sequence[str],
    records: Iterable[Any] | Mapping[Any, Any],
    metadata_index: FactorMetadataIndex | None = None,
    candidate_gate: bool = True,
    output_path: str | os.PathLike[str] | None = None,
    fit_ids_sha256: str | None = None,
    metadata_index_sha256: str | None = None,
    expected_object_ids_sha256: str | None = None,
    primary_node_ids: Sequence[int] = PRIMARY_NODE_IDS,
) -> LearnedFactorManifest:
    """Validate observed factors and build a digest-bound manifest."""

    if candidate_gate is not True:
        raise ValueError("candidate gate failed; manifest write is forbidden")
    if _text(condition, "condition") != "F0":
        raise ValueError("learned factor manifest condition must be F0")
    if checkpoint_role != "calibration_last":
        raise ValueError("factor manifest requires calibration_last checkpoint")
    _hash(checkpoint_sha256, "checkpoint_sha256")
    if not isinstance(metadata_index, FactorMetadataIndex):
        raise ValueError("metadata_index binding is required")
    resolved_path = resolve_provenance_path(checkpoint_path)
    actual_sha256 = sha256_file(resolved_path)
    if actual_sha256 != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    expected_fit_ids = _ids(fit_ids, "fit_ids")
    expected_fit_hash = digest_ids(expected_fit_ids)
    if fit_ids_sha256 is not None and _hash(fit_ids_sha256, "fit_ids_sha256") != expected_fit_hash:
        raise ValueError("fit ID hash mismatch")
    bound_metadata_hash = metadata_index.sha256
    expected_tuple = expected_eligible_cyclist_object_ids(metadata_index, expected_fit_ids)
    if metadata_index_sha256 is not None and _hash(metadata_index_sha256, "metadata_index_sha256") != bound_metadata_hash:
        raise ValueError("metadata index hash mismatch")
    expected = set(expected_tuple)
    accepted: list[LearnedObjectFactor] = []
    seen: set[tuple[str, str]] = set()
    normalized = tuple(_coerce_factor(raw) for raw in _flatten_records(records))
    for factor in normalized:
        if factor.image_id not in expected_fit_ids:
            raise ValueError("fit image coverage contains an image outside fit IDs")
        identity = (factor.image_id, factor.object_id)
        if factor.eligible_cyclist:
            if identity not in expected:
                raise ValueError("object identity coverage mismatch")
            if identity in seen:
                raise ValueError("duplicate learned object factor")
            seen.add(identity)
            accepted.append(factor)
    if seen != expected:
        raise ValueError("object identity coverage mismatch")
    expected_object_hash = digest_ids(tuple(sorted(expected)))
    if expected_object_ids_sha256 is not None and _hash(expected_object_ids_sha256, "expected_object_ids_sha256") != expected_object_hash:
        raise ValueError("expected object hash mismatch")
    if tuple(primary_node_ids) != PRIMARY_NODE_IDS:
        raise ValueError("primary_node_ids must equal the registered nodes")
    manifest = _manifest_from_objects(
        condition=_text(condition, "condition"),
        checkpoint_path=resolved_path,
        checkpoint_role=checkpoint_role,
        checkpoint_sha256=checkpoint_sha256,
        fit_ids=expected_fit_ids,
        metadata_index=metadata_index,
        objects=tuple(accepted),
        fit_ids_sha256=fit_ids_sha256,
        metadata_index_sha256=bound_metadata_hash,
        expected_object_ids_sha256=expected_object_ids_sha256,
        primary_node_ids=tuple(primary_node_ids),
    )
    if output_path is not None:
        _write_manifest(manifest, Path(output_path))
    return manifest


def _batch_image_ids(batch: Any) -> tuple[str, ...]:
    if isinstance(batch, Mapping):
        for key in ("image_ids", "image_id"):
            if key in batch:
                values = batch[key]
                if isinstance(values, str):
                    return (values,)
                return tuple(values)
    for key in ("image_ids", "image_id"):
        if hasattr(batch, key):
            values = getattr(batch, key)
            if isinstance(values, str):
                return (values,)
            return tuple(values)
    raise ValueError("deterministic loader batch must expose image_ids")


def deterministic_no_augmentation_loader(loader: Iterable[Any], expected_images: Sequence[str]) -> tuple[Any, ...]:
    """Materialize a loader while requiring its explicit deterministic image IDs."""

    expected = set(_ids(expected_images, "expected_images"))
    observed: set[str] = set()
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        image_ids = _batch_image_ids(batch)
        for image_id in image_ids:
            if not isinstance(image_id, str) or not image_id or image_id.strip() != image_id:
                raise ValueError("fit image IDs must be exact strings")
            if image_id not in expected:
                raise ValueError("fit image coverage contains an image outside fit IDs")
            if image_id in observed:
                raise ValueError("deterministic loader contains duplicate images")
            observed.add(image_id)
        yield batch
        del batch
    if observed != expected:
        raise ValueError("fit image coverage mismatch")


def evaluate_primary_nodes(model: Any, batch: Any, primary_node_ids: Sequence[int] = PRIMARY_NODE_IDS) -> tuple[LearnedObjectFactor, ...]:
    """Evaluate an IFDR batch and pool the registered primary-node contexts."""

    nodes = tuple(primary_node_ids)
    if nodes != PRIMARY_NODE_IDS:
        raise ValueError("primary_node_ids must equal the registered nodes")
    explicit_hook = getattr(model, "evaluate_primary_nodes", None)
    if callable(explicit_hook):
        result = explicit_hook(batch, nodes)
        return _normalize_evaluated_factors(result)
    if not isinstance(batch, Mapping):
        raise ValueError("IFDR evaluation batch must be a mapping")
    try:
        import torch
        from ifdr_yolo.eval.factor_observer import LetterboxGeometry
        from ifdr_yolo.eval.factor_observer_runtime import (
            DEFAULT_REQUIRED_NODES,
            pool_reliability_contexts,
        )
    except ImportError as exc:
        raise ValueError("IFDR runtime dependencies are unavailable") from exc
    images = batch.get("img", batch.get("images"))
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("IFDR batch images must be a BCHW tensor")
    image_ids = _batch_image_ids(batch)
    if len(image_ids) != images.shape[0] or len(set(image_ids)) != len(image_ids):
        raise ValueError("IFDR batch image_ids must match the image batch exactly")
    geometries = batch.get("geometries")
    objects = batch.get("objects")
    if not isinstance(geometries, Mapping) or set(geometries) != set(image_ids):
        raise ValueError("IFDR batch geometries must cover image_ids exactly")
    if not isinstance(objects, Mapping) or set(objects) != set(image_ids):
        raise ValueError("IFDR batch objects must cover image_ids exactly")
    if any(not isinstance(geometries[image_id], LetterboxGeometry) for image_id in image_ids):
        raise ValueError("IFDR batch geometries must be LetterboxGeometry records")
    for image_id in image_ids:
        records = objects[image_id]
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise ValueError("IFDR batch objects must be FactorObjectRecord sequences")
        if any(not isinstance(record, FactorObjectRecord) for record in records):
            raise ValueError("IFDR batch objects must be FactorObjectRecord sequences")
    device = torch.device("cpu")
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            device = parameter.device
            break
    if device.type == "cpu":
        buffers = getattr(model, "buffers", None)
        if callable(buffers):
            for buffer in buffers():
                device = buffer.device
                break
    model(images.to(device))
    consume = getattr(model, "consume_reliability_context", None)
    if not callable(consume):
        raise ValueError("IFDR model must expose consume_reliability_context")
    contexts = consume()
    factors: list[LearnedObjectFactor] = []
    for batch_index, image_id in enumerate(image_ids):
        for record in objects[image_id]:
            if (
                record.class_name != "Cyclist"
                or not record.sampling_valid
                or not record.visibility_valid
            ):
                continue
            pooled = pool_reliability_contexts(
                contexts,
                batch_index=batch_index,
                bbox_xyxy=record.bbox_xyxy,
                geometry=geometries[image_id],
                required_nodes=DEFAULT_REQUIRED_NODES,
            )
            node_values = {
                pooled_node.node: (
                    pooled_node.predicted_sampling,
                    pooled_node.predicted_visibility,
                )
                for pooled_node in pooled
                if pooled_node.node in nodes
            }
            aggregate = aggregate_primary_node_factors(node_values)
            factors.append(
                LearnedObjectFactor(
                    image_id=image_id,
                    object_id=record.object_id,
                    sampling=aggregate.sampling,
                    visibility=aggregate.visibility,
                    learned_joint=aggregate.learned_joint,
                    eligible_cyclist=True,
                )
            )
    return tuple(sorted(factors, key=lambda item: (item.image_id, item.object_id)))


def _normalize_evaluated_factors(result: Any) -> tuple[LearnedObjectFactor, ...]:
    if isinstance(result, LearnedObjectFactor):
        return (result,)
    if isinstance(result, Mapping):
        if "image_id" in result:
            return (_coerce_factor(result),)
        result = result.get("learned_factors", result.get("objects", ()))
    return tuple(_coerce_factor(item) for item in result)


def _stable_state(value: Any) -> Any:
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) or not key or key.strip() != key for key in keys):
            raise ValueError("model state mapping keys must be exact strings")
        if len(set(keys)) != len(keys):
            raise ValueError("model state mapping keys must be unique")
        return {key: _stable_state(value[key]) for key in sorted(keys)}
    if isinstance(value, (tuple, list)):
        return [_stable_state(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return {"dtype": str(tensor.dtype), "shape": tuple(tensor.shape), "values": tensor.tolist()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def full_model_state_sha256(model: Any) -> str:
    """Hash parameters and all named buffers without materializing JSON lists."""

    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_parameters) and callable(named_buffers):
        digest = sha256()
        entries = [
            ("parameter", name, value) for name, value in named_parameters()
        ] + [
            ("buffer", name, value) for name, value in named_buffers()
        ]
        for kind, name, value in sorted(entries, key=lambda item: (item[0], item[1])):
            digest.update(kind.encode("utf-8"))
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            if hasattr(value, "detach"):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(repr(tuple(tensor.shape)).encode("ascii"))
                import torch

                raw = tensor.reshape(-1).view(torch.uint8).reshape(-1)
                chunk_size = 1024 * 1024
                for start in range(0, raw.numel(), chunk_size):
                    digest.update(memoryview(raw[start : start + chunk_size].numpy()))
            else:
                digest.update(sha256_canonical(_stable_state(value)).encode("ascii"))
        return digest.hexdigest()
    state = model.state_dict() if hasattr(model, "state_dict") else getattr(model, "__dict__", {})
    return sha256_canonical(_stable_state(state))


def capture_training_flags(model: Any) -> tuple[tuple[str, bool], ...]:
    modules: list[tuple[str, Any]] = []
    if hasattr(model, "named_modules"):
        modules = list(model.named_modules())
    if not modules:
        modules = [("", model)]
    return tuple((name, bool(getattr(module, "training"))) for name, module in modules if hasattr(module, "training"))


def restore_training_flags(model: Any, flags: Sequence[tuple[str, bool]]) -> None:
    modules = dict(model.named_modules()) if hasattr(model, "named_modules") else {"": model}
    for name, value in flags:
        module = modules.get(name)
        if module is None:
            continue
        if name == "" and hasattr(module, "train"):
            module.train(value)
        elif hasattr(module, "training"):
            module.training = value


def _capture_model_state(model: Any) -> tuple[tuple[str, str, Any], ...]:
    """Clone parameters/buffers to CPU without constructing a JSON snapshot."""

    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_parameters) and callable(named_buffers):
        snapshot: list[tuple[str, str, Any]] = []
        for kind, entries in (
            ("parameter", named_parameters()),
            ("buffer", named_buffers()),
        ):
            for name, value in entries:
                if hasattr(value, "detach"):
                    snapshot.append((kind, name, value.detach().cpu().clone()))
                else:
                    snapshot.append((kind, name, deepcopy(value)))
        return tuple(sorted(snapshot, key=lambda item: (item[0], item[1])))
    state_dict = getattr(model, "state_dict", None)
    if callable(state_dict):
        return (("state_dict", "", deepcopy(state_dict())),)
    return ()


def _restore_model_state(model: Any, snapshot: Sequence[tuple[str, str, Any]]) -> None:
    if not snapshot:
        return
    if snapshot[0][0] == "state_dict":
        load_state_dict = getattr(model, "load_state_dict", None)
        if callable(load_state_dict):
            load_state_dict(snapshot[0][2], strict=True)
        return
    parameters = dict(model.named_parameters()) if hasattr(model, "named_parameters") else {}
    buffers = dict(model.named_buffers()) if hasattr(model, "named_buffers") else {}
    import torch

    with torch.no_grad():
        for kind, name, value in snapshot:
            target = parameters.get(name) if kind == "parameter" else buffers.get(name)
            if target is None:
                raise ValueError(f"model state member disappeared during manifest evaluation: {name}")
            if hasattr(target, "copy_") and hasattr(value, "to"):
                target.copy_(value.to(device=target.device, dtype=target.dtype))
            else:
                raise ValueError(f"model state member is not copyable: {name}")


def load_validated_checkpoint(model: Any, checkpoint_path: Path, *, role: str) -> None:
    if role != "calibration_last":
        raise ValueError("checkpoint role must be calibration_last")
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        raise ValueError("model must expose load_state_dict for calibration checkpoint")
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("unable to load calibration checkpoint") from exc
    state = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "ema", "model"):
            if key in checkpoint:
                state = checkpoint[key]
                break
    state_dict_method = getattr(state, "state_dict", None)
    if callable(state_dict_method):
        state = state_dict_method()
    if not isinstance(state, Mapping):
        raise ValueError("calibration checkpoint has no state dict")
    before_model_state = _capture_model_state(model)
    try:
        load_state_dict(state, strict=True)
    except Exception as exc:
        _restore_model_state(model, before_model_state)
        raise ValueError("calibration checkpoint state_dict does not match model") from exc


def build_learned_factor_manifest(
    *,
    condition: str,
    checkpoint_path: str | os.PathLike[str],
    checkpoint_role: str,
    checkpoint_sha256: str,
    model: Any,
    loader: Iterable[Any],
    fit_ids: Sequence[str],
    metadata_index: FactorMetadataIndex,
) -> LearnedFactorManifest:
    """Evaluate the deterministic fit loader and bind the resulting factors."""

    if _text(condition, "condition") != "F0":
        raise ValueError("learned factor manifest condition must be F0")
    if checkpoint_role != "calibration_last":
        raise ValueError("factor manifest requires calibration_last checkpoint")
    _hash(checkpoint_sha256, "checkpoint_sha256")
    if not isinstance(metadata_index, FactorMetadataIndex):
        raise TypeError("metadata_index must be a FactorMetadataIndex")
    resolved_path = resolve_provenance_path(checkpoint_path)
    if sha256_file(resolved_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    expected_images = _ids(fit_ids, "fit_ids")
    expected_objects = set(expected_eligible_cyclist_object_ids(metadata_index, expected_images))
    load_validated_checkpoint(model, resolved_path, role=checkpoint_role)
    before_model_state = _capture_model_state(model)
    before_state = full_model_state_sha256(model)
    before_flags = capture_training_flags(model)
    observed_images: list[str] = []
    observed_objects: list[LearnedObjectFactor] = []
    batches = deterministic_no_augmentation_loader(loader, expected_images)
    state_changed = False
    try:
        model.eval()
        try:
            import torch

            no_grad = torch.no_grad()
        except ImportError:
            no_grad = nullcontext()
        with no_grad:
            for batch in batches:
                observed_images.extend(_batch_image_ids(batch))
                observed_objects.extend(evaluate_primary_nodes(model, batch, PRIMARY_NODE_IDS))
    finally:
        try:
            state_changed = full_model_state_sha256(model) != before_state
        finally:
            try:
                _restore_model_state(model, before_model_state)
            finally:
                restore_training_flags(model, before_flags)
    if tuple(sorted(observed_images)) != expected_images:
        raise ValueError("fit image coverage mismatch")
    observed_ids = tuple(sorted((item.image_id, item.object_id) for item in observed_objects if item.eligible_cyclist))
    if observed_ids != tuple(sorted(expected_objects)):
        raise ValueError("object identity coverage mismatch")
    if state_changed:
        raise ValueError("manifest generation changed model state")
    return _manifest_from_objects(
        condition=_text(condition, "condition"),
        checkpoint_path=resolved_path,
        checkpoint_role=checkpoint_role,
        checkpoint_sha256=checkpoint_sha256,
        fit_ids=expected_images,
        metadata_index=metadata_index,
        objects=tuple(item for item in observed_objects if item.eligible_cyclist),
    )


def build_learned_focus_distribution(
    *,
    manifest: LearnedFactorManifest,
    metadata_index: FactorMetadataIndex,
    metadata_priorities: ValidatedMetadataPriorities,
    epoch: int,
) -> ReplayDistribution:
    """Build the registered 50/50 metadata/learned factor replay distribution."""

    if not isinstance(metadata_priorities, ValidatedMetadataPriorities):
        raise ValueError("metadata priorities must be a validated wrapper")
    verify_manifest_binding(manifest, metadata_index)
    if metadata_priorities.metadata_index_sha256 != metadata_index.sha256:
        raise ValueError("metadata priorities are bound to a different metadata index")
    if set(metadata_priorities.values) != set(manifest.fit_ids):
        raise ValueError("metadata priorities must cover exactly the fit IDs")
    learned_priority = dict(image_max_eligible_cyclist_joint(manifest.objects))
    if not learned_priority:
        raise ValueError("no eligible Cyclist learned factors for focus distribution")
    learned_percentile = average_tie_percentile_rank(learned_priority)
    focus_scores = {
        image_id: 0.5 * metadata_priorities.values[image_id]
        + 0.5 * learned_percentile[image_id]
        for image_id in sorted(learned_priority)
    }
    weights = {image_id: score + 0.05 for image_id, score in focus_scores.items()}
    total = sum(weights.values())
    focus = {image_id: value / total for image_id, value in weights.items()}
    original = uniform_probabilities(manifest.fit_ids)
    probabilities = mix_m3_probabilities(original=original, focus=focus, epoch=epoch)
    distribution_sha256 = digest_distribution(
        "factor_guided",
        epoch,
        probabilities,
        manifest.manifest_sha256,
        manifest.checkpoint_sha256,
        manifest.metadata_index_sha256,
        focus_scores=focus_scores,
    )
    return ReplayDistribution(
        mode="factor_guided",
        epoch=epoch,
        eta=replay_eta(epoch),
        image_ids=manifest.fit_ids,
        original_probabilities=probabilities.original,
        focus_probabilities=probabilities.focus,
        probabilities=probabilities.final,
        focus_scores=focus_scores,
        source_sha256=manifest.manifest_sha256,
        manifest_sha256=manifest.manifest_sha256,
        calibration_checkpoint_sha256=manifest.checkpoint_sha256,
        metadata_index_sha256=manifest.metadata_index_sha256,
        distribution_sha256=distribution_sha256,
    )


__all__ = [
    "PRIMARY_NODE_IDS",
    "SCHEMA_VERSION",
    "LearnedObjectFactor",
    "LearnedFactorManifest",
    "ValidatedMetadataPriorities",
    "aggregate_primary_node_factors",
    "average_tie_percentile_rank",
    "build_manifest_from_records",
    "build_learned_factor_manifest",
    "build_learned_focus_distribution",
    "capture_training_flags",
    "deterministic_no_augmentation_loader",
    "digest_ids",
    "evaluate_primary_nodes",
    "expected_eligible_cyclist_object_ids",
    "full_model_state_sha256",
    "image_max_eligible_cyclist_joint",
    "load_validated_checkpoint",
    "manifest_digest",
    "resolve_provenance_path",
    "restore_training_flags",
    "verify_manifest_binding",
]
