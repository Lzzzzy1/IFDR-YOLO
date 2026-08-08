"""Build development factor evidence from persisted observer/audit artifacts.

The observer is deliberately kept out of this module.  A calibration
``last.pt`` checkpoint and its role manifest are verified here, then persisted
observer rows are reduced to the registered six-node gate rows.  The raw rows
remain part of the immutable evidence object so paired image-cluster draws
can recompute the four endpoints; a point estimate is never treated as a
per-image sample.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from ifdr_yolo.data.replay_sampler import sha256_canonical
from ifdr_yolo.eval.factor_repair_gate import (
    DIAGNOSTIC_NODE_IDS,
    PRIMARY_ENDPOINTS,
    PRIMARY_NODE_IDS,
    FactorRepairEvidence,
    FactorRepairGateDecision,
    evaluate_factor_repair_gate,
)
from ifdr_yolo.eval.natural_factor_audit import (
    NaturalFactorObservation,
    intervention_statistics,
    partial_spearman,
)


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_NAME = "last.pt"
_CHECKPOINT_ROLE = "calibration_last"
_MISSING = object()


def _jsonable(value: object) -> object:
    """Convert nested producer objects into deterministic JSON values."""

    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evidence contains a non-finite number")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_json(item) for item in value)
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value.lower()) is None:
        raise ValueError(f"{field} must be a 64-hex SHA256")
    return value.lower()


def _read_json(value: object, field: str) -> object:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return converted
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve(strict=False)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"{field} is missing or empty: {path}")
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (OSError, UnicodeDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"{field} is malformed: {path}") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field} must contain a JSON object")
        return payload
    raise ValueError(f"{field} must be a mapping or JSON path")


def _read_observation_jsonl(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve(strict=False)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"observations are missing or empty: {path}")
        rows: list[object] = []
        try:
            with path.open("rb") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    if not raw.endswith(b"\n"):
                        raise ValueError(
                            f"observations JSONL has an unterminated line: {path}:{line_number}"
                        )
                    try:
                        row = json.loads(
                            raw.decode("utf-8"),
                            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                        )
                    except (UnicodeDecodeError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"observations JSONL is malformed: {path}:{line_number}"
                        ) from error
                    if not isinstance(row, Mapping):
                        raise ValueError("observation rows must be JSON objects")
                    rows.append(row)
        except OSError as error:
            raise ValueError(f"unable to read observations: {path}") from error
        if not rows:
            raise ValueError(f"observations are empty: {path}")
        return tuple(rows)
    if isinstance(value, Mapping) or isinstance(value, NaturalFactorObservation):
        return (value,)
    try:
        rows = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("observations must be an iterable or JSONL path") from error
    if not rows:
        raise ValueError("observations must not be empty")
    return rows


def _observation_to_dict(row: NaturalFactorObservation) -> dict[str, object]:
    return {
        "seed": row.seed,
        "node_id": row.node_id,
        "image_id": row.image_id,
        "object_id": row.object_id,
        "class_id": row.class_id,
        "class_name": row.class_name,
        "box_height": row.box_height,
        "region_role": row.region_role,
        "intervention_kind": row.intervention_kind,
        "intervention_factor": row.intervention_factor,
        "intervention_severity": row.intervention_severity,
        "pair_id": row.pair_id,
        "natural_sampling": row.natural_sampling,
        "natural_visibility": row.natural_visibility,
        "predicted_sampling": row.predicted_sampling,
        "predicted_visibility": row.predicted_visibility,
        "branch_weights": list(row.branch_weights),
    }


def _mapping_observation(row: Mapping[str, object]) -> tuple[NaturalFactorObservation, dict[str, object]]:
    source = row.get("factor_observation")
    if isinstance(source, Mapping):
        row = source
    try:
        observation = NaturalFactorObservation(
            seed=row["seed"],
            node_id=row["node_id"],
            image_id=row["image_id"],
            object_id=row["object_id"],
            class_id=row["class_id"],
            class_name=row.get("class_name"),
            box_height=row["box_height"],
            region_role=row["region_role"],
            intervention_kind=row["intervention_kind"],
            intervention_factor=row.get("intervention_factor"),
            intervention_severity=row["intervention_severity"],
            pair_id=row.get("pair_id"),
            natural_sampling=row["natural_sampling"],
            natural_visibility=row["natural_visibility"],
            predicted_sampling=row["predicted_sampling"],
            predicted_visibility=row["predicted_visibility"],
            branch_weights=tuple(row["branch_weights"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("observer row is incomplete or invalid") from error
    # Keep all provenance fields supplied by the observer.  The canonical
    # evidence payload therefore remains auditable without retaining a model.
    raw = dict(row)
    raw.setdefault("seed", observation.seed)
    raw.setdefault("node_id", observation.node_id)
    raw.setdefault("image_id", observation.image_id)
    raw.setdefault("object_id", observation.object_id)
    return observation, raw


def _normalise_observations(
    value: object,
    *,
    checkpoint_sha256: str,
    stage: str,
) -> tuple[tuple[NaturalFactorObservation, ...], tuple[dict[str, object], ...]]:
    source_rows = _read_observation_jsonl(value)
    observations: list[NaturalFactorObservation] = []
    raw_rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for index, source in enumerate(source_rows):
        if isinstance(source, NaturalFactorObservation):
            observation = source
            raw = _observation_to_dict(source)
        elif isinstance(source, Mapping):
            observation, raw = _mapping_observation(source)
            supplied_hash = raw.get("checkpoint_sha256")
            if supplied_hash is None and any(
                name in raw for name in ("schema_version", "observation_id", "manifest_sha256")
            ):
                raise ValueError(f"observer checkpoint hash is missing at row {index}")
            if supplied_hash is not None and _require_hash(supplied_hash, "observation checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(f"observation checkpoint hash mismatch at row {index}")
        else:
            raise ValueError("observations must contain NaturalFactorObservation or mappings")
        if stage == "development" and observation.seed != 17:
            raise ValueError("development evidence requires seed 17 only")
        identity = (
            observation.seed,
            observation.node_id,
            observation.image_id,
            observation.object_id,
            observation.intervention_kind,
            observation.intervention_factor,
            observation.intervention_severity,
            observation.region_role,
            observation.pair_id,
        )
        if identity in seen:
            raise ValueError("observer observations contain duplicate identities")
        seen.add(identity)
        observations.append(observation)
        raw_rows.append(raw)
    if not observations:
        raise ValueError("observations must not be empty")
    order = sorted(
        range(len(observations)),
        key=lambda index: (
            observations[index].image_id,
            observations[index].seed,
            observations[index].node_id,
            observations[index].object_id,
            observations[index].intervention_kind,
            observations[index].intervention_factor or "",
            observations[index].intervention_severity,
            observations[index].region_role,
        ),
    )
    return (
        tuple(observations[index] for index in order),
        tuple(raw_rows[index] for index in order),
    )


def _checkpoint_entry(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if any(name in value for name in ("path", "sha256", "hash", "role", "checkpoint_role")):
        return value
    for name in (
        "primary_checkpoint",
        "calibration_last",
        "primary",
        "last_checkpoint",
        "last",
    ):
        nested = value.get(name)
        if isinstance(nested, Mapping):
            return nested
    nested_roles = value.get("checkpoint_roles")
    if isinstance(nested_roles, Mapping):
        return _checkpoint_entry(nested_roles)
    return None


def _resolve_checkpoint(
    checkpoint: object,
    checkpoint_roles: object,
) -> tuple[dict[str, object], dict[str, object]]:
    roles_payload = _read_json(checkpoint_roles, "checkpoint_roles")
    role_entry = _checkpoint_entry(roles_payload)
    if role_entry is None:
        raise ValueError("checkpoint_roles must contain a primary calibration checkpoint role/hash")

    checkpoint_entry = checkpoint if isinstance(checkpoint, Mapping) else {"path": checkpoint}
    if not isinstance(checkpoint_entry, Mapping):
        raise ValueError("checkpoint must be a path or mapping")
    supplied_role = checkpoint_entry.get("role", checkpoint_entry.get("checkpoint_role"))
    if supplied_role is not None and supplied_role not in {"primary", _CHECKPOINT_ROLE}:
        raise ValueError("calibration checkpoint role must be primary/calibration_last")
    raw_path = checkpoint_entry.get("path", role_entry.get("path"))
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        raise ValueError("calibration checkpoint path is required")
    path = Path(raw_path).expanduser().resolve(strict=False)
    if path.name != _CHECKPOINT_NAME:
        raise ValueError("calibration evidence requires last.pt, never best.pt")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("calibration last.pt is missing or empty")

    role = role_entry.get("role", role_entry.get("checkpoint_role"))
    checkpoint_role = role_entry.get("checkpoint_role")
    if checkpoint_role is None and role == _CHECKPOINT_ROLE:
        checkpoint_role = _CHECKPOINT_ROLE
    if role is not None and role not in {"primary", _CHECKPOINT_ROLE}:
        raise ValueError("calibration checkpoint role must be primary/calibration_last")
    if checkpoint_role is not None and checkpoint_role != _CHECKPOINT_ROLE:
        raise ValueError("calibration checkpoint role must be calibration_last")
    expected_hashes: list[str] = []
    for source in (checkpoint_entry, role_entry):
        raw_hash = source.get("sha256", source.get("hash"))
        if raw_hash is not None:
            expected_hashes.append(_require_hash(raw_hash, "calibration checkpoint sha256"))
    if not expected_hashes:
        raise ValueError("checkpoint_roles must include calibration checkpoint hash")
    if len(set(expected_hashes)) != 1:
        raise ValueError("calibration checkpoint role/hash mismatch")
    actual_hash = _sha256_bytes(path.read_bytes())
    if actual_hash != expected_hashes[0]:
        raise ValueError("calibration checkpoint hash mismatch")
    if role_entry.get("path") is not None:
        role_path = Path(str(role_entry["path"])).expanduser().resolve(strict=False)
        # Trainers commonly persist the role path as the registered basename
        # ``last.pt`` while callers pass the resolved absolute path.  A
        # relative basename is still bound to the verified caller path; an
        # explicit relative subpath or absolute path must resolve identically.
        relative_role_path = Path(str(role_entry["path"]))
        if not relative_role_path.is_absolute() and relative_role_path.as_posix() == _CHECKPOINT_NAME:
            pass
        elif role_path != path:
            raise ValueError("checkpoint_roles path does not match calibration checkpoint")
    resolved = {
        "path": str(path),
        "role": "primary",
        "checkpoint_role": _CHECKPOINT_ROLE,
        "sha256": actual_hash,
    }
    return resolved, dict(_jsonable(roles_payload))  # type: ignore[arg-type]


def _finite_stat(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and -1.0 <= result <= 1.0 else None


def _residual_rho(rows: Sequence[NaturalFactorObservation], factor: str) -> float | None:
    natural = tuple(row for row in rows if row.intervention_kind == "natural")
    if not natural:
        return None
    result = partial_spearman(
        tuple(
            row.natural_sampling if factor == "sampling" else row.natural_visibility
            for row in natural
        ),
        tuple(
            row.predicted_sampling if factor == "sampling" else row.predicted_visibility
            for row in natural
        ),
        tuple(row.box_height for row in natural),
        tuple(row.class_id for row in natural),
    )
    if not isinstance(result, Mapping) or result.get("success") is not True:
        return None
    return _finite_stat(result.get("rho"))


def _specificity_gap(rows: Sequence[NaturalFactorObservation], factor: str) -> tuple[float | None, int]:
    result = intervention_statistics(rows, factor=factor)
    malformed = result.get("malformed", 0)
    malformed_count = int(malformed) if isinstance(malformed, (int, float)) and math.isfinite(float(malformed)) else 1
    target = result.get("target_mean_response")
    background = result.get("background_mean_response")
    if target is None or background is None:
        return None, malformed_count
    gap = float(target) - float(background)
    if not math.isfinite(gap) or not -1.0 <= gap <= 1.0:
        return None, malformed_count
    return gap, malformed_count


def _endpoint_rows(observations: Sequence[NaturalFactorObservation]) -> tuple[dict[str, object], ...]:
    by_seed_node: dict[tuple[int, int], list[NaturalFactorObservation]] = defaultdict(list)
    for row in observations:
        by_seed_node[(row.seed, row.node_id)].append(row)
    seeds = sorted({row.seed for row in observations})
    expected_nodes = (*PRIMARY_NODE_IDS, *DIAGNOSTIC_NODE_IDS)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for node in expected_nodes:
            subset = tuple(by_seed_node.get((seed, node), ()))
            sampling_rho = _residual_rho(subset, "sampling")
            visibility_rho = _residual_rho(subset, "visibility")
            sampling_gap, sampling_malformed = _specificity_gap(subset, "sampling")
            visibility_gap, visibility_malformed = _specificity_gap(subset, "visibility")
            endpoints = {
                PRIMARY_ENDPOINTS[0]: sampling_rho,
                PRIMARY_ENDPOINTS[1]: visibility_rho,
                PRIMARY_ENDPOINTS[2]: sampling_gap,
                PRIMARY_ENDPOINTS[3]: visibility_gap,
            }
            finite = [value for value in endpoints.values() if _finite_stat(value) is not None]
            direction = (
                float(sampling_rho + visibility_rho) / 2.0
                if sampling_rho is not None and visibility_rho is not None
                else None
            )
            rows.append(
                {
                    "seed": seed,
                    "node_id": node,
                    "endpoints": endpoints,
                    "direction": direction,
                    "malformed": sampling_malformed + visibility_malformed,
                    "complete": len(finite) == len(PRIMARY_ENDPOINTS),
                }
            )
    return tuple(rows)


def _pooled_endpoints(observations: Sequence[NaturalFactorObservation]) -> dict[str, float]:
    by_node = _endpoint_rows(observations)
    primary = [row for row in by_node if row.get("node_id") in PRIMARY_NODE_IDS]
    result: dict[str, float] = {}
    for name in PRIMARY_ENDPOINTS:
        values = [
            float(row["endpoints"][name])
            for row in primary
            if isinstance(row.get("endpoints"), Mapping)
            and _finite_stat(row["endpoints"].get(name)) is not None
        ]
        if len(values) != len(primary) or not values:
            raise ValueError("observer observations do not contain complete endpoint evidence")
        result[name] = float(sum(values) / len(values))
    return result


def _same_image_ids(
    observations: Sequence[NaturalFactorObservation],
    image_ids: Sequence[str] | None,
    image_ids_hash: str | None,
) -> tuple[tuple[str, ...], str]:
    observed = tuple(sorted({row.image_id for row in observations}))
    if not observed:
        raise ValueError("observer observations contain no image IDs")
    if image_ids is not None:
        supplied = tuple(sorted(str(item) for item in image_ids))
        if not supplied or len(set(supplied)) != len(supplied):
            raise ValueError("image IDs must be unique and non-empty")
        if supplied != observed:
            raise ValueError("observer image IDs do not match development image IDs")
    else:
        supplied = observed
    derived_hash = sha256_canonical(list(supplied))
    if image_ids_hash is not None:
        checked_hash = _require_hash(image_ids_hash, "image_ids_hash")
        if checked_hash != derived_hash:
            raise ValueError("image IDs hash mismatch")
        return supplied, checked_hash
    return supplied, derived_hash


def _audit_image_identity(audit: Mapping[str, object], image_ids: tuple[str, ...], image_hash: str) -> None:
    def walk(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        for key, nested in value.items():
            name = str(key)
            if name in {"image_ids", "development_image_ids", "observed_image_ids"}:
                if isinstance(nested, (str, bytes)) or tuple(sorted(str(item) for item in nested)) != image_ids:
                    raise ValueError("audit image IDs do not match development image IDs")
            elif name in {"image_ids_hash", "development_image_ids_hash", "image_id_sha256"}:
                if _require_hash(nested, f"audit {name}") != image_hash:
                    raise ValueError("audit image IDs hash mismatch")
            walk(nested)

    walk(audit)


def _gate_dict(gate: object) -> Mapping[str, object]:
    if isinstance(gate, FactorRepairGateDecision):
        return gate.to_dict()
    if isinstance(gate, Mapping):
        return gate
    to_dict = getattr(gate, "to_dict", None)
    if callable(to_dict) and isinstance(to_dict(), Mapping):
        return to_dict()
    raise ValueError("absolute gate must be a FactorRepairGateDecision")


@dataclass(frozen=True)
class FactorRepairEvidenceBundle(FactorRepairEvidence):
    """Immutable evidence plus its absolute development gate and raw rows."""

    absolute_gate: FactorRepairGateDecision | Mapping[str, object] | None = None
    stage: str = "development"
    checkpoint: Mapping[str, object] | None = None
    checkpoint_roles: Mapping[str, object] | None = None
    raw_observations: tuple[Mapping[str, object], ...] = ()
    endpoint_rows: tuple[Mapping[str, object], ...] = ()
    audit: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.endpoint_samples is not None:
            raise ValueError("point endpoint samples are not admissible; retain raw observations")
        if self.stage != "development":
            raise ValueError("factor repair evidence stage must be development")
        if self.absolute_gate is None:
            raise ValueError("absolute_gate is required")
        if self.checkpoint is None or self.checkpoint_roles is None or self.audit is None:
            raise ValueError("checkpoint, checkpoint_roles, and audit evidence are required")
        object.__setattr__(self, "checkpoint", _freeze_json(_jsonable(self.checkpoint)))
        object.__setattr__(self, "checkpoint_roles", _freeze_json(_jsonable(self.checkpoint_roles)))
        object.__setattr__(self, "raw_observations", tuple(_freeze_json(_jsonable(item)) for item in self.raw_observations))
        object.__setattr__(self, "endpoint_rows", tuple(_freeze_json(_jsonable(item)) for item in self.endpoint_rows))
        object.__setattr__(self, "audit", _freeze_json(_jsonable(self.audit)))

    @property
    def evidence(self) -> "FactorRepairEvidenceBundle":
        """Compatibility alias: the bundle is itself a FactorRepairEvidence."""

        return self

    @property
    def gate(self) -> FactorRepairGateDecision | Mapping[str, object]:
        return self.absolute_gate  # type: ignore[return-value]

    def __iter__(self):
        """Allow ``evidence, absolute_gate = build_factor_repair_evidence(...)``."""

        yield self
        yield self.absolute_gate

    def __getitem__(self, key: str) -> object:
        if key == "evidence":
            return self
        if key in {"absolute_gate", "gate"}:
            return self.absolute_gate
        return self.to_dict()[key]

    @property
    def canonical_payload(self) -> Mapping[str, object]:
        return canonical_evidence_payload(self)

    def verify_digest(self) -> bool:
        return sha256_canonical(self.canonical_payload) == self.evidence_sha256

    def recompute_endpoints(self, indices: Sequence[int]) -> Mapping[str, float]:
        """Recompute pooled primary endpoints for one image-cluster draw."""

        if len(indices) != len(self.image_ids):
            raise ValueError("image-cluster draw length does not match evidence image count")
        by_image: dict[str, list[object]] = defaultdict(list)
        for raw in self.raw_observations:
            observation, _ = _mapping_observation(raw)
            by_image[observation.image_id].append(observation)
        sampled: list[NaturalFactorObservation] = []
        for draw_index, index in enumerate(indices):
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 0 or index >= len(self.image_ids):
                raise ValueError("image-cluster draw index is out of bounds")
            # ``natural_factor_audit`` rejects duplicate natural identities.
            # A bootstrap draw can select one image more than once, so clone
            # each cluster with a draw-local image identity.  The values and
            # pair structure are unchanged; only the audit grouping key is
            # made unique for this resample.
            sampled.extend(
                replace(
                    observation,
                    image_id=f"{observation.image_id}\0cluster-{draw_index}",
                )
                for observation in by_image[self.image_ids[int(index)]]  # type: ignore[index]
            )
        return _pooled_endpoints(tuple(sampled))

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.canonical_payload)
        payload["evidence_sha256"] = self.evidence_sha256
        return payload


def canonical_evidence_payload(value: FactorRepairEvidenceBundle | Mapping[str, object]) -> dict[str, object]:
    """Return the canonical SHA payload without ``evidence_sha256``."""

    if isinstance(value, FactorRepairEvidenceBundle):
        gate = _gate_dict(value.absolute_gate)
        payload = {
            "schema_version": 1,
            "stage": value.stage,
            "condition": value.condition,
            "image_ids": tuple(value.image_ids),
            "image_ids_hash": value.image_ids_hash,
            "endpoints": value.endpoints,
            "absolute_gate": gate,
            "checkpoint": value.checkpoint,
            "checkpoint_roles": value.checkpoint_roles,
            "raw_observations": value.raw_observations,
            "endpoint_rows": value.endpoint_rows,
            "audit": value.audit,
            "complete": value.complete,
            "absolute_gate_passed": value.absolute_gate_passed,
        }
    else:
        payload = {str(key): item for key, item in value.items() if key != "evidence_sha256"}
    return _jsonable(payload)  # type: ignore[return-value]


def validate_shared_image_identity(*evidence: object) -> tuple[tuple[str, ...], str]:
    """Require F0/F1/F2/F3 evidence to share one exact image cluster."""

    if not evidence:
        raise ValueError("at least one evidence record is required")
    first = evidence[0]
    first_ids = first.get("image_ids") if isinstance(first, Mapping) else getattr(first, "image_ids", None)
    first_hash = first.get("image_ids_hash") if isinstance(first, Mapping) else getattr(first, "image_ids_hash", None)
    if isinstance(first_ids, (str, bytes)) or not isinstance(first_ids, (tuple, list)):
        raise ValueError("evidence image IDs are missing")
    if not isinstance(first_hash, str):
        raise ValueError("evidence image IDs hash is missing")
    canonical_ids = tuple(sorted(str(item) for item in first_ids))
    canonical_hash = _require_hash(first_hash, "evidence image_ids_hash")
    for item in evidence[1:]:
        ids = item.get("image_ids") if isinstance(item, Mapping) else getattr(item, "image_ids", None)
        image_hash = item.get("image_ids_hash") if isinstance(item, Mapping) else getattr(item, "image_ids_hash", None)
        if isinstance(ids, (str, bytes)) or not isinstance(ids, (tuple, list)):
            raise ValueError("evidence image IDs are missing")
        if tuple(sorted(str(value) for value in ids)) != canonical_ids or _require_hash(image_hash, "evidence image_ids_hash") != canonical_hash:
            raise ValueError("F0-F3 evidence image IDs mismatch")
    return canonical_ids, canonical_hash


def build_factor_repair_evidence(
    condition: str,
    stage: str = "development",
    checkpoint: object | None = None,
    observation_rows: object | None = None,
    audit_decision: object | None = None,
    *,
    checkpoint_roles: object | None = None,
    observations: object | None = None,
    audit: object | None = None,
    image_ids: Sequence[str] | None = None,
    image_ids_hash: str | None = None,
    development_image_ids: Sequence[str] | None = None,
    development_image_ids_hash: str | None = None,
    shared_image_ids: Sequence[str] | None = None,
    shared_image_ids_hash: str | None = None,
) -> FactorRepairEvidenceBundle:
    """Build immutable development evidence from existing observer/audit files.

    This function performs no model loading or inference.  It accepts only a
    ``calibration_last`` primary checkpoint and complete persisted observer
    rows, and invokes :func:`evaluate_factor_repair_gate` exactly once.
    """

    if condition not in {"F0", "F1", "F2", "F3"}:
        raise ValueError("condition must be F0, F1, F2, or F3")
    if stage != "development":
        raise ValueError("factor repair evidence is development-only")
    if checkpoint is None:
        raise ValueError("calibration checkpoint is required")
    if checkpoint_roles is None:
        raise ValueError("checkpoint_roles is required")
    if observations is not None and observation_rows is not None:
        raise ValueError("observer observations were supplied more than once")
    if audit is not None and audit_decision is not None:
        raise ValueError("audit evidence was supplied more than once")
    observations = observations if observations is not None else observation_rows
    audit = audit if audit is not None else audit_decision
    if observations is None:
        raise ValueError("observations are required")
    if image_ids is not None and development_image_ids is not None and tuple(image_ids) != tuple(development_image_ids):
        raise ValueError("development image IDs were supplied more than once")
    if image_ids_hash is not None and development_image_ids_hash is not None and image_ids_hash != development_image_ids_hash:
        raise ValueError("development image IDs hash was supplied more than once")
    if shared_image_ids is not None:
        if (image_ids is not None and tuple(image_ids) != tuple(shared_image_ids)) or (
            development_image_ids is not None and tuple(development_image_ids) != tuple(shared_image_ids)
        ):
            raise ValueError("shared image IDs do not match development image IDs")
        image_ids = shared_image_ids
    if shared_image_ids_hash is not None:
        if (image_ids_hash is not None and image_ids_hash != shared_image_ids_hash) or (
            development_image_ids_hash is not None and development_image_ids_hash != shared_image_ids_hash
        ):
            raise ValueError("shared image IDs hash does not match development image IDs hash")
        image_ids_hash = shared_image_ids_hash
    expected_ids = image_ids if image_ids is not None else development_image_ids
    expected_hash = image_ids_hash if image_ids_hash is not None else development_image_ids_hash
    checkpoint_payload, roles_payload = _resolve_checkpoint(checkpoint, checkpoint_roles)
    normalized, raw_rows = _normalise_observations(
        observations,
        checkpoint_sha256=str(checkpoint_payload["sha256"]),
        stage=stage,
    )
    ids, ids_hash = _same_image_ids(normalized, expected_ids, expected_hash)
    audit_payload_raw = _read_json(audit, "audit") if audit is not None else None
    if not isinstance(audit_payload_raw, Mapping):
        raise ValueError("complete raw audit evidence is required")
    audit_payload = dict(_jsonable(audit_payload_raw))
    _audit_image_identity(audit_payload, ids, ids_hash)
    endpoint_rows = _endpoint_rows(normalized)
    endpoints = _pooled_endpoints(normalized)
    absolute_gate = evaluate_factor_repair_gate(
        {"rows": endpoint_rows, "audit": audit_payload},
        stage="development",
    )
    gate_payload = _gate_dict(absolute_gate)
    passed = bool(gate_payload.get("passed", False))
    digest_payload = {
        "schema_version": 1,
        "stage": stage,
        "condition": condition,
        "image_ids": ids,
        "image_ids_hash": ids_hash,
        "endpoints": endpoints,
        "absolute_gate": gate_payload,
        "checkpoint": checkpoint_payload,
        "checkpoint_roles": roles_payload,
        "raw_observations": tuple(raw_rows),
        "endpoint_rows": endpoint_rows,
        "audit": audit_payload,
        "complete": True,
        "absolute_gate_passed": passed,
    }
    evidence_hash = sha256_canonical(_jsonable(digest_payload))
    return FactorRepairEvidenceBundle(
        condition=condition,
        image_ids_hash=ids_hash,
        image_ids=ids,
        endpoints=endpoints,
        evidence_sha256=evidence_hash,
        absolute_gate_passed=passed,
        complete=True,
        endpoint_samples=None,
        absolute_gate=absolute_gate,
        stage=stage,
        checkpoint=checkpoint_payload,
        checkpoint_roles=roles_payload,
        raw_observations=tuple(raw_rows),
        endpoint_rows=endpoint_rows,
        audit=audit_payload,
    )


def _gate_from_payload(value: object) -> FactorRepairGateDecision:
    if isinstance(value, FactorRepairGateDecision):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("absolute_gate must be a gate decision mapping")
    try:
        return FactorRepairGateDecision(
            passed=bool(value["passed"]),
            stage=str(value["stage"]),
            primary_nodes=tuple(value["primary_nodes"]),
            diagnostic_nodes=tuple(value["diagnostic_nodes"]),
            checks=value["checks"],
            failures=tuple(value["failures"]),
            evidence_sha256=str(value["evidence_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("absolute_gate is malformed") from error


def load_factor_repair_evidence(path: str | Path) -> FactorRepairEvidenceBundle:
    """Load and verify one persisted evidence bundle without running observer."""

    payload = _read_json(path, "factor repair evidence")
    if not isinstance(payload, Mapping):
        raise ValueError("factor repair evidence must be a JSON object")
    for forbidden in ("endpoint_samples", "per_image_endpoints", "image_endpoint_table"):
        if forbidden in payload and payload.get(forbidden) is not None:
            raise ValueError("point endpoint samples are not admissible evidence")
    expected_hash = _require_hash(payload.get("evidence_sha256"), "evidence_sha256")
    gate = _gate_from_payload(payload.get("absolute_gate", payload.get("gate")))
    checkpoint = payload.get("checkpoint")
    checkpoint_roles = payload.get("checkpoint_roles", {"primary_checkpoint": checkpoint})
    if checkpoint is None:
        raise ValueError("factor repair evidence checkpoint is missing")
    checkpoint_payload, roles_payload = _resolve_checkpoint(checkpoint, checkpoint_roles)
    raw_rows = payload.get("raw_observations", payload.get("observations"))
    endpoint_rows = payload.get("endpoint_rows")
    audit = payload.get("audit")
    if not isinstance(raw_rows, (tuple, list)) or not raw_rows:
        raise ValueError("factor repair evidence raw observations are missing")
    if not isinstance(endpoint_rows, (tuple, list)) or not endpoint_rows:
        raise ValueError("factor repair evidence endpoint rows are missing")
    if not isinstance(audit, Mapping):
        raise ValueError("factor repair evidence audit is missing")
    observations, normalized_raw = _normalise_observations(
        tuple(raw_rows), checkpoint_sha256=str(checkpoint_payload["sha256"]), stage="development"
    )
    ids = payload.get("image_ids")
    ids_hash = payload.get("image_ids_hash")
    if isinstance(ids, (str, bytes)) or not isinstance(ids, (tuple, list)):
        raise ValueError("factor repair evidence image IDs are missing")
    image_ids, image_hash = _same_image_ids(observations, tuple(str(item) for item in ids), str(ids_hash) if ids_hash is not None else None)
    audit_payload = dict(_jsonable(audit))
    _audit_image_identity(audit_payload, image_ids, image_hash)
    absolute_passed = bool(payload.get("absolute_gate_passed", gate.passed))
    if absolute_passed != gate.passed:
        raise ValueError("absolute gate pass flag mismatch")
    bundle = FactorRepairEvidenceBundle(
        condition=str(payload.get("condition", "")),
        image_ids_hash=image_hash,
        image_ids=image_ids,
        endpoints=payload.get("endpoints", {}),
        evidence_sha256=expected_hash,
        absolute_gate_passed=absolute_passed,
        complete=bool(payload.get("complete", True)),
        endpoint_samples=None,
        absolute_gate=gate,
        stage=str(payload.get("stage", "development")),
        checkpoint=checkpoint_payload,
        checkpoint_roles=roles_payload,
        raw_observations=tuple(normalized_raw),
        endpoint_rows=tuple(endpoint_rows),
        audit=audit_payload,
    )
    if not bundle.verify_digest():
        raise ValueError("factor repair evidence SHA256 mismatch")
    # The persisted rows must still describe the stored common image cluster;
    # do not silently accept a point-only replacement.
    if _jsonable(bundle.raw_observations) != _jsonable(tuple(raw_rows)):
        raise ValueError("factor repair evidence raw observations changed during load")
    return bundle


__all__ = [
    "FactorRepairEvidence",
    "FactorRepairGateDecision",
    "FactorRepairEvidenceBundle",
    "build_factor_repair_evidence",
    "canonical_evidence_payload",
    "load_factor_repair_evidence",
    "validate_shared_image_identity",
]
