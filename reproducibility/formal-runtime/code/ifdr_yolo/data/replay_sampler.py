"""Deterministic, provenance-bound replay distributions and draw journals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from uuid import uuid4


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MODES = frozenset({"M1", "M2", "M3", "factor_guided"})
_JOURNAL_SCHEMA_VERSION = 1
JOURNAL_FILENAME = "replay_journal.json"


def _validate_mapping_key(key: Any, field: str) -> str:
    if (
        not isinstance(key, str)
        or not key
        or key.strip() != key
        or any(character.isspace() for character in key)
    ):
        raise ValueError(f"{field} keys must be exact non-empty strings")
    return key


def _jsonable(value: Any) -> Any:
    """Convert immutable containers to values accepted by ``json.dumps``."""
    if isinstance(value, Mapping):
        return {
            _validate_mapping_key(key, "canonical mapping"): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("canonical payload contains a non-finite float")
    return value


def sha256_canonical(payload: Any) -> str:
    """Hash a JSON payload with the repository's stable canonical encoding."""
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_hash(value: Any, field: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA256")
    return value


def _validate_ids(image_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(image_ids, (str, bytes)):
        raise ValueError("image_ids must be a non-empty sequence of IDs")
    normalized: list[str] = []
    seen: set[str] = set()
    for image_id in image_ids:
        if not isinstance(image_id, str) or not image_id or image_id.strip() != image_id:
            raise ValueError("image IDs must be exact non-empty text")
        if any(character.isspace() for character in image_id):
            raise ValueError("image IDs must not contain whitespace")
        if image_id in seen:
            raise ValueError(f"duplicate image_id: {image_id}")
        seen.add(image_id)
        normalized.append(image_id)
    if not normalized:
        raise ValueError("replay sampler requires at least one fit image")
    return tuple(sorted(normalized))


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and non-negative")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be finite and non-negative") from error
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _finite_score(value: Any, field: str) -> float:
    number = _finite_probability(value, field)
    if number > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return number


def _normalize_scores(
    scores: Mapping[str, Any] | None,
    image_ids: Sequence[str],
    *,
    field: str,
) -> dict[str, float]:
    if scores is None:
        return {}
    if not isinstance(scores, Mapping):
        raise ValueError(f"{field} must be a mapping")
    fit_ids = set(image_ids)
    result: dict[str, float] = {}
    for image_id, value in scores.items():
        _validate_mapping_key(image_id, field)
        if image_id not in fit_ids:
            continue
        result[image_id] = _finite_score(value, f"{field}[{image_id}]")
    return dict(sorted(result.items()))


def _probability_map(
    values: Mapping[str, Any],
    image_ids: Sequence[str],
    *,
    field: str,
    allow_subset: bool = False,
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field} must be a mapping")
    expected = set(image_ids)
    result: dict[str, float] = {}
    for image_id, value in values.items():
        _validate_mapping_key(image_id, field)
        if image_id not in expected:
            raise ValueError(f"{field} contains image outside fit IDs: {image_id}")
        result[image_id] = _finite_probability(value, f"{field}[{image_id}]")
    if allow_subset:
        if not result:
            raise ValueError(f"{field} must not be empty")
    elif set(result) != expected:
        raise ValueError(f"{field} must cover every fit ID")
    total = sum(result.values())
    if not isfinite(total) or total <= 0.0:
        raise ValueError(f"{field} must have a positive finite total")
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"{field} must be normalized")
    return MappingProxyType(dict(sorted(result.items())))


def replay_eta(epoch: int) -> float:
    """Return the registered six-to-sixty epoch replay schedule."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 60:
        raise ValueError("replay epoch must be in [1, 60]")
    if epoch <= 5:
        return 0.30 * (epoch - 1) / 4.0
    if epoch <= 40:
        return 0.30
    return 0.30 * (60 - epoch) / 20.0


def mixture_probability(original: float, focus: float, epoch: int) -> float:
    """Mix one original/focus probability pair at a registered epoch."""
    eta = replay_eta(epoch)
    return (1.0 - eta) * original + eta * focus


def uniform_probabilities(image_ids: Sequence[str]) -> Mapping[str, float]:
    ids = _validate_ids(image_ids)
    probability = 1.0 / len(ids)
    return MappingProxyType({image_id: probability for image_id in ids})


@dataclass(frozen=True)
class ReplayProbabilities:
    """Normalized probability maps returned by :func:`mix_m3_probabilities`."""

    original: Mapping[str, float]
    focus: Mapping[str, float]
    final: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original",
            MappingProxyType(dict(sorted(self.original.items()))),
        )
        object.__setattr__(
            self,
            "focus",
            MappingProxyType(dict(sorted(self.focus.items()))),
        )
        object.__setattr__(
            self,
            "final",
            MappingProxyType(dict(sorted(self.final.items()))),
        )


def _normalize_for_mixture(
    values: Mapping[str, Any], *, field: str, expected_ids: set[str] | None = None
) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{field} must be a non-empty mapping")
    result = {
        _validate_mapping_key(image_id, field): _finite_probability(
            value, f"{field}[{image_id}]"
        )
        for image_id, value in values.items()
    }
    if expected_ids is not None and set(result) != expected_ids:
        raise ValueError(f"{field} must cover every fit ID")
    total = sum(result.values())
    if total <= 0.0 or not isfinite(total):
        raise ValueError(f"{field} must have a positive finite total")
    return {image_id: value / total for image_id, value in sorted(result.items())}


def mix_m3_probabilities(
    *,
    original: Mapping[str, Any],
    focus: Mapping[str, Any],
    epoch: int,
) -> ReplayProbabilities:
    """Mix a fit-uniform map with a focus map using the frozen eta schedule."""
    original_map = _normalize_for_mixture(original, field="original")
    focus_map = _normalize_for_mixture(focus, field="focus")
    expected_ids = set(original_map)
    if not set(focus_map).issubset(expected_ids):
        raise ValueError("focus probabilities must be a subset of fit IDs")
    eta = replay_eta(epoch)
    final = {
        image_id: (1.0 - eta) * original_map[image_id]
        + eta * focus_map.get(image_id, 0.0)
        for image_id in sorted(expected_ids)
    }
    total = sum(final.values())
    if not isfinite(total) or total <= 0.0:
        raise ValueError("mixed replay probabilities must have a positive total")
    final = {image_id: value / total for image_id, value in final.items()}
    return ReplayProbabilities(
        original=original_map,
        focus=focus_map,
        final=final,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _derived_source_sha256(
    image_ids: Sequence[str], cyclist_joint: Mapping[str, Any] | None
) -> str:
    scores = {}
    if cyclist_joint is not None:
        scores = {
            str(image_id): float(value)
            for image_id, value in sorted(cyclist_joint.items())
        }
    return sha256_canonical(
        {"image_ids": tuple(sorted(image_ids)), "cyclist_joint": scores}
    )


def _distribution_payload(
    *,
    mode: str,
    epoch: int,
    image_ids: Sequence[str],
    original_probabilities: Mapping[str, float],
    focus_probabilities: Mapping[str, float],
    probabilities: Mapping[str, float],
    focus_scores: Mapping[str, float],
    source_sha256: str | None,
    manifest_sha256: str | None,
    calibration_checkpoint_sha256: str | None,
    metadata_index_sha256: str | None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "epoch": epoch,
        "eta": replay_eta(epoch),
        "image_ids": tuple(sorted(image_ids)),
        "original_probabilities": tuple(sorted(original_probabilities.items())),
        "focus_probabilities": tuple(sorted(focus_probabilities.items())),
        "probabilities": tuple(sorted(probabilities.items())),
        "focus_scores": tuple(sorted(focus_scores.items())),
        "source_sha256": source_sha256,
        "manifest_sha256": manifest_sha256,
        "calibration_checkpoint_sha256": calibration_checkpoint_sha256,
        "metadata_index_sha256": metadata_index_sha256,
    }


def _digest_distribution_payload(payload: Mapping[str, Any]) -> str:
    return sha256_canonical(payload)


def digest_distribution(
    mode: str,
    epoch: int,
    probabilities: ReplayProbabilities,
    *provenance_hashes: str | None,
    source_sha256: str | None = None,
    manifest_sha256: str | None = None,
    calibration_checkpoint_sha256: str | None = None,
    metadata_index_sha256: str | None = None,
    focus_scores: Mapping[str, float] | None = None,
) -> str:
    """Compute the canonical digest for a complete replay distribution.

    Three positional provenance hashes follow the Task 3A call shape and mean
    ``manifest``, ``calibration checkpoint`` and ``metadata index``; in that
    shape the source identity is the manifest identity.  Four positional
    hashes are accepted for callers that provide all fields explicitly.
    """
    if provenance_hashes:
        if any(
            value is not None
            for value in (
                source_sha256,
                manifest_sha256,
                calibration_checkpoint_sha256,
                metadata_index_sha256,
            )
        ):
            raise TypeError("provenance hashes must be positional or keyword-only")
        if len(provenance_hashes) == 3:
            source_sha256 = manifest_sha256 = provenance_hashes[0]
            calibration_checkpoint_sha256 = provenance_hashes[1]
            metadata_index_sha256 = provenance_hashes[2]
        elif len(provenance_hashes) == 4:
            (
                source_sha256,
                manifest_sha256,
                calibration_checkpoint_sha256,
                metadata_index_sha256,
            ) = provenance_hashes
        else:
            raise TypeError("digest_distribution expects three or four hashes")
    original = dict(probabilities.original)
    focus = dict(probabilities.focus)
    final = dict(probabilities.final)
    image_ids = tuple(sorted(final))
    scores = dict(sorted((focus_scores or {}).items()))
    payload = _distribution_payload(
        mode=mode,
        epoch=epoch,
        image_ids=image_ids,
        original_probabilities=original,
        focus_probabilities=focus,
        probabilities=final,
        focus_scores=scores,
        source_sha256=source_sha256,
        manifest_sha256=manifest_sha256,
        calibration_checkpoint_sha256=calibration_checkpoint_sha256,
        metadata_index_sha256=metadata_index_sha256,
    )
    return _digest_distribution_payload(payload)


@dataclass(frozen=True)
class ReplayDistribution:
    mode: str
    epoch: int
    eta: float
    image_ids: tuple[str, ...]
    original_probabilities: Mapping[str, float]
    focus_probabilities: Mapping[str, float]
    probabilities: Mapping[str, float]
    source_sha256: str
    manifest_sha256: str | None
    calibration_checkpoint_sha256: str | None
    metadata_index_sha256: str | None
    distribution_sha256: str
    focus_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"unsupported replay mode: {self.mode}")
        ids = _validate_ids(self.image_ids)
        object.__setattr__(self, "image_ids", ids)
        expected_eta = replay_eta(self.epoch)
        if not isfinite(float(self.eta)) or abs(float(self.eta) - expected_eta) > 1e-12:
            raise ValueError("replay eta does not match the registered schedule")
        object.__setattr__(self, "eta", expected_eta)
        original = _probability_map(
            self.original_probabilities, ids, field="original_probabilities"
        )
        focus = _probability_map(
            self.focus_probabilities,
            ids,
            field="focus_probabilities",
            allow_subset=True,
        )
        final = _probability_map(self.probabilities, ids, field="probabilities")
        object.__setattr__(self, "original_probabilities", original)
        object.__setattr__(self, "focus_probabilities", focus)
        object.__setattr__(self, "probabilities", final)
        scores = _normalize_scores(self.focus_scores, ids, field="focus_scores")
        object.__setattr__(self, "focus_scores", MappingProxyType(scores))
        _validate_hash(self.source_sha256, "source_sha256")
        _validate_hash(self.manifest_sha256, "manifest_sha256", allow_none=True)
        _validate_hash(
            self.calibration_checkpoint_sha256,
            "calibration_checkpoint_sha256",
            allow_none=True,
        )
        _validate_hash(
            self.metadata_index_sha256,
            "metadata_index_sha256",
            allow_none=True,
        )
        if self.mode == "factor_guided":
            for field_name, value in (
                ("manifest_sha256", self.manifest_sha256),
                ("calibration_checkpoint_sha256", self.calibration_checkpoint_sha256),
                ("metadata_index_sha256", self.metadata_index_sha256),
            ):
                if value is None:
                    raise ValueError(f"factor-guided replay requires {field_name}")
        elif self.manifest_sha256 is not None or self.calibration_checkpoint_sha256 is not None:
            raise ValueError(
                "legacy metadata replay does not accept manifest/checkpoint hashes"
            )
        expected_digest = _digest_distribution_payload(
            _distribution_payload(
                mode=self.mode,
                epoch=self.epoch,
                image_ids=ids,
                original_probabilities=original,
                focus_probabilities=focus,
                probabilities=final,
                focus_scores=scores,
                source_sha256=self.source_sha256,
                manifest_sha256=self.manifest_sha256,
                calibration_checkpoint_sha256=self.calibration_checkpoint_sha256,
                metadata_index_sha256=self.metadata_index_sha256,
            )
        )
        if not isinstance(self.distribution_sha256, str) or self.distribution_sha256 != expected_digest:
            raise ValueError("distribution digest mismatch")

    @property
    def focus_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.focus_probabilities))


def normalize_replay_distribution(distribution: ReplayDistribution) -> dict[str, Any]:
    """Return and verify the canonical scientific identity payload."""
    if not isinstance(distribution, ReplayDistribution):
        raise TypeError("distribution must be a ReplayDistribution")
    payload = _distribution_payload(
        mode=distribution.mode,
        epoch=distribution.epoch,
        image_ids=distribution.image_ids,
        original_probabilities=distribution.original_probabilities,
        focus_probabilities=distribution.focus_probabilities,
        probabilities=distribution.probabilities,
        focus_scores=distribution.focus_scores,
        source_sha256=distribution.source_sha256,
        manifest_sha256=distribution.manifest_sha256,
        calibration_checkpoint_sha256=distribution.calibration_checkpoint_sha256,
        metadata_index_sha256=distribution.metadata_index_sha256,
    )
    if _digest_distribution_payload(payload) != distribution.distribution_sha256:
        raise ValueError("distribution digest mismatch")
    return payload


def build_replay_distribution(
    image_ids: Sequence[str],
    cyclist_joint: Mapping[str, Any] | None = None,
    *,
    mode: str,
    epoch: int,
    source_sha256: str | None = None,
    manifest_sha256: str | None = None,
    calibration_checkpoint_sha256: str | None = None,
    metadata_index_sha256: str | None = None,
    focus_scores: Mapping[str, Any] | None = None,
) -> ReplayDistribution:
    """Build one immutable M1/M2/M3 or factor-guided distribution."""
    if mode not in _MODES:
        raise ValueError(f"unsupported replay mode: {mode}")
    ids = _validate_ids(image_ids)
    replay_eta(epoch)
    if mode == "factor_guided":
        if (
            source_sha256 is None
            or manifest_sha256 is None
            or calibration_checkpoint_sha256 is None
            or metadata_index_sha256 is None
        ):
            raise ValueError("factor-guided replay requires complete provenance hashes")
    elif manifest_sha256 is not None or calibration_checkpoint_sha256 is not None:
        raise ValueError(
            "legacy metadata replay does not accept manifest/checkpoint hashes"
        )
    raw_scores = focus_scores if focus_scores is not None else cyclist_joint
    scores = _normalize_scores(raw_scores, ids, field="focus_scores")
    if source_sha256 is None:
        source_sha256 = _derived_source_sha256(ids, cyclist_joint)
    _validate_hash(source_sha256, "source_sha256")
    if metadata_index_sha256 is None:
        metadata_index_sha256 = source_sha256
    _validate_hash(metadata_index_sha256, "metadata_index_sha256")
    _validate_hash(manifest_sha256, "manifest_sha256", allow_none=True)
    _validate_hash(
        calibration_checkpoint_sha256,
        "calibration_checkpoint_sha256",
        allow_none=True,
    )
    if mode == "factor_guided":
        if manifest_sha256 is None or calibration_checkpoint_sha256 is None:
            raise ValueError("factor-guided replay requires complete provenance hashes")
        if not scores:
            raise ValueError("factor-guided replay requires focus scores")
    original = uniform_probabilities(ids)
    if mode == "M1":
        focus = original
        final = mix_m3_probabilities(
            original=original, focus=focus, epoch=epoch
        )
        focus_scores_for_record: Mapping[str, float] = {}
    elif mode == "M2":
        focus_ids = tuple(sorted(scores))
        if not focus_ids:
            raise ValueError("M2 replay requires a non-empty Cyclist pool")
        focus = uniform_probabilities(focus_ids)
        final = mix_m3_probabilities(
            original=original, focus=focus, epoch=epoch
        )
        focus_scores_for_record = {image_id: 1.0 for image_id in focus_ids}
    else:
        if not scores:
            raise ValueError(f"{mode} replay requires a non-empty focus pool")
        # Percentile clipping uses fit-only focus IDs; held-out scores never
        # affect the cap or enter the resulting probability map.
        cap = _percentile(tuple(scores.values()), 0.95)
        clipped = {
            image_id: min(value, cap) for image_id, value in scores.items()
        }
        weights = {image_id: value + 0.05 for image_id, value in clipped.items()}
        total = sum(weights.values())
        focus = {image_id: value / total for image_id, value in weights.items()}
        final = mix_m3_probabilities(
            original=original, focus=focus, epoch=epoch
        )
        focus_scores_for_record = clipped
    digest = _digest_distribution_payload(
        _distribution_payload(
            mode=mode,
            epoch=epoch,
            image_ids=ids,
            original_probabilities=final.original,
            focus_probabilities=final.focus,
            probabilities=final.final,
            focus_scores=focus_scores_for_record,
            source_sha256=source_sha256,
            manifest_sha256=manifest_sha256,
            calibration_checkpoint_sha256=calibration_checkpoint_sha256,
            metadata_index_sha256=metadata_index_sha256,
        )
    )
    return ReplayDistribution(
        mode=mode,
        epoch=epoch,
        eta=replay_eta(epoch),
        image_ids=ids,
        original_probabilities=final.original,
        focus_probabilities=final.focus,
        probabilities=final.final,
        source_sha256=source_sha256,
        manifest_sha256=manifest_sha256,
        calibration_checkpoint_sha256=calibration_checkpoint_sha256,
        metadata_index_sha256=metadata_index_sha256,
        distribution_sha256=digest,
        focus_scores=focus_scores_for_record,
    )


def deterministic_choice(
    probabilities: Mapping[str, float], *, key: Any
) -> tuple[str, float]:
    """Select from a probability map without mutable RNG state."""
    normalized = _normalize_for_mixture(probabilities, field="probabilities")
    token = sha256_canonical(key)
    unit = int(token, 16) / float(1 << 256)
    cumulative = 0.0
    items = tuple(sorted(normalized.items()))
    for image_id, probability in items:
        cumulative += probability
        if unit < cumulative:
            return image_id, probability
    return items[-1]


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("replay seed must be an integer")
    return seed


class ReplayDrawJournal:
    """Atomic, resumable exactly-once draw journal for one distribution."""

    def __init__(
        self,
        root: Path,
        *,
        seed: int,
        distribution: ReplayDistribution,
        class_counts: Mapping[str, Mapping[str, int]] | None = None,
        records: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    ) -> None:
        self._root = Path(root)
        self._journal_path = self._root / JOURNAL_FILENAME
        self._seed = _validate_seed(seed)
        self._distribution = distribution
        self._class_counts = self._normalize_class_counts(class_counts)
        unknown_ids = set(self._class_counts).difference(self._distribution.image_ids)
        if unknown_ids:
            raise ValueError(
                "class_counts contains image ID outside the distribution: "
                + ", ".join(sorted(unknown_ids))
            )
        self._identity = self._build_identity()
        self._records: dict[tuple[int, int], dict[str, Any]] = {
            key: dict(value) for key, value in (records or {}).items()
        }

    @property
    def root(self) -> Path:
        return self._root

    @property
    def journal_path(self) -> Path:
        return self._journal_path

    @property
    def state_path(self) -> Path:
        return self._journal_path

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def distribution(self) -> ReplayDistribution:
        return self._distribution

    def _build_identity(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "distribution": normalize_replay_distribution(self.distribution),
            "distribution_sha256": self.distribution.distribution_sha256,
            "manifest_sha256": self.distribution.manifest_sha256,
            "calibration_checkpoint_sha256": self.distribution.calibration_checkpoint_sha256,
            "metadata_index_sha256": self.distribution.metadata_index_sha256,
            "class_counts": tuple(
                (image_id, tuple(sorted(counts.items())))
                for image_id, counts in sorted(self._class_counts.items())
            ),
        }

    @staticmethod
    def _normalize_class_counts(
        class_counts: Mapping[str, Mapping[str, int]] | None,
    ) -> dict[str, dict[str, int]]:
        if class_counts is None:
            return {}
        if not isinstance(class_counts, Mapping):
            raise ValueError("class_counts must be a mapping")
        result: dict[str, dict[str, int]] = {}
        for image_id, counts in class_counts.items():
            _validate_mapping_key(image_id, "class_counts")
            if not isinstance(counts, Mapping):
                raise ValueError("class_counts must map image IDs to mappings")
            normalized: dict[str, int] = {}
            for class_name, count in counts.items():
                _validate_mapping_key(class_name, "class_counts")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("realized class counts must be non-negative integers")
                normalized[class_name] = count
            result[image_id] = dict(sorted(normalized.items()))
        return result

    @staticmethod
    def _state_from_bytes(content: bytes) -> dict[str, Any]:
        value = json.loads(content.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("replay journal state must be an object")
        if value.get("schema_version") != _JOURNAL_SCHEMA_VERSION:
            raise ValueError("unsupported replay journal schema")
        if not isinstance(value.get("scientific_identity"), dict):
            raise ValueError("replay journal scientific identity is missing")
        if not isinstance(value.get("draws"), list):
            raise ValueError("replay journal draws must be a list")
        return value

    def _restore_bytes(self, content: bytes) -> None:
        temporary = self.journal_path.with_name(
            f".{self.journal_path.name}.recover-{uuid4().hex}.tmp"
        )
        try:
            temporary.write_bytes(content)
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, self.journal_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_state(self) -> dict[str, Any]:
        backup = self.journal_path.with_name(self.journal_path.name + ".bak")
        candidates: list[tuple[Path, bytes, dict[str, Any]]] = []
        errors: list[Exception] = []
        for path in (self.journal_path, backup):
            if not path.exists():
                continue
            content = path.read_bytes()
            try:
                candidates.append((path, content, self._state_from_bytes(content)))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                errors.append(error)
        if not candidates:
            if errors:
                raise errors[0]
            raise FileNotFoundError(self.journal_path)

        matching = [
            candidate
            for candidate in candidates
            if sha256_canonical(candidate[2]["scientific_identity"])
            == sha256_canonical(self._identity)
        ]
        if not matching:
            raise ValueError("scientific identity mismatch")

        validated: list[
            tuple[Path, bytes, dict[str, Any], dict[tuple[int, int], dict[str, Any]]]
        ] = []
        for path, content, state in matching:
            try:
                records = self._load_records(state)
            except Exception:
                # A matching replica with a semantic or duplicate-record error
                # must never be used as a recovery source.
                raise
            validated.append((path, content, state, records))

        if len(validated) > 1:
            first_records = validated[0][3]
            second_records = validated[1][3]

            def _is_subset(
                smaller: Mapping[tuple[int, int], Mapping[str, Any]],
                larger: Mapping[tuple[int, int], Mapping[str, Any]],
            ) -> bool:
                if not set(smaller).issubset(larger):
                    return False
                return all(
                    _jsonable(smaller[key]) == _jsonable(larger[key])
                    for key in smaller
                )

            if not (
                _is_subset(first_records, second_records)
                or _is_subset(second_records, first_records)
            ):
                raise ValueError("replay journal replicas disagree")

        validated.sort(
            key=lambda candidate: (
                len(candidate[3]),
                candidate[0] == self.journal_path,
            ),
            reverse=True,
        )
        selected_path, selected_content, selected_state, _ = validated[0]
        if selected_path != self.journal_path:
            self._restore_bytes(selected_content)
        return selected_state

    def _assert_identity(self, state: Mapping[str, Any]) -> None:
        if sha256_canonical(state["scientific_identity"]) != sha256_canonical(self._identity):
            raise ValueError("scientific identity mismatch")

    def _record_for_key(self, *, epoch: int, draw_index: int) -> dict[str, Any]:
        image_id, probability = deterministic_choice(
            self.distribution.probabilities,
            key=(
                self.seed,
                epoch,
                draw_index,
                self.distribution.distribution_sha256,
                self.distribution.manifest_sha256,
                self.distribution.calibration_checkpoint_sha256,
                self.distribution.metadata_index_sha256,
            ),
        )
        return {
            "epoch": epoch,
            "draw_index": draw_index,
            "image_id": image_id,
            "probability": probability,
            "realized_image_count": 1,
            "realized_class_counts": dict(self._class_counts.get(image_id, {})),
        }

    @staticmethod
    def _record_key(record: Mapping[str, Any]) -> tuple[int, int]:
        epoch = record.get("epoch")
        draw_index = record.get("draw_index")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 60:
            raise ValueError("journal draw epoch must be in [1, 60]")
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise ValueError("journal draw_index must be a non-negative integer")
        return epoch, draw_index

    def _validate_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise ValueError("journal draw must be an object")
        key = self._record_key(record)
        if key[0] != self.distribution.epoch:
            raise ValueError("journal draw epoch does not match distribution epoch")
        expected = self._record_for_key(epoch=key[0], draw_index=key[1])
        if record.get("image_id") != expected["image_id"]:
            raise ValueError("draw record conflicts with deterministic choice")
        try:
            probability = float(record.get("probability"))
        except (TypeError, ValueError):
            raise ValueError("draw probability is invalid") from None
        if not isfinite(probability) or abs(probability - expected["probability"]) > 1e-12:
            raise ValueError("draw record probability conflict")
        if record.get("realized_image_count") != 1:
            raise ValueError("draw realized image count must be one")
        counts = record.get("realized_class_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("draw realized class counts are missing")
        normalized = self._normalize_class_counts({"_": counts})["_"]
        if normalized != expected["realized_class_counts"]:
            raise ValueError("draw realized class counts conflict")
        return {
            "epoch": key[0],
            "draw_index": key[1],
            "image_id": expected["image_id"],
            "probability": expected["probability"],
            "realized_image_count": 1,
            "realized_class_counts": normalized,
        }

    def _load_records(self, state: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
        self._assert_identity(state)
        records: dict[tuple[int, int], dict[str, Any]] = {}
        raw_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
        for raw_record in state["draws"]:
            if not isinstance(raw_record, Mapping):
                raise ValueError("journal draw must be an object")
            raw_key = self._record_key(raw_record)
            prior_raw = raw_by_key.get(raw_key)
            if prior_raw is not None and _jsonable(prior_raw) != _jsonable(raw_record):
                raise ValueError("conflicting duplicate draw record")
            raw_by_key[raw_key] = raw_record
            record = self._validate_record(raw_record)
            key = (record["epoch"], record["draw_index"])
            prior = records.get(key)
            if prior is not None and prior != record:
                raise ValueError("conflicting duplicate draw record")
            records[key] = record
        return records

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _JOURNAL_SCHEMA_VERSION,
            "scientific_identity": self._identity,
            "draws": [
                self._records[key]
                for key in sorted(self._records)
            ],
        }

    def _fsync_directory(self) -> None:
        try:
            descriptor = os.open(str(self.root), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _atomic_replace(path: Path, content: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _atomic_write(self) -> None:
        content = (
            json.dumps(
                _jsonable(self._state_payload()),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        backup = self.journal_path.with_name(self.journal_path.name + ".bak")
        self._atomic_replace(backup, content)
        self._atomic_replace(self.journal_path, content)
        self._fsync_directory()

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        seed: int,
        distribution: ReplayDistribution,
        class_counts: Mapping[str, Mapping[str, int]] | None = None,
    ) -> "ReplayDrawJournal":
        root = Path(root)
        if root.exists() and not root.is_dir():
            raise ValueError(f"journal root is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        journal = cls(
            root,
            seed=seed,
            distribution=distribution,
            class_counts=class_counts,
        )
        if journal.journal_path.exists() or journal.journal_path.with_name(journal.journal_path.name + ".bak").exists():
            state = journal._read_state()
            journal._records = journal._load_records(state)
            return journal
        journal._atomic_write()
        return journal

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        seed: int,
        distribution: ReplayDistribution,
        class_counts: Mapping[str, Mapping[str, int]] | None = None,
    ) -> "ReplayDrawJournal":
        root = Path(root)
        journal = cls(
            root,
            seed=seed,
            distribution=distribution,
            class_counts=class_counts,
        )
        state = journal._read_state()
        journal._records = journal._load_records(state)
        return journal

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(deepcopy(self._records[key]) for key in sorted(self._records))

    @property
    def scientific_identity(self) -> Mapping[str, Any]:
        return deepcopy(self._identity)

    def draw(self, *, epoch: int, draw_index: int) -> Mapping[str, Any]:
        if epoch != self.distribution.epoch:
            raise ValueError("draw epoch must match distribution epoch")
        key = self._record_key({"epoch": epoch, "draw_index": draw_index})
        expected = self._record_for_key(epoch=key[0], draw_index=key[1])
        prior = self._records.get(key)
        if prior is not None:
            if prior != expected:
                raise ValueError("conflicting duplicate draw record")
            return deepcopy(prior)
        self._records[key] = expected
        try:
            self._atomic_write()
        except Exception:
            self._records.pop(key, None)
            raise
        return deepcopy(expected)

    def draw_epoch(self, *, epoch: int, fit_count: int | None = None) -> list[Mapping[str, Any]]:
        replay_eta(epoch)
        count = len(self.distribution.image_ids) if fit_count is None else fit_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("fit_count must be a non-negative integer")
        if count != len(self.distribution.image_ids):
            raise ValueError("fit_count must equal the fit image count")
        return [self.draw(epoch=epoch, draw_index=index) for index in range(count)]

    sample_epoch = draw_epoch


__all__ = [
    "JOURNAL_FILENAME",
    "ReplayDistribution",
    "ReplayDrawJournal",
    "ReplayProbabilities",
    "build_replay_distribution",
    "deterministic_choice",
    "digest_distribution",
    "mix_m3_probabilities",
    "mixture_probability",
    "normalize_replay_distribution",
    "replay_eta",
    "sha256_canonical",
    "uniform_probabilities",
]
