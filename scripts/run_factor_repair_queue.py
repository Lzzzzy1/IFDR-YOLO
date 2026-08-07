"""A fail-closed, one-process queue for registered factor-repair jobs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from ifdr_yolo.data.replay_sampler import sha256_canonical
from ifdr_yolo.eval.factor_repair_gate import (
    FactorRepairEvidence,
    FactorRepairSelectionDecision,
    digest_selection_decision,
)
from ifdr_yolo.experiments.run_store import atomic_write_json
from scripts.train_factor_repair import (
    CALIBRATION_CONDITIONS,
    DIAGNOSTIC_CHECKPOINT,
    PRIMARY_CHECKPOINT,
    CHECKPOINT_ROLE,
    ProcessLock,
    file_sha256,
)


DEFAULT_JOBS = (
    "F0-calibration",
    "F1-calibration",
    "F2-calibration",
    "F3-calibration",
    "F0-adaptation",
    "selected-repair-adaptation",
)
_JOB_RE = re.compile(r"^(F[0-3])-(calibration|adaptation)$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_STATES = {"pending", "running", "complete", "blocked", "failed"}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    if hasattr(value, "__dict__"):
        converted = vars(value)
        if isinstance(converted, Mapping):
            return converted
    raise ValueError(f"{field} must be a mapping")


def _hash(value: object, field: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or _HASH_RE.fullmatch(value.lower()) is None:
        raise ValueError(f"{field} must be a 64-hex SHA256")
    return value.lower()


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical(value.to_dict())
    return repr(value)


def _condition_from_job(name: str) -> str:
    match = _JOB_RE.fullmatch(name)
    if not match:
        raise ValueError(f"unregistered factor-repair job: {name!r}")
    return match.group(1)


def _checkpoint(artifacts: Mapping[str, object], role: str) -> Mapping[str, object] | None:
    aliases = {
        "primary": ("primary_checkpoint", "checkpoint", "calibration_checkpoint"),
        "diagnostic": ("diagnostic_checkpoint", "best_checkpoint"),
    }
    for alias in aliases[role]:
        value = artifacts.get(alias)
        if isinstance(value, Mapping):
            return value
        if value is not None:
            try:
                return _as_mapping(value, f"{role} checkpoint")
            except ValueError:
                pass
    return None


def _artifact_field(artifacts: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in artifacts:
            return artifacts[name]
    manifest = artifacts.get("manifest")
    if isinstance(manifest, Mapping):
        for name in names:
            if name in manifest:
                return manifest[name]
    return None


def _normalize_artifacts(job: str, raw: Mapping[str, object]) -> dict[str, object]:
    artifacts = deepcopy(dict(raw))
    if not _has_explicit_gate(artifacts):
        raise ValueError("completed artifacts must carry an explicit gate result")
    if job == "selected-repair-adaptation":
        condition = artifacts.get("condition")
        if condition not in CALIBRATION_CONDITIONS:
            raise ValueError("selected adaptation artifact condition is required")
    else:
        condition = _condition_from_job(job)
    supplied_condition = artifacts.get("condition")
    if supplied_condition is not None and supplied_condition != condition:
        raise ValueError("artifact condition mismatch")
    artifacts["condition"] = condition

    gate_passed = _artifact_gate_passed(artifacts)
    primary = _checkpoint(artifacts, "primary")
    diagnostic = _checkpoint(artifacts, "diagnostic")
    if primary is None:
        raise ValueError("primary checkpoint artifact is required")
    if primary is not None:
        path = str(primary.get("path", ""))
        role = primary.get("role")
        checkpoint_role = primary.get("checkpoint_role")
        if (
            DIAGNOSTIC_CHECKPOINT in Path(path).name
            or role == "best"
            or checkpoint_role == "best"
        ):
            raise ValueError("best.pt cannot be used as a primary checkpoint")
        if role != "primary":
            raise ValueError("primary role must be primary")
        expected_checkpoint_role = (
            "task_adaptation_last" if job.endswith("adaptation") else CHECKPOINT_ROLE
        )
        if checkpoint_role != expected_checkpoint_role:
            raise ValueError(
                f"{job} checkpoint role must be {expected_checkpoint_role}"
            )
        expected = primary.get("sha256", primary.get("hash"))
        if expected is None:
            raise ValueError("primary checkpoint sha256 is required")
        _hash(expected, "primary checkpoint sha256", required=True)
        if not path:
            raise ValueError("primary checkpoint path is required")
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise ValueError("primary checkpoint is missing or empty")
        actual = file_sha256(candidate)
        if actual != str(expected).lower():
            raise ValueError("primary checkpoint hash mismatch")
    if diagnostic is not None:
        path = str(diagnostic.get("path", ""))
        diagnostic_role = diagnostic.get("role")
        if diagnostic_role != "diagnostic":
            raise ValueError("diagnostic role must be diagnostic")
        if DIAGNOSTIC_CHECKPOINT not in Path(path).name:
            raise ValueError("diagnostic checkpoint must be best.pt")
        expected = diagnostic.get("sha256", diagnostic.get("hash"))
        if expected is None:
            raise ValueError("diagnostic checkpoint sha256 is required")
        _hash(expected, "diagnostic checkpoint sha256", required=True)
        if not path:
            raise ValueError("diagnostic checkpoint path is required")
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise ValueError("diagnostic checkpoint is missing or empty")
        actual = file_sha256(candidate)
        if actual != str(expected).lower():
            raise ValueError("diagnostic checkpoint hash mismatch")
    if diagnostic is None:
        raise ValueError("diagnostic checkpoint artifact is required")

    manifest = artifacts.get("manifest")
    if not gate_passed and job in {"F1-calibration", "F2-calibration", "F3-calibration"}:
        # Failed candidate runs are retained as negative evidence only.  They
        # must never publish a promotable factor-guided manifest.
        manifest = None
        artifacts.pop("manifest", None)
    elif not isinstance(manifest, Mapping):
        raise ValueError("completed artifacts require a complete manifest")

    if isinstance(manifest, Mapping):
        manifest_role = manifest.get("checkpoint_role")
        manifest_path = str(manifest.get("checkpoint_path", manifest.get("path", "")))
        if manifest_role == "best" or DIAGNOSTIC_CHECKPOINT in Path(manifest_path).name:
            raise ValueError("best.pt manifest is not admissible")
        expected_checkpoint_role = (
            "task_adaptation_last" if job.endswith("adaptation") else CHECKPOINT_ROLE
        )
        if manifest_role != expected_checkpoint_role:
            raise ValueError(
                f"manifest checkpoint role must be {expected_checkpoint_role}"
            )
        primary_checkpoint = _checkpoint(artifacts, "primary")
        primary_hash = None if primary_checkpoint is None else primary_checkpoint.get("sha256", primary_checkpoint.get("hash"))
        manifest_checkpoint = _hash(manifest.get("checkpoint_sha256"), "manifest checkpoint_sha256", required=True)
        if primary_hash is None or manifest_checkpoint != _hash(primary_hash, "primary checkpoint sha256", required=True):
            raise ValueError("manifest checkpoint hash mismatch")
        required_manifest_hashes = (
            "metadata_index_sha256",
            "semantic_state_sha256",
            "fit_ids_sha256",
            "image_ids_sha256",
            "manifest_sha256",
        )
        for name in required_manifest_hashes:
            _hash(manifest.get(name), f"manifest.{name}", required=True)
            direct = artifacts.get(name)
            if direct is not None and _hash(direct, name, required=True) != manifest[name]:
                raise ValueError(f"manifest {name} mismatch")
        manifest_digest = str(manifest["manifest_sha256"]).lower()
        digest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if sha256_canonical(digest_payload) != manifest_digest:
            raise ValueError("manifest SHA256 mismatch")

    # Validate every supplied identity field, including nested manifest fields.
    hash_aliases = (
        "checkpoint_sha256", "calibration_checkpoint_sha256", "semantic_state_sha256",
        "metadata_index_sha256", "metadata_sha256", "fit_ids_sha256", "image_ids_sha256",
        "expected_object_ids_sha256", "manifest_sha256", "evidence_sha256",
        "factor_evidence_sha256", "image_id_sha256",
    )
    for name in hash_aliases:
        if name in artifacts and artifacts[name] is not None:
            _hash(artifacts[name], name, required=True)
    if isinstance(manifest, Mapping):
        for name in hash_aliases:
            if name in manifest and manifest[name] is not None:
                _hash(manifest[name], f"manifest.{name}", required=True)
    return artifacts


def _artifact_gate_passed(artifacts: Mapping[str, object]) -> bool:
    for name in ("gate_passed", "absolute_gate_passed", "passed"):
        if name in artifacts:
            return bool(artifacts[name])
    gate = artifacts.get("gate")
    if isinstance(gate, Mapping) and "passed" in gate:
        return bool(gate["passed"])
    for name in ("gate_decision", "factor_gate"):
        candidate = artifacts.get(name)
        if candidate is not None and hasattr(candidate, "passed"):
            return bool(getattr(candidate, "passed"))
    return True


def _has_explicit_gate(artifacts: Mapping[str, object]) -> bool:
    if any(name in artifacts for name in ("gate_passed", "absolute_gate_passed", "passed")):
        return True
    gate = artifacts.get("gate")
    if isinstance(gate, Mapping) and "passed" in gate:
        return True
    return any(
        artifacts.get(name) is not None and hasattr(artifacts.get(name), "passed")
        for name in ("gate_decision", "factor_gate")
    )


def _artifact_evidence_hash(artifacts: Mapping[str, object]) -> str | None:
    return _hash(
        _artifact_field(artifacts, "evidence_sha256", "factor_evidence_sha256"),
        "artifact evidence_sha256",
    )


def _artifact_checkpoint_hash(artifacts: Mapping[str, object]) -> str | None:
    primary = _checkpoint(artifacts, "primary")
    value = None if primary is None else primary.get("sha256", primary.get("hash"))
    value = value or _artifact_field(artifacts, "checkpoint_sha256", "calibration_checkpoint_sha256")
    return _hash(value, "artifact checkpoint_sha256")


def _artifact_manifest(artifacts: Mapping[str, object]) -> Mapping[str, object] | None:
    manifest = artifacts.get("manifest")
    return manifest if isinstance(manifest, Mapping) else None


def validate_factor_guided_manifest(
    manifest: Mapping[str, object],
    *,
    condition: str | None = None,
    expected: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Validate an immutable factor-guided manifest and its bound hashes."""

    value = _as_mapping(manifest, "factor-guided manifest")
    actual_condition = value.get("condition")
    if not isinstance(actual_condition, str) or actual_condition not in CALIBRATION_CONDITIONS:
        raise ValueError("manifest condition must be F0, F1, F2, or F3")
    if condition is not None and actual_condition != condition:
        raise ValueError("manifest condition mismatch")
    role = value.get("checkpoint_role", value.get("role"))
    path = str(value.get("checkpoint_path", value.get("path", "")))
    if role != CHECKPOINT_ROLE or Path(path).name == DIAGNOSTIC_CHECKPOINT:
        raise ValueError("manifest must bind calibration_last, never best.pt")
    for name in ("checkpoint_sha256", "fit_ids_sha256", "metadata_index_sha256", "expected_object_ids_sha256", "manifest_sha256"):
        if name in value:
            _hash(value[name], f"manifest.{name}", required=True)
    if value.get("manifest_sha256") is not None:
        payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
        if sha256_canonical(payload) != value["manifest_sha256"]:
            raise ValueError("learned-factor manifest SHA256 mismatch")
    for name, expected_value in (expected or {}).items():
        if name in value and value[name] != expected_value:
            raise ValueError(f"manifest {name} mismatch")
    return value


def _validate_decision_type(decision: object) -> FactorRepairSelectionDecision:
    if not isinstance(decision, FactorRepairSelectionDecision):
        raise ValueError("selection decision object is required; manual condition strings are forbidden")
    if not decision.verify_digest():
        raise ValueError("selection decision SHA256 mismatch")
    for field, value in (
        ("reference_evidence_sha256", decision.reference_evidence_sha256),
        ("selected_evidence_sha256", decision.selected_evidence_sha256),
        ("decision_sha256", decision.decision_sha256),
    ):
        _hash(value, field, required=True)
    # The immutable gate implementation represents endpoint tables as sorted
    # tuples; a mutable mapping supplied by an untrusted caller is rejected.
    if not isinstance(decision.endpoint_table, tuple):
        raise ValueError("selection endpoint table must be immutable")
    return decision


def _rebuild_selection_evidence(
    artifacts: Mapping[str, object], expected_condition: str
) -> FactorRepairEvidence:
    """Rebuild Task8's immutable evidence record from persisted artifacts."""

    payload = artifacts.get("evidence_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("selection evidence payload is required")
    condition = payload.get("condition", artifacts.get("condition"))
    if condition != expected_condition:
        raise ValueError("selection evidence condition mismatch")
    raw_ids = artifacts.get("image_ids", payload.get("image_ids"))
    if isinstance(raw_ids, (str, bytes)) or raw_ids is None:
        raise ValueError("selection evidence image IDs are required")
    image_ids = tuple(sorted(str(item) for item in raw_ids))
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("selection evidence image IDs are duplicated or empty")
    payload_ids = payload.get("image_ids")
    if payload_ids is None or tuple(sorted(str(item) for item in payload_ids)) != image_ids:
        raise ValueError("selection evidence image IDs mismatch")
    endpoints = artifacts.get(
        "endpoint_table", artifacts.get("endpoints", payload.get("endpoints"))
    )
    if not isinstance(endpoints, Mapping):
        raise ValueError("selection evidence endpoints are required")
    payload_endpoints = payload.get("endpoints")
    if not isinstance(payload_endpoints, Mapping) or dict(payload_endpoints) != dict(endpoints):
        raise ValueError("selection evidence endpoints mismatch")
    evidence_hash = _artifact_evidence_hash(artifacts)
    if evidence_hash is None:
        raise ValueError("selection evidence SHA256 is required")
    if sha256_canonical(payload) != evidence_hash:
        raise ValueError("artifact evidence canonical SHA256 mismatch")
    image_ids_hash = payload.get(
        "image_ids_hash", artifacts.get("image_ids_hash", sha256_canonical(image_ids))
    )
    return FactorRepairEvidence(
        condition=str(condition),
        image_ids_hash=str(image_ids_hash),
        image_ids=image_ids,
        endpoints=endpoints,
        evidence_sha256=evidence_hash,
        absolute_gate_passed=_artifact_gate_passed(artifacts),
        complete=bool(payload.get("complete", artifacts.get("complete", True))),
    )


def _validate_selection_artifacts(
    decision: FactorRepairSelectionDecision,
    f0: Mapping[str, object],
    selected: Mapping[str, object],
) -> None:
    f0_evidence = _rebuild_selection_evidence(f0, "F0")
    selected_evidence = _rebuild_selection_evidence(selected, decision.selected_condition)
    if not f0_evidence.complete or not selected_evidence.complete:
        raise ValueError("selection evidence must be complete")
    f0_hash = _artifact_evidence_hash(f0)
    selected_hash = _artifact_evidence_hash(selected)
    if f0_hash != decision.reference_evidence_sha256:
        raise ValueError("F0 evidence SHA256 does not match selection decision")
    if selected_hash != decision.selected_evidence_sha256:
        raise ValueError("selected evidence SHA256 does not match selection decision")
    f0_checkpoint = _artifact_checkpoint_hash(f0)
    selected_checkpoint = _artifact_checkpoint_hash(selected)
    if f0_checkpoint is None or selected_checkpoint is None:
        raise ValueError("calibration checkpoint identity is required")
    if f0_checkpoint == selected_checkpoint:
        raise ValueError("F0 and selected calibration checkpoints must be distinct")
    f0_semantic = _hash(_artifact_field(f0, "semantic_state_sha256"), "F0 semantic_state_sha256", required=True)
    selected_semantic = _hash(_artifact_field(selected, "semantic_state_sha256"), "selected semantic_state_sha256", required=True)
    if f0_semantic == selected_semantic:
        raise ValueError("F0 and selected semantic state hashes must be distinct")
    shared_hash_fields = (
        ("metadata_index_sha256", "metadata_sha256"),
        ("fit_ids_sha256", "image_ids_sha256"),
    )
    for aliases in shared_hash_fields:
        f0_value = _hash(_artifact_field(f0, *aliases), aliases[0], required=True)
        selected_value = _hash(_artifact_field(selected, *aliases), aliases[0], required=True)
        if f0_value != selected_value:
            raise ValueError(f"{aliases[0]} mismatch between F0 and selected artifacts")
    for name in ("image_ids_sha256", "fit_ids_sha256"):
        f0_value = _hash(_artifact_field(f0, name), name, required=True)
        selected_value = _hash(_artifact_field(selected, name), name, required=True)
        if f0_value != selected_value:
            raise ValueError(f"{name} mismatch between F0 and selected artifacts")
    for name, artifacts, expected_condition in (
        ("F0", f0, "F0"),
        (decision.selected_condition, selected, decision.selected_condition),
    ):
        if artifacts.get("condition") not in {None, expected_condition}:
            raise ValueError(f"{name} artifact condition mismatch")
        primary = _checkpoint(artifacts, "primary")
        if primary is None:
            raise ValueError(f"{name} calibration primary checkpoint is missing")
        role = primary.get("role")
        checkpoint_role = primary.get("checkpoint_role")
        path = str(primary.get("path", ""))
        if role != "primary" or checkpoint_role != CHECKPOINT_ROLE or Path(path).name == DIAGNOSTIC_CHECKPOINT:
            raise ValueError(f"{name} calibration must use calibration_last")
        manifest = _artifact_manifest(artifacts)
        if manifest is None:
            raise ValueError("validated learned-factor manifest is required")
        if manifest.get("condition") not in {None, expected_condition}:
            raise ValueError("learned-factor manifest condition mismatch")
        manifest_role = manifest.get("checkpoint_role", manifest.get("role", CHECKPOINT_ROLE))
        if manifest_role != CHECKPOINT_ROLE:
            raise ValueError("learned-factor manifest role mismatch")
        manifest_hash = _hash(manifest.get("checkpoint_sha256"), "manifest checkpoint_sha256", required=True)
        if manifest_hash != _artifact_checkpoint_hash(artifacts):
            raise ValueError("learned-factor manifest checkpoint hash mismatch")
        manifest_digest = _hash(manifest.get("manifest_sha256"), "manifest_sha256", required=True)
        digest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if sha256_canonical(digest_payload) != manifest_digest:
            raise ValueError("learned-factor manifest SHA256 mismatch")
        metadata_hash = manifest.get("metadata_index_sha256", manifest.get("metadata_sha256"))
        artifact_metadata = _artifact_field(artifacts, "metadata_index_sha256", "metadata_sha256")
        if metadata_hash is not None and artifact_metadata is not None and metadata_hash != artifact_metadata:
            raise ValueError("learned-factor manifest metadata hash mismatch")
    f0_endpoints = dict(decision.endpoint_table["F0"])
    selected_endpoints = dict(decision.endpoint_table[decision.selected_condition])
    def _image_ids(artifacts: Mapping[str, object]) -> tuple[str, ...]:
        value = artifacts.get("image_ids")
        if value is None:
            payload = artifacts.get("evidence_payload", artifacts.get("evidence"))
            value = payload.get("image_ids") if isinstance(payload, Mapping) else None
        if isinstance(value, (str, bytes)) or value is None:
            raise ValueError("selection image IDs are required")
        result = tuple(sorted(str(item) for item in value))
        if not result or len(set(result)) != len(result):
            raise ValueError("selection image IDs are duplicated or empty")
        return result

    f0_image_ids = _image_ids(f0)
    selected_image_ids = _image_ids(selected)
    if f0_image_ids != selected_image_ids:
        raise ValueError("selection image IDs mismatch")
    for artifacts, expected in ((f0, f0_endpoints), (selected, selected_endpoints)):
        endpoints = artifacts.get("endpoint_table", artifacts.get("endpoints"))
        if not isinstance(endpoints, Mapping) or set(endpoints) != set(expected):
            raise ValueError("selection endpoint evidence is incomplete")
        for key, value in expected.items():
            if float(endpoints[key]) != float(value):
                raise ValueError("selection endpoint evidence mismatch")


@dataclass
class QueueJob:
    name: str
    state: str = "pending"
    artifacts: dict[str, object] | None = None
    error: str | None = None


class FactorRepairQueue:
    """Persistent state machine with one GPU lock and no auto-promotion."""

    def __init__(self, root: Path, jobs: Mapping[str, Mapping[str, object]], *, selected_condition: str | None = None, decision: object | None = None, identity: Mapping[str, object] | None = None):
        self.root = Path(root).expanduser().resolve()
        self.state_path = self.root / "queue_state.json"
        self.lock = ProcessLock(self.root / ".gpu.lock")
        self.jobs = {
            name: QueueJob(name=name, state=str(payload.get("state", "pending")), artifacts=deepcopy(payload.get("artifacts")), error=payload.get("error"))
            for name, payload in jobs.items()
        }
        self.selected_condition = selected_condition
        self.selection_decision = decision
        self.identity = deepcopy(dict(identity or {}))

    @classmethod
    def create(cls, root: Path, jobs: Sequence[str] = DEFAULT_JOBS, *, identity: Mapping[str, object] | None = None) -> "FactorRepairQueue":
        root = Path(root).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            state_path = root / "queue_state.json"
            if state_path.is_file():
                existing = cls.load(root)
                if (
                    not isinstance(identity, Mapping)
                    or not identity
                    or not existing.identity
                    or _canonical(identity) != _canonical(existing.identity)
                ):
                    raise ValueError("queue create identity mismatch")
                return existing
            raise FileExistsError(f"queue directory already exists: {root}")
        root.mkdir(parents=True, exist_ok=True)
        names = tuple(jobs)
        if not names or len(set(names)) != len(names):
            raise ValueError("queue jobs must be a non-empty unique sequence")
        for name in names:
            _condition_from_job(name) if name != "selected-repair-adaptation" else None
            if name != "selected-repair-adaptation" and not _JOB_RE.fullmatch(name):
                raise ValueError(f"unregistered factor-repair job: {name!r}")
        queue = cls(root, {name: {"state": "pending"} for name in names}, identity=identity)
        queue._persist()
        return queue

    @classmethod
    def load(cls, root: Path) -> "FactorRepairQueue":
        path = Path(root).expanduser().resolve() / "queue_state.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("queue state must be a mapping")
        jobs = payload.get("jobs")
        if not isinstance(jobs, Mapping):
            raise ValueError("queue state jobs are missing")
        raw_decision = payload.get("selection_decision")
        decision: object | None = raw_decision
        if isinstance(raw_decision, Mapping):
            try:
                decision = FactorRepairSelectionDecision(
                    reference_condition=raw_decision["reference_condition"],
                    selected_condition=raw_decision["selected_condition"],
                    delta_s_point=raw_decision["delta_s_point"],
                    delta_s_ci95=tuple(raw_decision["delta_s_ci95"]),
                    endpoint_table=raw_decision["endpoint_table"],
                    reference_evidence_sha256=raw_decision["reference_evidence_sha256"],
                    selected_evidence_sha256=raw_decision["selected_evidence_sha256"],
                    decision_sha256=raw_decision["decision_sha256"],
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("queue selection decision is invalid") from error
        return cls(Path(root), jobs, selected_condition=payload.get("selected_condition"), decision=decision, identity=payload.get("identity"))

    def _persist(self) -> None:
        payload = {
            "schema_version": 1,
            "jobs": {
                name: {
                    "state": job.state,
                    "artifacts": deepcopy(job.artifacts),
                    **({"error": job.error} if job.error else {}),
                }
                for name, job in self.jobs.items()
            },
            "selected_condition": self.selected_condition,
            "selection_decision": _canonical(self.selection_decision) if self.selection_decision is not None else None,
            "identity": deepcopy(self.identity),
        }
        atomic_write_json(self.state_path, payload)

    @property
    def status(self) -> dict[str, str]:
        return {name: job.state for name, job in self.jobs.items()}

    def job_status(self, name: str) -> str:
        return self._job(name).state

    def _job(self, name: str) -> QueueJob:
        if name not in self.jobs:
            raise ValueError(f"unknown queue job: {name}")
        return self.jobs[name]

    def acquire_gpu_lock(self) -> None:
        self.lock.acquire()

    def release_gpu_lock(self) -> None:
        self.lock.release()

    def transition(self, name: str, state: str, *, artifacts: Mapping[str, object] | None = None, error: BaseException | None = None) -> None:
        job = self._job(name)
        if state not in _ALLOWED_STATES:
            raise ValueError(f"unknown queue state: {state}")
        allowed = {
            "pending": {"running", "blocked"},
            "running": {"complete", "failed"},
            "failed": {"running"},
            "complete": set(),
            "blocked": set(),
        }
        if state not in allowed[job.state]:
            raise ValueError(f"illegal queue transition: {job.state} -> {state}")
        if artifacts is not None:
            job.artifacts = _normalize_artifacts(name, artifacts)
        job.state = state
        job.error = None if error is None else f"{type(error).__name__}: {error}"
        self._persist()

    def start(self, name: str) -> None:
        if not self.launchable(name):
            raise ValueError(f"queue job is not launchable: {name}")
        self.transition(name, "running")

    def complete(self, name: str, *, artifacts: Mapping[str, object]) -> None:
        mapping = _as_mapping(artifacts, "queue artifacts")
        normalized = _normalize_artifacts(name, mapping)
        state = "complete" if _artifact_gate_passed(normalized) else "failed"
        if state == "failed" and name in {"F1-calibration", "F2-calibration", "F3-calibration"}:
            normalized.pop("manifest", None)
            normalized["gate_failure"] = True
        self.transition(name, state, artifacts=normalized)

    def fail(self, name: str, error: BaseException) -> None:
        self.transition(name, "failed", error=error)

    def resume(self, name: str, *, identity: Mapping[str, object] | None = None) -> None:
        job = self._job(name)
        if job.state != "failed":
            raise ValueError("only failed queue jobs may resume")
        if not job.artifacts:
            raise ValueError("failed job artifacts are missing")
        if (
            not isinstance(identity, Mapping)
            or not identity
            or not self.identity
            or _canonical(identity) != _canonical(self.identity)
        ):
            raise ValueError("queue resume identity mismatch")
        _normalize_artifacts(name, job.artifacts)
        self.transition(name, "running")

    def run_job(self, name: str, runner: Callable[[], Mapping[str, object]]) -> Mapping[str, object]:
        self.acquire_gpu_lock()
        try:
            self.start(name)
            try:
                artifacts = runner()
                self.complete(name, artifacts=artifacts)
                return artifacts
            except Exception as error:
                self.fail(name, error)
                raise
        finally:
            self.release_gpu_lock()

    def launchable(self, name: str) -> bool:
        if name not in self.jobs or self.jobs[name].state != "pending":
            return False
        if name == "selected-repair-adaptation":
            if not self.selected_condition or "F0-calibration" not in self.jobs:
                return False
            candidate_name = f"{self.selected_condition}-calibration"
            ready = (
                self.jobs["F0-calibration"].state == "complete"
                and candidate_name in self.jobs
                and self.jobs[candidate_name].state == "complete"
            )
            if not ready or not isinstance(self.selection_decision, FactorRepairSelectionDecision):
                return False
            try:
                _validate_selection_artifacts(
                    self.selection_decision,
                    self.jobs["F0-calibration"].artifacts or {},
                    self.jobs[candidate_name].artifacts or {},
                )
            except (TypeError, ValueError, KeyError):
                return False
            return True
        if name == "F0-adaptation":
            if self.jobs.get("F0-calibration", QueueJob(name, "blocked")).state != "complete":
                return False
            if not self.selected_condition or not isinstance(self.selection_decision, FactorRepairSelectionDecision):
                return False
            candidate_name = f"{self.selected_condition}-calibration"
            if candidate_name not in self.jobs or self.jobs[candidate_name].state != "complete":
                return False
            try:
                _validate_selection_artifacts(
                    self.selection_decision,
                    self.jobs["F0-calibration"].artifacts or {},
                    self.jobs[candidate_name].artifacts or {},
                )
            except (TypeError, ValueError, KeyError):
                return False
            return True
        if name.endswith("-adaptation"):
            condition = _condition_from_job(name)
            if condition != "F0":
                return False
            return False
        condition = _condition_from_job(name)
        if condition == "F0":
            return True
        f0 = self.jobs.get("F0-calibration")
        return f0 is not None and f0.state == "complete"

    def consume_selection_decision(self, decision: object | None) -> None:
        if decision is None:
            candidate_jobs = [self.jobs.get(f"F{i}-calibration") for i in (1, 2, 3)]
            if any(job is not None and job.state not in {"complete", "blocked", "failed"} for job in candidate_jobs):
                return
            self.selected_condition = None
            self.selection_decision = None
            for name in ("F0-adaptation", "selected-repair-adaptation"):
                if name in self.jobs and self.jobs[name].state == "pending":
                    self.jobs[name].state = "blocked"
            self._persist()
            return
        checked = _validate_decision_type(decision)
        if self.selection_decision is not None:
            existing = self.selection_decision
            if isinstance(existing, FactorRepairSelectionDecision) and existing.decision_sha256 == checked.decision_sha256:
                return
            raise ValueError("selection decision already consumed")
        selected = checked.selected_condition
        if "F0-calibration" not in self.jobs or self.jobs["F0-calibration"].state != "complete":
            raise ValueError("selection decision requires complete F0 calibration")
        candidate_name = f"{selected}-calibration"
        if candidate_name not in self.jobs or self.jobs[candidate_name].state != "complete":
            raise ValueError("selection decision candidate calibration is incomplete")
        f0_artifacts = self.jobs["F0-calibration"].artifacts or {}
        candidate_artifacts = self.jobs[candidate_name].artifacts or {}
        if not _artifact_gate_passed(f0_artifacts) or not _artifact_gate_passed(candidate_artifacts):
            raise ValueError("selection decision requires passing calibration artifacts")
        _validate_selection_artifacts(checked, f0_artifacts, candidate_artifacts)
        self.selected_condition = selected
        self.selection_decision = checked
        self._persist()

    def track_f_adaptation_status(self) -> str:
        if self.selected_condition is None:
            return "blocked"
        if not self.launchable("F0-adaptation"):
            return "blocked"
        if not self.launchable("selected-repair-adaptation"):
            return "blocked"
        return "ready"


__all__ = [
    "DEFAULT_JOBS",
    "FactorRepairQueue",
    "QueueJob",
    "validate_factor_guided_manifest",
]
