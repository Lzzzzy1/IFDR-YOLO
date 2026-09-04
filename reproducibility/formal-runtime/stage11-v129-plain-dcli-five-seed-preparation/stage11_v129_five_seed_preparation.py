"""Freeze a result-blind PLAIN_P2 versus DCLI five-seed preparation contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import tarfile


EXPECTED_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
HISTORICAL_SEEDS_FORBIDDEN: tuple[int, ...] = (17, 29, 41)
CANDIDATES: tuple[str, ...] = ("PLAIN_P2", "DCLI")
PROTOCOL: str = "KITTI_FIT3341_DEV371"
UPSTREAM_PACKAGE_SHA256: str = "69bf31ccebdec38188c9646d51612c71507280c670cb2f6d8f4b0eae5781b957"
UPSTREAM_PACKAGE_SIZE: int = 56_709_120
UPSTREAM_MANIFEST_SHA256: str = "957e0a2121a331c9758937ff03f316a1a55d82951293e7a641853d5fa28583fc"
SEED0_CODE_GIT_HEAD: str = "44fd2b0a83e070487fff6523b5447a1aa230708f"
TRUSTED_AUTHORIZATION_SHA256: None = None
SOURCE_CONFIGS: dict[str, tuple[str, str]] = {
    "PLAIN_P2": (
        "code/configs/experiments/selection/kitti_plain_p2_s0.yaml",
        "7789e8ee69ce28bc5ddf6dbd591d9920faa0d137fe0d39e347cab8958e5a1a0a",
    ),
    "DCLI": (
        "code/configs/experiments/selection/kitti_dcli_s0.yaml",
        "c7b23e0654921d9b46fae45ffa134e5c12baf518ed8a5549caa1443896cf1f29",
    ),
}
ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "PLAIN_P2": (
        "code/scripts/run_p2_fit_reference.py",
        "35db5831963ac4a5f80871756dd5b2174108405b677470a685e60d503e8d1afe",
    ),
    "DCLI": (
        "code/scripts/run_p2_interaction_s0.py",
        "bcfeb4b6daacd77e84939823ea2d4f1a63a52279becbcad411301b2033d4493f",
    ),
}
SEED0_EVIDENCE: dict[str, dict[str, str]] = {
    "PLAIN_P2": {
        "metrics_sha256": "42b8d9606035bffd3b9d321e6f424741ac9caa63076798160facd3aef8621ec7",
        "results_sha256": "da2b6ab260438575efc81fd6f458a39b1f994130db38b5918336981ae86955f9",
        "status_sha256": "1cb13107e8972ab347a63c05d32e9a5acdbf9ec61a95890796d5e5da98d3b4bf",
        "candidate_receipt_sha256": "dd58bd7b3f286c144078ed2a619df2cff3183df9d6aac86b93e44a938dc7386b",
    },
    "DCLI": {
        "metrics_sha256": "578bf36f6ff6edd0d07a1faea5c30fcaf8146408e80ea71ddc7efeaf581688c6",
        "results_sha256": "c1cfc92c1f840ce8073f5d32b1216aafda6becbf9cb3cdcc4ea0e3dfa67ea37f",
        "status_sha256": "ba1d79e53b0bdf45baa3dd71b168b8c6faa09a1f0410446ff4eb0fbcb0f3236c",
        "candidate_receipt_sha256": "c5c5eff509859adbb3bca1632782149fa3b4f93b5680231cecc5f75e4d1577ed",
    },
}
MISSING_INPUTS: tuple[str, ...] = (
    "PAID_REMOTE_AUTHORIZATION_MISSING",
    "TRUSTED_ENDPOINT_AND_CREDENTIAL_REGISTRATION_MISSING",
    "FRESH_REMOTE_GPU_DISK_PROCESS_PREFLIGHT_MISSING",
    "SEEDS_1_4_RESULTS_MISSING",
    "TRUSTED_RESULT_AND_COST_THRESHOLD_REGISTRY_MISSING",
    "INDEPENDENT_PRELAUNCH_REVIEW_MISSING",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be an object")
    return value


def _sequence(value: object, role: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{role} must be a sequence")
    return value


def _exact(value: Mapping[str, object], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{role} schema differs")


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{role} must be lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{role} must be lowercase SHA256")
    return value


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{role} must be finite numeric")
    return number


def _ap(value: object, role: str) -> float:
    number = _finite(value, role)
    if number < 0.0 or number > 100.0:
        raise ValueError(f"{role} must be within [0, 100]")
    return number


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError(f"upstream member must be a regular file: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"upstream member cannot be read: {name}")
    return stream.read()


def _validate_upstream_package(path: Path) -> tuple[dict[str, str], Mapping[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("upstream package must be a regular non-symlink file")
    if path.stat().st_size != UPSTREAM_PACKAGE_SIZE or _file_sha256(path) != UPSTREAM_PACKAGE_SHA256:
        raise ValueError("upstream package identity differs")
    with tarfile.open(path, "r:") as archive:
        manifest_bytes = _read_member(archive, "package-manifest.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != UPSTREAM_MANIFEST_SHA256:
            raise ValueError("upstream package manifest differs")
        manifest = _mapping(
            json.loads(manifest_bytes),
            "upstream manifest",
        )
        files = _mapping(manifest.get("files"), "upstream manifest files")
        contents: dict[str, str] = {}
        for candidate in CANDIDATES:
            config_path, expected_config_sha = SOURCE_CONFIGS[candidate]
            entrypoint_path, expected_entrypoint_sha = ENTRYPOINTS[candidate]
            for member_path, expected_sha in (
                (config_path, expected_config_sha),
                (entrypoint_path, expected_entrypoint_sha),
            ):
                identity = _mapping(files.get(member_path), f"upstream {member_path} identity")
                if identity.get("sha256") != expected_sha:
                    raise ValueError(f"upstream package source differs: {member_path}")
                content = _read_member(archive, member_path)
                if hashlib.sha256(content).hexdigest() != expected_sha:
                    raise ValueError(f"upstream package bytes differ: {member_path}")
                if member_path == config_path:
                    contents[candidate] = content.decode("utf-8")
        pretrained = _mapping(files.get("code/yolov8m.pt"), "upstream pretrained identity")
        if pretrained.get("sha256") != "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5":
            raise ValueError("upstream pretrained identity differs")
    return contents, manifest


def derive_seed_config(source: str, candidate: str, seed: int) -> str:
    """Derive one registered config by changing only its seed fields."""

    if candidate not in CANDIDATES:
        raise ValueError("candidate is not registered")
    if isinstance(seed, bool) or seed not in EXPECTED_SEEDS:
        raise ValueError("seed is not registered")
    if source.count("  seed: 0\n") != 1:
        raise ValueError("source config seed marker differs")
    result = source.replace("  seed: 0\n", f"  seed: {seed}\n", 1)
    if candidate == "DCLI":
        if result.count("    base_seed: 0\n") != 1:
            raise ValueError("DCLI source config base seed marker differs")
        result = result.replace("    base_seed: 0\n", f"    base_seed: {seed}\n", 1)
    elif "base_seed:" in result:
        raise ValueError("PLAIN_P2 config contains an unexpected base seed")
    return result


def _config_path(candidate: str, seed: int) -> str:
    slug = "plain_p2" if candidate == "PLAIN_P2" else "dcli"
    return f"configs/stage11-v129/kitti_{slug}_s{seed}.yaml"


def _matrix_order() -> tuple[tuple[int, str], ...]:
    result: list[tuple[int, str]] = []
    for seed in EXPECTED_SEEDS:
        order = CANDIDATES if seed % 2 == 0 else tuple(reversed(CANDIDATES))
        result.extend((seed, candidate) for candidate in order)
    return tuple(result)


def _receipt_schemas() -> dict[str, list[str]]:
    return {
        "identity": [
            "schema", "state", "plan_sha256", "task_sha256", "seed", "candidate",
            "git_head", "config_sha256", "entrypoint_sha256", "pretrained_sha256",
            "fit_ids_sha256", "development_ids_sha256", "runtime_sha256",
        ],
        "budget": [
            "schema", "state", "task_sha256", "epochs", "imgsz", "batch", "workers",
            "optimizer", "prediction_protocol_sha256", "training_runtime_seconds",
            "parameters", "flops", "median_latency_ms", "fps", "peak_vram_bytes",
        ],
        "checkpoint": [
            "schema", "state", "task_sha256", "generation", "epoch_completed",
            "epoch_prefix", "checkpoint_sha256", "optimizer_sha256", "scheduler_sha256",
            "ema_sha256", "scaler_sha256", "rng_sha256", "sampler_sha256",
            "dataloader_sha256", "interval_seconds", "created_at_utc",
        ],
        "recovery": [
            "schema", "state", "task_sha256", "interrupted_generation",
            "uninterrupted_generation", "resume_epoch", "common_prefix_sha256",
            "final_trajectory_sha256", "final_artifact_manifest_sha256",
            "equivalence", "maximum_checkpoint_interval_seconds",
        ],
        "failure": [
            "schema", "state", "task_sha256", "generation", "failed_step",
            "returncode", "stdout_sha256", "stderr_sha256", "last_checkpoint_sha256",
            "gpu_after", "preserved_paths", "next_task_started",
        ],
        "publication": [
            "schema", "state", "task_sha256", "generation", "primary_root_identity",
            "mirror_root_identity", "files", "byte_identical", "independent_roots",
        ],
    }


def _upstream_contract() -> dict[str, object]:
    return {
        "immutable_code_package_sha256": UPSTREAM_PACKAGE_SHA256,
        "immutable_code_package_size": UPSTREAM_PACKAGE_SIZE,
        "package_manifest_sha256": UPSTREAM_MANIFEST_SHA256,
    }


def _code_identity_contract() -> dict[str, object]:
    return {
        "git_head": SEED0_CODE_GIT_HEAD,
        "clean_required": True,
        "pretrained_sha256": "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5",
        "fit_ids_sha256": "50a1f8d72b747d1e2f460a4e0a355469484d33fb1e9928b64ba6594c1873362",
        "development_ids_sha256": "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8",
    }


def _budget_contract() -> dict[str, object]:
    return {
        "epochs": 30,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "device": "0",
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "amp": True,
        "deterministic": True,
        "primary_checkpoint": "last.pt",
        "prediction": {"conf": 0.001, "iou": 0.7, "max_det": 300, "half": False},
    }


def _checkpoint_contract() -> dict[str, object]:
    return {
        "maximum_interval_seconds": 300,
        "durable_boundary": "END_OF_EACH_EPOCH",
        "strict_epoch_prefix": list(range(1, 31)),
        "required_state": ["model", "optimizer", "scheduler", "ema", "scaler", "rng", "sampler", "dataloader"],
    }


def _recovery_contract() -> dict[str, object]:
    return {
        "required": True,
        "interruption_boundary": "AFTER_DURABLE_EPOCH",
        "equivalence": "EXACT_REGISTERED_TRAJECTORY_AND_FINAL_ARTIFACTS",
        "common_prefix_byte_identical": True,
        "uninterrupted_and_resumed_required": True,
    }


def _failure_contract() -> dict[str, object]:
    return {
        "stop_on_first_failure": True,
        "preserve_failure_generation": True,
        "next_task_must_not_start": True,
        "terminal_success_forbidden_after_failure": True,
    }


def _publication_contract() -> dict[str, object]:
    return {
        "primary_mirror_byte_identical": True,
        "independent_roots_required": True,
        "fresh_generation_required": True,
        "exact_file_closure_required": True,
        "atomic_receipts_required": True,
    }


def _paired_statistics_contract() -> dict[str, object]:
    return {
        "input_schema": "stage11-v129-paired-statistics-input-v1",
        "exact_seeds": list(EXPECTED_SEEDS),
        "difference": "PLAIN_P2_MINUS_DCLI",
        "macro": "UNWEIGHTED_PEDESTRIAN_CYCLIST_MODERATE_AP_R40",
        "ci": "PAIRED_T_95_PERCENT_DF4_T_CRITICAL_2.776445",
        "no_result_based_seed_deletion": True,
        "historical_seeds_excluded": list(HISTORICAL_SEEDS_FORBIDDEN),
        "required_per_seed_receipts": ["identity", "budget", "recovery", "publication"],
    }


def build_prepared_plan(upstream_package: Path) -> dict[str, object]:
    """Build the exact local PREPARED_NO_GO five-seed plan."""

    source_contents, _manifest = _validate_upstream_package(upstream_package)
    derived_configs: dict[str, dict[str, dict[str, object]]] = {}
    for candidate in CANDIDATES:
        derived_configs[candidate] = {}
        for seed in EXPECTED_SEEDS:
            content = derive_seed_config(source_contents[candidate], candidate, seed)
            encoded = content.encode("utf-8")
            derived_configs[candidate][str(seed)] = {
                "path": _config_path(candidate, seed),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size": len(encoded),
                "content": content,
            }
    matrix: list[dict[str, object]] = []
    for ordinal, (seed, candidate) in enumerate(_matrix_order(), start=1):
        entrypoint_path, entrypoint_sha = ENTRYPOINTS[candidate]
        config = derived_configs[candidate][str(seed)]
        matrix.append(
            {
                "ordinal": ordinal,
                "seed": seed,
                "candidate": candidate,
                "config_path": config["path"],
                "config_sha256": config["sha256"],
                "entrypoint_path": entrypoint_path,
                "entrypoint_sha256": entrypoint_sha,
                "execution_state": "RETAINED_EXISTING_SEED0" if seed == 0 else "PENDING_AUTHORIZATION",
                "seed0_evidence": SEED0_EVIDENCE[candidate] if seed == 0 else None,
            }
        )
    return {
        "schema": "stage11-v129-plain-dcli-five-seed-preparation-v1",
        "state": "PREPARED_NO_GO",
        "protocol": PROTOCOL,
        "method": "PLAIN_P2",
        "comparator": "DCLI",
        "seeds": list(EXPECTED_SEEDS),
        "historical_seeds_forbidden": list(HISTORICAL_SEEDS_FORBIDDEN),
        "result_blind": True,
        "no_seed_deletion": True,
        "authorization": {"remote": False, "gpu": False, "paid_server": False, "five_seed": False},
        "upstream": _upstream_contract(),
        "code_identity": _code_identity_contract(),
        "config_sources": {
            candidate: {
                "path": SOURCE_CONFIGS[candidate][0],
                "sha256": SOURCE_CONFIGS[candidate][1],
                "content": source_contents[candidate],
            }
            for candidate in CANDIDATES
        },
        "derived_configs": derived_configs,
        "matrix": matrix,
        "budget": _budget_contract(),
        "checkpoint_contract": _checkpoint_contract(),
        "recovery_contract": _recovery_contract(),
        "failure_contract": _failure_contract(),
        "publication_contract": _publication_contract(),
        "receipt_schemas": _receipt_schemas(),
        "paired_statistics_contract": _paired_statistics_contract(),
        "missing": list(MISSING_INPUTS),
    }


def validate_prepared_plan(value: object) -> dict[str, object]:
    """Validate the exact result-blind plan and return a detached copy."""

    plan = _mapping(value, "plan")
    _exact(
        plan,
        {
            "schema", "state", "protocol", "method", "comparator", "seeds",
            "historical_seeds_forbidden", "result_blind", "no_seed_deletion",
            "authorization", "upstream", "code_identity", "config_sources",
            "derived_configs", "matrix", "budget", "checkpoint_contract",
            "recovery_contract", "failure_contract", "publication_contract",
            "receipt_schemas", "paired_statistics_contract", "missing",
        },
        "plan",
    )
    if (
        plan["schema"] != "stage11-v129-plain-dcli-five-seed-preparation-v1"
        or plan["state"] != "PREPARED_NO_GO"
        or plan["protocol"] != PROTOCOL
        or plan["method"] != "PLAIN_P2"
        or plan["comparator"] != "DCLI"
    ):
        raise ValueError("plan identity differs")
    if plan["seeds"] != list(EXPECTED_SEEDS) or plan["historical_seeds_forbidden"] != list(HISTORICAL_SEEDS_FORBIDDEN):
        raise ValueError("plan seeds differ")
    if plan["result_blind"] is not True or plan["no_seed_deletion"] is not True:
        raise ValueError("plan result-blind rules differ")
    if plan["authorization"] != {"remote": False, "gpu": False, "paid_server": False, "five_seed": False}:
        raise ValueError("plan authorization differs")
    exact_contracts = (
        ("upstream", _upstream_contract()),
        ("code_identity", _code_identity_contract()),
        ("budget", _budget_contract()),
        ("checkpoint_contract", _checkpoint_contract()),
        ("recovery_contract", _recovery_contract()),
        ("failure_contract", _failure_contract()),
        ("publication_contract", _publication_contract()),
        ("receipt_schemas", _receipt_schemas()),
        ("paired_statistics_contract", _paired_statistics_contract()),
    )
    for role, expected_contract in exact_contracts:
        if plan[role] != expected_contract:
            raise ValueError(f"{role} differs")
    config_sources = _mapping(plan["config_sources"], "config sources")
    derived_configs = _mapping(plan["derived_configs"], "derived configs")
    if set(config_sources) != set(CANDIDATES) or set(derived_configs) != set(CANDIDATES):
        raise ValueError("config candidates differ")
    for candidate in CANDIDATES:
        source = _mapping(config_sources[candidate], f"{candidate} config source")
        _exact(source, {"path", "sha256", "content"}, f"{candidate} config source")
        content = source["content"]
        if not isinstance(content, str):
            raise ValueError(f"{candidate} config content differs")
        if (
            source["path"] != SOURCE_CONFIGS[candidate][0]
            or source["sha256"] != SOURCE_CONFIGS[candidate][1]
            or hashlib.sha256(content.encode("utf-8")).hexdigest() != SOURCE_CONFIGS[candidate][1]
        ):
            raise ValueError(f"{candidate} config source differs")
        candidate_configs = _mapping(derived_configs[candidate], f"{candidate} derived configs")
        if set(candidate_configs) != {str(seed) for seed in EXPECTED_SEEDS}:
            raise ValueError(f"{candidate} config seeds differ")
        for seed in EXPECTED_SEEDS:
            derived = _mapping(candidate_configs[str(seed)], f"{candidate} seed {seed} config")
            _exact(derived, {"path", "sha256", "size", "content"}, f"{candidate} seed {seed} config")
            expected_content = derive_seed_config(content, candidate, seed)
            expected_bytes = expected_content.encode("utf-8")
            if derived != {
                "path": _config_path(candidate, seed),
                "sha256": hashlib.sha256(expected_bytes).hexdigest(),
                "size": len(expected_bytes),
                "content": expected_content,
            }:
                raise ValueError(f"{candidate} seed {seed} config differs")
    matrix = _sequence(plan["matrix"], "matrix")
    if len(matrix) != 10:
        raise ValueError("matrix must contain exact ten slots")
    expected_order = _matrix_order()
    for index, raw_item in enumerate(matrix):
        item = _mapping(raw_item, f"matrix item {index}")
        _exact(
            item,
            {
                "ordinal", "seed", "candidate", "config_path", "config_sha256",
                "entrypoint_path", "entrypoint_sha256", "execution_state", "seed0_evidence",
            },
            f"matrix item {index}",
        )
        seed, candidate = expected_order[index]
        expected_config = _mapping(
            derived_configs[candidate],
            f"{candidate} configs",
        )[str(seed)]
        config = _mapping(expected_config, f"{candidate} seed {seed} config")
        entrypoint_path, entrypoint_sha = ENTRYPOINTS[candidate]
        expected_state = "RETAINED_EXISTING_SEED0" if seed == 0 else "PENDING_AUTHORIZATION"
        if (
            item["ordinal"] != index + 1
            or item["seed"] != seed
            or item["candidate"] != candidate
            or item["config_path"] != config["path"]
            or item["config_sha256"] != config["sha256"]
            or item["entrypoint_path"] != entrypoint_path
            or item["entrypoint_sha256"] != entrypoint_sha
            or item["execution_state"] != expected_state
        ):
            raise ValueError("matrix identity differs")
        if seed == 0:
            if item["seed0_evidence"] != SEED0_EVIDENCE[candidate]:
                raise ValueError("matrix seed0 evidence differs")
        elif item["seed0_evidence"] is not None:
            raise ValueError("matrix future task contains result evidence")
    if plan["missing"] != list(MISSING_INPUTS):
        raise ValueError("missing input list differs")
    return json.loads(_canonical(plan))


def evaluate_launch_readiness(
    plan_value: object,
    authorization_value: object | None,
    prerequisite_value: object,
) -> dict[str, object]:
    """Return PREPARED_NO_GO or reject caller-minted launch authorization."""

    plan = validate_prepared_plan(plan_value)
    prerequisites = _mapping(prerequisite_value, "launch prerequisites")
    if authorization_value is not None:
        if TRUSTED_AUTHORIZATION_SHA256 is None:
            raise ValueError("trusted authorization is not registered")
        raise ValueError("trusted authorization validation is unavailable")
    return {
        "schema": "stage11-v129-launch-readiness-v1",
        "state": "PREPARED_NO_GO",
        "plan_sha256": canonical_sha256(plan),
        "missing": list(MISSING_INPUTS),
        "observed_prerequisite_keys": sorted(str(key) for key in prerequisites),
        "authorization": {"remote": False, "gpu": False, "paid_server": False, "five_seed": False},
        "runner_calls_authorized": 0,
    }


def validate_paired_statistics_input(value: object) -> dict[str, object]:
    """Validate complete five-seed paired AP inputs without seed deletion."""

    root = _mapping(value, "paired statistics input")
    _exact(root, {"schema", "protocol", "method", "comparator", "seeds"}, "paired statistics input")
    if (
        root["schema"] != "stage11-v129-paired-statistics-input-v1"
        or root["protocol"] != PROTOCOL
        or root["method"] != "PLAIN_P2"
        or root["comparator"] != "DCLI"
    ):
        raise ValueError("paired statistics identity differs")
    seeds = _sequence(root["seeds"], "paired statistics seeds")
    if len(seeds) != len(EXPECTED_SEEDS):
        raise ValueError("paired statistics require exact seeds [0,1,2,3,4]")
    validated: list[dict[str, object]] = []
    for index, raw_seed in enumerate(seeds):
        item = _mapping(raw_seed, f"paired seed {index}")
        _exact(
            item,
            {
                "seed", "PLAIN_P2", "DCLI", "identity_receipt_sha256",
                "budget_receipt_sha256", "recovery_receipt_sha256",
                "publication_receipt_sha256",
            },
            f"paired seed {index}",
        )
        if item["seed"] != EXPECTED_SEEDS[index]:
            raise ValueError("paired statistics require exact seeds [0,1,2,3,4]")
        arms: dict[str, dict[str, float]] = {}
        for candidate in CANDIDATES:
            arm = _mapping(item[candidate], f"seed {index} {candidate}")
            _exact(arm, {"pedestrian_ap_r40", "cyclist_ap_r40"}, f"seed {index} {candidate}")
            arms[candidate] = {
                "pedestrian_ap_r40": _ap(arm["pedestrian_ap_r40"], f"seed {index} {candidate} pedestrian"),
                "cyclist_ap_r40": _ap(arm["cyclist_ap_r40"], f"seed {index} {candidate} cyclist"),
            }
        validated.append(
            {
                "seed": EXPECTED_SEEDS[index],
                "PLAIN_P2": arms["PLAIN_P2"],
                "DCLI": arms["DCLI"],
                "identity_receipt_sha256": _sha(item["identity_receipt_sha256"], "identity receipt"),
                "budget_receipt_sha256": _sha(item["budget_receipt_sha256"], "budget receipt"),
                "recovery_receipt_sha256": _sha(item["recovery_receipt_sha256"], "recovery receipt"),
                "publication_receipt_sha256": _sha(item["publication_receipt_sha256"], "publication receipt"),
            }
        )
    return {
        "schema": "stage11-v129-paired-statistics-input-v1",
        "protocol": PROTOCOL,
        "method": "PLAIN_P2",
        "comparator": "DCLI",
        "seeds": validated,
    }


__all__: tuple[str, ...] = (
    "EXPECTED_SEEDS",
    "build_prepared_plan",
    "canonical_sha256",
    "derive_seed_config",
    "evaluate_launch_readiness",
    "validate_paired_statistics_input",
    "validate_prepared_plan",
)
