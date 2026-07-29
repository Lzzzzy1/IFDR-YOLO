from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from ifdr_yolo.data.phase1_audit import audit_generated_dataset
from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.experiments.config import BaselineConfig


def canonical_ids_sha256(image_ids: tuple[str, ...]) -> str:
    content = "".join(f"{image_id}\n" for image_id in image_ids)
    return sha256(content.encode("utf-8")).hexdigest()


def verify_file_sha256(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ValueError(
            f"{label} SHA256 mismatch: path={path}, "
            f"expected={expected.lower()}, actual={actual}"
        )
    return actual


def classify_porcelain_status(
    lines: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tracked: list[str] = []
    untracked: list[str] = []
    for line in lines:
        if not line:
            continue
        target = untracked if line.startswith("?? ") else tracked
        target.append(line)
    return tuple(tracked), tuple(untracked)


def find_repository_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(f"no Git repository found from: {start}")


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def collect_git_provenance(root: Path) -> dict[str, object]:
    sha = _run_git(root, "rev-parse", "HEAD").strip()
    branch = _run_git(root, "branch", "--show-current").strip()
    status_lines = tuple(
        _run_git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).splitlines()
    )
    tracked, untracked = classify_porcelain_status(status_lines)
    return {
        "commit": sha,
        "branch": branch,
        "tracked_changes": list(tracked),
        "untracked_files": list(untracked),
        "tracked_clean": not tracked,
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def collect_environment() -> dict[str, object]:
    import torch

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os_name": os.name,
        "torch": torch.__version__,
        "ultralytics": _package_version("ultralytics"),
        "numpy": _package_version("numpy"),
        "pillow": _package_version("Pillow"),
        "pyyaml": _package_version("PyYAML"),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": device_count,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(device_count)
        ],
    }


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _verify_source_split(
    *,
    source: dict[str, object],
    split_name: str,
    path: Path,
    image_ids: tuple[str, ...],
) -> str:
    expected_count = source.get(f"{split_name}_count")
    if expected_count != len(image_ids):
        raise ValueError(
            f"source {split_name} count mismatch: "
            f"expected={expected_count}, actual={len(image_ids)}"
        )
    expected_hash = source.get(f"{split_name}_sha256")
    if not isinstance(expected_hash, str):
        raise ValueError(f"source {split_name}_sha256 must be a string")
    return verify_file_sha256(
        path,
        expected_hash,
        label=f"source {split_name} split",
    )


def _verify_generated_split_hash(
    *,
    manifest: dict[str, object],
    split_name: str,
    image_ids: tuple[str, ...],
) -> str:
    actual = canonical_ids_sha256(image_ids)
    expected = manifest.get(f"{split_name}_split_sha256")
    if expected != actual:
        raise ValueError(
            f"generated {split_name} split SHA256 mismatch: "
            f"expected={expected}, actual={actual}"
        )
    return actual


def _verify_metadata_membership(
    *,
    path: Path,
    train_ids: tuple[str, ...],
    val_ids: tuple[str, ...],
) -> None:
    expected = {
        **{image_id: "train" for image_id in train_ids},
        **{image_id: "val" for image_id in val_ids},
    }
    actual: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        image_id = str(record.get("image_id"))
        split = str(record.get("split"))
        if image_id in actual:
            raise ValueError(
                f"duplicate image metadata ID at {path}:{line_number}: {image_id}"
            )
        actual[image_id] = split
    if actual != expected:
        raise ValueError("image metadata split membership mismatch")


def verify_dataset(
    config: BaselineConfig,
    *,
    verify_all_hashes: bool,
) -> dict[str, object]:
    train_ids = load_ids(config.paths.train_ids)
    val_ids = load_ids(config.paths.val_ids)
    source_path = config.paths.train_ids.parent / "source.json"
    source = _load_json(source_path, "split source")
    train_file_hash = _verify_source_split(
        source=source,
        split_name="train",
        path=config.paths.train_ids,
        image_ids=train_ids,
    )
    val_file_hash = _verify_source_split(
        source=source,
        split_name="val",
        path=config.paths.val_ids,
        image_ids=val_ids,
    )

    manifest_path = config.paths.generated_data / "manifest.json"
    manifest = _load_json(manifest_path, "generated dataset manifest")
    train_ids_hash = _verify_generated_split_hash(
        manifest=manifest,
        split_name="train",
        image_ids=train_ids,
    )
    val_ids_hash = _verify_generated_split_hash(
        manifest=manifest,
        split_name="val",
        image_ids=val_ids,
    )
    _verify_metadata_membership(
        path=config.paths.generated_data / "metadata" / "images.jsonl",
        train_ids=train_ids,
        val_ids=val_ids,
    )

    summary = audit_generated_dataset(
        source_image_dir=config.paths.raw_images,
        source_label_dir=config.paths.raw_labels,
        train_ids=train_ids,
        val_ids=val_ids,
        generated_dir=config.paths.generated_data,
        verify_all_source_hashes=verify_all_hashes,
    )
    return {
        **asdict(summary),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_file_sha256": train_file_hash,
        "val_file_sha256": val_file_hash,
        "train_ids_sha256": train_ids_hash,
        "val_ids_sha256": val_ids_hash,
        "generated_manifest": manifest,
        "split_source": source,
    }
