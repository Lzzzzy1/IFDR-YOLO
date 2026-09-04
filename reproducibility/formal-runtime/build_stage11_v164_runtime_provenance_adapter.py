"""Build a runtime successor that records verified gitless provenance."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile


V163_SHA256 = "d24d7841f33d6992b225336e904382ee1e0d14352f41f59631ae6c8f60daf278"
V156_SHA256 = "1f04ad62b7cec844f8169f0f29a0cf220663e454ad1d19dc84470fc682409f08"
V156_MANIFEST_SHA256 = "dfa73559be6a566cb2aaf61789eb1069711fd35b2bed152ab2058f92b38c94e2"
SOURCE_FILES = (
    "build_stage11_v164_runtime_provenance_adapter.py",
    "test_stage11_v164_runtime_provenance_adapter.py",
)
V163_MEMBER = "upstream/stage11-v163-gitless-provenance-envelope.tar"
V156_MEMBER = "upstream/stage11-v156-runtime-config-closure.tar"
ENTRYPOINT = "code/scripts/run_p2_interaction_s0.py"
PROVENANCE_PATH = "code/.stage11-provenance.json"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _regular_bytes(path: Path, label: str) -> bytes:
    source = Path(path)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise RuntimeError(f"{label} must be a nonempty regular file")
            content = stream.read()
            after = os.fstat(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    current = source.lstat()
    identity = before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or identity != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ) or len(content) != before.st_size:
        raise RuntimeError(f"{label} changed while being read")
    return content


def _read_archive(content: bytes, label: str) -> tuple[dict[str, bytes], dict[str, object]]:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError(f"{label} contains duplicate members")
        files: dict[str, bytes] = {}
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(f"{label} contains an unsafe member")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"{label} member cannot be read")
            files[member.name] = stream.read()
    manifest_raw = files.get("package-manifest.json")
    manifest = json.loads(manifest_raw) if isinstance(manifest_raw, bytes) else None
    identities = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(identities, Mapping) or manifest_raw != _canonical(manifest) or names != [*sorted(identities), "package-manifest.json"]:
        raise RuntimeError(f"{label} manifest differs")
    for name, raw_identity in identities.items():
        identity = raw_identity if isinstance(raw_identity, Mapping) else {}
        member_content = files.get(name)
        if (
            set(identity) != {"sha256", "size"}
            or not isinstance(member_content, bytes)
            or identity.get("sha256") != _sha(member_content)
            or identity.get("size") != len(member_content)
        ):
            raise RuntimeError(f"{label} member differs: {name}")
    return files, dict(manifest)


def _read_v163(content: bytes) -> tuple[bytes, dict[str, object]]:
    if _sha(content) != V163_SHA256:
        raise RuntimeError("v163 provenance envelope SHA256 differs")
    files, manifest = _read_archive(content, "v163 provenance envelope")
    if (
        manifest.get("schema") != "stage11-v163-gitless-provenance-envelope-v1"
        or manifest.get("state") != "PREPARED_NO_GO"
        or manifest.get("authorization") != {"gpu": False, "remote": False, "training": False, "upload": False}
    ):
        raise RuntimeError("v163 provenance envelope contract differs")
    raw_provenance = files.get("runtime-provenance.json")
    provenance = json.loads(raw_provenance) if isinstance(raw_provenance, bytes) else None
    if not isinstance(provenance, dict) or raw_provenance != _canonical(provenance):
        raise RuntimeError("v163 provenance record differs")
    upstream = files.get("upstream/stage11-v156-runtime-config-closure.tar")
    if not isinstance(upstream, bytes) or _sha(upstream) != V156_SHA256:
        raise RuntimeError("v163 v156 source differs")
    return upstream, provenance


def _read_v156(content: bytes) -> tuple[dict[str, bytes], bytes]:
    if _sha(content) != V156_SHA256:
        raise RuntimeError("v156 runtime archive SHA256 differs")
    files, manifest = _read_archive(content, "v156 runtime archive")
    raw_manifest = files["package-manifest.json"]
    if _sha(raw_manifest) != V156_MANIFEST_SHA256 or manifest.get("schema") != "stage11-v156-runtime-config-closure-package-v1":
        raise RuntimeError("v156 runtime manifest differs")
    return {name: value for name, value in files.items() if name != "package-manifest.json"}, raw_manifest


def _runtime_provenance(source: Mapping[str, object]) -> dict[str, object]:
    tree = source.get("source_runtime_tree_sha256")
    if not isinstance(tree, str) or len(tree) != 64 or set(tree) - set("0123456789abcdef"):
        raise RuntimeError("v163 source runtime tree identity differs")
    expected = {
        "schema": "stage11-v163-gitless-runtime-provenance-v1",
        "source_runtime_manifest_sha256": V156_MANIFEST_SHA256,
        "source_runtime_package_sha256": V156_SHA256,
        "source_runtime_tree_sha256": tree,
        "state": "PREPARED_NO_GO",
    }
    if dict(source) != expected:
        raise RuntimeError("v163 source provenance contract differs")
    return {
        "runtime_identity": f"gitless-v156:{tree}",
        "schema": "stage11-v164-runtime-provenance-v1",
        "source_provenance_package_sha256": V163_SHA256,
        "source_runtime_manifest_sha256": V156_MANIFEST_SHA256,
        "source_runtime_package_sha256": V156_SHA256,
        "source_runtime_tree_sha256": tree,
        "state": "PREPARED_NO_GO",
    }


def _patch_entrypoint(content: bytes, provenance: Mapping[str, object]) -> bytes:
    source = content.decode("utf-8")
    start = source.find("def _git_commit(root: Path) -> str:\n")
    end = source.find("\n\ndef _code_sha256", start)
    if start < 0 or end < 0:
        raise RuntimeError("v156 DCLI entrypoint has no replaceable Git identity function")
    expected = json.dumps(dict(provenance), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    replacement = (
        "def _git_commit(root: Path) -> str:\n"
        "    provenance_path = Path(root) / '.stage11-provenance.json'\n"
        "    try:\n"
        "        payload = json.loads(provenance_path.read_text(encoding='utf-8'))\n"
        "    except (OSError, json.JSONDecodeError) as error:\n"
        "        raise ValueError('gitless runtime provenance is unavailable') from error\n"
        f"    expected = {expected}\n"
        "    if payload != expected:\n"
        "        raise ValueError('gitless runtime provenance differs')\n"
        "    return str(expected['runtime_identity'])\n"
    )
    return (source[:start] + replacement + source[end:]).encode("utf-8")


def _manifest(payload: Mapping[str, bytes]) -> dict[str, object]:
    return {
        "authorization": {"gpu": False, "remote": False, "training": False, "upload": False},
        "files": {
            name: {"sha256": _sha(content), "size": len(content)}
            for name, content in sorted(payload.items())
        },
        "schema": "stage11-v164-runtime-provenance-adapter-v1",
        "state": "PREPARED_NO_GO",
    }


def _add(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = 0o600
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(content))


def _tar(payload: Mapping[str, bytes], manifest: Mapping[str, object]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name, content in [*sorted(payload.items()), ("package-manifest.json", _canonical(manifest))]:
            _add(archive, name, content)
    return output.getvalue()


def _write_fresh(path: Path, content: bytes) -> None:
    target = Path(path)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _payload(v163: bytes) -> dict[str, bytes]:
    upstream, source_provenance = _read_v163(v163)
    runtime, raw_v156_manifest = _read_v156(upstream)
    provenance = _runtime_provenance(source_provenance)
    if provenance["source_runtime_tree_sha256"] != _sha(raw_v156_manifest):
        raise RuntimeError("v163 runtime tree receipt differs")
    runtime[ENTRYPOINT] = _patch_entrypoint(runtime[ENTRYPOINT], provenance)
    runtime[PROVENANCE_PATH] = _canonical(provenance)
    runtime[V163_MEMBER] = v163
    runtime[V156_MEMBER] = upstream
    return runtime


def build_package(package_root: Path, v163_package: Path, output: Path) -> dict[str, object]:
    """Build the runtime adapter without claiming a Git commit that is unavailable."""
    v163 = _regular_bytes(Path(v163_package), "v163 provenance envelope")
    payload = _payload(v163)
    root = Path(package_root)
    payload.update({name: _regular_bytes(root / name, name) for name in SOURCE_FILES})
    manifest = _manifest(payload)
    _write_fresh(Path(output), _tar(payload, manifest))
    if verify_package(Path(output)) != manifest:
        raise RuntimeError("v164 runtime provenance adapter readback differs")
    return manifest


def verify_package(package: Path) -> dict[str, object]:
    """Verify the runtime adapter and its inherited provenance closure."""
    raw = _regular_bytes(Path(package), "v164 runtime provenance adapter")
    files, manifest = _read_archive(raw, "v164 runtime provenance adapter")
    if (
        manifest.get("schema") != "stage11-v164-runtime-provenance-adapter-v1"
        or manifest.get("state") != "PREPARED_NO_GO"
        or manifest.get("authorization") != {"gpu": False, "remote": False, "training": False, "upload": False}
    ):
        raise RuntimeError("v164 runtime provenance adapter contract differs")
    for name in SOURCE_FILES:
        if name not in files:
            raise RuntimeError(f"v164 package source is missing: {name}")
    expected = _payload(files[V163_MEMBER])
    expected.update({name: files[name] for name in SOURCE_FILES})
    observed = {name: content for name, content in files.items() if name != "package-manifest.json"}
    if observed != expected:
        raise RuntimeError("v164 runtime provenance adapter closure differs")
    return manifest


__all__ = ["build_package", "verify_package"]
