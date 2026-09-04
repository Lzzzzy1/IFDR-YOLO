from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "create_evidence_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("create_evidence_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_is_deterministic_and_excludes_itself() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "b").mkdir()
        (root / "a.txt").write_text("alpha\n", encoding="utf-8")
        (root / "b" / "z.json").write_text("{}\n", encoding="utf-8")
        manifest = root / "MANIFEST.sha256"
        metadata = root / "manifest.json"

        first = MODULE.create_manifest(
            root=root,
            manifest_path=manifest,
            metadata_path=metadata,
        )
        first_text = manifest.read_text(encoding="utf-8")
        second = MODULE.create_manifest(
            root=root,
            manifest_path=manifest,
            metadata_path=metadata,
        )

        assert first["files"] == second["files"]
        assert first_text == manifest.read_text(encoding="utf-8")
        assert [row["path"] for row in first["files"]] == ["a.txt", "b/z.json"]
        assert "MANIFEST.sha256" not in first_text
        assert "manifest.json" not in first_text
        assert first["manifest_sha256"] == MODULE.sha256_file(manifest)


def test_manifest_changes_when_evidence_changes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        item = root / "evidence.txt"
        item.write_text("before\n", encoding="utf-8")
        before = MODULE.create_manifest(
            root=root,
            manifest_path=root / "MANIFEST.sha256",
            metadata_path=root / "manifest.json",
        )
        item.write_text("after\n", encoding="utf-8")
        after = MODULE.create_manifest(
            root=root,
            manifest_path=root / "MANIFEST.sha256",
            metadata_path=root / "manifest.json",
        )
        assert before["manifest_sha256"] != after["manifest_sha256"]


if __name__ == "__main__":
    test_manifest_is_deterministic_and_excludes_itself()
    test_manifest_changes_when_evidence_changes()
    print("PASS: deterministic evidence manifest")
