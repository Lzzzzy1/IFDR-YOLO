from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(
    *,
    root: Path,
    manifest_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    excluded = {manifest_path, metadata_path}
    files: list[dict[str, object]] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        resolved = path.resolve()
        if resolved in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("evidence root contains no files")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in files
    )
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temporary.write_text(
        manifest_text,
        encoding="utf-8",
        newline="\n",
    )
    manifest_temporary.replace(manifest_path)

    report = {
        "schema_version": 1,
        "root": str(root),
        "file_count": len(files),
        "files": files,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata_temporary.replace(metadata_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic SHA-256 manifest for an evidence tree."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = create_manifest(
        root=args.root,
        manifest_path=args.manifest,
        metadata_path=args.metadata,
    )
    print(f"manifest={report['manifest_path']}")
    print(f"file_count={report['file_count']}")
    print(f"manifest_sha256={report['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
