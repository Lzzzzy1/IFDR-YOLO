from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.experiments.evidence import write_evidence_configs


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    paths = write_evidence_configs(
        repository_root=repository_root,
        output_dir=repository_root / "configs/experiments/evidence",
    )
    for key, path in paths.items():
        print(f"{key}={path.relative_to(repository_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
