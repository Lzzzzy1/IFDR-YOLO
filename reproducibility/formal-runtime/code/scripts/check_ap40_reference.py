from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.eval.reference_check import run_reference_check


OPENPCDET_COMMIT = "233f849829b6ac19afb8af8837a0246890908755"
OFFICIAL_DEVKIT_SHA256 = (
    "ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare IFDR-YOLO AP40 with the OpenPCDet KITTI port."
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/ap40-reference-check.json"),
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_reference_check(args.reference_dir)
    payload.update(
        {
            "openpcdet_commit": OPENPCDET_COMMIT,
            "openpcdet_url": (
                "https://github.com/open-mmlab/OpenPCDet/tree/"
                f"{OPENPCDET_COMMIT}/pcdet/datasets/kitti/"
                "kitti_object_eval_python"
            ),
            "official_devkit_url": (
                "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
                "devkit_object.zip"
            ),
            "official_devkit_sha256": OFFICIAL_DEVKIT_SHA256,
            "tolerance": args.tolerance,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    max_difference = float(payload["max_absolute_difference"])
    print(
        f"cases={payload['case_count']} "
        f"max_absolute_difference={max_difference:.12g}"
    )
    if max_difference > args.tolerance:
        print("REFERENCE AP40 CHECK FAILED")
        return 1
    print("REFERENCE AP40 CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
