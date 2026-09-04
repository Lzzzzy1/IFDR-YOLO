from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EVIDENCE_SEEDS = (29, 41)
SOURCE_CONFIGS = OrderedDict(
    (
        (
            "baseline",
            "configs/experiments/kitti_yolov8m_baseline_s17.yaml",
        ),
        ("p2", "configs/experiments/kitti_yolov8m_p2_s17.yaml"),
        (
            "fusion_only",
            "configs/experiments/ablations/"
            "kitti_ifdr_fusion_only_s17.yaml",
        ),
    )
)
OUTPUT_PREFIXES = {
    "baseline": "kitti_yolov8m_baseline",
    "p2": "kitti_yolov8m_p2",
    "fusion_only": "kitti_ifdr_fusion_only",
}


def build_seed_payload(payload: object, seed: int) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    result = deepcopy(payload)
    experiment = result.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("configuration experiment must be a mapping")
    experiment["seed"] = seed
    method = result.get("ifdr")
    if method is not None:
        if not isinstance(method, dict):
            raise ValueError("ifdr must be a mapping")
        intervention = method.get("intervention")
        if not isinstance(intervention, dict):
            raise ValueError("ifdr.intervention must be a mapping")
        intervention["base_seed"] = seed
    return result


def write_evidence_configs(
    *,
    repository_root: Path,
    output_dir: Path,
) -> dict[str, Path]:
    root = repository_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for condition, relative_source in SOURCE_CONFIGS.items():
        source = root / relative_source
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        for seed in EVIDENCE_SEEDS:
            key = f"{condition}_s{seed}"
            path = output_dir / f"{OUTPUT_PREFIXES[condition]}_s{seed}.yaml"
            path.write_text(
                yaml.safe_dump(
                    build_seed_payload(payload, seed),
                    sort_keys=False,
                ),
                encoding="utf-8",
                newline="\n",
            )
            written[key] = path
    return written
