from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


COMPONENT_FACTORIAL = OrderedDict(
    (
        ("factor_control", (False, False)),
        ("fusion_only", (True, False)),
        ("dcli_only", (False, True)),
        ("full", (True, True)),
    )
)


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{field} must be a mapping with string keys")
    return value


def build_component_ablation_payloads(
    base_payload: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    if not isinstance(base_payload, Mapping):
        raise ValueError("base payload must be a mapping")
    base = deepcopy(dict(base_payload))
    experiment = _mapping(base.get("experiment"), "experiment")
    method = _mapping(base.get("ifdr"), "ifdr")
    components = _mapping(method.get("components"), "ifdr.components")
    required = {
        "fusion_gate",
        "dcli",
        "factor_supervision",
        "interventions",
    }
    if set(components) != required:
        raise ValueError("base IFDR components must use the locked schema")

    result: dict[str, dict[str, object]] = {}
    for name, (fusion_gate, dcli) in COMPONENT_FACTORIAL.items():
        payload = deepcopy(base)
        payload_experiment = _mapping(
            payload["experiment"],
            "experiment",
        )
        payload_components = _mapping(
            _mapping(payload["ifdr"], "ifdr")["components"],
            "ifdr.components",
        )
        payload_experiment["variant"] = f"ifdr-{name.replace('_', '-')}"
        payload_components.update(
            {
                "fusion_gate": fusion_gate,
                "dcli": dcli,
                "factor_supervision": True,
                "interventions": True,
            }
        )
        result[name] = payload
    return result


def write_component_ablation_configs(
    *,
    base_config: Path,
    output_dir: Path,
) -> dict[str, Path]:
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    payloads = build_component_ablation_payloads(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name, condition in payloads.items():
        path = output_dir / f"kitti_ifdr_{name}_s17.yaml"
        path.write_text(
            yaml.safe_dump(condition, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        result[name] = path
    return result
