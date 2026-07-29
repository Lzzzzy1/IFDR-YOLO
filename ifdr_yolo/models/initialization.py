from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

import torch


_MODEL_KEY = re.compile(r"^model\.(\d+)\.")


@dataclass(frozen=True)
class InitializationReport:
    strategy: str
    max_layer: int
    expected_items: int
    transferred_items: int
    source_items: int
    target_items: int
    untransferred_items: int
    transferred_keys: tuple[str, ...]
    transferred_shapes: dict[str, list[int]]

    def to_payload(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "max_layer": self.max_layer,
            "expected_items": self.expected_items,
            "transferred_items": self.transferred_items,
            "source_items": self.source_items,
            "target_items": self.target_items,
            "untransferred_items": self.untransferred_items,
            "transferred_keys": list(self.transferred_keys),
            "transferred_shapes": self.transferred_shapes,
        }


def select_semantic_prefix_state(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    max_layer: int,
) -> dict[str, torch.Tensor]:
    if max_layer < 0:
        raise ValueError("max_layer must be >= 0")
    selected: dict[str, torch.Tensor] = {}
    for key in sorted(source):
        match = _MODEL_KEY.match(key)
        if match is None or int(match.group(1)) > max_layer:
            continue
        target_tensor = target.get(key)
        if target_tensor is None or source[key].shape != target_tensor.shape:
            continue
        selected[key] = source[key]
    return selected


def apply_semantic_prefix_initialization(
    target_model: Any,
    source_model: Any,
    *,
    max_layer: int,
    expected_items: int,
) -> InitializationReport:
    source_state = source_model.float().state_dict()
    target_state = target_model.state_dict()
    selected = select_semantic_prefix_state(
        source_state,
        target_state,
        max_layer=max_layer,
    )
    if len(selected) != expected_items:
        raise RuntimeError(
            "semantic initialization item mismatch: "
            f"expected={expected_items}, actual={len(selected)}"
        )
    target_model.load_state_dict(selected, strict=False)
    keys = tuple(selected)
    return InitializationReport(
        strategy="semantic_prefix",
        max_layer=max_layer,
        expected_items=expected_items,
        transferred_items=len(keys),
        source_items=len(source_state),
        target_items=len(target_state),
        untransferred_items=len(target_state) - len(keys),
        transferred_keys=keys,
        transferred_shapes={
            key: list(selected[key].shape)
            for key in keys
        },
    )
