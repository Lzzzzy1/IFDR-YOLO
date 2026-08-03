from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


CLASSES = ("Car", "Cyclist", "Pedestrian")


@dataclass(frozen=True)
class SelectionDecision:
    advance: bool
    candidate_mean_ap40: float
    full_mean_ap40: float
    fusion_mean_ap40: float
    failed_checks: tuple[str, ...]


def _moderate_ap40(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = {
            name: float(payload["classes"][name]["moderate"]["ap40"])
            for name in CLASSES
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid AP40 metrics: {path}") from error
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"AP40 metrics must be finite: {path}")
    return values


def _gradient_checks(path: Path) -> tuple[bool, bool, bool]:
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        anchors = [
            record["parameter_groups"]["semantic_anchor"]["gradient_norms"]
            for record in records
        ]
        fusion = [
            record["parameter_groups"]["fusion_adapters"]["gradient_norms"]
            for record in records
        ]
        localization = [
            record["parameter_groups"]["localization_adapter"]["gradient_norms"]
            for record in records
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid gradient diagnostics: {path}") from error
    if not records:
        raise ValueError(f"gradient diagnostics are empty: {path}")
    norms = [
        float(value)
        for group in (*anchors, *fusion, *localization)
        for value in group.values()
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in norms):
        raise ValueError(f"gradient norms must be finite and non-negative: {path}")
    protected = all(float(group["detection"]) <= 1e-12 for group in anchors)
    supervised = any(float(group["factor"]) > 0.0 for group in anchors) and any(
        float(group["counterfactual"]) > 0.0 for group in anchors
    )
    adapters_train = any(float(group["detection"]) > 0.0 for group in fusion) and any(
        float(group["detection"]) > 0.0 for group in localization
    )
    return protected, supervised, adapters_train


def evaluate_selection_gate(
    *,
    full_metrics: Path,
    fusion_metrics: Path,
    candidate_metrics: Path,
    gradient_diagnostics: Path,
) -> SelectionDecision:
    full = _moderate_ap40(full_metrics)
    fusion = _moderate_ap40(fusion_metrics)
    candidate = _moderate_ap40(candidate_metrics)
    full_mean = sum(full.values()) / len(full)
    fusion_mean = sum(fusion.values()) / len(fusion)
    candidate_mean = sum(candidate.values()) / len(candidate)
    protected, supervised, adapters_train = _gradient_checks(
        gradient_diagnostics
    )

    failed: list[str] = []
    if not (
        candidate_mean >= full_mean + 1.0
        or candidate_mean >= fusion_mean - 0.5
    ):
        failed.append("performance")
    if any(candidate[name] < full[name] - 1.0 for name in CLASSES):
        failed.append("class_regression")
    if not protected:
        failed.append("semantic_protection")
    if not supervised:
        failed.append("semantic_supervision")
    if not adapters_train:
        failed.append("adapter_training")

    return SelectionDecision(
        advance=not failed,
        candidate_mean_ap40=candidate_mean,
        full_mean_ap40=full_mean,
        fusion_mean_ap40=fusion_mean,
        failed_checks=tuple(failed),
    )
