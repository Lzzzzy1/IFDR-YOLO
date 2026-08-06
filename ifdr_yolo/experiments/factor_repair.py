"""Registered semantic calibration phases for factor repair experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from ifdr_yolo.models.gated_fusion import ReliabilityGatedConcat


_CALIBRATION_MASKS: dict[str, dict[str, float]] = {
    "F0": {"synthetic": 1.0, "natural": 0.0, "specificity": 0.0},
    "F1": {"synthetic": 1.0, "natural": 1.0, "specificity": 0.0},
    "F2": {"synthetic": 1.0, "natural": 0.0, "specificity": 1.0},
    "F3": {"synthetic": 1.0, "natural": 1.0, "specificity": 1.0},
}
_CALIBRATION_VARIANTS = tuple(_CALIBRATION_MASKS)


def _freeze_audit_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_audit_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_audit_value(item) for item in value)
    return value


@dataclass(frozen=True)
class SemanticCalibrationPhase:
    """Immutable, auditable freeze contract for one F0--F3 calibration."""

    variant: str
    epochs: int
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    loss_mask: Mapping[str, float]
    fusion_schedule: float = 0.0
    dcli_schedule: float = 0.0
    factor_supervision_schedule: float = 1.0
    early_stopping: bool = False
    diagnostic_group_names: tuple[str, ...] = ()
    diagnostic_group_provenance: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variant not in _CALIBRATION_VARIANTS:
            raise ValueError("variant must be one of F0, F1, F2, or F3")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs != 30
        ):
            raise ValueError("semantic calibration requires exactly 30 epochs")
        if self.fusion_schedule != 0.0 or self.dcli_schedule != 0.0:
            raise ValueError("semantic calibration schedules must be zero")
        if self.factor_supervision_schedule != 1.0:
            raise ValueError("factor supervision schedule must be one")
        if self.early_stopping:
            raise ValueError("semantic calibration disables early stopping")
        for field_name in (
            "trainable_parameter_names",
            "frozen_parameter_names",
            "diagnostic_group_names",
        ):
            value = _freeze_audit_value(getattr(self, field_name))
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) for item in value
            ):
                raise TypeError(f"{field_name} must contain parameter names")
            object.__setattr__(self, field_name, value)
        loss_mask = _freeze_audit_value(self.loss_mask)
        expected_mask = _CALIBRATION_MASKS[self.variant]
        if not isinstance(loss_mask, Mapping) or dict(loss_mask) != expected_mask:
            raise ValueError("loss mask does not match registered variant")
        object.__setattr__(self, "loss_mask", loss_mask)
        diagnostic_provenance = _freeze_audit_value(
            self.diagnostic_group_provenance
        )
        if not isinstance(diagnostic_provenance, Mapping) or any(
            not isinstance(name, str)
            or not isinstance(names, tuple)
            or not all(isinstance(item, str) for item in names)
            for name, names in diagnostic_provenance.items()
        ):
            raise TypeError("diagnostic_group_provenance must map names to tuples")
        object.__setattr__(self, "diagnostic_group_provenance", diagnostic_provenance)
        provenance = _freeze_audit_value(self.provenance)
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", provenance)


def _named_parameters(model: object) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    method = getattr(model, "named_parameters", None)
    if not callable(method):
        raise TypeError("calibration model must expose named_parameters()")
    result = tuple(method())
    if any(
        not isinstance(name, str) or not isinstance(parameter, torch.nn.Parameter)
        for name, parameter in result
    ):
        raise TypeError("model named_parameters() returned an invalid item")
    return result


def _group_named_parameters(
    model: object,
) -> dict[str, tuple[tuple[str, torch.nn.Parameter], ...]]:
    """Build ordered projection/shared groups from registered identities."""

    method = getattr(model, "factor_semantic_named_parameters", None)
    if not callable(method):
        raise TypeError("calibration model must expose factor_semantic_named_parameters()")
    semantic_named = tuple(method())
    if any(
        not isinstance(name, str) or not isinstance(parameter, torch.nn.Parameter)
        for name, parameter in semantic_named
    ):
        raise TypeError("factor_semantic_named_parameters() returned an invalid item")
    names_by_id = {id(parameter): (name, parameter) for name, parameter in semantic_named}

    nodes = tuple(getattr(model, "fusion_node_indices", ()))
    graph = getattr(model, "model", None)
    if len(nodes) != 6 or graph is None:
        raise ValueError("semantic calibration requires six fusion nodes")

    groups: dict[str, tuple[tuple[str, torch.nn.Parameter], ...]] = {}
    projection_count = 0
    seen_ids: set[int] = set()
    for node_ordinal, index in enumerate(nodes):
        try:
            layer = graph[index]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(f"fusion node {index} is not registered") from error
        if not isinstance(layer, ReliabilityGatedConcat):
            raise TypeError(f"fusion node {index} is not ReliabilityGatedConcat")
        if len(layer.projections) != 2:
            raise ValueError(f"fusion node {index} must expose exactly two projections")
        for branch, projection in enumerate(layer.projections):
            projection_count += 1
            group_name = f"projection_{node_ordinal * 2 + branch:02d}"
            members: list[tuple[str, torch.nn.Parameter]] = []
            for parameter in projection.parameters():
                parameter_id = id(parameter)
                if parameter_id not in names_by_id:
                    raise ValueError("projection parameter is not registered")
                if parameter_id in seen_ids:
                    raise ValueError("semantic calibration projection identities overlap")
                seen_ids.add(parameter_id)
                members.append(names_by_id[parameter_id])
            groups[group_name] = tuple(sorted(members, key=lambda item: item[0]))

    first_layer = graph[nodes[0]]
    assert isinstance(first_layer, ReliabilityGatedConcat)
    for group_name, module in (
        ("shared_core", first_layer.reliability_estimator.shared_core),
        ("factor_head", first_layer.reliability_estimator.factor_head),
    ):
        members = []
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter_id not in names_by_id:
                raise ValueError(f"{group_name} parameter is not registered")
            if parameter_id in seen_ids:
                raise ValueError(f"{group_name} overlaps another semantic group")
            seen_ids.add(parameter_id)
            members.append(names_by_id[parameter_id])
        groups[group_name] = tuple(sorted(members, key=lambda item: item[0]))
    if projection_count != 12:
        raise ValueError("semantic calibration requires 12 projection modules")
    if len(groups) != 14:
        raise ValueError("semantic calibration requires 12 projections plus two shared groups")
    return groups


def factor_calibration_named_parameter_groups(
    model: object,
) -> dict[str, tuple[tuple[str, torch.nn.Parameter], ...]]:
    """Return ordered calibration groups with complete parameter names."""

    return _group_named_parameters(model)


def factor_calibration_parameter_groups(
    model: object,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Return identity-deduplicated parameter groups for an optimizer."""

    return {
        group_name: tuple(parameter for _, parameter in members)
        for group_name, members in _group_named_parameters(model).items()
    }


def split_three_view_contexts(
    contexts: dict[int, object],
    batch_size: int,
    *,
    required_nodes: tuple[int, ...] | None = None,
) -> dict[str, dict[int, object]]:
    """Thin named-view wrapper around the model's canonical splitter."""

    from ifdr_yolo.models.ifdr_model import split_three_view_contexts as split_contexts

    clean, target, background = split_contexts(
        contexts,
        batch_size,
        required_nodes=required_nodes,
    )
    return {
        "clean": clean,
        "target": target,
        "background": background,
    }


def _optimizer_clear_callable(optimizer: object):
    if optimizer is None:
        return None
    state = getattr(optimizer, "state", None)
    clear = getattr(state, "clear", None) if state is not None else None
    if not callable(clear):
        raise TypeError("optimizer must expose mutable state")
    return clear


def semantic_calibration_phase(
    model: object,
    *,
    variant: str,
    epochs: int = 30,
    optimizer: object | None = None,
) -> SemanticCalibrationPhase:
    """Apply and return the registered F0--F3 semantic freeze contract."""

    if not isinstance(variant, str) or variant not in _CALIBRATION_VARIANTS:
        raise ValueError("variant must be one of F0, F1, F2, or F3")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != 30:
        raise ValueError("semantic calibration requires exactly 30 epochs")
    schedule_setter = getattr(model, "set_component_schedules", None)
    if not callable(schedule_setter):
        raise TypeError("calibration model must expose set_component_schedules()")
    # Validate the optimizer interface without changing its state yet.  Model
    # structure validation below must also complete before a clear is called.
    clear_optimizer_state = _optimizer_clear_callable(optimizer)

    groups = _group_named_parameters(model)
    named_parameters = _named_parameters(model)
    trainable_ids = {
        id(parameter)
        for members in groups.values()
        for _, parameter in members
    }
    all_ids = {id(parameter) for _, parameter in named_parameters}
    if not trainable_ids <= all_ids:
        raise ValueError("semantic calibration group contains unknown parameters")
    if clear_optimizer_state is not None:
        # Clear before mutating any model flags/schedules; a clear failure is
        # therefore also atomic with respect to the model.
        clear_optimizer_state()
    for _, parameter in named_parameters:
        parameter.requires_grad = id(parameter) in trainable_ids
    schedule_setter(fusion=0.0, dcli=0.0, factor_supervision=1.0)

    trainable_names = tuple(
        sorted(name for name, parameter in named_parameters if id(parameter) in trainable_ids)
    )
    frozen_names = tuple(
        sorted(name for name, parameter in named_parameters if id(parameter) not in trainable_ids)
    )
    provenance = MappingProxyType(
        {
            group_name: tuple(name for name, _ in members)
            for group_name, members in groups.items()
        }
    )
    phase_provenance = MappingProxyType(
        {
            "parameter_groups": provenance,
            "identity_deduplicated": True,
            "projection_group_count": 12,
        }
    )
    return SemanticCalibrationPhase(
        variant=variant,
        epochs=epochs,
        trainable_parameter_names=trainable_names,
        frozen_parameter_names=frozen_names,
        loss_mask=MappingProxyType(dict(_CALIBRATION_MASKS[variant])),
        fusion_schedule=0.0,
        dcli_schedule=0.0,
        factor_supervision_schedule=1.0,
        early_stopping=False,
        diagnostic_group_names=tuple(groups),
        diagnostic_group_provenance=provenance,
        provenance=phase_provenance,
    )


def run_calibration_validation(
    model: object,
    batch: object,
    *,
    optimizer: object | None = None,
) -> object:
    """Run a no-grad validation forward; optimizer is intentionally unused."""

    del optimizer
    was_training = bool(getattr(model, "training", False))
    train_method = getattr(model, "train", None)
    try:
        eval_method = getattr(model, "eval", None)
        if not callable(eval_method):
            raise TypeError("validation model must expose eval()")
        eval_method()
        forward = getattr(model, "__call__", None)
        if not callable(forward):
            raise TypeError("validation model must be callable")
        with torch.no_grad():
            return forward(batch)
    finally:
        if callable(train_method):
            train_method(was_training)


__all__ = [
    "SemanticCalibrationPhase",
    "factor_calibration_named_parameter_groups",
    "factor_calibration_parameter_groups",
    "run_calibration_validation",
    "semantic_calibration_phase",
    "split_three_view_contexts",
]
