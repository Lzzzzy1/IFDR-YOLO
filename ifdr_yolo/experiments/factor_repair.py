"""Registered semantic calibration phases for factor repair experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import re
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
    phase = SemanticCalibrationPhase(
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
    # Bind the immutable condition contract to the model consumed by the
    # calibration loss.  Without this hand-off, F0--F3 all silently use the
    # unmasked route and natural/specificity gradients leak into F0.
    setattr(model, "_semantic_calibration_phase", phase)
    return phase


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


# ---------------------------------------------------------------------------
# Fixed-budget task adaptation
# ---------------------------------------------------------------------------


_TASK_CONDITIONS = frozenset(("F0", "F1", "F2", "F3"))
_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
_TASK_PARAMETER_RE = re.compile(r"^model\.\d+\.")
_TASK_PARAMETER_CATEGORIES = frozenset(
    {
        "graph_task",
        "fusion_adapter",
        "fusion_gate",
        "localization_adapter",
    }
)


def _semantic_entries(model: object) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    method = getattr(model, "factor_semantic_named_parameters", None)
    if not callable(method):
        raise TypeError("adaptation model must expose factor_semantic_named_parameters()")
    entries = tuple(method())
    if not entries or any(
        not isinstance(name, str) or not isinstance(parameter, torch.nn.Parameter)
        for name, parameter in entries
    ):
        raise TypeError("factor_semantic_named_parameters() returned invalid entries")
    deduplicated: list[tuple[str, torch.nn.Parameter]] = []
    seen: dict[int, str] = {}
    for name, parameter in entries:
        parameter_id = id(parameter)
        previous_name = seen.get(parameter_id)
        if previous_name is not None:
            if previous_name != name:
                raise ValueError(
                    "shared semantic identity has ambiguous parameter names"
                )
            continue
        seen[parameter_id] = name
        deduplicated.append((name, parameter))
    return tuple(deduplicated)


def semantic_module_ids(model: object) -> frozenset[int]:
    """Return the identity set of the twelve projections and two shared modules.

    The model hook is preferred so small test models and future registered
    variants can describe the same contract without relying on graph internals.
    Existing IFDR models expose the fusion graph and are resolved from that
    canonical registration when the hook is absent.
    """

    hook = getattr(model, "factor_semantic_modules", None)
    modules: tuple[object, ...]
    if callable(hook):
        modules = tuple(hook())
    else:
        nodes = tuple(getattr(model, "fusion_node_indices", ()))
        graph = getattr(model, "model", None)
        if len(nodes) != 6 or graph is None:
            raise TypeError("adaptation model must expose semantic modules")
        selected: list[object] = []
        try:
            for index in nodes:
                layer = graph[index]
                selected.extend(tuple(layer.projections))
            first = graph[nodes[0]]
            estimator = first.reliability_estimator
            selected.extend((estimator.shared_core, estimator.factor_head))
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise TypeError("adaptation model has invalid semantic modules") from error
        modules = tuple(selected)
    if len(modules) != 14 or any(not isinstance(module, torch.nn.Module) for module in modules):
        raise ValueError("task adaptation requires exactly 12 projections plus two shared modules")
    ids = tuple(id(module) for module in modules)
    if len(set(ids)) != len(ids):
        raise ValueError("semantic module identities must be distinct")
    # Ensure all semantic modules are registered by the model; unregistered
    # modules would make the checkpoint/hash contract incomplete.
    registered = (
        {id(module) for module in model.modules()}
        if callable(getattr(model, "modules", None))
        else set()
    )
    if not set(ids) <= registered:
        raise ValueError("semantic module is not registered on the model")
    return frozenset(ids)


def _semantic_module_names(model: object, module_ids: frozenset[int]) -> tuple[str, ...]:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("adaptation model must expose named_modules()")
    names = tuple(sorted(name for name, module in named_modules() if id(module) in module_ids))
    if not names:
        raise ValueError("semantic module identity set is not registered")
    return names


def _tensor_record(digest: object, kind: str, name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"semantic {kind} {name} is not a tensor")
    value = tensor.detach().cpu().contiguous()
    metadata = f"{kind}\0{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode("utf-8")
    update = getattr(digest, "update", None)
    if not callable(update):
        raise TypeError("semantic digest must expose update()")
    update(len(metadata).to_bytes(8, "big"))
    update(metadata)
    # Viewing as bytes avoids dtype-specific NumPy conversion failures (most
    # notably bfloat16) while preserving the exact storage representation.
    raw = value.view(torch.uint8).numpy().tobytes()
    update(len(raw).to_bytes(8, "big"))
    update(raw)


def semantic_state_sha256(model: object, module_ids: object | None = None) -> str:
    """Hash semantic module identities, parameters, and buffers only.

    Parameters and buffers are selected by identity, so shared modules are
    represented once even when referenced by all six fusion nodes.
    """

    ids = semantic_module_ids(model) if module_ids is None else frozenset(module_ids)
    if not ids or any(not isinstance(value, int) for value in ids):
        raise ValueError("semantic module ids must be a non-empty identity set")
    module_names = _semantic_module_names(model, ids)
    all_parameters = getattr(model, "named_parameters", None)
    all_buffers = getattr(model, "named_buffers", None)
    if not callable(all_parameters) or not callable(all_buffers):
        raise TypeError("adaptation model must expose named parameters and buffers")
    parameter_ids: set[int] = set()
    buffer_ids: set[int] = set()
    for _, module in model.named_modules():
        if id(module) not in ids:
            continue
        parameter_ids.update(id(parameter) for parameter in module.parameters())
        buffer_ids.update(id(buffer) for buffer in module.buffers())
    parameter_records = tuple(sorted(
        (name, parameter)
        for name, parameter in all_parameters()
        if id(parameter) in parameter_ids
    ))
    buffer_records = tuple(sorted(
        (name, buffer)
        for name, buffer in all_buffers()
        if id(buffer) in buffer_ids
    ))
    if not parameter_records and not buffer_records:
        raise ValueError("semantic module state is empty")
    digest = hashlib.sha256()
    digest.update(b"ifdr-semantic-state-v1\0")
    for name in module_names:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for name, parameter in parameter_records:
        _tensor_record(digest, "parameter", name, parameter)
    for name, buffer in buffer_records:
        _tensor_record(digest, "buffer", name, buffer)
    return digest.hexdigest()


def enforce_semantic_eval_mode(model: object, module_ids: object | None = None) -> None:
    """Train task modules while keeping every semantic module in eval mode."""

    ids = semantic_module_ids(model) if module_ids is None else frozenset(module_ids)
    train = getattr(model, "train", None)
    modules = getattr(model, "modules", None)
    if not callable(train) or not callable(modules):
        raise TypeError("adaptation model must expose train() and modules()")
    train()
    for module in modules():
        if id(module) not in ids:
            continue
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False


def _require_registered_task_path(
    trainable_parameter_names: tuple[str, ...] | list[str],
    model: object | None = None,
) -> tuple[str, ...]:
    raw_names = tuple(trainable_parameter_names)
    if not raw_names or any(
        not isinstance(name, str) or not name for name in raw_names
    ):
        raise ValueError("task adaptation requires non-empty named trainable parameters")
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("task adaptation parameter names must be unique")
    names = tuple(sorted(raw_names))
    layer_count = None
    graph = getattr(model, "model", None) if model is not None else None
    registered_names: set[str] | None = None
    task_categories: Mapping[str, str] | None = None
    if model is not None:
        if graph is None:
            raise ValueError("task adaptation model graph is not registered")
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            raise TypeError("adaptation model must expose named_parameters()")
        registered_names = {name for name, _ in named_parameters()}
        task_categories = _registered_task_parameter_categories(model, names)
    try:
        layer_count = len(graph) if graph is not None else None
    except TypeError as error:
        if model is not None:
            raise ValueError("task adaptation model graph is not indexable") from error
    for name in names:
        if name.startswith("localization_adapter."):
            if registered_names is not None and name not in registered_names:
                raise ValueError(f"unregistered localization adapter parameter: {name}")
            if task_categories is not None and name not in task_categories:
                raise ValueError(f"unregistered localization adapter parameter: {name}")
            continue
        match = _TASK_PARAMETER_RE.match(name)
        if match is None:
            raise ValueError(f"unregistered task adaptation parameter path: {name}")
        index = int(name.split(".", 2)[1])
        if layer_count is not None and index >= layer_count:
            raise ValueError(f"unregistered task adaptation layer: {name}")
        if ".projections." in name or ".reliability_estimator." in name:
            raise ValueError(f"semantic parameter leaked into task adaptation: {name}")
        if registered_names is not None and name not in registered_names:
            raise ValueError(f"unregistered task adaptation parameter path: {name}")
        if task_categories is not None and name not in task_categories:
            raise ValueError(f"unregistered task adaptation parameter path: {name}")
    return names


def _registered_task_parameter_categories(
    model: object,
    trainable_parameter_names: tuple[str, ...] | list[str],
) -> dict[str, str]:
    """Resolve task parameters through registered module identities.

    A path-shaped name is not sufficient for adaptation: an arbitrary
    parameter attached to a semantic module must not silently become a task
    parameter.  Ownership is resolved from ``named_modules`` and the module
    identity contract, then categorized for checkpoint provenance.
    """

    names = tuple(trainable_parameter_names)
    named_parameters = getattr(model, "named_parameters", None)
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_parameters) or not callable(named_modules):
        raise TypeError("adaptation model must expose named parameters and modules")
    named = tuple(named_parameters())
    by_name = {name: parameter for name, parameter in named}
    if set(names) - set(by_name):
        missing = sorted(set(names) - set(by_name))
        raise ValueError(f"unregistered task adaptation parameter path: {missing[0]}")
    semantic_parameter_ids = {
        id(parameter) for _, parameter in _semantic_entries(model)
    }

    # Small registered variants must declare their complete task path.  The
    # identity check prevents a path-only hook from smuggling in an alias or
    # an extra non-semantic parameter.
    task_hook = getattr(model, "task_adaptation_named_parameters", None)
    hook_names: set[str] | None = None
    if callable(task_hook):
        entries = tuple(task_hook())
        if not entries or any(
            not isinstance(name, str)
            or not isinstance(parameter, torch.nn.Parameter)
            for name, parameter in entries
        ):
            raise TypeError("task_adaptation_named_parameters() returned invalid entries")
        hook_names = set()
        hook_ids: dict[int, str] = {}
        for name, parameter in entries:
            if name in hook_names or name not in by_name or by_name[name] is not parameter:
                raise ValueError("task adaptation hook identity does not match model")
            previous = hook_ids.get(id(parameter))
            if previous is not None and previous != name:
                raise ValueError("task adaptation hook contains ambiguous aliases")
            hook_names.add(name)
            hook_ids[id(parameter)] = name
            if id(parameter) in semantic_parameter_ids:
                raise ValueError("task adaptation hook contains semantic parameters")
        nonsemantic_names = {
            name
            for name, parameter in named
            if id(parameter) not in semantic_parameter_ids
        }
        if nonsemantic_names != hook_names or set(names) != hook_names:
            raise ValueError(
                "task adaptation hook must register every non-semantic parameter"
            )

    owners: dict[int, tuple[str, torch.nn.Module, str]] = {}
    module_by_name = dict(named_modules())
    for module_name, module in module_by_name.items():
        for local_name, parameter in module.named_parameters(recurse=False):
            owners.setdefault(id(parameter), (module_name, module, local_name))
    graph = getattr(model, "model", None)
    try:
        layer_count = len(graph) if graph is not None else None
    except TypeError as error:
        raise ValueError("task adaptation model graph is not indexable") from error

    try:
        from ultralytics.nn.modules import C2f, Conv, Detect, SPPF
        allowed_graph_types = (Conv, C2f, SPPF, Detect)
    except ImportError:
        allowed_graph_types = ()

    categories: dict[str, str] = {}
    for name in names:
        parameter = by_name[name]
        owner = owners.get(id(parameter))
        if owner is None:
            raise ValueError(f"task parameter has no registered module owner: {name}")
        module_name, module, local_name = owner
        if id(module) in semantic_module_ids(model):
            raise ValueError(
                f"task adaptation parameter is attached to a semantic module: {name}"
            )
        if name.startswith("localization_adapter."):
            categories[name] = "localization_adapter"
            continue
        match = _TASK_PARAMETER_RE.match(name)
        if match is None:
            raise ValueError(f"unregistered task adaptation parameter path: {name}")
        index = int(name.split(".", 2)[1])
        if graph is None or layer_count is None or index >= layer_count:
            raise ValueError(f"unregistered task adaptation layer: {name}")
        if ".projections." in name or ".reliability_estimator." in name:
            raise ValueError(f"semantic parameter leaked into task adaptation: {name}")
        if module_name == "model" or not module_name.startswith(f"model.{index}"):
            raise ValueError(f"unregistered task adaptation module owner: {name}")
        root = graph[index]
        suffix = name[len(f"model.{index}.") :]
        if hook_names is not None:
            category = "fusion_gate" if local_name == "gate_logit" else "graph_task"
        elif isinstance(root, ReliabilityGatedConcat):
            if suffix == "gate_logit":
                category = "fusion_gate"
            elif suffix.startswith("router."):
                category = "graph_task"
            elif suffix.startswith("fusion_adapter."):
                category = "fusion_adapter"
            else:
                raise ValueError(f"unregistered ReliabilityGatedConcat parameter: {name}")
        elif isinstance(root, allowed_graph_types):
            category = "graph_task"
        else:
            raise ValueError(f"unregistered task adaptation layer type: {name}")
        categories[name] = category
    return categories


def build_optimizer(
    optimizer_name: str,
    model: object,
    trainable_parameter_names: tuple[str, ...] | list[str],
    optimizer_hparams: Mapping[str, object],
) -> torch.optim.Optimizer:
    """Build a fresh optimizer from an explicit named-parameter allowlist."""

    if not isinstance(optimizer_name, str) or optimizer_name.lower() not in _OPTIMIZERS:
        raise ValueError("optimizer_name must be Adam, AdamW, or SGD")
    if not isinstance(optimizer_hparams, Mapping):
        raise TypeError("optimizer_hparams must be a mapping")
    names = _require_registered_task_path(tuple(trainable_parameter_names), model)
    named = dict(model.named_parameters()) if callable(getattr(model, "named_parameters", None)) else {}
    if set(names) - set(named):
        raise ValueError("task adaptation parameter is not present on model")
    parameters = []
    for name in names:
        parameter = named[name]
        if not parameter.requires_grad:
            raise ValueError(f"task adaptation parameter is frozen: {name}")
        parameters.append(parameter)
    try:
        optimizer = _OPTIMIZERS[optimizer_name.lower()](parameters, **dict(optimizer_hparams))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid optimizer hyperparameters") from error
    if optimizer.state:
        raise RuntimeError("fresh task adaptation optimizer unexpectedly has state")
    return optimizer


def registered_update_count(epochs: int, updates_per_epoch: int = 1) -> int:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs != 60:
        raise ValueError("registered task adaptation requires exactly 60 epochs")
    if (
        isinstance(updates_per_epoch, bool)
        or not isinstance(updates_per_epoch, int)
        or updates_per_epoch <= 0
    ):
        raise ValueError("updates_per_epoch must be a positive integer")
    return epochs * updates_per_epoch


def _freeze_sequence(values: object, field_name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{field_name} must be a sequence") from error
    return result


@dataclass(frozen=True)
class TaskAdaptationPhase:
    """Registered, condition-local contract for the fixed 60-epoch phase."""

    condition: str
    calibration_checkpoint_path: Path
    calibration_checkpoint_role: str
    calibration_checkpoint_sha256: str
    semantic_state_sha256: str
    semantic_module_names: tuple[str, ...]
    epochs: int
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    optimizer: torch.optim.Optimizer
    optimizer_name: str
    optimizer_hparams: Mapping[str, object]
    eta_schedule: tuple[object, ...]
    updates_per_epoch: int
    update_count: int
    early_stopping: bool
    primary_checkpoint: str
    task_parameter_categories: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    semantic_state_journal: list[dict[str, object]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.condition, str) or self.condition not in _TASK_CONDITIONS:
            raise ValueError("condition must be one of F0, F1, F2, or F3")
        if self.calibration_checkpoint_role != "calibration_last":
            raise ValueError("task adaptation requires calibration_last checkpoint")
        if self.epochs != 60:
            raise ValueError("task adaptation requires exactly 60 epochs")
        if self.primary_checkpoint != "last.pt":
            raise ValueError("task adaptation primary checkpoint must be last.pt")
        if not re.fullmatch(r"[0-9a-f]{64}", self.calibration_checkpoint_sha256):
            raise ValueError("calibration checkpoint SHA256 must be a 64-hex digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.semantic_state_sha256):
            raise ValueError("semantic state SHA256 must be a 64-hex digest")
        if self.early_stopping:
            raise ValueError("task adaptation disables early stopping")
        if self.updates_per_epoch <= 0 or isinstance(self.updates_per_epoch, bool):
            raise ValueError("updates_per_epoch must be a positive integer")
        if self.update_count != registered_update_count(
            self.epochs,
            self.updates_per_epoch,
        ):
            raise ValueError("task adaptation update count is not registered")
        object.__setattr__(
            self,
            "calibration_checkpoint_path",
            Path(self.calibration_checkpoint_path).resolve(),
        )
        object.__setattr__(self, "trainable_parameter_names", tuple(self.trainable_parameter_names))
        object.__setattr__(self, "frozen_parameter_names", tuple(self.frozen_parameter_names))
        object.__setattr__(self, "semantic_module_names", tuple(self.semantic_module_names))
        object.__setattr__(self, "eta_schedule", tuple(self.eta_schedule))
        categories = MappingProxyType(dict(self.task_parameter_categories))
        if set(categories) != set(self.trainable_parameter_names):
            raise ValueError("task parameter categories must cover trainable parameters")
        if any(value not in _TASK_PARAMETER_CATEGORIES for value in categories.values()):
            raise ValueError("task parameter category is not registered")
        object.__setattr__(self, "task_parameter_categories", categories)
        object.__setattr__(self, "optimizer_hparams", MappingProxyType(dict(self.optimizer_hparams)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def task_adaptation_phase(
    model: object,
    *,
    condition: str,
    calibration_checkpoint_path: str | os.PathLike[str],
    calibration_provenance: Mapping[str, object] | None = None,
    calibration_checkpoint_role: str = "calibration_last",
    epochs: int = 60,
    optimizer_name: str,
    optimizer_hparams: Mapping[str, object],
    eta_schedule: object,
    primary_checkpoint: str = "last.pt",
    updates_per_epoch: int = 1,
    load_calibration: bool = True,
) -> TaskAdaptationPhase:
    """Create an independent, fixed-budget task adaptation phase."""

    if calibration_checkpoint_role != "calibration_last" or epochs != 60 or primary_checkpoint != "last.pt":
        raise ValueError("registered adaptation requires calibration_last, 60 epochs, and last.pt")
    if not isinstance(condition, str) or condition not in _TASK_CONDITIONS:
        raise ValueError("condition must be one of F0, F1, F2, or F3")
    from ifdr_yolo.data.learned_factor_manifest import (
        load_validated_checkpoint,
        resolve_provenance_path,
    )
    from ifdr_yolo.data.splits import sha256_file

    if not isinstance(calibration_provenance, Mapping):
        raise ValueError("calibration_provenance is required")
    expected_condition = calibration_provenance.get("condition")
    expected_path_value = calibration_provenance.get("checkpoint_path")
    expected_sha256 = calibration_provenance.get("checkpoint_sha256")
    if expected_condition != condition:
        raise ValueError("calibration checkpoint condition provenance mismatch")
    if not isinstance(expected_path_value, (str, os.PathLike)):
        raise ValueError("calibration provenance checkpoint_path is required")
    resolved_path = resolve_provenance_path(calibration_checkpoint_path)
    expected_path = resolve_provenance_path(expected_path_value)
    if expected_path != resolved_path:
        raise ValueError("calibration checkpoint path provenance mismatch")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("calibration provenance SHA256 is required")
    checkpoint_sha256 = sha256_file(resolved_path)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError("calibration checkpoint SHA256 provenance mismatch")
    # Validate every provenance field before mutating the model.  A resume
    # lifecycle already loaded the task checkpoint into ``model``; in that
    # case the immutable calibration bytes are checked above but must not be
    # loaded again, otherwise task progress is silently discarded.
    if not isinstance(load_calibration, bool):
        raise TypeError("load_calibration must be a boolean")
    if load_calibration:
        load_validated_checkpoint(model, resolved_path, role=calibration_checkpoint_role)
    semantic_entries = _semantic_entries(model)
    semantic_ids = {id(parameter) for _, parameter in semantic_entries}
    named_parameters = tuple(model.named_parameters())
    all_ids = {id(parameter) for _, parameter in named_parameters}
    if not semantic_ids <= all_ids:
        raise ValueError("semantic calibration parameter is not registered on model")
    trainable = tuple(
        sorted(
            name
            for name, parameter in named_parameters
            if id(parameter) not in semantic_ids
        )
    )
    task_parameter_categories = _registered_task_parameter_categories(model, trainable)
    _require_registered_task_path(trainable, model)
    for _, parameter in named_parameters:
        parameter.requires_grad = id(parameter) not in semantic_ids
    frozen = tuple(sorted(name for name, parameter in named_parameters if not parameter.requires_grad))
    if set(frozen) != {name for name, parameter in named_parameters if id(parameter) in semantic_ids}:
        raise ValueError("task adaptation froze a non-semantic parameter")
    module_ids = semantic_module_ids(model)
    optimizer = build_optimizer(optimizer_name, model, trainable, optimizer_hparams)
    schedule = _freeze_sequence(eta_schedule, "eta_schedule")
    if len(schedule) != epochs or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in schedule
    ):
        raise ValueError("eta_schedule must contain exactly 60 finite numeric values")
    update_count = registered_update_count(epochs, updates_per_epoch)
    semantic_hash = semantic_state_sha256(model, module_ids)
    provenance = {
        "condition": condition,
        "checkpoint_role": calibration_checkpoint_role,
        "checkpoint_path": resolved_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "semantic_module_names": _semantic_module_names(model, module_ids),
        "task_parameter_categories": dict(task_parameter_categories),
        "updates_per_epoch": updates_per_epoch,
        "expected_optimizer_steps": update_count,
    }
    return TaskAdaptationPhase(
        condition=condition,
        calibration_checkpoint_path=resolved_path,
        calibration_checkpoint_role=calibration_checkpoint_role,
        calibration_checkpoint_sha256=checkpoint_sha256,
        semantic_state_sha256=semantic_hash,
        semantic_module_names=_semantic_module_names(model, module_ids),
        epochs=epochs,
        trainable_parameter_names=trainable,
        frozen_parameter_names=frozen,
        optimizer=optimizer,
        optimizer_name=optimizer_name,
        optimizer_hparams=dict(optimizer_hparams),
        eta_schedule=tuple(schedule),
        updates_per_epoch=updates_per_epoch,
        update_count=update_count,
        early_stopping=False,
        primary_checkpoint=primary_checkpoint,
        task_parameter_categories=task_parameter_categories,
        provenance=provenance,
    )


def verify_semantic_state(
    model: object,
    phase: TaskAdaptationPhase,
    *,
    event: str | None = None,
    epoch: int | None = None,
    optimizer_steps: int | None = None,
) -> str:
    """Fail closed when a condition's semantic bytes or journal diverge."""

    current = semantic_state_sha256(model, semantic_module_ids(model))
    if current != phase.semantic_state_sha256:
        raise RuntimeError(
            f"semantic state changed for condition {phase.condition}"
        )
    for record in phase.semantic_state_journal:
        if (
            not isinstance(record, Mapping)
            or record.get("event")
            not in {"epoch_commit", "resume_check", "final_checkpoint"}
            or record.get("semantic_state_sha256") != phase.semantic_state_sha256
        ):
            raise RuntimeError("semantic state journal is missing or changed")
        record_epoch = record.get("epoch")
        record_steps = record.get("optimizer_steps")
        if (
            isinstance(record_epoch, bool)
            or not isinstance(record_epoch, int)
            or record_epoch < 0
            or record_epoch >= phase.epochs
            or isinstance(record_steps, bool)
            or not isinstance(record_steps, int)
            or record_steps < 0
            or record_steps > phase.update_count
        ):
            raise RuntimeError("semantic state journal progress is missing or changed")
    if event is not None:
        if event not in {"epoch_commit", "resume_check", "final_checkpoint"}:
            raise ValueError("unknown semantic journal event")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or epoch >= phase.epochs
            or isinstance(optimizer_steps, bool)
            or not isinstance(optimizer_steps, int)
            or optimizer_steps < 0
            or optimizer_steps > phase.update_count
        ):
            raise ValueError("semantic journal progress is required")
        phase.semantic_state_journal.append(
            {
                "event": event,
                "semantic_state_sha256": current,
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
            }
        )
    return current


__all__ = [
    "SemanticCalibrationPhase",
    "factor_calibration_named_parameter_groups",
    "factor_calibration_parameter_groups",
    "run_calibration_validation",
    "semantic_calibration_phase",
    "split_three_view_contexts",
    "TaskAdaptationPhase",
    "build_optimizer",
    "enforce_semantic_eval_mode",
    "registered_update_count",
    "semantic_module_ids",
    "semantic_state_sha256",
    "task_adaptation_phase",
    "verify_semantic_state",
]
