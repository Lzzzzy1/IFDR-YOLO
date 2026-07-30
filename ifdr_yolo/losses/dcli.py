from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _bounded_scalar(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return float(value)


def _validate_unit_tensor(tensor: torch.Tensor, field: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{field} must be a tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{field} must be floating point")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{field} must contain finite values")
    if torch.any((tensor < 0.0) | (tensor > 1.0)):
        raise ValueError(f"{field} must be within [0, 1]")


def dcli_localization_error(
    localization_error: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Apply bounded uncertainty scaling without a shortcut gradient."""

    beta = _bounded_scalar(beta, "beta")
    if not isinstance(localization_error, torch.Tensor):
        raise ValueError("localization_error must be a tensor")
    if localization_error.shape != uncertainty.shape:
        raise ValueError(
            "localization_error and uncertainty must share shape"
        )
    if not localization_error.is_floating_point():
        raise ValueError("localization_error must be floating point")
    if not torch.isfinite(localization_error).all():
        raise ValueError("localization_error must contain finite values")
    _validate_unit_tensor(uncertainty, "uncertainty")
    if beta == 0.0:
        return localization_error
    scale = 1.0 + beta * uncertainty.detach()
    return localization_error / scale + scale.log()


def normalized_dfl_entropy(
    distribution_logits: torch.Tensor,
    *,
    reg_max: int,
) -> torch.Tensor:
    """Return mean normalized entropy of the four DFL box sides."""

    if (
        isinstance(reg_max, bool)
        or not isinstance(reg_max, int)
        or reg_max <= 1
    ):
        raise ValueError("reg_max must be an integer greater than one")
    if (
        not isinstance(distribution_logits, torch.Tensor)
        or distribution_logits.ndim != 3
        or distribution_logits.shape[-1] != 4 * reg_max
        or not distribution_logits.is_floating_point()
    ):
        raise ValueError(
            "distribution_logits must have shape [batch, anchors, 4*reg_max]"
        )
    logits = distribution_logits.reshape(
        *distribution_logits.shape[:2],
        4,
        reg_max,
    )
    probabilities = logits.softmax(dim=-1)
    log_probabilities = logits.log_softmax(dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    entropy = entropy.mean(dim=-1) / math.log(reg_max)
    return entropy.clamp(0.0, 1.0)


def _non_negative_weight(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def derive_localization_uncertainty(
    factors: torch.Tensor,
    dfl_entropy: torch.Tensor,
    *,
    factor_weights: tuple[float, float] = (1.0, 1.0),
    entropy_weight: float = 1.0,
) -> torch.Tensor:
    """Fuse sampling, visibility and DFL ambiguity into one auditable score."""

    _validate_unit_tensor(factors, "factors")
    _validate_unit_tensor(dfl_entropy, "dfl_entropy")
    if factors.ndim < 1 or factors.shape[-1] != 2:
        raise ValueError("factors must end with sampling and visibility")
    if factors.shape[:-1] != dfl_entropy.shape:
        raise ValueError("factor and DFL entropy shapes are incompatible")
    if not isinstance(factor_weights, tuple) or len(factor_weights) != 2:
        raise ValueError("factor_weights must contain exactly two values")
    sampling_weight = _non_negative_weight(
        factor_weights[0],
        "factor_weights[0]",
    )
    visibility_weight = _non_negative_weight(
        factor_weights[1],
        "factor_weights[1]",
    )
    entropy_weight = _non_negative_weight(
        entropy_weight,
        "entropy_weight",
    )
    denominator = sampling_weight + visibility_weight + entropy_weight
    if denominator <= 0.0:
        raise ValueError("at least one uncertainty weight must be positive")
    return (
        sampling_weight * factors[..., 0]
        + visibility_weight * factors[..., 1]
        + entropy_weight * dfl_entropy
    ) / denominator


def uncertainty_calibration_loss(
    predicted_uncertainty: torch.Tensor,
    localization_residual: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calibrate uncertainty to detached residuals without moving box targets."""

    if (
        not isinstance(predicted_uncertainty, torch.Tensor)
        or not isinstance(localization_residual, torch.Tensor)
        or predicted_uncertainty.shape != localization_residual.shape
    ):
        raise ValueError(
            "predicted uncertainty and localization residual must share shape"
        )
    target = localization_residual.detach().clamp(0.0, 1.0)
    elementwise = F.smooth_l1_loss(
        predicted_uncertainty,
        target,
        reduction="none",
    )
    if valid_mask is None:
        return elementwise.mean()
    if (
        not isinstance(valid_mask, torch.Tensor)
        or valid_mask.dtype != torch.bool
        or valid_mask.shape != elementwise.shape
    ):
        raise ValueError("valid_mask must be a boolean tensor with matching shape")
    if not valid_mask.any():
        return predicted_uncertainty.sum() * 0.0
    return elementwise[valid_mask].mean()
