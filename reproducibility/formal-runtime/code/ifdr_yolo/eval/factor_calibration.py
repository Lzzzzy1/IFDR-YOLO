from __future__ import annotations

import math

import torch


def _validated_vectors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(
        isinstance(value, torch.Tensor)
        for value in (prediction, target, weight)
    ):
        raise ValueError("prediction, target and weight must be tensors")
    if prediction.shape != target.shape or prediction.shape != weight.shape:
        raise ValueError("prediction, target and weight shapes must match")
    if prediction.numel() == 0:
        raise ValueError("factor tensors must not be empty")
    prediction = prediction.detach().to(dtype=torch.float64).reshape(-1)
    target = target.detach().to(dtype=torch.float64).reshape(-1)
    weight = weight.detach().to(dtype=torch.float64).reshape(-1)
    if not (
        torch.isfinite(prediction).all()
        and torch.isfinite(target).all()
        and torch.isfinite(weight).all()
    ):
        raise ValueError("factor tensors must contain finite values")
    if torch.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("prediction values must be within [0, 1]")
    if torch.any((target < 0.0) | (target > 1.0)):
        raise ValueError("target values must be within [0, 1]")
    if torch.any(weight < 0.0):
        raise ValueError("weight values must be non-negative")
    support = weight > 0.0
    if not torch.any(support):
        raise ValueError("at least one element must have positive weight")
    return prediction[support], target[support], weight[support]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum()


def factor_calibration_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    bins: int = 10,
) -> dict[str, object]:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    prediction, target, weight = _validated_vectors(
        prediction,
        target,
        weight,
    )
    error = prediction - target
    mae = _weighted_mean(error.abs(), weight)
    rmse = _weighted_mean(error.square(), weight).sqrt()
    bias = _weighted_mean(error, weight)

    prediction_mean = _weighted_mean(prediction, weight)
    target_mean = _weighted_mean(target, weight)
    prediction_delta = prediction - prediction_mean
    target_delta = target - target_mean
    prediction_variance = _weighted_mean(
        prediction_delta.square(),
        weight,
    )
    target_variance = _weighted_mean(target_delta.square(), weight)
    pearson: float | None
    if prediction_variance <= 0.0 or target_variance <= 0.0:
        pearson = None
    else:
        covariance = _weighted_mean(
            prediction_delta * target_delta,
            weight,
        )
        pearson = float(
            covariance / (prediction_variance * target_variance).sqrt()
        )
        pearson = min(max(pearson, -1.0), 1.0)

    bin_payload: list[dict[str, object]] = []
    weighted_calibration_error = 0.0
    total_weight = float(weight.sum())
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (prediction >= lower) & (
            prediction <= upper if index == bins - 1 else prediction < upper
        )
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        selected_weight = weight[selected]
        weight_sum = float(selected_weight.sum())
        mean_prediction = float(
            _weighted_mean(prediction[selected], selected_weight)
        )
        mean_target = float(_weighted_mean(target[selected], selected_weight))
        gap = abs(mean_prediction - mean_target)
        weighted_calibration_error += gap * weight_sum
        bin_payload.append(
            {
                "lower": lower,
                "upper": upper,
                "count": selected_count,
                "weight": weight_sum,
                "mean_prediction": mean_prediction,
                "mean_target": mean_target,
                "absolute_gap": gap,
            }
        )

    ece = weighted_calibration_error / total_weight
    assert math.isfinite(ece)
    return {
        "count": int(prediction.numel()),
        "weight": total_weight,
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "pearson": pearson,
        "ece": ece,
        "bins": bin_payload,
    }


def summarize_factor_calibration(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    bins: int = 10,
) -> dict[str, object]:
    if prediction.ndim < 2 or prediction.shape[1] != 2:
        raise ValueError("factor tensors must have two channels at dimension 1")
    if target.shape != prediction.shape or weight.shape != prediction.shape:
        raise ValueError("prediction, target and weight shapes must match")
    return {
        "schema_version": 1,
        "factors": {
            "sampling": factor_calibration_metrics(
                prediction[:, 0],
                target[:, 0],
                weight[:, 0],
                bins=bins,
            ),
            "visibility": factor_calibration_metrics(
                prediction[:, 1],
                target[:, 1],
                weight[:, 1],
                bins=bins,
            ),
        },
    }
