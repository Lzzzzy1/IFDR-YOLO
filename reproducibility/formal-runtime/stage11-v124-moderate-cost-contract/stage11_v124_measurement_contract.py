"""Approved Stage11 v124 Moderate-slice and matched-cost contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Callable


CURRENT_PROTOCOL: str = "KITTI_FIT3341_DEV371"
REGISTERED_CLASSES: tuple[str, ...] = ("Pedestrian", "Cyclist")
REGISTERED_SLICES: tuple[str, ...] = (
    "small_25_40",
    "large_gt_80",
    "far_gt_40m",
    "near_0_20m",
)
COST_FIELDS: tuple[str, ...] = (
    "parameters",
    "flops",
    "median_latency_ms",
    "fps",
    "peak_vram_bytes",
    "training_runtime_seconds",
)
SHA256_LENGTH: int = 64
DEVELOPMENT_IDS_ORDERED_SHA256: str = (
    "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
)
KITTI_TYPES_SHA256: str = (
    "131bb9079283d2bb86deb0dc4a9c346c423838a8ab786e5fa3ed5db7e5f0b7e6"
)
KITTI_AP40_SHA256: str = (
    "ce7257c43bd405da93e43455ccb230db921956142ad1b18208eb5924575a7af4"
)


def canonical_bytes(value: object) -> bytes:
    """Encode a JSON-compatible value in the registered canonical form."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash a canonical JSON-compatible value."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def slice_measurement_contract() -> dict[str, object]:
    """Return the approved Moderate-slice measurement contract."""

    return {
        "schema": "stage11-v124-slice-measurement-contract-v1",
        "protocol": CURRENT_PROTOCOL,
        "metric": "KITTI_2D_MODERATE_CONDITIONAL_AP_R40",
        "difficulty": "moderate",
        "classes": list(REGISTERED_CLASSES),
        "slices": {
            "small_25_40": {
                "axis": "height_px",
                "selector": "25 < bbox.height <= 40",
                "role": "FORMAL_NO_HARM_TARGET",
            },
            "large_gt_80": {
                "axis": "height_px",
                "selector": "bbox.height > 80",
                "role": "FORMAL_NO_HARM_CONTROL",
            },
            "far_gt_40m": {
                "axis": "depth_m",
                "selector": "location_xyz[2] > 40",
                "role": "FORMAL_NO_HARM_TARGET",
            },
            "near_0_20m": {
                "axis": "depth_m",
                "selector": "0 < location_xyz[2] <= 20",
                "role": "FORMAL_NO_HARM_CONTROL",
            },
        },
        "reporting": ["per_class", "two_class_unweighted_macro"],
        "zero_gt_state": "NOT_ESTIMABLE",
        "derived_false_negatives": "num_valid_gt - true_positives",
        "hard_compatibility": {
            "metric": "KITTI_2D_CONDITIONAL_AP40",
            "difficulty": "hard",
            "decision_role": "DESCRIPTIVE_ONLY",
            "formal_gate": False,
        },
        "source_sha256": {
            "ifdr_yolo/data/kitti_types.py": KITTI_TYPES_SHA256,
            "ifdr_yolo/eval/kitti_ap40.py": KITTI_AP40_SHA256,
        },
    }


def cost_measurement_contract() -> dict[str, object]:
    """Return the approved matched inference/training-cost contract."""

    prediction_protocol = {
        "imgsz": 640,
        "batch": 1,
        "fp32": True,
        "rect": True,
        "augment": False,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "max_nms": 30000,
    }
    return {
        "schema": "stage11-v124-cost-measurement-contract-v1",
        "protocol": CURRENT_PROTOCOL,
        "dataset": {
            "role": "development",
            "count": 371,
            "ordered_ids_sha256": DEVELOPMENT_IDS_ORDERED_SHA256,
        },
        "runtime_precedent": {
            "ultralytics": "8.4.98",
            "torch": "2.8.0+cu128",
            "torchvision": "0.23.0+cu128",
            "cudnn": 91002,
        },
        "prediction_protocol": prediction_protocol,
        "latency": {
            "clock": "time.perf_counter_ns",
            "warmup_images": 50,
            "measured_passes": 5,
            "sample_count": 1855,
            "order": "registered_371_order_repeated_five_times",
            "synchronization": "torch.cuda.synchronize_before_and_after_each_invocation",
            "formal_statistic": "median_per_image_synchronized_latency_ms",
            "diagnostics": ["per_pass_median_latency_ms", "p95_latency_ms"],
        },
        "fps": {
            "definition": "measured_images / total_synchronized_seconds",
            "sample_count": 1855,
        },
        "peak_vram": {
            "reset": "torch.cuda.reset_peak_memory_stats_after_model_and_first_input",
            "read": "torch.cuda.max_memory_allocated_after_measured_passes",
            "includes_resident_model": True,
        },
        "parameters": {
            "definition": "sum(parameter.numel() for parameter in model.parameters())",
        },
        "flops": {
            "input": "fp32_1x3x640x640",
            "requires_tool_name": True,
            "requires_tool_version": True,
            "requires_tool_source_sha256": True,
            "requires_counting_convention": True,
            "missing_tool_state": "BLOCKED_FLOPS_TOOL_MISSING",
            "identity_mismatch_state": "BLOCKED_FLOPS_TOOL_IDENTITY_MISMATCH",
        },
        "training_runtime": {
            "source": "canonical_mirrored_ultralytics_results_csv",
            "epochs": list(range(1, 31)),
            "statistic": "final_cumulative_time_seconds",
        },
        "required_pair_identity_equal": [
            "protocol",
            "development_ids_ordered_sha256",
            "hardware_runtime_sha256",
            "prediction_protocol_sha256",
            "profiler_identity_sha256",
        ],
        "metrics": list(COST_FIELDS),
    }


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{role} schema differs")


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise ValueError(f"{role} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{role} must be a lowercase SHA256")
    return value


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{role} must be finite numeric")
    return number


def _positive(value: object, role: str) -> float:
    number = _finite(value, role)
    if number <= 0.0:
        raise ValueError(f"{role} must be positive")
    return number


def _integer(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{role} must be an integer")
    return value


def _slice_selectors() -> dict[str, Callable[[object], bool]]:
    return {
        "small_25_40": lambda value: 25.0 < value.bbox.height <= 40.0,
        "large_gt_80": lambda value: value.bbox.height > 80.0,
        "far_gt_40m": lambda value: value.location_xyz[2] > 40.0,
        "near_0_20m": lambda value: 0.0 < value.location_xyz[2] <= 20.0,
    }


def evaluate_moderate_slices(
    gt_by_image: dict[str, tuple[object, ...]],
    detections_by_image: dict[str, tuple[object, ...]],
) -> dict[str, object]:
    """Evaluate the four approved slices with the locked Moderate AP core."""

    from ifdr_yolo.data.kitti_types import Difficulty
    from ifdr_yolo.eval.kitti_ap40 import evaluate_class

    classes: dict[str, dict[str, object]] = {}
    selectors = _slice_selectors()
    for class_name in REGISTERED_CLASSES:
        per_slice: dict[str, object] = {}
        for slice_name in REGISTERED_SLICES:
            metrics = evaluate_class(
                gt_by_image=gt_by_image,
                detections_by_image=detections_by_image,
                class_name=class_name,
                difficulty=Difficulty.MODERATE,
                valid_selector=selectors[slice_name],
            )
            state = "PASS" if metrics.num_valid_gt > 0 else "NOT_ESTIMABLE"
            per_slice[slice_name] = {
                "state": state,
                "ap_r40": metrics.ap40 if state == "PASS" else None,
                "num_valid_gt": metrics.num_valid_gt,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "derived_false_negatives": metrics.num_valid_gt
                - metrics.true_positives,
                "ignored_detections": metrics.ignored_detections,
            }
        classes[class_name] = per_slice

    macro: dict[str, object] = {}
    for slice_name in REGISTERED_SLICES:
        values = [classes[name][slice_name]["ap_r40"] for name in REGISTERED_CLASSES]
        estimable = all(value is not None for value in values)
        macro[slice_name] = {
            "state": "PASS" if estimable else "NOT_ESTIMABLE",
            "ap_r40": (
                sum(float(value) for value in values) / len(values)
                if estimable
                else None
            ),
        }

    return {
        "schema": "stage11-v124-slice-measurement-v1",
        "state": "PASS"
        if all(value["state"] == "PASS" for value in macro.values())
        else "NOT_ESTIMABLE",
        "protocol": CURRENT_PROTOCOL,
        "difficulty": "moderate",
        "measurement_contract_sha256": canonical_sha256(
            slice_measurement_contract()
        ),
        "classes": classes,
        "macro": macro,
        "hard_compatibility": {
            "decision_role": "DESCRIPTIVE_ONLY",
            "formal_gate": False,
        },
    }


def validate_slice_measurement(value: object) -> dict[str, object]:
    """Validate and return an immutable canonical copy of a rich slice record."""

    root = _mapping(value, "slice measurement")
    _exact(
        root,
        {
            "schema",
            "state",
            "protocol",
            "difficulty",
            "measurement_contract_sha256",
            "classes",
            "macro",
            "hard_compatibility",
        },
        "slice measurement",
    )
    if (
        root["schema"] != "stage11-v124-slice-measurement-v1"
        or root["protocol"] != CURRENT_PROTOCOL
        or root["difficulty"] != "moderate"
        or root["measurement_contract_sha256"]
        != canonical_sha256(slice_measurement_contract())
    ):
        raise ValueError("slice measurement identity differs")
    if root["state"] not in {"PASS", "NOT_ESTIMABLE"}:
        raise ValueError("slice measurement state differs")
    hard = _mapping(root["hard_compatibility"], "hard compatibility")
    _exact(hard, {"decision_role", "formal_gate"}, "hard compatibility")
    if hard != {"decision_role": "DESCRIPTIVE_ONLY", "formal_gate": False}:
        raise ValueError("hard compatibility identity differs")

    classes = _mapping(root["classes"], "slice classes")
    if set(classes) != set(REGISTERED_CLASSES):
        raise ValueError("slice class schema differs")
    expected_macro: dict[str, float | None] = {}
    for class_name in REGISTERED_CLASSES:
        per_slice = _mapping(classes[class_name], f"{class_name} slices")
        if set(per_slice) != set(REGISTERED_SLICES):
            raise ValueError(f"{class_name} slice schema differs")
        for slice_name in REGISTERED_SLICES:
            measurement = _mapping(
                per_slice[slice_name], f"{class_name} {slice_name}"
            )
            _exact(
                measurement,
                {
                    "state",
                    "ap_r40",
                    "num_valid_gt",
                    "true_positives",
                    "false_positives",
                    "derived_false_negatives",
                    "ignored_detections",
                },
                f"{class_name} {slice_name}",
            )
            count = _integer(
                measurement["num_valid_gt"], f"{class_name} {slice_name} GT"
            )
            true_positives = _integer(
                measurement["true_positives"], f"{class_name} {slice_name} TP"
            )
            false_positives = _integer(
                measurement["false_positives"], f"{class_name} {slice_name} FP"
            )
            false_negatives = _integer(
                measurement["derived_false_negatives"],
                f"{class_name} {slice_name} FN",
            )
            ignored = _integer(
                measurement["ignored_detections"],
                f"{class_name} {slice_name} ignored",
            )
            if min(count, true_positives, false_positives, false_negatives, ignored) < 0:
                raise ValueError(f"{class_name} {slice_name} count is negative")
            if true_positives + false_negatives != count:
                raise ValueError(f"{class_name} {slice_name} derived FN differs")
            if count == 0:
                if measurement["state"] != "NOT_ESTIMABLE" or measurement["ap_r40"] is not None:
                    raise ValueError(f"{class_name} {slice_name} zero-GT state differs")
            else:
                ap = _finite(measurement["ap_r40"], f"{class_name} {slice_name} AP")
                if measurement["state"] != "PASS" or not 0.0 <= ap <= 100.0:
                    raise ValueError(f"{class_name} {slice_name} AP state differs")

    macro = _mapping(root["macro"], "slice macro")
    if set(macro) != set(REGISTERED_SLICES):
        raise ValueError("slice macro schema differs")
    all_estimable = True
    for slice_name in REGISTERED_SLICES:
        measurement = _mapping(macro[slice_name], f"macro {slice_name}")
        _exact(measurement, {"state", "ap_r40"}, f"macro {slice_name}")
        class_values = [classes[name][slice_name]["ap_r40"] for name in REGISTERED_CLASSES]
        estimable = all(value is not None for value in class_values)
        all_estimable = all_estimable and estimable
        if not estimable:
            if measurement != {"state": "NOT_ESTIMABLE", "ap_r40": None}:
                raise ValueError(f"macro {slice_name} state differs")
            expected_macro[slice_name] = None
        else:
            expected = sum(float(value) for value in class_values) / len(class_values)
            observed = _finite(measurement["ap_r40"], f"macro {slice_name} AP")
            if measurement["state"] != "PASS" or abs(observed - expected) > 1e-12:
                raise ValueError(f"macro {slice_name} arithmetic differs")
            expected_macro[slice_name] = expected
    expected_state = "PASS" if all_estimable else "NOT_ESTIMABLE"
    if root["state"] != expected_state:
        raise ValueError("slice measurement aggregate state differs")
    return json.loads(canonical_bytes(root).decode("utf-8"))


def build_v123_slice_projection(
    control_value: object,
    method_value: object,
    seed: int,
    report_sha256: str,
) -> dict[str, object]:
    """Project a validated rich pair into the frozen v123 macro schema."""

    if isinstance(seed, bool) or seed not in range(5):
        raise ValueError("slice projection seed is not registered")
    report_sha = _sha(report_sha256, "slice projection report SHA")
    control = validate_slice_measurement(control_value)
    method = validate_slice_measurement(method_value)
    if control["state"] != "PASS" or method["state"] != "PASS":
        raise ValueError("slice projection requires estimable rich measurements")
    slices: dict[str, object] = {}
    for projected, source in (("small", "small_25_40"), ("far", "far_gt_40m")):
        control_ap = float(control["macro"][source]["ap_r40"])
        method_ap = float(method["macro"][source]["ap_r40"])
        slices[projected] = {
            "control_ap_r40": control_ap,
            "method_ap_r40": method_ap,
            "difference_ap_r40": method_ap - control_ap,
        }
    return {
        "schema": "stage11-v123-slice-pair-receipt-v1",
        "state": "PASS",
        "protocol": CURRENT_PROTOCOL,
        "seed": seed,
        "report_sha256": report_sha,
        "slice_measurement_contract_sha256": canonical_sha256(
            slice_measurement_contract()
        ),
        "slices": slices,
    }


def validate_cost_measurement(value: object) -> dict[str, object]:
    """Validate one candidate's matched-cost measurement."""

    root = _mapping(value, "cost measurement")
    _exact(
        root,
        {
            "schema",
            "state",
            "measurement_contract_sha256",
            "identity",
            "metrics",
            "diagnostics",
        },
        "cost measurement",
    )
    if (
        root["schema"] != "stage11-v124-cost-measurement-v1"
        or root["state"] != "PASS"
        or root["measurement_contract_sha256"]
        != canonical_sha256(cost_measurement_contract())
    ):
        raise ValueError("cost measurement identity differs")
    identity = _mapping(root["identity"], "cost identity")
    _exact(
        identity,
        {
            "protocol",
            "development_ids_ordered_sha256",
            "hardware_runtime_sha256",
            "prediction_protocol_sha256",
            "profiler_identity_sha256",
            "model_sha256",
            "checkpoint_sha256",
        },
        "cost identity",
    )
    if identity["protocol"] != CURRENT_PROTOCOL:
        raise ValueError("cost identity protocol differs")
    if identity["development_ids_ordered_sha256"] != DEVELOPMENT_IDS_ORDERED_SHA256:
        raise ValueError("cost identity development order differs")
    if identity["prediction_protocol_sha256"] != canonical_sha256(
        cost_measurement_contract()["prediction_protocol"]
    ):
        raise ValueError("cost identity prediction protocol differs")
    for name in (
        "hardware_runtime_sha256",
        "prediction_protocol_sha256",
        "profiler_identity_sha256",
        "model_sha256",
        "checkpoint_sha256",
    ):
        _sha(identity[name], f"cost identity {name}")

    metrics = _mapping(root["metrics"], "cost metrics")
    if set(metrics) != set(COST_FIELDS):
        raise ValueError("cost metric schema differs")
    for name in COST_FIELDS:
        _positive(metrics[name], f"cost metric {name}")

    diagnostics = _mapping(root["diagnostics"], "cost diagnostics")
    _exact(
        diagnostics,
        {
            "warmup_images",
            "measured_passes",
            "latency_sample_count",
            "per_pass_median_latency_ms",
            "p95_latency_ms",
            "total_synchronized_seconds",
        },
        "cost diagnostics",
    )
    if diagnostics["warmup_images"] != 50 or diagnostics["measured_passes"] != 5 or diagnostics["latency_sample_count"] != 1855:
        raise ValueError("cost diagnostic sampling differs")
    medians = diagnostics["per_pass_median_latency_ms"]
    if not isinstance(medians, Sequence) or isinstance(medians, (str, bytes)) or len(medians) != 5:
        raise ValueError("cost diagnostic per-pass medians differ")
    for index, median in enumerate(medians):
        _positive(median, f"cost diagnostic pass {index} median")
    _positive(diagnostics["p95_latency_ms"], "cost diagnostic p95")
    total_seconds = _positive(
        diagnostics["total_synchronized_seconds"], "cost diagnostic total time"
    )
    expected_fps = 1855.0 / total_seconds
    observed_fps = _positive(metrics["fps"], "cost metric fps")
    if abs(observed_fps - expected_fps) > max(1e-9, expected_fps * 1e-9):
        raise ValueError("cost metric fps differs from synchronized total")
    return json.loads(canonical_bytes(root).decode("utf-8"))


def build_v123_cost_projection(
    control_value: object,
    method_value: object,
    seed: int,
    report_sha256: str,
) -> dict[str, object]:
    """Project a validated matched pair into the frozen v123 cost schema."""

    if isinstance(seed, bool) or seed not in range(5):
        raise ValueError("cost projection seed is not registered")
    report_sha = _sha(report_sha256, "cost projection report SHA")
    control = validate_cost_measurement(control_value)
    method = validate_cost_measurement(method_value)
    equal_fields = cost_measurement_contract()["required_pair_identity_equal"]
    for name in equal_fields:
        if control["identity"][name] != method["identity"][name]:
            raise ValueError(f"cost matched identity differs: {name}")
    metrics: dict[str, object] = {}
    for name in COST_FIELDS:
        control_value_number = float(control["metrics"][name])
        method_value_number = float(method["metrics"][name])
        ratio = (
            control_value_number / method_value_number
            if name == "fps"
            else method_value_number / control_value_number
        )
        metrics[name] = {
            "control": control_value_number,
            "method": method_value_number,
            "overhead_ratio": ratio,
        }
    return {
        "schema": "stage11-v123-cost-pair-receipt-v1",
        "state": "PASS",
        "protocol": CURRENT_PROTOCOL,
        "seed": seed,
        "report_sha256": report_sha,
        "measurement_contract_sha256": canonical_sha256(
            cost_measurement_contract()
        ),
        "metrics": metrics,
    }


__all__ = [
    "COST_FIELDS",
    "CURRENT_PROTOCOL",
    "REGISTERED_CLASSES",
    "REGISTERED_SLICES",
    "build_v123_cost_projection",
    "build_v123_slice_projection",
    "canonical_bytes",
    "canonical_sha256",
    "cost_measurement_contract",
    "evaluate_moderate_slices",
    "slice_measurement_contract",
    "validate_cost_measurement",
    "validate_slice_measurement",
]
