from __future__ import annotations

from typing import Any


EXPECTED_STRIDES = [4.0, 8.0, 16.0, 32.0]
EXPECTED_PARAMETERS = 25_052_620
EXPECTED_STATE_ITEMS = 581


def inspect_p2_model(model: Any) -> dict[str, object]:
    strides = [float(value) for value in model.stride.tolist()]
    detect = model.model[-1]
    detect_inputs = len(detect.cv2)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    state_items = len(model.state_dict())
    if strides != EXPECTED_STRIDES:
        raise RuntimeError(f"P2 stride mismatch: {strides}")
    if detect_inputs != 4:
        raise RuntimeError(f"P2 Detect input mismatch: {detect_inputs}")
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"P2 parameter mismatch: {parameters}")
    if state_items != EXPECTED_STATE_ITEMS:
        raise RuntimeError(f"P2 state item mismatch: {state_items}")
    return {
        "strides": strides,
        "detect_inputs": detect_inputs,
        "parameters": parameters,
        "state_items": state_items,
    }
