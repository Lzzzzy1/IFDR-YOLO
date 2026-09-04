from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTRACT_ROOT = ROOT.parent / "stage11-v124-moderate-cost-contract"
EXECUTOR_ROOT = ROOT.parent / "stage11-v128-seed0-measurement"
sys.path.insert(0, str(CONTRACT_ROOT))
sys.path.insert(0, str(EXECUTOR_ROOT))


def _load_module() -> object:
    path = ROOT / "stage11_v200_slice_cost_measurement.py"
    spec = importlib.util.spec_from_file_location("stage11_v200_slice_cost_measurement", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load slice/cost module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measurement = _load_module()


class SliceCostMeasurementTests(unittest.TestCase):
    def test_cost_identity_uses_exact_frozen_prediction_protocol(self) -> None:
        from stage11_v128_measurement_executor import build_cost_measurement

        identity = measurement.build_cost_identity(
            {"python": "3.12.3", "torch": "2.8.0+cu128", "cuda": "12.8", "gpu": "RTX 5090"},
            {"tool": "torch.profiler", "torch": "2.8.0+cu128", "input": "fp32_1x3x640x640"},
            "1" * 64,
            "2" * 64,
        )
        observed = build_cost_measurement(
            identity,
            1,
            1.0,
            [[1.0] * 371 for _ in range(5)],
            1,
            1.0,
        )
        self.assertEqual(observed["state"], "PASS")

    def test_registered_specs_are_exact_five_seed_pairs(self) -> None:
        validated = measurement.validate_run_specs(measurement.RUN_SPECS)
        self.assertEqual(len(validated), 10)
        self.assertEqual(sorted({item["seed"] for item in validated}), [0, 1, 2, 3, 4])
        for seed in range(5):
            self.assertEqual(
                {item["candidate"] for item in validated if item["seed"] == seed},
                {"PLAIN_P2", "DCLI"},
            )

    def test_historical_seed_is_rejected(self) -> None:
        invalid = [dict(item) for item in measurement.RUN_SPECS]
        invalid[-1]["seed"] = 17
        with self.assertRaisesRegex(ValueError, "exact seeds"):
            measurement.validate_run_specs(invalid)

    def test_yolo_detection_format_is_deterministic(self) -> None:
        rows = [(1, 0.75, 0.1, 0.2, 0.3, 0.4), (0, 0.5, 0.6, 0.7, 0.2, 0.1)]
        self.assertEqual(
            measurement.format_yolo_detections(rows),
            b"1 0.100000000 0.200000000 0.300000000 0.400000000 0.750000000\n"
            b"0 0.600000000 0.700000000 0.200000000 0.100000000 0.500000000\n",
        )


if __name__ == "__main__":
    unittest.main()
