from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.eval.selection_gate import evaluate_selection_gate


CLASSES = ("Car", "Cyclist", "Pedestrian")


def _write_metrics(path: Path, values: tuple[float, float, float]) -> None:
    payload = {
        "classes": {
            name: {"moderate": {"ap40": value}}
            for name, value in zip(CLASSES, values)
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_protected_gradients(path: Path, *, detection_norm: float = 0.0) -> None:
    records = []
    for step in (50, 100):
        records.append(
            {
                "schema_version": 2,
                "step": step,
                "parameter_groups": {
                    "semantic_anchor": {
                        "gradient_norms": {
                            "detection": detection_norm,
                            "factor": 0.04,
                            "counterfactual": 0.003,
                        },
                        "pairs": {},
                        "schema_version": 1,
                    },
                    "fusion_adapters": {
                        "gradient_norms": {
                            "detection": 0.02,
                            "factor": 0.001,
                            "counterfactual": 0.0001,
                        },
                        "pairs": {},
                        "schema_version": 1,
                    },
                    "localization_adapter": {
                        "gradient_norms": {
                            "detection": 0.01,
                            "factor": 0.0,
                            "counterfactual": 0.0,
                        },
                        "pairs": {},
                        "schema_version": 1,
                    },
                },
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class SelectionGateTest(unittest.TestCase):
    def _evaluate(
        self,
        directory: str,
        *,
        candidate: tuple[float, float, float] = (91.0, 42.0, 62.0),
        detection_norm: float = 0.0,
    ):
        root = Path(directory)
        full = root / "full.json"
        fusion = root / "fusion.json"
        candidate_path = root / "candidate.json"
        gradients = root / "gradient_diagnostics.jsonl"
        _write_metrics(full, (90.0, 40.0, 60.0))
        _write_metrics(fusion, (92.0, 44.0, 63.0))
        _write_metrics(candidate_path, candidate)
        _write_protected_gradients(gradients, detection_norm=detection_norm)
        return evaluate_selection_gate(
            full_metrics=full,
            fusion_metrics=fusion,
            candidate_metrics=candidate_path,
            gradient_diagnostics=gradients,
        )

    def test_accepts_performance_gain_with_protected_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            decision = self._evaluate(directory)

        self.assertTrue(decision.advance)
        self.assertAlmostEqual(decision.candidate_mean_ap40, 65.0)
        self.assertEqual(decision.failed_checks, ())

    def test_rejects_candidate_with_large_class_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            decision = self._evaluate(
                directory,
                candidate=(94.0, 46.0, 58.9),
            )

        self.assertFalse(decision.advance)
        self.assertIn("class_regression", decision.failed_checks)

    def test_rejects_downstream_gradient_leak_into_semantic_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            decision = self._evaluate(directory, detection_norm=1e-4)

        self.assertFalse(decision.advance)
        self.assertIn("semantic_protection", decision.failed_checks)

    def test_rejects_candidate_without_required_performance_gain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            decision = self._evaluate(
                directory,
                candidate=(90.2, 40.2, 60.2),
            )

        self.assertFalse(decision.advance)
        self.assertIn("performance", decision.failed_checks)


if __name__ == "__main__":
    unittest.main()
