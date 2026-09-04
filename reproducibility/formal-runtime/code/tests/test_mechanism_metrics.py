import unittest

import torch

from ifdr_yolo.eval.mechanism_metrics import (
    MechanismResponse,
    measure_paired_mechanism_response,
    summarize_mechanism_responses,
)
from ifdr_yolo.models.gated_fusion import ReliabilityContext


def _context(
    factors: tuple[float, float],
    branches: tuple[float, float],
) -> ReliabilityContext:
    factor_map = torch.tensor(factors).reshape(1, 2, 1, 1)
    branch_map = torch.tensor(branches).reshape(1, 2, 1, 1)
    return ReliabilityContext(
        factors=factor_map,
        branch_weights=branch_map,
        gate_strength=0.5,
    )


class PairedMechanismMetricsTest(unittest.TestCase):
    def test_sampling_response_separates_target_leakage_and_routing(self) -> None:
        clean = {11: _context((0.2, 0.3), (0.5, 0.5))}
        intervention = {11: _context((0.6, 0.3), (0.7, 0.3))}
        delta = torch.tensor([0.4, 0.0]).reshape(1, 2, 1, 1)
        weight = torch.ones_like(delta)

        responses = measure_paired_mechanism_response(
            intervention,
            clean,
            delta,
            weight,
            kind="sampling",
            role="object",
            strength=0.4,
        )

        self.assertEqual(len(responses), 1)
        response = responses[0]
        self.assertEqual(response.node, 11)
        self.assertAlmostEqual(response.target_response, 0.4, places=6)
        self.assertAlmostEqual(response.expected_response, 0.4, places=6)
        self.assertAlmostEqual(response.target_mae, 0.0, places=6)
        self.assertAlmostEqual(response.leakage, 0.0, places=6)
        self.assertAlmostEqual(response.selectivity, 1.0, places=6)
        self.assertAlmostEqual(response.routing_shift, 0.2, places=6)

    def test_non_target_delta_reduces_selectivity(self) -> None:
        clean = {11: _context((0.2, 0.3), (0.5, 0.5))}
        intervention = {11: _context((0.6, 0.4), (0.5, 0.5))}
        delta = torch.tensor([0.4, 0.0]).reshape(1, 2, 1, 1)

        response = measure_paired_mechanism_response(
            intervention,
            clean,
            delta,
            torch.ones_like(delta),
            kind="sampling",
            role="background",
            strength=0.4,
        )[0]

        self.assertAlmostEqual(response.leakage, 0.1, places=6)
        self.assertAlmostEqual(response.selectivity, 0.8, places=6)

    def test_summary_reports_strength_monotonicity_without_node_pseudoreplication(
        self,
    ) -> None:
        responses = []
        for strength in (0.2, 0.4, 0.6):
            for node in (11, 14):
                responses.append(
                    MechanismResponse(
                        kind="visibility",
                        role="object",
                        strength=strength,
                        node=node,
                        target_response=strength,
                        expected_response=strength,
                        target_mae=0.0,
                        leakage=0.0,
                        selectivity=1.0,
                        routing_shift=strength / 2.0,
                    )
                )

        summary = summarize_mechanism_responses(responses)

        curve = summary["conditions"]["visibility"]["object"]["aggregate"]
        self.assertEqual(curve["strengths"], [0.2, 0.4, 0.6])
        self.assertEqual(curve["samples_per_strength"], [2, 2, 2])
        self.assertAlmostEqual(curve["spearman"], 1.0, places=6)
        self.assertEqual(curve["monotonic_violations"], 0)
        self.assertAlmostEqual(curve["direction_agreement"], 1.0)
        self.assertIn("11", summary["conditions"]["visibility"]["object"]["nodes"])

    def test_summary_detects_non_monotonic_response(self) -> None:
        responses = [
            MechanismResponse(
                kind="sampling",
                role="object",
                strength=strength,
                node=11,
                target_response=response,
                expected_response=strength,
                target_mae=abs(response - strength),
                leakage=0.0,
                selectivity=1.0,
                routing_shift=0.1,
            )
            for strength, response in ((0.2, 0.2), (0.4, 0.5), (0.6, 0.3))
        ]

        curve = summarize_mechanism_responses(responses)["conditions"][
            "sampling"
        ]["object"]["aggregate"]

        self.assertEqual(curve["monotonic_violations"], 1)
        self.assertLess(curve["spearman"], 1.0)

    def test_rejects_context_node_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same nodes"):
            measure_paired_mechanism_response(
                {11: _context((0.2, 0.3), (0.5, 0.5))},
                {14: _context((0.2, 0.3), (0.5, 0.5))},
                torch.zeros(1, 2, 1, 1),
                torch.ones(1, 2, 1, 1),
                kind="sampling",
                role="object",
                strength=0.2,
            )


if __name__ == "__main__":
    unittest.main()
