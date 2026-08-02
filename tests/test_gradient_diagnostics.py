import unittest

import torch

from ifdr_yolo.experiments.gradient_diagnostics import (
    GradientConflictAccumulator,
    ScheduledGradientDiagnostics,
    gradient_conflict_snapshot,
)


class GradientDiagnosticsTest(unittest.TestCase):
    def test_detects_opposing_gradients_without_populating_parameter_grad(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        first = parameter.sum()
        second = -parameter.sum()

        snapshot = gradient_conflict_snapshot(
            {"detection": first, "factor": second},
            (parameter,),
        )

        self.assertAlmostEqual(
            snapshot["gradient_norms"]["detection"],
            2.0**0.5,
        )
        pair = snapshot["pairs"]["detection::factor"]
        self.assertAlmostEqual(pair["cosine"], -1.0)
        self.assertTrue(pair["conflict"])
        self.assertIsNone(parameter.grad)

    def test_zero_gradient_has_no_defined_cosine(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        active = parameter.square().sum()
        inactive = active * 0.0

        snapshot = gradient_conflict_snapshot(
            {"active": active, "inactive": inactive},
            (parameter,),
        )

        self.assertEqual(snapshot["gradient_norms"]["inactive"], 0.0)
        pair = snapshot["pairs"]["active::inactive"]
        self.assertIsNone(pair["cosine"])
        self.assertFalse(pair["conflict"])

    def test_accumulator_reports_conflict_frequency_and_negative_cosine(self) -> None:
        accumulator = GradientConflictAccumulator()
        accumulator.update(
            {
                "gradient_norms": {"a": 1.0, "b": 1.0},
                "pairs": {
                    "a::b": {"cosine": -0.5, "conflict": True},
                },
            }
        )
        accumulator.update(
            {
                "gradient_norms": {"a": 1.0, "b": 1.0},
                "pairs": {
                    "a::b": {"cosine": 0.25, "conflict": False},
                },
            }
        )

        summary = accumulator.summary()

        pair = summary["pairs"]["a::b"]
        self.assertEqual(pair["observations"], 2)
        self.assertEqual(pair["defined_cosines"], 2)
        self.assertEqual(pair["conflict_frequency"], 0.5)
        self.assertEqual(pair["mean_negative_cosine"], -0.5)

    def test_scheduled_recorder_samples_exact_interval_and_drains_once(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        recorder = ScheduledGradientDiagnostics(interval=2)

        first = recorder.observe(
            {"a": parameter.sum(), "b": -parameter.sum()},
            (parameter,),
        )
        second = recorder.observe(
            {"a": parameter.sum(), "b": -parameter.sum()},
            (parameter,),
        )

        self.assertIsNone(first)
        self.assertEqual(second["step"], 2)
        self.assertEqual(recorder.drain(), (second,))
        self.assertEqual(recorder.drain(), ())


if __name__ == "__main__":
    unittest.main()
