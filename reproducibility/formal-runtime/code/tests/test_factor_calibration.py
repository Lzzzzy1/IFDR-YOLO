import unittest

import torch

from ifdr_yolo.eval.factor_calibration import (
    factor_calibration_metrics,
    summarize_factor_calibration,
)


class FactorCalibrationTest(unittest.TestCase):
    def test_perfect_prediction_has_zero_error_and_unit_correlation(self) -> None:
        prediction = torch.tensor([0.1, 0.4, 0.8])
        target = prediction.clone()
        weight = torch.ones_like(prediction)

        result = factor_calibration_metrics(
            prediction,
            target,
            weight,
            bins=5,
        )

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["ece"], 0.0)
        self.assertAlmostEqual(result["pearson"], 1.0, places=6)

    def test_zero_weight_elements_do_not_affect_metrics(self) -> None:
        prediction = torch.tensor([0.2, 1.0])
        target = torch.tensor([0.0, 0.0])
        weight = torch.tensor([1.0, 0.0])

        result = factor_calibration_metrics(
            prediction,
            target,
            weight,
            bins=2,
        )

        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["mae"], 0.2)
        self.assertAlmostEqual(result["bias"], 0.2)
        self.assertIsNone(result["pearson"])

    def test_summarizes_sampling_and_visibility_channels(self) -> None:
        prediction = torch.tensor([[[[0.2]], [[0.7]]]])
        target = torch.tensor([[[[0.0]], [[1.0]]]])
        weight = torch.ones_like(prediction)

        result = summarize_factor_calibration(
            prediction,
            target,
            weight,
            bins=4,
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertAlmostEqual(result["factors"]["sampling"]["mae"], 0.2)
        self.assertAlmostEqual(result["factors"]["visibility"]["mae"], 0.3)

    def test_rejects_empty_valid_support(self) -> None:
        values = torch.zeros(2)

        with self.assertRaisesRegex(ValueError, "positive weight"):
            factor_calibration_metrics(values, values, values)


if __name__ == "__main__":
    unittest.main()
