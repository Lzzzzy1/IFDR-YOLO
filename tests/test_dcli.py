import math
import unittest

import torch


class DCLITest(unittest.TestCase):
    def test_beta_zero_is_exactly_original_localization_error(self) -> None:
        from ifdr_yolo.losses.dcli import dcli_localization_error

        error = torch.tensor([0.0, 0.2, 0.8, 1.0])
        uncertainty = torch.tensor([1.0, 0.7, 0.3, 0.0])

        actual = dcli_localization_error(
            error,
            uncertainty,
            beta=0.0,
        )

        self.assertTrue(torch.equal(actual, error))

    def test_uses_bounded_heteroscedastic_scaling(self) -> None:
        from ifdr_yolo.losses.dcli import dcli_localization_error

        error = torch.tensor([0.2, 0.8])
        uncertainty = torch.tensor([0.0, 1.0])

        actual = dcli_localization_error(
            error,
            uncertainty,
            beta=0.5,
        )

        expected = torch.tensor(
            [0.2, 0.8 / 1.5 + math.log(1.5)]
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_dcli_stops_uncertainty_gradient_but_keeps_box_gradient(self) -> None:
        from ifdr_yolo.losses.dcli import dcli_localization_error

        error = torch.tensor([0.3, 0.6], requires_grad=True)
        uncertainty = torch.tensor([0.4, 0.9], requires_grad=True)

        loss = dcli_localization_error(
            error,
            uncertainty,
            beta=0.7,
        ).sum()
        loss.backward()

        self.assertIsNotNone(error.grad)
        self.assertTrue(torch.isfinite(error.grad).all())
        self.assertIsNone(uncertainty.grad)

    def test_rejects_invalid_values_and_shapes(self) -> None:
        from ifdr_yolo.losses.dcli import dcli_localization_error

        error = torch.ones(2)
        uncertainty = torch.ones(3)
        with self.assertRaises(ValueError):
            dcli_localization_error(error, uncertainty, beta=0.5)
        with self.assertRaises(ValueError):
            dcli_localization_error(error, error, beta=1.01)
        with self.assertRaises(ValueError):
            dcli_localization_error(error, torch.tensor([0.0, 1.1]), beta=0.5)


class LocalizationUncertaintyTest(unittest.TestCase):
    def test_normalized_dfl_entropy_has_known_limits(self) -> None:
        from ifdr_yolo.losses.dcli import normalized_dfl_entropy

        uniform_logits = torch.zeros(1, 2, 4 * 16)
        peaked_logits = torch.full((1, 2, 4, 16), -20.0)
        peaked_logits[..., 3] = 20.0

        uniform = normalized_dfl_entropy(uniform_logits, reg_max=16)
        peaked = normalized_dfl_entropy(
            peaked_logits.flatten(2),
            reg_max=16,
        )

        self.assertTrue(torch.allclose(uniform, torch.ones_like(uniform)))
        self.assertTrue((peaked < 1e-5).all())

    def test_combines_two_factors_and_entropy_interpretably(self) -> None:
        from ifdr_yolo.losses.dcli import derive_localization_uncertainty

        factors = torch.tensor([[[0.2, 0.8], [1.0, 0.0]]])
        entropy = torch.tensor([[0.5, 0.5]])

        uncertainty = derive_localization_uncertainty(
            factors,
            entropy,
            factor_weights=(2.0, 1.0),
            entropy_weight=1.0,
        )

        expected = torch.tensor(
            [[(0.4 + 0.8 + 0.5) / 4.0, (2.0 + 0.0 + 0.5) / 4.0]]
        )
        self.assertTrue(torch.allclose(uncertainty, expected))

    def test_calibration_only_updates_uncertainty_branch(self) -> None:
        from ifdr_yolo.losses.dcli import uncertainty_calibration_loss

        predicted = torch.tensor([0.2, 0.8], requires_grad=True)
        residual = torch.tensor([0.9, 0.1], requires_grad=True)
        mask = torch.tensor([True, False])

        loss = uncertainty_calibration_loss(
            predicted,
            residual,
            valid_mask=mask,
        )
        loss.backward()

        self.assertIsNotNone(predicted.grad)
        self.assertNotEqual(float(predicted.grad[0]), 0.0)
        self.assertEqual(float(predicted.grad[1]), 0.0)
        self.assertIsNone(residual.grad)


if __name__ == "__main__":
    unittest.main()
