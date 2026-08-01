import unittest

import torch

from ifdr_yolo.models.gated_fusion import (
    ReliabilityContext,
    ReliabilityGatedConcat,
    ResidualFactorAdapter,
)


class ReliabilityGatedConcatTest(unittest.TestCase):
    def inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(17)
        first = torch.randn(2, 8, 16, 20, generator=generator)
        second = torch.randn(2, 12, 16, 20, generator=generator)
        return first, second

    def test_zero_schedule_is_exactly_original_concat(self) -> None:
        module = ReliabilityGatedConcat(
            input_channels=(8, 12),
            reliability_channels=4,
        )
        first, second = self.inputs()

        output = module((first, second))

        self.assertTrue(torch.equal(output, torch.cat((first, second), dim=1)))
        context = module.consume_context()
        self.assertEqual(context.factors.shape, (2, 2, 16, 20))
        self.assertEqual(context.branch_weights.shape, (2, 2, 16, 20))
        self.assertTrue(
            torch.allclose(
                context.branch_weights.sum(dim=1),
                torch.ones(2, 16, 20),
            )
        )
        self.assertEqual(context.gate_strength, 0.0)

    def test_factor_maps_and_branch_weights_have_valid_ranges(self) -> None:
        module = ReliabilityGatedConcat(
            input_channels=(8, 12),
            reliability_channels=4,
        )
        module.set_schedule(1.0)
        module(self.inputs())

        context = module.consume_context()

        self.assertTrue(torch.all(context.factors >= 0.0))
        self.assertTrue(torch.all(context.factors <= 1.0))
        self.assertTrue(torch.all(context.branch_weights >= 0.0))
        self.assertTrue(torch.all(context.branch_weights <= 1.0))
        self.assertGreater(context.gate_strength, 0.0)
        self.assertLess(context.gate_strength, 1.0)

    def test_context_can_only_be_consumed_once(self) -> None:
        module = ReliabilityGatedConcat(
            input_channels=(8, 12),
            reliability_channels=4,
        )
        module(self.inputs())
        self.assertIsInstance(module.consume_context(), ReliabilityContext)

        with self.assertRaisesRegex(RuntimeError, "no reliability context"):
            module.consume_context()

    def test_open_gate_backpropagates_to_inputs_and_parameters(self) -> None:
        module = ReliabilityGatedConcat(
            input_channels=(8, 12),
            reliability_channels=4,
        )
        module.set_schedule(1.0)
        first, second = self.inputs()
        first.requires_grad_(True)
        second.requires_grad_(True)

        output = module((first, second))
        context = module.consume_context()
        loss = output.square().mean()
        loss = loss + context.factors.mean()
        loss.backward()

        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())
        parameter_grads = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in parameter_grads))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in parameter_grads)
        )

    def test_rejects_invalid_schedule_and_input_shapes(self) -> None:
        module = ReliabilityGatedConcat(
            input_channels=(8, 12),
            reliability_channels=4,
        )
        with self.assertRaisesRegex(ValueError, "schedule"):
            module.set_schedule(1.1)
        first, second = self.inputs()
        with self.assertRaisesRegex(ValueError, "spatial"):
            module((first, second[:, :, :-1]))
        with self.assertRaisesRegex(ValueError, "channels"):
            module((first[:, :7], second))

    def test_ephemeral_context_is_not_in_state_dict(self) -> None:
        module = ReliabilityGatedConcat(
            input_channels=(8, 12),
            reliability_channels=4,
        )
        module(self.inputs())

        self.assertFalse(
            any("context" in name for name in module.state_dict())
        )

    def test_protected_detection_gradient_updates_adapter_not_anchor(self) -> None:
        module = ReliabilityGatedConcat(
            input_channels=(8, 12),
            reliability_channels=4,
            semantic_protection=True,
        )
        module.set_schedule(1.0)
        output = module(self.inputs())
        module.consume_context()

        output.square().mean().backward()

        anchor_gradients = [
            parameter.grad
            for parameter in module.reliability_estimator.parameters()
        ]
        adapter_gradients = [
            parameter.grad
            for parameter in module.fusion_adapter.parameters()
        ]
        self.assertTrue(all(gradient is None for gradient in anchor_gradients))
        self.assertTrue(
            any(
                gradient is not None and torch.count_nonzero(gradient) > 0
                for gradient in adapter_gradients
            )
        )

    def test_protected_factor_supervision_still_updates_anchor(self) -> None:
        module = ReliabilityGatedConcat(
            input_channels=(8, 12),
            reliability_channels=4,
            semantic_protection=True,
        )
        module(self.inputs())
        context = module.consume_context()

        context.factors.mean().backward()

        anchor_gradients = [
            parameter.grad
            for parameter in module.reliability_estimator.parameters()
        ]
        self.assertTrue(
            any(
                gradient is not None and torch.count_nonzero(gradient) > 0
                for gradient in anchor_gradients
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in module.fusion_adapter.parameters()
            )
        )


class ResidualFactorAdapterTest(unittest.TestCase):
    def test_initially_preserves_factors_and_blocks_input_gradient(self) -> None:
        adapter = ResidualFactorAdapter(hidden_channels=4)
        factors = torch.tensor(
            [[[0.2, 0.7], [0.6, 0.1]]],
            requires_grad=True,
        )

        adapted = adapter(factors.detach())
        self.assertTrue(torch.equal(adapted, factors.detach()))
        adapted.square().mean().backward()

        self.assertIsNone(factors.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.count_nonzero(parameter.grad) > 0
                for parameter in adapter.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
