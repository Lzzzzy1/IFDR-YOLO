from pathlib import Path
import unittest

import torch

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kitti-p2-m.yaml"
bootstrap_ultralytics_config(ROOT)


class DCLIBboxLossTest(unittest.TestCase):
    def _inputs(self) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(43)
        pred_dist = torch.randn(1, 2, 64, generator=generator)
        pred_bboxes = torch.tensor(
            [[[1.0, 1.0, 4.0, 5.0], [2.0, 2.0, 6.0, 7.0]]]
        )
        anchor_points = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
        target_bboxes = torch.tensor(
            [[[1.2, 1.0, 4.2, 5.5], [2.0, 2.5, 6.5, 7.0]]]
        )
        target_scores = torch.tensor(
            [[[0.8, 0.0, 0.0], [0.0, 0.7, 0.0]]]
        )
        target_scores_sum = target_scores.sum()
        fg_mask = torch.tensor([[True, True]])
        imgsz = torch.tensor([128.0, 128.0])
        stride = torch.ones(1, 2, 1)
        return (
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )

    def test_beta_and_calibration_zero_match_ultralytics_exactly(self) -> None:
        from ultralytics.utils.loss import BboxLoss

        from ifdr_yolo.losses.ifdr_detection import DCLIBboxLoss

        inputs = self._inputs()
        original = BboxLoss(reg_max=16)
        dcli = DCLIBboxLoss(
            reg_max=16,
            beta=0.0,
            calibration_gain=0.0,
        )
        dcli.set_uncertainty(torch.tensor([[0.3, 0.9]]))

        expected = original(*inputs)
        actual = dcli(*inputs)

        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_zero_schedule_disables_nonzero_dcli_configuration(self) -> None:
        from ultralytics.utils.loss import BboxLoss

        from ifdr_yolo.losses.ifdr_detection import DCLIBboxLoss

        inputs = self._inputs()
        original = BboxLoss(reg_max=16)
        dcli = DCLIBboxLoss(
            reg_max=16,
            beta=0.7,
            calibration_gain=0.3,
        )
        dcli.set_schedule(0.0)
        dcli.set_uncertainty(torch.tensor([[0.3, 0.9]]))

        expected = original(*inputs)
        actual = dcli(*inputs)

        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_requires_fresh_anchor_aligned_uncertainty(self) -> None:
        from ifdr_yolo.losses.ifdr_detection import DCLIBboxLoss

        dcli = DCLIBboxLoss(reg_max=16)
        with self.assertRaises(RuntimeError):
            dcli(*self._inputs())
        with self.assertRaises(ValueError):
            dcli.set_uncertainty(torch.ones(1, 3, 1))

    def test_consumes_uncertainty_once(self) -> None:
        from ifdr_yolo.losses.ifdr_detection import DCLIBboxLoss

        dcli = DCLIBboxLoss(reg_max=16)
        dcli.set_uncertainty(torch.tensor([[0.3, 0.9]]))
        first = dcli(*self._inputs())

        self.assertTrue(all(torch.isfinite(value) for value in first))
        with self.assertRaises(RuntimeError):
            dcli(*self._inputs())

    def test_ciou_promotes_half_precision_geometry_to_avoid_overflow(
        self,
    ) -> None:
        from ifdr_yolo.losses.ifdr_detection import stable_ciou

        predicted = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0]],
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.tensor(
            [[150.0, 150.0, 160.0, 160.0]],
            dtype=torch.float16,
        )

        overlap = stable_ciou(predicted, target)
        overlap.sum().backward()

        self.assertEqual(overlap.dtype, torch.float32)
        self.assertTrue(torch.isfinite(overlap).all())
        self.assertTrue(torch.isfinite(predicted.grad).all())


class PyramidFactorAlignmentTest(unittest.TestCase):
    def test_flattens_factors_in_detect_anchor_order(self) -> None:
        from ifdr_yolo.losses.ifdr_detection import (
            flatten_pyramid_factors,
        )
        from ifdr_yolo.models.gated_fusion import ReliabilityContext

        contexts = {}
        shapes = ((4, 4), (2, 2), (1, 1), (1, 1))
        for index, (node, shape) in enumerate(
            zip((17, 20, 23, 26), shapes)
        ):
            factors = torch.full((1, 2, *shape), float(index) / 3.0)
            contexts[node] = ReliabilityContext(
                factors=factors,
                branch_weights=torch.full((1, 2, *shape), 0.5),
                gate_strength=1.0,
            )
        features = [torch.empty(1, 8, *shape) for shape in shapes]

        flattened = flatten_pyramid_factors(contexts, features)

        self.assertEqual(flattened.shape, (1, 22, 2))
        self.assertTrue(torch.equal(flattened[:, :16], torch.zeros(1, 16, 2)))
        self.assertTrue(
            torch.equal(
                flattened[:, 20:21],
                torch.full((1, 1, 2), 2.0 / 3.0),
            )
        )
        self.assertTrue(
            torch.equal(flattened[:, 21:], torch.ones(1, 1, 2))
        )

    def test_multiscale_supervision_is_zero_at_target_and_has_gradients(self) -> None:
        from ifdr_yolo.losses.ifdr_detection import (
            multiscale_factor_supervision,
        )
        from ifdr_yolo.models.gated_fusion import ReliabilityContext

        target = torch.zeros(1, 2, 16, 16)
        target[:, 0] = 0.25
        target[:, 1] = 0.75
        weight = torch.ones_like(target)
        contexts = {}
        for node, size in zip((11, 14, 17, 20, 23, 26), (4, 8, 16, 8, 4, 2)):
            factors = torch.zeros(1, 2, size, size, requires_grad=True)
            factors.data[:, 0] = 0.25
            factors.data[:, 1] = 0.75
            contexts[node] = ReliabilityContext(
                factors=factors,
                branch_weights=torch.full_like(factors, 0.5),
                gate_strength=1.0,
            )

        exact = multiscale_factor_supervision(contexts, target, weight)
        contexts[17].factors.data[:, 0] = 0.9
        mismatch = multiscale_factor_supervision(contexts, target, weight)
        mismatch.backward()

        self.assertEqual(float(exact.detach()), 0.0)
        self.assertGreater(float(mismatch.detach()), 0.0)
        self.assertIsNotNone(contexts[17].factors.grad)

    def test_counterfactual_delta_is_selective_and_updates_both_views(
        self,
    ) -> None:
        from ifdr_yolo.losses.ifdr_detection import (
            multiscale_counterfactual_consistency,
        )
        from ifdr_yolo.models.gated_fusion import ReliabilityContext

        delta_target = torch.zeros(1, 2, 16, 16)
        delta_target[:, 0] = 0.2
        weight = torch.ones_like(delta_target)
        clean_contexts = {}
        intervention_contexts = {}
        for node, size in zip(
            (11, 14, 17, 20, 23, 26),
            (4, 8, 16, 8, 4, 2),
        ):
            clean = torch.empty(
                1,
                2,
                size,
                size,
            ).fill_(0.3)
            clean[:, 1] = 0.4
            clean.requires_grad_(True)
            intervention = clean.detach().clone()
            intervention[:, 0] += 0.2
            intervention.requires_grad_(True)
            clean_contexts[node] = ReliabilityContext(
                factors=clean,
                branch_weights=torch.full_like(clean, 0.5),
                gate_strength=1.0,
            )
            intervention_contexts[node] = ReliabilityContext(
                factors=intervention,
                branch_weights=torch.full_like(intervention, 0.5),
                gate_strength=1.0,
            )

        exact = multiscale_counterfactual_consistency(
            intervention_contexts,
            clean_contexts,
            delta_target,
            weight,
        )
        intervention_contexts[17].factors.data[:, 1] = 0.8
        mismatch = multiscale_counterfactual_consistency(
            intervention_contexts,
            clean_contexts,
            delta_target,
            weight,
        )
        mismatch.backward()

        self.assertLess(float(exact.detach()), 1e-12)
        self.assertGreater(float(mismatch.detach()), 0.0)
        self.assertIsNotNone(intervention_contexts[17].factors.grad)
        self.assertIsNotNone(clean_contexts[17].factors.grad)

    def test_counterfactual_delta_rejects_missing_scale(self) -> None:
        from ifdr_yolo.losses.ifdr_detection import (
            multiscale_counterfactual_consistency,
        )

        with self.assertRaisesRegex(ValueError, "missing.*node 11"):
            multiscale_counterfactual_consistency(
                {},
                {},
                torch.zeros(1, 2, 4, 4),
                torch.ones(1, 2, 4, 4),
            )


class IFDRDetectionLossIntegrationTest(unittest.TestCase):
    def test_gradient_diagnostics_observe_protected_anchor_contract(self) -> None:
        from ultralytics.utils import DEFAULT_CFG

        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        batch = {
            "img": torch.randn(1, 3, 128, 128),
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "ifdr_factor_target": torch.ones(1, 2, 128, 128),
            "ifdr_factor_weight": torch.ones(1, 2, 128, 128),
        }
        groups_by_protection: dict[bool, dict[str, object]] = {}
        for protected in (False, True):
            model = IFDRDetectionModel(
                str(MODEL_PATH),
                nc=3,
                verbose=False,
                semantic_protection=protected,
                gradient_diagnostic_interval=1,
            )
            model.args = DEFAULT_CFG
            model.set_component_schedules(
                fusion=1.0,
                dcli=1.0,
                factor_supervision=1.0,
            )
            model.train()

            total, _ = model.loss(batch)
            records = model.drain_gradient_diagnostics()

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["schema_version"], 2)
            groups = records[0]["parameter_groups"]
            groups_by_protection[protected] = groups
            anchor_norms = groups["semantic_anchor"]["gradient_norms"]
            self.assertGreater(anchor_norms["factor"], 0.0)
            self.assertTrue(
                all(parameter.grad is None for parameter in model.parameters())
            )
            total.sum().backward()

        unprotected = groups_by_protection[False]
        self.assertEqual(set(unprotected), {"semantic_anchor"})
        self.assertGreater(
            unprotected["semantic_anchor"]["gradient_norms"]["detection"],
            0.0,
        )

        protected = groups_by_protection[True]
        self.assertEqual(
            set(protected),
            {
                "semantic_anchor",
                "fusion_adapters",
                "localization_adapter",
            },
        )
        self.assertEqual(
            protected["semantic_anchor"]["gradient_norms"]["detection"],
            0.0,
        )
        fusion_norms = protected["fusion_adapters"]["gradient_norms"]
        self.assertGreater(fusion_norms["detection"], 0.0)
        self.assertGreater(fusion_norms["factor"], 0.0)
        self.assertIsNotNone(
            protected["fusion_adapters"]["pairs"][
                "detection::factor"
            ]["cosine"]
        )

        localization_norms = protected[
            "localization_adapter"
        ]["gradient_norms"]
        self.assertGreater(localization_norms["detection"], 0.0)
        self.assertEqual(localization_norms["factor"], 0.0)

    def test_counterfactual_pair_uses_one_joint_forward_and_one_bn_update(
        self,
    ) -> None:
        from torch.nn.modules.batchnorm import _BatchNorm
        from ultralytics.utils import DEFAULT_CFG

        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        torch.manual_seed(66)
        model = IFDRDetectionModel(
            str(MODEL_PATH),
            nc=3,
            verbose=False,
            semantic_protection=True,
            counterfactual_gain=0.2,
        )
        model.args = DEFAULT_CFG
        model.set_reliability_schedule(1.0)
        model.train()
        first_batch_norm = next(
            module for module in model.modules() if isinstance(module, _BatchNorm)
        )
        tracked_before = int(first_batch_norm.num_batches_tracked)
        observed_batch_sizes: list[int] = []

        def record_input_batch(_module, inputs) -> None:
            observed_batch_sizes.append(int(inputs[0].shape[0]))

        handle = model.model[0].register_forward_pre_hook(record_input_batch)
        image = torch.randn(1, 3, 128, 128)
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "ifdr_factor_target": torch.zeros(1, 2, 128, 128),
            "ifdr_factor_weight": torch.ones(1, 2, 128, 128),
            "ifdr_counterfactual_img": image.clone(),
            "ifdr_counterfactual_delta": torch.zeros(1, 2, 128, 128),
            "ifdr_counterfactual_weight": torch.ones(1, 2, 128, 128),
        }

        try:
            total, _ = model.loss(batch)
        finally:
            handle.remove()

        self.assertTrue(torch.isfinite(total).all())
        self.assertEqual(observed_batch_sizes, [2])
        self.assertEqual(
            int(first_batch_norm.num_batches_tracked),
            tracked_before + 1,
        )

    def test_real_model_uses_dcli_and_backpropagates_to_factor_heads(self) -> None:
        from ultralytics.utils import DEFAULT_CFG

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        torch.manual_seed(59)
        model = IFDRDetectionModel(str(MODEL_PATH), nc=3, verbose=False)
        model.args = DEFAULT_CFG
        model.set_reliability_schedule(1.0)
        model.train()
        batch = {
            "img": torch.randn(1, 3, 128, 128),
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "ifdr_factor_target": torch.zeros(1, 2, 128, 128),
            "ifdr_factor_weight": torch.ones(1, 2, 128, 128),
        }

        total, components = model.loss(batch)
        total.sum().backward()

        self.assertIsInstance(model.criterion, IFDRDetectionLoss)
        self.assertEqual(components.shape, (5,))
        self.assertTrue(torch.isfinite(total).all())
        factor_gradients = [
            model.model[index].factor_head.weight.grad
            for index in (17, 20, 23, 26)
        ]
        self.assertTrue(
            all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in factor_gradients
            )
        )

    def test_factor_supervision_remains_active_when_gate_and_dcli_are_off(
        self,
    ) -> None:
        from ultralytics.utils import DEFAULT_CFG

        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        torch.manual_seed(61)
        model = IFDRDetectionModel(str(MODEL_PATH), nc=3, verbose=False)
        model.args = DEFAULT_CFG
        model.set_component_schedules(
            fusion=0.0,
            dcli=0.0,
            factor_supervision=1.0,
        )
        model.train()
        batch = {
            "img": torch.randn(1, 3, 128, 128),
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "ifdr_factor_target": torch.ones(1, 2, 128, 128),
            "ifdr_factor_weight": torch.ones(1, 2, 128, 128),
        }

        total, _ = model.loss(batch)
        total.sum().backward()

        gradient = model.model[17].factor_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_protected_counterfactual_loss_runs_second_view(self) -> None:
        from ultralytics.utils import DEFAULT_CFG

        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        torch.manual_seed(67)
        model = IFDRDetectionModel(
            str(MODEL_PATH),
            nc=3,
            verbose=False,
            semantic_protection=True,
            counterfactual_gain=0.2,
        )
        model.args = DEFAULT_CFG
        model.set_reliability_schedule(1.0)
        model.train()
        image = torch.randn(1, 3, 128, 128)
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "ifdr_factor_target": torch.zeros(1, 2, 128, 128),
            "ifdr_factor_weight": torch.ones(1, 2, 128, 128),
            "ifdr_counterfactual_img": image.clone(),
            "ifdr_counterfactual_delta": torch.zeros(1, 2, 128, 128),
            "ifdr_counterfactual_weight": torch.ones(1, 2, 128, 128),
        }

        total, components = model.loss(batch)
        total.sum().backward()

        self.assertTrue(torch.isfinite(total).all())
        self.assertEqual(components.shape, (5,))
        self.assertGreater(float(components[3]), 0.0)
        self.assertLess(float(components[4].abs()), 1e-12)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.localization_adapter.parameters()
            )
        )

    def test_frozen_factor_schedule_skips_counterfactual_forward(self) -> None:
        from ultralytics.utils import DEFAULT_CFG

        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model = IFDRDetectionModel(
            str(MODEL_PATH),
            nc=3,
            verbose=False,
            semantic_protection=True,
            counterfactual_gain=0.2,
        )
        model.args = DEFAULT_CFG
        model.set_component_schedules(
            fusion=0.0,
            dcli=0.0,
            factor_supervision=0.0,
        )
        model.train()
        batch = {
            "img": torch.randn(1, 3, 128, 128),
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "ifdr_factor_target": torch.zeros(1, 2, 128, 128),
            "ifdr_factor_weight": torch.ones(1, 2, 128, 128),
            "ifdr_counterfactual_img": torch.zeros(1, 3, 128),
            "ifdr_counterfactual_delta": torch.zeros(1, 2, 128, 128),
            "ifdr_counterfactual_weight": torch.ones(1, 2, 128, 128),
        }

        total, components = model.loss(batch)

        self.assertTrue(torch.isfinite(total).all())
        self.assertEqual(components.shape, (5,))
        self.assertEqual(float(components[3]), 0.0)
        self.assertEqual(float(components[4]), 0.0)


if __name__ == "__main__":
    unittest.main()
