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

        self.assertEqual(float(exact), 0.0)
        self.assertGreater(float(mismatch), 0.0)
        self.assertIsNotNone(contexts[17].factors.grad)


class IFDRDetectionLossIntegrationTest(unittest.TestCase):
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
        self.assertEqual(components.shape, (3,))
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


if __name__ == "__main__":
    unittest.main()
