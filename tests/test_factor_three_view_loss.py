from __future__ import annotations

import unittest

import torch

from pathlib import Path

from ifdr_yolo.experiments.ultralytics_runtime import bootstrap_ultralytics_config

bootstrap_ultralytics_config(Path(__file__).resolve().parents[1])

from ifdr_yolo.models.gated_fusion import ReliabilityContext
from ifdr_yolo.losses.factor_alignment import ObjectFactorTarget


class ThreeViewLossTest(unittest.TestCase):
    def _contexts(self, value: float = 0.0, batch: int = 1):
        return {
            node: ReliabilityContext(
                factors=torch.full((batch, 2, size, size), value + node / 100.0, requires_grad=True),
                branch_weights=torch.ones(batch, 2, size, size),
                gate_strength=1.0,
            )
            for node, size in ((11, 4), (14, 3), (17, 2), (20, 3), (23, 4), (26, 5))
        }

    def test_specificity_margin_uses_delta_from_clean(self):
        from ifdr_yolo.losses.factor_alignment import factor_specificity_loss

        self.assertTrue(torch.allclose(
            factor_specificity_loss(torch.tensor([0.20]), torch.tensor([0.50]), torch.tensor([0.26])),
            torch.tensor(0.0),
        ))
        self.assertTrue(torch.allclose(
            factor_specificity_loss(torch.tensor([0.20]), torch.tensor([0.30]), torch.tensor([0.26])),
            torch.tensor(0.01),
        ))

    def test_specificity_margin_is_registered(self):
        from ifdr_yolo.losses.factor_alignment import factor_specificity_loss

        with self.assertRaisesRegex(ValueError, "registered specificity margin"):
            factor_specificity_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), margin=0.1)

    def test_specificity_uses_background_clean_roi_baseline(self):
        from ifdr_yolo.losses.factor_alignment import factor_specificity_from_contexts

        def contexts(clean_background: float):
            result = {}
            for node, size in ((17, 2), (20, 2), (23, 2), (26, 2)):
                values = torch.full((1, 2, size, size), clean_background)
                result[node] = ReliabilityContext(
                    factors=values,
                    branch_weights=torch.ones_like(values),
                    gate_strength=1.0,
                )
            return result

        clean = contexts(0.2)
        target = contexts(0.4)
        background = contexts(0.3)
        pair = {
            "batch_index": 0,
            "target_box_xyxy_normalized": (0.0, 0.0, 0.5, 0.5),
            "background_box_xyxy_normalized": (0.5, 0.5, 1.0, 1.0),
            "factor_kind": "sampling",
            "severity": 0.5,
            "weight": 1.0,
        }
        baseline = factor_specificity_from_contexts(clean, target, background, (pair,))
        changed = contexts(0.2)
        for context in changed.values():
            context.factors[:, :, 1:, 1:] = 0.1
        altered = factor_specificity_from_contexts(changed, target, background, (pair,))
        self.assertNotEqual(float(baseline), float(altered))

    def test_no_effective_specificity_pair_is_graph_connected_zero(self):
        from ifdr_yolo.losses.factor_alignment import factor_specificity_from_contexts

        clean = self._contexts(0.2)
        target = self._contexts(0.3)
        background = self._contexts(0.4)
        for contexts in (clean, target, background):
            for context in contexts.values():
                context.factors.retain_grad()
        pair = {
            "batch_index": 0,
            "target_box_xyxy_normalized": (0.0, 0.0, 1.0, 1.0),
            "background_box_xyxy_normalized": (0.0, 0.0, 1.0, 1.0),
            "factor_kind": "sampling",
            "severity": 0.24,
            "weight": 0.0,
        }
        loss = factor_specificity_from_contexts(clean, target, background, (pair,))
        loss.backward()
        for contexts in (clean, target, background):
            self.assertTrue(
                all(contexts[node].factors.grad is not None for node in (17, 20, 23, 26))
            )

    def test_synthetic_loss_uses_target_context_only(self):
        from ifdr_yolo.losses.ifdr_detection import synthetic_factor_loss_from_context

        target = self._contexts(0.2)
        clean = self._contexts(0.3)
        background = self._contexts(0.4)
        dense_target = torch.full((1, 2, 8, 8), 0.8)
        dense_weight = torch.ones_like(dense_target)
        a = synthetic_factor_loss_from_context(target, dense_target, dense_weight)
        for context in clean.values():
            context.factors.data.add_(100.0)
        for context in background.values():
            context.factors.data.sub_(100.0)
        b = synthetic_factor_loss_from_context(target, dense_target, dense_weight)
        self.assertTrue(torch.equal(a, b))

    def test_route_calibration_losses_exposes_components_and_weight(self):
        from ifdr_yolo.losses.ifdr_detection import route_calibration_losses

        contexts = self._contexts()
        targets = (ObjectFactorTarget(0, 0, (0.0, 0.0, 1.0, 1.0), (0.1, 0.1), (True, True)),)
        intervention = ()
        losses = route_calibration_losses(
            clean_context=contexts,
            target_context=contexts,
            background_context=contexts,
            dense_target=torch.zeros(1, 2, 8, 8),
            dense_weight=torch.ones(1, 2, 8, 8),
            natural_object_targets=targets,
            intervention=intervention,
        )
        self.assertEqual(set(losses), {"synthetic_factor_loss", "natural_factor_loss", "specificity_loss", "total"})
        self.assertTrue(torch.equal(losses["total"], losses["synthetic_factor_loss"] + losses["natural_factor_loss"] + 0.5 * losses["specificity_loss"]))

    @staticmethod
    def _fake_model(*, context_batch: int = 3):
        from ifdr_yolo.data.ifdr_dataset import (
            BACKGROUND_IMAGE_KEY,
            CLEAN_IMAGE_KEY,
            FACTOR_OBJECT_TARGETS_KEY,
            FACTOR_TARGET_KEY,
            FACTOR_WEIGHT_KEY,
            SPECIFICITY_PAIRS_KEY,
            TARGET_IMAGE_KEY,
        )
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        nodes = (11, 14, 17, 20, 23, 26)
        contexts = {}
        for node in nodes:
            factors = torch.cat(
                tuple(
                    torch.full((1, 2, 2, 2), float(index + 1))
                    for index in range(3 if context_batch == 3 else 2)
                ),
                dim=0,
            )
            contexts[node] = ReliabilityContext(
                factors=factors,
                branch_weights=torch.ones_like(factors),
                gate_strength=1.0,
            )

        class Criterion:
            def __init__(self):
                self.detection_calls = 0
                self.calibration_kwargs = None

            def __call__(self, *_args, **_kwargs):
                self.detection_calls += 1
                raise AssertionError("detection criterion must not run in calibration")

            def calibration_loss(self, **kwargs):
                self.calibration_kwargs = kwargs
                return torch.tensor(2.0, requires_grad=True), torch.tensor([0.5, 0.25, 0.125])

        class Model:
            _calibration_view_keys = staticmethod(IFDRDetectionModel._calibration_view_keys)
            _calibration_view_batch = IFDRDetectionModel._calibration_view_batch

            def __init__(self):
                self._fusion_node_indices = nodes
                self.contexts = contexts
                self.criterion = Criterion()
                self.forward_inputs = []
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, image):
                self.forward_inputs.append(image.detach().clone())
                self.bn(image)
                return {"ignored": image}

            def consume_reliability_context(self):
                return self.contexts

            def init_criterion(self):
                return self.criterion

            def _counterfactual_pair_is_active(self, _batch):
                return False

        model = Model()
        batch = {
            CLEAN_IMAGE_KEY: torch.full((1, 3, 4, 4), 1.0),
            TARGET_IMAGE_KEY: torch.full((1, 3, 4, 4), 2.0),
            BACKGROUND_IMAGE_KEY: torch.full((1, 3, 4, 4), 3.0),
            FACTOR_TARGET_KEY: torch.zeros(1, 2, 8, 8),
            FACTOR_WEIGHT_KEY: torch.ones(1, 2, 8, 8),
            FACTOR_OBJECT_TARGETS_KEY: (),
            SPECIFICITY_PAIRS_KEY: (),
        }
        return model, batch

    def test_calibration_does_not_call_detection_loss(self):
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model, batch = self._fake_model()
        total, components = IFDRDetectionModel.loss(model, batch)
        self.assertEqual(float(total), 2.0)
        self.assertEqual(tuple(components.shape), (3,))
        self.assertEqual(model.criterion.detection_calls, 0)

    def test_calibration_uses_one_ordered_three_b_forward_and_splits_all_nodes(self):
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model, batch = self._fake_model()
        IFDRDetectionModel.loss(model, batch)
        self.assertEqual(len(model.forward_inputs), 1)
        self.assertEqual(tuple(model.forward_inputs[0][:, 0, 0, 0].tolist()), (1.0, 2.0, 3.0))
        self.assertEqual(int(model.bn.num_batches_tracked), 1)
        kwargs = model.criterion.calibration_kwargs
        self.assertEqual(set(kwargs["clean_context"]), {11, 14, 17, 20, 23, 26})
        self.assertEqual(float(kwargs["clean_context"][17].factors.mean()), 1.0)
        self.assertEqual(float(kwargs["target_context"][17].factors.mean()), 2.0)
        self.assertEqual(float(kwargs["background_context"][17].factors.mean()), 3.0)

    def test_uint8_views_are_normalized_before_joint_forward(self):
        from ifdr_yolo.data.ifdr_dataset import (
            BACKGROUND_IMAGE_KEY,
            CLEAN_IMAGE_KEY,
            TARGET_IMAGE_KEY,
        )
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model, batch = self._fake_model()
        batch[CLEAN_IMAGE_KEY] = torch.full((1, 3, 4, 4), 64, dtype=torch.uint8)
        batch[TARGET_IMAGE_KEY] = torch.full((1, 3, 4, 4), 128, dtype=torch.uint8)
        batch[BACKGROUND_IMAGE_KEY] = torch.full((1, 3, 4, 4), 192, dtype=torch.uint8)
        IFDRDetectionModel.loss(model, batch)
        forwarded = model.forward_inputs[0]
        self.assertTrue(forwarded.is_floating_point())
        self.assertEqual(tuple(forwarded.shape), (3, 3, 4, 4))
        self.assertTrue(torch.allclose(forwarded[:, 0, 0, 0], torch.tensor([64, 128, 192], dtype=torch.float32) / 255.0))

    def test_mixed_view_dtypes_fail_closed(self):
        from ifdr_yolo.data.ifdr_dataset import TARGET_IMAGE_KEY
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model, batch = self._fake_model()
        batch[TARGET_IMAGE_KEY] = batch[TARGET_IMAGE_KEY].double()
        with self.assertRaisesRegex(RuntimeError, "share dtype"):
            IFDRDetectionModel.loss(model, batch)

    def test_partial_calibration_view_keys_fail_closed(self):
        from ifdr_yolo.data.ifdr_dataset import CLEAN_IMAGE_KEY
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model, _ = self._fake_model()
        with self.assertRaisesRegex(RuntimeError, "missing view key"):
            IFDRDetectionModel.loss(model, {CLEAN_IMAGE_KEY: torch.zeros(1, 3, 4, 4)})

    def test_calibration_with_preds_fails_closed(self):
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model, batch = self._fake_model()
        with self.assertRaisesRegex(RuntimeError, "preds=None"):
            IFDRDetectionModel.loss(model, batch, preds={})

    def test_calibration_requires_registered_object_target_and_pair_keys(self):
        from ifdr_yolo.data.ifdr_dataset import (
            FACTOR_OBJECT_TARGETS_KEY,
            SPECIFICITY_PAIRS_KEY,
        )
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model, batch = self._fake_model()
        batch.pop(FACTOR_OBJECT_TARGETS_KEY)
        with self.assertRaisesRegex(RuntimeError, "factor object targets"):
            IFDRDetectionModel.loss(model, batch)
        model, batch = self._fake_model()
        batch.pop(SPECIFICITY_PAIRS_KEY)
        with self.assertRaisesRegex(RuntimeError, "specificity pairs"):
            IFDRDetectionModel.loss(model, batch)

    def test_contexts_must_be_three_b(self):
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        model, batch = self._fake_model(context_batch=2)
        with self.assertRaisesRegex(RuntimeError, "leading dimension 3B"):
            IFDRDetectionModel.loss(model, batch)


if __name__ == "__main__":
    unittest.main()

