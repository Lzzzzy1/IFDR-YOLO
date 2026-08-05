from __future__ import annotations

import math
from pathlib import Path
import unittest
import weakref
from unittest.mock import patch

import torch
from torch.nn import functional as F

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)

ROOT = Path(__file__).resolve().parents[1]
bootstrap_ultralytics_config(ROOT)

from ifdr_yolo.losses.factor_alignment import (
    ObjectFactorTarget,
    map_normalized_box_to_feature_roi,
    object_balanced_factor_loss,
)


class FactorAlignmentLossTest(unittest.TestCase):
    @staticmethod
    def factor_map(
        sampling: float,
        visibility: float,
        *,
        batch_size: int = 1,
        height: int = 4,
        width: int = 4,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        result = torch.empty(
            batch_size,
            2,
            height,
            width,
        )
        result[:, 0].fill_(sampling)
        result[:, 1].fill_(visibility)
        if requires_grad:
            result.requires_grad_()
        return result

    @staticmethod
    def target(
        *,
        batch_index: int = 0,
        class_id: int = 0,
        box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        values: tuple[float, float] = (0.8, 0.4),
        valid: tuple[bool, bool] = (True, True),
    ) -> ObjectFactorTarget:
        return ObjectFactorTarget(
            batch_index=batch_index,
            class_id=class_id,
            box_xyxy_normalized=box,
            target=values,
            valid=valid,
        )

    def test_object_balanced_loss_is_invariant_to_roi_area(self) -> None:
        small = self.target(
            class_id=2,
            box=(0.0, 0.0, 0.25, 0.25),
        )
        large = self.target(
            class_id=2,
            box=(0.0, 0.0, 1.0, 1.0),
        )
        factor_map = self.factor_map(0.5, 0.5)
        self.assertTrue(
            torch.allclose(
                object_balanced_factor_loss([factor_map], [small]),
                object_balanced_factor_loss([factor_map], [large]),
            )
        )

    def test_invalid_sampling_keeps_visibility_channel(self) -> None:
        target = self.target(
            class_id=1,
            values=(0.0, 0.9),
            valid=(False, True),
            box=(0.0, 0.0, 0.5, 0.5),
        )
        loss = object_balanced_factor_loss(
            [self.factor_map(0.7, 0.2)],
            [target],
        )
        expected = F.smooth_l1_loss(torch.tensor(0.2), torch.tensor(0.9))
        self.assertTrue(torch.allclose(loss, expected))

    def test_class_macro_average_ignores_object_frequency(self) -> None:
        factor_map = self.factor_map(0.0, 0.0)
        targets = [
            self.target(class_id=3, values=(0.0, 0.0))
            for _ in range(20)
        ]
        targets.append(self.target(class_id=8, values=(1.0, 1.0)))

        loss = object_balanced_factor_loss([factor_map], targets)
        car_loss = torch.tensor(0.0)
        cyclist_loss = F.smooth_l1_loss(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )
        expected = (car_loss + cyclist_loss) / 2.0
        object_mean = cyclist_loss / 21.0
        self.assertTrue(torch.allclose(loss, expected))
        self.assertFalse(torch.allclose(loss, object_mean))

    def test_normalized_box_maps_per_node_size(self) -> None:
        box = (0.11, 0.17, 0.43, 0.71)
        rois = [
            map_normalized_box_to_feature_roi(box, height, width)
            for height, width in ((80, 120), (40, 60), (20, 30), (10, 15))
        ]
        self.assertEqual(
            rois,
            [
                (13, 13, 52, 57),
                (6, 6, 26, 29),
                (3, 3, 13, 15),
                (1, 1, 7, 8),
            ],
        )
        self.assertEqual(len(set(rois)), 4)

    def test_clip_out_of_range_box_then_pool(self) -> None:
        roi = map_normalized_box_to_feature_roi(
            (-0.01, 0.10, 1.01, 0.80),
            4,
            5,
        )
        self.assertEqual(roi, (0, 0, 5, 4))
        target = self.target(box=(-0.01, 0.10, 1.01, 0.80))
        actual = object_balanced_factor_loss(
            [self.factor_map(0.8, 0.4, height=4, width=5)],
            [target],
        )
        self.assertTrue(torch.isfinite(actual))

    def test_reverse_or_nonfinite_normalized_box_fails(self) -> None:
        with self.assertRaises(ValueError):
            map_normalized_box_to_feature_roi((0.5, 0.0, 0.2, 1.0), 4, 4)
        with self.assertRaises(ValueError):
            map_normalized_box_to_feature_roi((0.0, 0.0, math.nan, 1.0), 4, 4)
        with self.assertRaises(ValueError):
            object_balanced_factor_loss(
                [self.factor_map(0.0, 0.0)],
                [self.target(box=(0.0, 0.0, math.inf, 1.0))],
            )

    def test_empty_roi_is_zero_weight_and_counted(self) -> None:
        factor_map = self.factor_map(0.8, 0.4, requires_grad=True)
        counter = [0]
        target = self.target(box=(-2.0, 0.1, -1.0, 0.8))
        loss = object_balanced_factor_loss(
            [factor_map],
            [target],
            empty_roi_counter=counter,
        )
        self.assertEqual(counter, [1])
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(factor_map.grad)
        self.assertTrue(torch.equal(factor_map.grad, torch.zeros_like(factor_map)))

    def test_empty_counter_counts_each_target_and_node(self) -> None:
        counter = [0]
        targets = [
            self.target(class_id=0, box=(-2.0, 0.1, -1.0, 0.8)),
            self.target(class_id=1, box=(2.0, 0.1, 3.0, 0.8)),
        ]
        maps = [self.factor_map(0.1, 0.2), self.factor_map(0.3, 0.4)]
        loss = object_balanced_factor_loss(
            maps,
            targets,
            empty_roi_counter=counter,
        )
        self.assertEqual(counter, [4])
        self.assertTrue(torch.equal(loss, maps[0].sum() * 0.0))

    def test_empty_node_maps_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            object_balanced_factor_loss([], [])

    def test_empty_node_maps_with_targets_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            object_balanced_factor_loss(
                [],
                [self.target()],
            )

    def test_no_targets_return_differentiable_zero(self) -> None:
        factor_map = self.factor_map(0.8, 0.4, requires_grad=True)
        loss = object_balanced_factor_loss([factor_map], [])
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(factor_map.grad)
        self.assertTrue(torch.equal(factor_map.grad, torch.zeros_like(factor_map)))

    def test_no_valid_channel_return_differentiable_zero(self) -> None:
        factor_map = self.factor_map(0.8, 0.4, requires_grad=True)
        target = self.target(valid=(False, False))
        loss = object_balanced_factor_loss([factor_map], [target])
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(factor_map.grad)
        self.assertTrue(torch.equal(factor_map.grad, torch.zeros_like(factor_map)))

    def test_gradient_flows_through_nonempty_roi(self) -> None:
        factor_map = self.factor_map(0.2, 0.4, requires_grad=True)
        target = self.target(values=(0.8, 0.9))
        loss = object_balanced_factor_loss([factor_map], [target])
        loss.backward()
        self.assertIsNotNone(factor_map.grad)
        self.assertTrue(torch.isfinite(factor_map.grad).all())
        self.assertGreater(float(factor_map.grad.abs().sum()), 0.0)

    def test_maps_must_be_floating_bchw_two_channel_and_finite(self) -> None:
        cases = (
            torch.zeros(2, 4, 4),
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 2, 4, 4, dtype=torch.int64),
            torch.full((1, 2, 4, 4), math.nan),
            torch.full((1, 2, 4, 4), math.inf),
        )
        for factor_map in cases:
            with self.subTest(shape=tuple(factor_map.shape), dtype=factor_map.dtype):
                with self.assertRaises(ValueError):
                    object_balanced_factor_loss([factor_map], [])

    def test_check_finite_false_propagates_nonfinite_loss(self) -> None:
        factor_map = torch.full((1, 2, 4, 4), math.nan)
        target = self.target(values=(0.8, 0.4))
        with self.assertRaises(ValueError):
            object_balanced_factor_loss([factor_map], [target])
        loss = object_balanced_factor_loss(
            [factor_map],
            [target],
            check_finite=False,
        )
        self.assertFalse(torch.isfinite(loss))

    def test_node_macro_is_numeric_average_across_nodes(self) -> None:
        target = self.target(values=(0.0, 0.0))
        maps = [self.factor_map(0.0, 0.0), self.factor_map(1.0, 1.0)]
        loss = object_balanced_factor_loss(maps, [target])
        expected = (
            F.smooth_l1_loss(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
            + F.smooth_l1_loss(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]))
        ) / 2.0
        self.assertTrue(torch.allclose(loss, expected))

    def test_nodes_must_share_batch_size_and_device(self) -> None:
        with self.assertRaises(ValueError):
            object_balanced_factor_loss(
                [
                    self.factor_map(0.0, 0.0, batch_size=1),
                    self.factor_map(0.0, 0.0, batch_size=2),
                ],
                [],
            )
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                object_balanced_factor_loss(
                    [
                        self.factor_map(0.0, 0.0),
                        self.factor_map(0.0, 0.0).cuda(),
                    ],
                    [],
                )

    def test_target_batch_class_types_and_bounds_are_strict(self) -> None:
        bad_targets = (
            self.target(batch_index=-1),
            self.target(batch_index=1),
            self.target(batch_index=True),
            self.target(class_id=-1),
            self.target(class_id=True),
        )
        for bad in bad_targets:
            with self.subTest(target=bad):
                with self.assertRaises((TypeError, ValueError)):
                    object_balanced_factor_loss([self.factor_map(0.0, 0.0)], [bad])

    def test_target_values_must_be_finite_two_real_numbers(self) -> None:
        for values in (
            (math.nan, 0.0),
            (0.0, math.inf),
            (0.0,),
            ("bad", 0.0),
            (-0.01, 0.0),
            (0.0, 1.01),
        ):
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    object_balanced_factor_loss(
                        [self.factor_map(0.0, 0.0)],
                        [self.target(values=values)],
                    )

    def test_empty_counter_must_start_nonnegative(self) -> None:
        with self.assertRaises(ValueError):
            object_balanced_factor_loss(
                [self.factor_map(0.0, 0.0)],
                [],
                empty_roi_counter=[-1],
            )

    def test_valid_values_must_be_exactly_two_booleans(self) -> None:
        for valid in (
            (True,),
            (True, False, True),
            (1, False),
            ("yes", False),
        ):
            with self.subTest(valid=valid):
                with self.assertRaises((TypeError, ValueError)):
                    object_balanced_factor_loss(
                        [self.factor_map(0.0, 0.0)],
                        [self.target(valid=valid)],
                    )

    def test_targets_must_be_a_sequence_of_object_factor_targets(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            object_balanced_factor_loss([self.factor_map(0.0, 0.0)], None)
        with self.assertRaises((TypeError, ValueError)):
            object_balanced_factor_loss([self.factor_map(0.0, 0.0)], [{"class_id": 0}])
        with self.assertRaises((TypeError, ValueError)):
            object_balanced_factor_loss([self.factor_map(0.0, 0.0)], "invalid")


class FactorAlignmentDetectionIntegrationTest(unittest.TestCase):
    class _Model:
        factor_supervision_schedule = 1.0
        dcli_schedule = 1.0

        def __init__(self, contexts, clean_contexts=None):
            self.contexts = contexts
            self.clean_contexts = clean_contexts
            self.observed = None

        def consume_loss_reliability_contexts(self):
            return self.contexts, self.clean_contexts

        @staticmethod
        def adapt_localization_factors(factors):
            return factors

        def observe_gradient_diagnostics(self, losses):
            self.observed = losses

    class _BBoxLoss:
        def set_schedule(self, _value):
            pass

        def set_uncertainty(self, _value):
            pass

        def discard_uncertainty(self):
            pass

    @staticmethod
    def _contexts(offset=0.0):
        from ifdr_yolo.models.gated_fusion import ReliabilityContext

        return {
            node: ReliabilityContext(
                factors=torch.full((1, 2, size, size), float(node) + offset),
                branch_weights=torch.ones(1, 2, size, size),
                gate_strength=1.0,
            )
            for node, size in (
                (11, 4),
                (14, 3),
                (17, 2),
                (20, 3),
                (23, 4),
                (26, 5),
            )
        }

    def _criterion(self, clean_contexts=None):
        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        model = self._Model(self._contexts(), clean_contexts)
        criterion = IFDRDetectionLoss.__new__(IFDRDetectionLoss)
        criterion._model_ref = weakref.ref(model)
        criterion.factor_weights = (1.0, 1.0)
        criterion.entropy_weight = 1.0
        criterion.factor_supervision_gain = 1.0
        criterion.counterfactual_gain = 0.0
        criterion.bbox_loss = self._BBoxLoss()
        criterion.reg_max = 16
        return criterion, model

    @staticmethod
    def _preds():
        return {
            "feats": [
                torch.zeros(1, 8, 2, 2),
                torch.zeros(1, 8, 3, 3),
                torch.zeros(1, 8, 4, 4),
                torch.zeros(1, 8, 5, 5),
            ],
            "boxes": torch.zeros(1, 4, 1),
        }

    @staticmethod
    def _batch(extra=None):
        batch = {
            "ifdr_factor_target": torch.zeros(1, 2, 8, 8),
            "ifdr_factor_weight": torch.ones(1, 2, 8, 8),
        }
        if extra:
            batch.update(extra)
        return batch

    def test_legacy_batch_does_not_call_natural_alignment(self):
        from ultralytics.utils.loss import v8DetectionLoss

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        criterion, _ = self._criterion()
        with patch.object(
            v8DetectionLoss,
            "get_assigned_targets_and_loss",
            return_value=(None, torch.zeros(3), None),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.multiscale_factor_supervision",
            return_value=torch.tensor(0.25),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.flatten_pyramid_factors",
            return_value=torch.zeros(1, 4, 2),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.normalized_dfl_entropy",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.derive_localization_uncertainty",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.object_balanced_factor_loss",
        ) as natural:
            IFDRDetectionLoss.get_assigned_targets_and_loss(
                criterion,
                self._preds(),
                self._batch(),
            )
        natural.assert_not_called()

    def test_factor_object_targets_use_only_four_primary_nodes_and_add_loss(self):
        from ultralytics.utils.loss import v8DetectionLoss

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        criterion, _ = self._criterion()
        natural_value = torch.tensor(0.5)

        def natural_side_effect(maps, targets, **_kwargs):
            self.assertEqual(len(maps), 4)
            self.assertEqual(
                [float(factor_map[0, 0, 0, 0]) for factor_map in maps],
                [17.0, 20.0, 23.0, 26.0],
            )
            self.assertEqual(len(targets), 1)
            return natural_value

        with patch.object(
            v8DetectionLoss,
            "get_assigned_targets_and_loss",
            return_value=(None, torch.zeros(3), None),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.multiscale_factor_supervision",
            return_value=torch.tensor(0.25),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.flatten_pyramid_factors",
            return_value=torch.zeros(1, 4, 2),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.normalized_dfl_entropy",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.derive_localization_uncertainty",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.object_balanced_factor_loss",
            side_effect=natural_side_effect,
        ):
            _assignments, loss, reported = IFDRDetectionLoss.get_assigned_targets_and_loss(
                criterion,
                self._preds(),
                self._batch(
                    {
                        "factor_object_targets": (
                            FactorAlignmentLossTest.target(class_id=4),
                        )
                    }
                ),
            )
        self.assertTrue(torch.equal(loss, torch.tensor([0.75, 0.0, 0.0])))
        self.assertTrue(torch.equal(reported[-2:], torch.tensor([0.75, 0.0])))

    def test_factor_object_targets_use_clean_contexts_when_paired(self):
        from ultralytics.utils.loss import v8DetectionLoss

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        clean_contexts = self._contexts(offset=100.0)
        criterion, _ = self._criterion(clean_contexts=clean_contexts)

        def natural_side_effect(maps, _targets, **_kwargs):
            self.assertEqual(
                [float(factor_map[0, 0, 0, 0]) for factor_map in maps],
                [117.0, 120.0, 123.0, 126.0],
            )
            return torch.tensor(0.0)

        with patch.object(
            v8DetectionLoss,
            "get_assigned_targets_and_loss",
            return_value=(None, torch.zeros(3), None),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.multiscale_factor_supervision",
            return_value=torch.tensor(0.25),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.flatten_pyramid_factors",
            return_value=torch.zeros(1, 4, 2),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.normalized_dfl_entropy",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.derive_localization_uncertainty",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.object_balanced_factor_loss",
            side_effect=natural_side_effect,
        ):
            IFDRDetectionLoss.get_assigned_targets_and_loss(
                criterion,
                self._preds(),
                self._batch(
                    {
                        "factor_object_targets": (
                            FactorAlignmentLossTest.target(class_id=4),
                        )
                    }
                ),
            )

    def test_production_natural_call_disables_repeated_finite_scan(self):
        from ultralytics.utils.loss import v8DetectionLoss

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        criterion, _ = self._criterion()
        with patch.object(
            v8DetectionLoss,
            "get_assigned_targets_and_loss",
            return_value=(None, torch.zeros(3), None),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.multiscale_factor_supervision",
            return_value=torch.tensor(0.25),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.flatten_pyramid_factors",
            return_value=torch.zeros(1, 4, 2),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.normalized_dfl_entropy",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.derive_localization_uncertainty",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.object_balanced_factor_loss",
            return_value=torch.tensor(0.0),
        ) as natural:
            IFDRDetectionLoss.get_assigned_targets_and_loss(
                criterion,
                self._preds(),
                self._batch(
                    {
                        "factor_object_targets": (
                            FactorAlignmentLossTest.target(class_id=4),
                        )
                    }
                ),
            )
        natural.assert_called_once()
        self.assertFalse(natural.call_args.kwargs["check_finite"])

    def test_missing_main_natural_context_fails_closed(self):
        from ultralytics.utils.loss import v8DetectionLoss

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        criterion, model = self._criterion()
        del model.contexts[20]
        with patch.object(
            v8DetectionLoss,
            "get_assigned_targets_and_loss",
            return_value=(None, torch.zeros(3), None),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.multiscale_factor_supervision",
            return_value=torch.tensor(0.25),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.flatten_pyramid_factors",
            return_value=torch.zeros(1, 4, 2),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.normalized_dfl_entropy",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.derive_localization_uncertainty",
            return_value=torch.zeros(1, 4),
        ):
            with self.assertRaisesRegex(ValueError, r"main.*20|20.*main"):
                IFDRDetectionLoss.get_assigned_targets_and_loss(
                    criterion,
                    self._preds(),
                    self._batch(
                        {
                            "factor_object_targets": (
                                FactorAlignmentLossTest.target(class_id=4),
                            )
                        }
                    ),
                )

    def test_missing_clean_natural_context_fails_closed(self):
        from ultralytics.utils.loss import v8DetectionLoss

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        clean_contexts = self._contexts(offset=100.0)
        del clean_contexts[23]
        criterion, _ = self._criterion(clean_contexts=clean_contexts)
        with patch.object(
            v8DetectionLoss,
            "get_assigned_targets_and_loss",
            return_value=(None, torch.zeros(3), None),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.multiscale_factor_supervision",
            return_value=torch.tensor(0.25),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.flatten_pyramid_factors",
            return_value=torch.zeros(1, 4, 2),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.normalized_dfl_entropy",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.derive_localization_uncertainty",
            return_value=torch.zeros(1, 4),
        ):
            with self.assertRaisesRegex(ValueError, r"clean.*23|23.*clean"):
                IFDRDetectionLoss.get_assigned_targets_and_loss(
                    criterion,
                    self._preds(),
                    self._batch(
                        {
                            "factor_object_targets": (
                                FactorAlignmentLossTest.target(class_id=4),
                            )
                        }
                    ),
                )

    def test_production_natural_alignment_backpropagates_to_four_primary_nodes(self):
        from ultralytics.utils.loss import v8DetectionLoss

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        criterion, model = self._criterion()
        for context in model.contexts.values():
            context.factors.requires_grad_()
        with patch.object(
            v8DetectionLoss,
            "get_assigned_targets_and_loss",
            return_value=(None, torch.zeros(3), None),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.multiscale_factor_supervision",
            return_value=torch.tensor(0.25),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.flatten_pyramid_factors",
            return_value=torch.zeros(1, 4, 2),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.normalized_dfl_entropy",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.derive_localization_uncertainty",
            return_value=torch.zeros(1, 4),
        ):
            _assignments, loss, _reported = IFDRDetectionLoss.get_assigned_targets_and_loss(
                criterion,
                self._preds(),
                self._batch(
                    {
                        "factor_object_targets": (
                            FactorAlignmentLossTest.target(class_id=4),
                        )
                    }
                ),
            )
        loss.sum().backward()
        for node in (17, 20, 23, 26):
            gradient = model.contexts[node].factors.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_malformed_factor_object_targets_fail_closed(self):
        from ultralytics.utils.loss import v8DetectionLoss

        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        criterion, _ = self._criterion()
        with patch.object(
            v8DetectionLoss,
            "get_assigned_targets_and_loss",
            return_value=(None, torch.zeros(3), None),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.multiscale_factor_supervision",
            return_value=torch.tensor(0.25),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.flatten_pyramid_factors",
            return_value=torch.zeros(1, 4, 2),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.normalized_dfl_entropy",
            return_value=torch.zeros(1, 4),
        ), patch(
            "ifdr_yolo.losses.ifdr_detection.derive_localization_uncertainty",
            return_value=torch.zeros(1, 4),
        ):
            with self.assertRaises((TypeError, ValueError)):
                IFDRDetectionLoss.get_assigned_targets_and_loss(
                    criterion,
                    self._preds(),
                    self._batch({"factor_object_targets": (object(),)}),
                )


if __name__ == "__main__":
    unittest.main()
