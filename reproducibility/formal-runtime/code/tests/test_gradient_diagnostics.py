import unittest

import torch

from ifdr_yolo.experiments.gradient_diagnostics import (
    GradientConflictAccumulator,
    ScheduledGradientDiagnostics,
    gradient_conflict_snapshot,
    grouped_gradient_conflict_snapshot,
    node_gradient_conflict_snapshot,
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

    def test_grouped_snapshot_exposes_protected_anchor_and_task_adapter(self) -> None:
        anchor = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        adapter = torch.nn.Parameter(torch.tensor([0.5, 0.25]))
        detection = (anchor.detach() * adapter).sum()
        factor = anchor.square().sum()

        snapshot = grouped_gradient_conflict_snapshot(
            {"detection": detection, "factor": factor},
            {
                "semantic_anchor": (anchor,),
                "fusion_adapters": (adapter,),
            },
        )

        self.assertEqual(snapshot["schema_version"], 2)
        groups = snapshot["parameter_groups"]
        self.assertEqual(
            groups["semantic_anchor"]["gradient_norms"]["detection"],
            0.0,
        )
        self.assertGreater(
            groups["semantic_anchor"]["gradient_norms"]["factor"],
            0.0,
        )
        self.assertGreater(
            groups["fusion_adapters"]["gradient_norms"]["detection"],
            0.0,
        )
        self.assertEqual(
            groups["fusion_adapters"]["gradient_norms"]["factor"],
            0.0,
        )
        self.assertIsNone(anchor.grad)
        self.assertIsNone(adapter.grad)
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

    def test_scheduled_recorder_samples_grouped_parameters(self) -> None:
        anchor = torch.nn.Parameter(torch.tensor([1.0]))
        adapter = torch.nn.Parameter(torch.tensor([2.0]))
        recorder = ScheduledGradientDiagnostics(interval=1)

        record = recorder.observe_groups(
            {
                "detection": adapter.square().sum(),
                "factor": anchor.square().sum(),
            },
            {
                "semantic_anchor": (anchor,),
                "fusion_adapters": (adapter,),
            },
        )

        self.assertEqual(record["step"], 1)
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(recorder.drain(), (record,))

    def test_scheduled_recorder_uses_node_losses_for_legacy_single_loss(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        recorder = ScheduledGradientDiagnostics(interval=1)
        node_tensors = {
            node: (parameter,)
            for node in (17, 20, 23, 26)
        }
        node_losses = {
            "detection_base": parameter.sum(),
            "dcli_conditioning": -parameter.sum(),
            "dcli_calibration": parameter.square().sum(),
            "factor_supervision": parameter.abs().sum(),
        }

        record = recorder.observe_groups(
            {"factor": node_losses["factor_supervision"]},
            {"semantic_anchor": (parameter,)},
            node_losses=node_losses,
            node_tensors=node_tensors,
        )

        self.assertEqual(record["step"], 1)
        self.assertEqual(
            set(record["parameter_groups"]["semantic_anchor"]["gradient_norms"]),
            set(node_losses),
        )
        self.assertEqual(
            set(record["node_diagnostics"]["nodes"]),
            {17, 20, 23, 26},
        )

    def test_node_snapshot_reports_registered_nodes_and_loss_components(self) -> None:
        nodes = {
            node: (torch.nn.Parameter(torch.tensor([float(node)])),)
            for node in (17, 20, 23, 26)
        }
        anchor = nodes[17][0]
        losses = {
            "detection_base": anchor.sum(),
            "dcli_conditioning": -anchor.sum(),
            "dcli_calibration": anchor.square().sum(),
            "factor_supervision": anchor.abs().sum(),
        }

        snapshot = node_gradient_conflict_snapshot(losses, nodes)

        self.assertEqual(set(snapshot["nodes"]), {17, 20, 23, 26})
        node17 = snapshot["nodes"][17]
        self.assertEqual(
            set(node17["gradient_norms"]),
            {
                "detection_base",
                "dcli_conditioning",
                "dcli_calibration",
                "factor_supervision",
            },
        )
        self.assertIsNotNone(
            node17["pairs"]["dcli_conditioning::detection_base"]["cosine"]
        )

if __name__ == "__main__":
    unittest.main()
