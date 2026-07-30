from pathlib import Path
import unittest

import torch

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kitti-p2-m.yaml"


class IFDRDetectionModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_ultralytics_config(ROOT)
        from ultralytics.nn.tasks import DetectionModel

        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

        torch.manual_seed(17)
        cls.baseline = DetectionModel(str(MODEL_PATH), verbose=False)
        torch.manual_seed(29)
        cls.ifdr = IFDRDetectionModel(str(MODEL_PATH), verbose=False)
        load_result = cls.ifdr.load_state_dict(
            cls.baseline.state_dict(),
            strict=False,
        )
        cls.missing_keys = tuple(load_result.missing_keys)
        cls.unexpected_keys = tuple(load_result.unexpected_keys)

    def test_installs_all_six_bidirectional_fusion_nodes(self) -> None:
        from ifdr_yolo.models.gated_fusion import ReliabilityGatedConcat

        self.assertEqual(
            self.ifdr.fusion_node_indices,
            (11, 14, 17, 20, 23, 26),
        )
        self.assertTrue(
            all(
                isinstance(self.ifdr.model[index], ReliabilityGatedConcat)
                for index in self.ifdr.fusion_node_indices
            )
        )
        self.assertFalse(self.unexpected_keys)
        self.assertTrue(self.missing_keys)
        self.assertTrue(
            all(
                any(
                    key.startswith(f"model.{index}.")
                    for index in self.ifdr.fusion_node_indices
                )
                for key in self.missing_keys
            )
        )

    def test_six_scales_share_one_factor_semantics_estimator(self) -> None:
        estimators = [
            self.ifdr.model[index].reliability_estimator
            for index in self.ifdr.fusion_node_indices
        ]

        self.assertTrue(
            all(estimator is estimators[0] for estimator in estimators[1:])
        )

    def test_zero_schedule_matches_original_p2_forward_exactly(self) -> None:
        self.ifdr.set_reliability_schedule(0.0)
        self.baseline.train()
        self.ifdr.train()
        image = torch.randn(
            1,
            3,
            128,
            128,
            generator=torch.Generator().manual_seed(101),
        )

        with torch.no_grad():
            baseline_output = self.baseline(image)
            ifdr_output = self.ifdr(image)

        self.assertEqual(set(ifdr_output), {"boxes", "scores", "feats"})
        for key in ("boxes", "scores"):
            self.assertTrue(
                torch.equal(ifdr_output[key], baseline_output[key]),
                key,
            )
        for actual, expected in zip(
            ifdr_output["feats"],
            baseline_output["feats"],
        ):
            self.assertTrue(torch.equal(actual, expected))

    def test_collects_one_context_per_fusion_node(self) -> None:
        self.ifdr.train()
        with torch.no_grad():
            self.ifdr(torch.zeros(1, 3, 128, 128))

        contexts = self.ifdr.consume_reliability_context()

        self.assertEqual(tuple(contexts), self.ifdr.fusion_node_indices)
        self.assertEqual(
            [tuple(contexts[index].factors.shape[-2:]) for index in contexts],
            [(8, 8), (16, 16), (32, 32), (16, 16), (8, 8), (4, 4)],
        )
        with self.assertRaisesRegex(RuntimeError, "no reliability context"):
            self.ifdr.consume_reliability_context()

    def test_schedule_updates_every_fusion_node(self) -> None:
        self.ifdr.set_reliability_schedule(0.75)

        for index in self.ifdr.fusion_node_indices:
            self.assertAlmostEqual(
                float(self.ifdr.model[index]._schedule),
                0.75,
            )
        self.assertAlmostEqual(self.ifdr.ifdr_schedule, 0.75)

    def test_component_schedules_can_disable_fusion_without_disabling_loss(self) -> None:
        self.ifdr.set_component_schedules(
            fusion=0.0,
            dcli=0.7,
            factor_supervision=0.4,
        )

        for index in self.ifdr.fusion_node_indices:
            self.assertEqual(float(self.ifdr.model[index]._schedule), 0.0)
        self.assertEqual(self.ifdr.fusion_schedule, 0.0)
        self.assertAlmostEqual(self.ifdr.dcli_schedule, 0.7)
        self.assertAlmostEqual(self.ifdr.factor_supervision_schedule, 0.4)

    def test_rejects_non_concat_fusion_node(self) -> None:
        from ultralytics.nn.tasks import DetectionModel

        from ifdr_yolo.models.ifdr_model import (
            FusionNodeSpec,
            install_reliability_fusion,
        )

        model = DetectionModel(str(MODEL_PATH), verbose=False)
        with self.assertRaisesRegex(ValueError, "Concat"):
            install_reliability_fusion(
                model,
                specs=(FusionNodeSpec(10, (576, 576)),),
                reliability_channels=4,
            )


if __name__ == "__main__":
    unittest.main()
