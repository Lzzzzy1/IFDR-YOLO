import unittest
from pathlib import Path

from ifdr_yolo.experiments.config import load_ifdr_config
from scripts.run_unprotected_counterfactual_queue import UNPROTECTED_SPECS


class UnprotectedCounterfactualQueueTest(unittest.TestCase):
    def test_specs_cover_three_seeds_with_equal_budget(self) -> None:
        self.assertEqual(tuple(spec.seed for spec in UNPROTECTED_SPECS), (17, 29, 41))
        self.assertTrue(all(spec.expected_epochs == 300 for spec in UNPROTECTED_SPECS))
        self.assertTrue(
            all(spec.variant == "ifdr-unprotected-counterfactual-joint" for spec in UNPROTECTED_SPECS)
        )

    def test_controls_disable_only_semantic_protection(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        for spec in UNPROTECTED_SPECS:
            config = load_ifdr_config(
                spec.config,
                repository_root=repository_root,
            )
            self.assertFalse(config.method.components.semantic_protection)
            self.assertTrue(config.method.components.counterfactual_consistency)
            self.assertEqual(config.method.loss.counterfactual_gain, 0.2)
            self.assertEqual(config.training.epochs, 300)
            self.assertTrue(config.method.components.fusion_gate)
            self.assertTrue(config.method.components.dcli)
            self.assertTrue(config.method.components.factor_supervision)


if __name__ == "__main__":
    unittest.main()
