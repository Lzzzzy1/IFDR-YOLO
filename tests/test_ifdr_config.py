from pathlib import Path
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/kitti_ifdr_yolov8m_s17.yaml"
PROTECTED_CONFIG_PATH = (
    ROOT
    / "configs/experiments/mechanisms/kitti_ifdr_protected_cf_s17.yaml"
)


class IFDRConfigTest(unittest.TestCase):
    def test_loads_locked_research_method_configuration(self) -> None:
        from ifdr_yolo.experiments.config import load_ifdr_config

        config = load_ifdr_config(CONFIG_PATH, repository_root=ROOT)

        self.assertEqual(config.experiment.variant, "ifdr")
        self.assertEqual(config.method.reliability_channels, 32)
        self.assertEqual(config.method.schedule.frozen_epochs, 5)
        self.assertEqual(config.method.schedule.ramp_epochs, 10)
        self.assertEqual(config.method.intervention.base_seed, 17)
        self.assertAlmostEqual(config.method.loss.dcli_beta, 0.5)
        self.assertEqual(config.method.loss.factor_weights, (1.0, 1.0))
        self.assertTrue(config.method.components.fusion_gate)
        self.assertTrue(config.method.components.dcli)
        self.assertTrue(config.method.components.factor_supervision)
        self.assertTrue(config.method.components.interventions)
        self.assertFalse(config.method.components.semantic_protection)
        self.assertFalse(
            config.method.components.counterfactual_consistency
        )
        self.assertEqual(config.method.loss.counterfactual_gain, 0.0)
        self.assertEqual(config.method.gradient_diagnostic_interval, 0)

    def test_loads_optional_gradient_diagnostic_interval(self) -> None:
        from ifdr_yolo.experiments.config import load_ifdr_config

        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["ifdr"]["gradient_diagnostic_interval"] = 50
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            config = load_ifdr_config(path, repository_root=ROOT)

        self.assertEqual(config.method.gradient_diagnostic_interval, 50)

        payload["ifdr"]["gradient_diagnostic_interval"] = -1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-diagnostics.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "diagnostic_interval"):
                load_ifdr_config(path, repository_root=ROOT)

    def test_loads_protected_counterfactual_mechanism(self) -> None:
        from ifdr_yolo.experiments.config import load_ifdr_config

        config = load_ifdr_config(
            PROTECTED_CONFIG_PATH,
            repository_root=ROOT,
        )

        self.assertEqual(
            config.experiment.variant,
            "ifdr-protected-counterfactual",
        )
        self.assertTrue(config.method.components.semantic_protection)
        self.assertTrue(
            config.method.components.counterfactual_consistency
        )
        self.assertAlmostEqual(
            config.method.loss.counterfactual_gain,
            0.2,
        )

    def test_rejects_unknown_or_invalid_method_fields(self) -> None:
        from ifdr_yolo.experiments.config import load_ifdr_config

        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["ifdr"]["unknown"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ifdr_config(path, repository_root=ROOT)

    def test_rejects_incoherent_counterfactual_controls(self) -> None:
        from ifdr_yolo.experiments.config import load_ifdr_config

        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["ifdr"]["components"]["counterfactual_consistency"] = True
        payload["ifdr"]["loss"]["counterfactual_gain"] = 0.2
        payload["ifdr"]["components"]["interventions"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-counterfactual.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires interventions"):
                load_ifdr_config(path, repository_root=ROOT)

        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["ifdr"]["loss"]["counterfactual_gain"] = 0.2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inactive-counterfactual.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be zero"):
                load_ifdr_config(path, repository_root=ROOT)

        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["ifdr"]["components"]["fusion_gate"] = "yes"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-components.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fusion_gate"):
                load_ifdr_config(path, repository_root=ROOT)

        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["ifdr"]["loss"]["dcli_beta"] = 1.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ifdr_config(path, repository_root=ROOT)


if __name__ == "__main__":
    unittest.main()
