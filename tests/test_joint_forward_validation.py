from pathlib import Path
import subprocess
import sys
import unittest

from ifdr_yolo.experiments.config import load_ifdr_config
from scripts.run_joint_forward_validation import SPECS


ROOT = Path(__file__).resolve().parents[1]


class JointForwardValidationTest(unittest.TestCase):
    def test_declares_one_recoverable_diagnostic_run(self) -> None:
        self.assertEqual(len(SPECS), 1)
        spec = SPECS[0]
        config = load_ifdr_config(ROOT / spec.config, repository_root=ROOT)

        self.assertEqual(spec.expected_epochs, 90)
        self.assertEqual(config.training.epochs, 90)
        self.assertTrue(config.method.components.semantic_protection)
        self.assertTrue(config.method.components.counterfactual_consistency)
        self.assertEqual(config.method.gradient_diagnostic_interval, 50)
        self.assertEqual(config.experiment.variant, spec.variant)

    def test_script_can_be_invoked_directly(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                str(ROOT / "scripts/run_joint_forward_validation.py"),
                "--help",
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
