from pathlib import Path
import subprocess
import sys
import unittest

from ifdr_yolo.experiments.config import load_ifdr_config
from scripts.run_mechanism_screen import SPECS


ROOT = Path(__file__).resolve().parents[1]


class MechanismScreenTest(unittest.TestCase):
    def test_script_can_be_invoked_directly(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                str(ROOT / "scripts" / "run_mechanism_screen.py"),
                "--help",
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_declares_locked_two_by_two_causal_matrix(self) -> None:
        self.assertEqual(
            tuple(spec.key for spec in SPECS),
            (
                "full_control_e90_s17",
                "protected_only_e90_s17",
                "counterfactual_only_e90_s17",
                "protected_counterfactual_e90_s17",
            ),
        )
        signatures = []
        for spec in SPECS:
            config = load_ifdr_config(
                ROOT / spec.config,
                repository_root=ROOT,
            )
            self.assertEqual(spec.expected_epochs, 90)
            self.assertEqual(config.training.epochs, 90)
            self.assertEqual(config.experiment.seed, 17)
            self.assertEqual(config.experiment.variant, spec.variant)
            signatures.append(
                (
                    config.method.components.semantic_protection,
                    config.method.components.counterfactual_consistency,
                )
            )

        self.assertEqual(
            signatures,
            [
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
