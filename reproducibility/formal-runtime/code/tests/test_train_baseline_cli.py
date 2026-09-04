from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainBaselineCliTest(unittest.TestCase):
    def test_script_has_direct_help_with_required_mode(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/train_baseline.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--config", completed.stdout)
        self.assertIn("--mode {dry-run,smoke,full}", completed.stdout)
        self.assertIn("--device", completed.stdout)


if __name__ == "__main__":
    unittest.main()
