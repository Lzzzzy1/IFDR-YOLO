from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.experiments.config import (
    load_baseline_config,
    load_ifdr_config,
)
from ifdr_yolo.experiments.evidence import write_evidence_configs


ROOT = Path(__file__).resolve().parents[1]


class EvidenceMatrixTest(unittest.TestCase):
    def test_writes_exact_locked_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_evidence_configs(
                repository_root=ROOT,
                output_dir=Path(directory),
            )
            self.assertEqual(
                tuple(paths),
                (
                    "baseline_s29",
                    "baseline_s41",
                    "p2_s29",
                    "p2_s41",
                    "fusion_only_s29",
                    "fusion_only_s41",
                ),
            )
            for key, path in paths.items():
                seed = int(key.rsplit("s", 1)[1])
                if key.startswith("fusion_only"):
                    config = load_ifdr_config(path, repository_root=ROOT)
                    self.assertEqual(config.method.intervention.base_seed, seed)
                    self.assertTrue(config.method.components.fusion_gate)
                    self.assertFalse(config.method.components.dcli)
                else:
                    config = load_baseline_config(path, repository_root=ROOT)
                self.assertEqual(config.experiment.seed, seed)
                self.assertEqual(config.training.epochs, 300)

    def test_rejects_invalid_seed(self) -> None:
        from ifdr_yolo.experiments.evidence import build_seed_payload

        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            build_seed_payload({}, True)


if __name__ == "__main__":
    unittest.main()
