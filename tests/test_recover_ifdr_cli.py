from pathlib import Path
import unittest

from scripts.recover_ifdr import build_parser


class RecoverIFDRCliTest(unittest.TestCase):
    def test_parses_config_run_and_device(self) -> None:
        args = build_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--run-dir",
                "runs/failed",
                "--device",
                "0",
            ]
        )

        self.assertEqual(args.config, Path("config.yaml"))
        self.assertEqual(args.run_dir, Path("runs/failed"))
        self.assertEqual(args.device, "0")


if __name__ == "__main__":
    unittest.main()
