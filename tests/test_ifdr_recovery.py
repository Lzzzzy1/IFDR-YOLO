from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.experiments.config import load_ifdr_config
from ifdr_yolo.experiments.recovery import (
    RecoveryServices,
    recover_ifdr_run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_ifdr_config(
    ROOT / "configs/experiments/kitti_ifdr_yolov8m_s17.yaml",
    repository_root=ROOT,
)


class FakeTrainer:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def train(self) -> None:
        run_dir = Path(self.kwargs["overrides"]["save_dir"])
        with (run_dir / "results.csv").open("a", encoding="utf-8") as output:
            for epoch in range(296, 300):
                output.write(f"{epoch},0.5\n")


class FakePredictor:
    def predict(self, **kwargs) -> Path:
        labels = Path(kwargs["output_dir"]) / "labels"
        labels.mkdir(parents=True)
        (labels / "000001.txt").write_text("", encoding="utf-8")
        return labels


class IFDRRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTrainer.instances.clear()

    def test_resumes_failed_run_then_completes_ap40_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            weights = run_dir / "weights"
            weights.mkdir(parents=True)
            (weights / "last.pt").write_bytes(b"last")
            (weights / "best.pt").write_bytes(b"best")
            (run_dir / "status.json").write_text(
                json.dumps({"state": "failed", "stage": "training"}),
                encoding="utf-8",
            )
            (run_dir / "results.csv").write_text(
                "epoch,metrics/mAP50-95(B)\n"
                + "".join(f"{epoch},0.4\n" for epoch in range(296)),
                encoding="utf-8",
            )
            split = root / "val.txt"
            split.write_text("000001\n", encoding="utf-8")
            paths = replace(
                CONFIG.paths,
                val_ids=split,
                raw_images=root / "raw-images",
                raw_labels=root / "raw-labels",
                generated_data=root / "generated",
            )
            config = replace(CONFIG, paths=paths)
            services = RecoveryServices(
                trainer_factory=FakeTrainer,
                prediction_adapter=FakePredictor(),
                evaluate=lambda **_: {"classes": {}, "split_count": 1},
                collect_git=lambda _: {
                    "commit": "abcdef123456",
                    "tracked_clean": True,
                },
                now=lambda: datetime(
                    2026,
                    7,
                    31,
                    tzinfo=timezone.utc,
                ),
            )

            result = recover_ifdr_run(
                config,
                run_dir=run_dir,
                repository_root=ROOT,
                device="0",
                services=services,
            )

            self.assertEqual(result.completed_epochs, 300)
            self.assertTrue(result.metrics_path.is_file())
            self.assertEqual(
                json.loads(
                    (run_dir / "status.json").read_text(encoding="utf-8")
                )["state"],
                "complete",
            )
            trainer = FakeTrainer.instances[-1]
            self.assertEqual(
                trainer.kwargs["overrides"]["resume"],
                str(weights / "last.pt"),
            )
            self.assertEqual(
                trainer.kwargs["overrides"]["save_dir"],
                str(run_dir),
            )
            self.assertTrue(
                (run_dir / "status.before-recovery.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
