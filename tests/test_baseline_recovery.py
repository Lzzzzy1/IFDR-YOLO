from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.experiments.baseline_recovery import (
    BaselineRecoveryServices,
    recover_baseline_run,
)
from ifdr_yolo.experiments.config import load_baseline_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_baseline_config(
    ROOT / "configs/experiments/kitti_yolov8m_p2_s17.yaml",
    repository_root=ROOT,
)


class FakePredictor:
    def predict(self, **kwargs) -> Path:
        labels = Path(kwargs["output_dir"]) / "labels"
        labels.mkdir(parents=True)
        (labels / "000001.txt").write_text("", encoding="utf-8")
        return labels


class BaselineRecoveryTest(unittest.TestCase):
    def _write_failed_run(
        self,
        run_dir: Path,
        *,
        epochs: tuple[int, ...] = (1,),
    ) -> None:
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
            + "".join(f"{epoch},0.4\n" for epoch in epochs),
            encoding="utf-8",
        )

    def test_resumes_failed_training_and_finishes_ap40(self) -> None:
        calls: list[tuple[Path, Path, str, int]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            weights = run_dir / "weights"
            weights.mkdir(parents=True)
            last = weights / "last.pt"
            last.write_bytes(b"last")
            (weights / "best.pt").write_bytes(b"best")
            (run_dir / "status.json").write_text(
                json.dumps({"state": "failed", "stage": "training"}),
                encoding="utf-8",
            )
            (run_dir / "results.csv").write_text(
                "epoch,metrics/mAP50-95(B)\n"
                + "".join(f"{epoch},0.4\n" for epoch in range(1, 297)),
                encoding="utf-8",
            )
            split = root / "val.txt"
            split.write_text("000001\n", encoding="utf-8")
            generated = root / "generated"
            image_dir = generated / "images" / "val"
            image_dir.mkdir(parents=True)
            (image_dir / "000001.png").write_bytes(b"png")
            config = replace(
                CONFIG,
                paths=replace(
                    CONFIG.paths,
                    val_ids=split,
                    raw_images=root / "raw-images",
                    raw_labels=root / "raw-labels",
                    generated_data=generated,
                ),
            )

            def resume(
                checkpoint: Path,
                destination: Path,
                device: str,
                workers: int,
            ) -> None:
                calls.append((checkpoint, destination, device, workers))
                with (destination / "results.csv").open(
                    "a", encoding="utf-8"
                ) as output:
                    for epoch in range(297, 301):
                        output.write(f"{epoch},0.5\n")

            services = BaselineRecoveryServices(
                resume_training=resume,
                prediction_adapter=FakePredictor(),
                evaluate=lambda **_: {"classes": {}, "split_count": 1},
                collect_git=lambda _: {
                    "commit": "abcdef123456",
                    "tracked_clean": True,
                },
                now=lambda: datetime(2026, 8, 1, tzinfo=timezone.utc),
            )

            result = recover_baseline_run(
                config,
                run_dir=run_dir,
                repository_root=ROOT,
                device="0",
                services=services,
            )

            self.assertEqual(result.completed_epochs, 300)
            self.assertEqual(
                calls,
                [(last, run_dir, "0", config.training.workers)],
            )
            self.assertTrue(result.metrics_path.is_file())
            self.assertTrue(
                (run_dir / "status.before-recovery.json").is_file()
            )
            status = json.loads(
                (run_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["completed_epochs"], 300)

    def test_rejects_non_training_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "status.json").write_text(
                json.dumps({"state": "failed", "stage": "evaluating"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "failed during training"):
                recover_baseline_run(
                    CONFIG,
                    run_dir=run_dir,
                    repository_root=ROOT,
                    device="0",
                )

    def test_rejects_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_failed_run(run_dir)
            (run_dir / "weights" / "last.pt").unlink()
            with self.assertRaisesRegex(
                FileNotFoundError,
                "recovery checkpoint does not exist",
            ):
                recover_baseline_run(
                    CONFIG,
                    run_dir=run_dir,
                    repository_root=ROOT,
                    device="0",
                )

    def test_rejects_non_contiguous_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_failed_run(run_dir, epochs=(1, 3))
            with self.assertRaisesRegex(ValueError, "must be contiguous"):
                recover_baseline_run(
                    CONFIG,
                    run_dir=run_dir,
                    repository_root=ROOT,
                    device="0",
                )

    def test_rejects_dirty_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_failed_run(run_dir)
            services = BaselineRecoveryServices(
                resume_training=lambda *_: None,
                prediction_adapter=FakePredictor(),
                evaluate=lambda **_: {},
                collect_git=lambda _: {
                    "commit": "abcdef123456",
                    "tracked_clean": False,
                },
                now=lambda: datetime(2026, 8, 1, tzinfo=timezone.utc),
            )
            with self.assertRaisesRegex(RuntimeError, "clean tracked"):
                recover_baseline_run(
                    CONFIG,
                    run_dir=run_dir,
                    repository_root=ROOT,
                    device="0",
                    services=services,
                )


if __name__ == "__main__":
    unittest.main()
