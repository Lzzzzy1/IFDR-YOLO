from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.experiments.run_store import RunStore, build_run_id


class RunStoreTest(unittest.TestCase):
    def test_build_run_id_contains_identity_seed_and_commit(self) -> None:
        value = build_run_id(
            timestamp=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
            dataset="kitti",
            model="yolov8m",
            variant="baseline",
            seed=17,
            git_sha="034aee29d105",
        )

        self.assertEqual(
            value,
            "20260729T120000Z-kitti-yolov8m-baseline-s17-034aee2",
        )

    def test_create_refuses_existing_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            path.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                RunStore.create(path)

    def test_create_records_prepared_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore.create(Path(directory) / "run")

            payload = json.loads(
                (store.root / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(store.state, "prepared")
            self.assertEqual(payload["state"], "prepared")
            self.assertIn("updated_at_utc", payload)

    def test_transitions_through_success_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore.create(Path(directory) / "run")

            for state in ("running", "trained", "evaluating", "complete"):
                store.transition(state)

            payload = json.loads(
                (store.root / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(store.state, "complete")
            self.assertEqual(payload["state"], "complete")

    def test_rejects_illegal_state_jump(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore.create(Path(directory) / "run")

            with self.assertRaisesRegex(ValueError, "prepared -> trained"):
                store.transition("trained")

    def test_fail_records_stage_and_exception_without_deleting_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore.create(Path(directory) / "run")
            artifact = store.root / "train.log"
            artifact.write_text("partial output\n", encoding="utf-8")
            error = RuntimeError("synthetic failure")

            store.fail(stage="training", error=error)

            payload = json.loads(
                (store.root / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(store.state, "failed")
            self.assertEqual(payload["stage"], "training")
            self.assertEqual(payload["error_type"], "RuntimeError")
            self.assertEqual(payload["error_message"], "synthetic failure")
            self.assertTrue(artifact.exists())


if __name__ == "__main__":
    unittest.main()
