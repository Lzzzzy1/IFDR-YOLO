"""Fail-closed single-GPU queue contract tests (Task 9)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import tempfile
import unittest
from dataclasses import replace

from ifdr_yolo.data.replay_sampler import sha256_canonical
from ifdr_yolo.eval.factor_repair_gate import (
    PRIMARY_ENDPOINTS,
    FactorRepairSelectionDecision,
    digest_selection_decision,
)
from scripts.run_factor_repair_queue import FactorRepairQueue
from scripts.train_factor_repair import file_sha256


_ARTIFACT_ROOT = Path(tempfile.mkdtemp(prefix="factor-repair-artifacts-"))


def _endpoints(value: float) -> dict[str, float]:
    return {name: value for name in PRIMARY_ENDPOINTS}


def _artifact(condition: str, checkpoint_hash: str, evidence_payload: dict[str, object]) -> dict[str, object]:
    checkpoint_path = _ARTIFACT_ROOT / f"{condition.lower()}-last.pt"
    checkpoint_path.write_bytes(f"{condition}-calibration".encode("utf-8"))
    checkpoint_hash = file_sha256(checkpoint_path)
    diagnostic_path = _ARTIFACT_ROOT / f"{condition.lower()}-best.pt"
    diagnostic_path.write_bytes(f"{condition}-diagnostic".encode("utf-8"))
    diagnostic_hash = file_sha256(diagnostic_path)
    evidence_hash = sha256_canonical(evidence_payload)
    manifest = {
        "condition": condition,
        "checkpoint_role": "calibration_last",
        "checkpoint_sha256": checkpoint_hash,
        "metadata_index_sha256": "3" * 64,
        "semantic_state_sha256": ("1" if condition == "F0" else "2") * 64,
        "fit_ids_sha256": "4" * 64,
        "image_ids_sha256": "5" * 64,
    }
    manifest["manifest_sha256"] = sha256_canonical(manifest)
    return {
        "condition": condition,
        "primary_checkpoint": {
            "path": checkpoint_path.as_posix(),
            "role": "primary",
            "checkpoint_role": "calibration_last",
            "sha256": checkpoint_hash,
        },
        "diagnostic_checkpoint": {
            "path": diagnostic_path.as_posix(),
            "role": "diagnostic",
            "sha256": diagnostic_hash,
        },
        "semantic_state_sha256": ("1" if condition == "F0" else "2") * 64,
        "metadata_index_sha256": "3" * 64,
        "fit_ids_sha256": "4" * 64,
        "image_ids_sha256": "5" * 64,
        "evidence_payload": evidence_payload,
        "evidence_sha256": evidence_hash,
        "image_ids": tuple(evidence_payload["image_ids"]),
        "endpoints": evidence_payload["endpoints"],
        "manifest": manifest,
        "passed": True,
    }


def _decision() -> tuple[FactorRepairSelectionDecision, dict[str, object], dict[str, object]]:
    f0_endpoints = _endpoints(0.10)
    f3_endpoints = _endpoints(0.30)
    f0_payload = {"condition": "F0", "image_ids": ["a", "b"], "endpoints": f0_endpoints}
    f3_payload = {"condition": "F3", "image_ids": ["a", "b"], "endpoints": f3_endpoints}
    f0_hash = sha256_canonical(f0_payload)
    f3_hash = sha256_canonical(f3_payload)
    f0 = _artifact("F0", "0" * 64, f0_payload)
    f3 = _artifact("F3", "0" * 64, f3_payload)
    table = {"F0": f0_endpoints, "F3": f3_endpoints}
    digest = digest_selection_decision("F0", "F3", 0.20, (0.05, 0.30), table, f0_hash, f3_hash)
    return FactorRepairSelectionDecision("F0", "F3", 0.20, (0.05, 0.30), table, f0_hash, f3_hash, digest), f0, f3


def _complete(queue: FactorRepairQueue, name: str, artifacts: dict[str, object]) -> None:
    if queue.job_status(name) == "pending":
        if queue.launchable(name):
            queue.start(name)
        else:
            queue.transition(name, "running")
    queue.complete(name, artifacts=artifacts)


class FactorRepairQueueTest(unittest.TestCase):
    def test_pending_complete_and_running_blocked_transitions_are_forbidden(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration",))
            _, f0, _ = _decision()
            with self.assertRaisesRegex(ValueError, "pending -> complete"):
                queue.transition("F0-calibration", "complete", artifacts=f0)
            queue.start("F0-calibration")
            with self.assertRaisesRegex(ValueError, "running -> blocked"):
                queue.transition("F0-calibration", "blocked")

    def test_start_persists_complete_identity_and_resume_rechecks_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {
                "config_sha256": "a" * 64,
                "condition": "F0",
                "commit": "b" * 40,
                "initialization_checkpoint_sha256": "c" * 64,
                "metadata_index_sha256": "d" * 64,
                "manifest_sha256": "e" * 64,
                "fit_ids_sha256": "f" * 64,
            }
            queue = FactorRepairQueue.create(root, jobs=("F0-calibration",), identity=identity)
            payload = __import__("json").loads((root / "queue_state.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["identity"], identity)
            _, f0, _ = _decision()
            queue.transition("F0-calibration", "running", artifacts=f0)
            queue.fail("F0-calibration", RuntimeError("interrupt"))
            Path(f0["primary_checkpoint"]["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                queue.resume("F0-calibration", identity=identity)

    def test_resume_requires_nonempty_identity_equal_to_persisted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"config_sha256": "a" * 64, "condition": "F0", "commit": "b" * 40}
            queue = FactorRepairQueue.create(root, jobs=("F0-calibration",), identity=identity)
            _, f0, _ = _decision()
            queue.transition("F0-calibration", "running", artifacts=f0)
            queue.fail("F0-calibration", RuntimeError("interrupt"))
            with self.assertRaisesRegex(ValueError, "identity"):
                queue.resume("F0-calibration")
            with self.assertRaisesRegex(ValueError, "identity"):
                queue.resume("F0-calibration", identity={"config_sha256": "c" * 64})
            queue.resume("F0-calibration", identity=identity)

    def test_create_existing_queue_rejects_missing_or_mismatched_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"config_sha256": "a" * 64, "condition": "F0", "commit": "b" * 40}
            FactorRepairQueue.create(root, jobs=("F0-calibration",), identity=identity)
            with self.assertRaisesRegex(ValueError, "identity"):
                FactorRepairQueue.create(root, jobs=("F0-calibration",), identity=None)
            with self.assertRaisesRegex(ValueError, "identity"):
                FactorRepairQueue.create(root, jobs=("F0-calibration",), identity={"config_sha256": "c" * 64})

    def test_completed_artifacts_require_explicit_gate_and_checkpoint(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration",))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                queue.transition("F0-calibration", "running")
                queue.complete("F0-calibration", artifacts={"condition": "F0", "gate_passed": True})

    def test_primary_schema_requires_primary_role_checkpoint_role_and_sha(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration",))
            _, f0, _ = _decision()
            bad_role = dict(f0)
            bad_role["primary_checkpoint"] = dict(f0["primary_checkpoint"])
            bad_role["primary_checkpoint"]["role"] = "calibration_last"
            queue.transition("F0-calibration", "running")
            with self.assertRaisesRegex(ValueError, "primary role"):
                queue.complete("F0-calibration", artifacts=bad_role)
            with TemporaryDirectory() as other_directory:
                queue = FactorRepairQueue.create(Path(other_directory), jobs=("F0-calibration",))
                missing_sha = dict(f0)
                missing_sha["primary_checkpoint"] = dict(f0["primary_checkpoint"])
                missing_sha["primary_checkpoint"].pop("sha256")
                queue.transition("F0-calibration", "running")
                with self.assertRaisesRegex(ValueError, "sha256"):
                    queue.complete("F0-calibration", artifacts=missing_sha)

    def test_selection_requires_manifest_digest_and_evidence_payload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = FactorRepairQueue.create(root / "manifest", jobs=("F0-calibration", "F3-calibration"))
            decision, f0, f3 = _decision()
            _complete(queue, "F0-calibration", f0)
            _complete(queue, "F3-calibration", f3)
            queue.jobs["F3-calibration"].artifacts["manifest"].pop("manifest_sha256")
            with self.assertRaisesRegex(ValueError, "manifest.*SHA256"):
                queue.consume_selection_decision(decision)
            queue = FactorRepairQueue.create(root / "evidence", jobs=("F0-calibration", "F3-calibration"))
            decision, f0, f3 = _decision()
            _complete(queue, "F0-calibration", f0)
            _complete(queue, "F3-calibration", f3)
            queue.jobs["F3-calibration"].artifacts.pop("evidence_payload")
            with self.assertRaisesRegex(ValueError, "evidence payload"):
                queue.consume_selection_decision(decision)

    def test_adaptation_rejects_best_primary_role(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("selected-repair-adaptation",))
            with self.assertRaisesRegex(ValueError, "best"):
                queue.transition("selected-repair-adaptation", "running")
                queue.complete("selected-repair-adaptation", artifacts={"condition": "F3", "gate_passed": True, "primary_checkpoint": {"path": "best.pt", "role": "best", "sha256": "a" * 64}})

    def test_selection_rejects_missing_endpoint_or_image_identity(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration", "F3-calibration"))
            decision, f0, f3 = _decision()
            f3["evidence_payload"] = dict(f3["evidence_payload"])
            f3["evidence_payload"]["endpoints"] = dict(f3["evidence_payload"]["endpoints"])
            f3["evidence_payload"]["endpoints"].pop(PRIMARY_ENDPOINTS[0])
            f3["endpoints"] = dict(f3["evidence_payload"]["endpoints"])
            f3["evidence_sha256"] = sha256_canonical(f3["evidence_payload"])
            decision = replace(
                decision,
                selected_evidence_sha256=f3["evidence_sha256"],
                decision_sha256=digest_selection_decision(
                    decision.reference_condition,
                    decision.selected_condition,
                    decision.delta_s_point,
                    decision.delta_s_ci95,
                    decision.endpoint_table,
                    decision.reference_evidence_sha256,
                    f3["evidence_sha256"],
                ),
            )
            _complete(queue, "F0-calibration", f0)
            _complete(queue, "F3-calibration", f3)
            with self.assertRaisesRegex(ValueError, "endpoint"):
                queue.consume_selection_decision(decision)

    def test_failed_candidate_never_persists_manifest(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F3-calibration",))
            _, _, f3 = _decision()
            f3["passed"] = False
            _complete(queue, "F3-calibration", f3)
            self.assertNotIn("manifest", queue.jobs["F3-calibration"].artifacts or {})

    def test_no_candidate_blocks_adaptation_only_after_all_candidates_terminal(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration", "F1-calibration", "F2-calibration", "F3-calibration", "F0-adaptation", "selected-repair-adaptation"))
            _, f0, _ = _decision()
            _complete(queue, "F0-calibration", f0)
            queue.consume_selection_decision(None)
            self.assertEqual(queue.job_status("F0-adaptation"), "pending")

    def test_selection_digest_is_locked_after_first_consumption(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration", "F3-calibration"))
            decision, f0, f3 = _decision()
            _complete(queue, "F0-calibration", f0)
            _complete(queue, "F3-calibration", f3)
            queue.consume_selection_decision(decision)
            altered = replace(decision, delta_s_point=decision.delta_s_point + 0.01, decision_sha256=digest_selection_decision("F0", "F3", decision.delta_s_point + 0.01, decision.delta_s_ci95, decision.endpoint_table, decision.reference_evidence_sha256, decision.selected_evidence_sha256))
            with self.assertRaisesRegex(ValueError, "already consumed"):
                queue.consume_selection_decision(altered)

    def test_queue_runs_f0_control_only_with_selected_repair(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = FactorRepairQueue.create(root, jobs=("F0-calibration", "F3-calibration", "F0-adaptation", "selected-repair-adaptation"))
            decision, f0, f3 = _decision()
            _complete(queue, "F0-calibration", f0)
            self.assertFalse(queue.launchable("F0-adaptation"))
            _complete(queue, "F3-calibration", f3)
            queue.consume_selection_decision(decision)
            self.assertTrue(queue.launchable("F0-adaptation"))
            self.assertTrue(queue.launchable("selected-repair-adaptation"))

    def test_queue_rejects_manual_condition_string(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration",))
            with self.assertRaisesRegex(ValueError, "selection decision"):
                queue.consume_selection_decision("F3")

    def test_no_factor_candidate_blocks_track_f_adaptation(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration", "F1-calibration", "F2-calibration", "F3-calibration"))
            _, f0, _ = _decision()
            _complete(queue, "F0-calibration", f0)
            failed = dict(f0)
            f1_path = _ARTIFACT_ROOT / "f1-last.pt"
            f1_path.write_bytes(b"F1-calibration")
            failed.update(condition="F1", passed=False, primary_checkpoint={"path": f1_path.as_posix(), "role": "primary", "checkpoint_role": "calibration_last", "sha256": file_sha256(f1_path)}, evidence_payload={"condition": "F1", "image_ids": ["a", "b"], "endpoints": _endpoints(0.0)}, evidence_sha256=sha256_canonical({"condition": "F1", "image_ids": ["a", "b"], "endpoints": _endpoints(0.0)}))
            for name, condition in (("F1-calibration", "F1"), ("F2-calibration", "F2"), ("F3-calibration", "F3")):
                current = dict(failed)
                current["condition"] = condition
                current["manifest"] = {"condition": condition, "checkpoint_role": "calibration_last", "checkpoint_sha256": current["primary_checkpoint"]["sha256"], "metadata_index_sha256": "3" * 64}
                _complete(queue, name, current)
            self.assertEqual(queue.track_f_adaptation_status(), "blocked")

    def test_queue_rejects_tampered_selection_sha(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration", "F3-calibration"))
            decision, f0, f3 = _decision()
            _complete(queue, "F0-calibration", f0)
            _complete(queue, "F3-calibration", f3)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                queue.consume_selection_decision(replace(decision, decision_sha256="0" * 64))

    def test_queue_rejects_best_manifest_or_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration",))
            _, f0, _ = _decision()
            bad = dict(f0)
            bad["manifest"] = {"checkpoint_role": "best", "checkpoint_path": "/runs/f0/best.pt", "checkpoint_sha256": "a" * 64}
            with self.assertRaisesRegex(ValueError, "best.pt manifest"):
                queue.transition("F0-calibration", "running")
                queue.complete("F0-calibration", artifacts=bad)

    def test_factor_guided_draw_identity_binds_manifest(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration",))
            _, f0, _ = _decision()
            f0["manifest"] = {"condition": "F0", "checkpoint_role": "calibration_last", "checkpoint_sha256": "c" * 64}
            f0["primary_checkpoint"] = dict(f0["primary_checkpoint"])
            with self.assertRaisesRegex(ValueError, "manifest checkpoint hash"):
                queue.transition("F0-calibration", "running")
                queue.complete("F0-calibration", artifacts=f0)

    def test_queue_persists_atomic_state(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = FactorRepairQueue.create(root, jobs=("F0-calibration",))
            self.assertTrue((root / "queue_state.json").is_file())
            self.assertEqual(FactorRepairQueue.load(root).status["F0-calibration"], "pending")


if __name__ == "__main__":
    unittest.main()
