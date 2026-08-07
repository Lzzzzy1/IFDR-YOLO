"""Fail-closed runner contract tests (Task 9)."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from scripts.train_factor_repair import (
    EpochDrawJournal,
    build_factor_repair_run,
    build_parser,
    evaluate_primary_last,
    file_sha256,
    run_registered_condition,
    validate_finite_loss,
    verify_checkpoint_artifacts,
)


def _hash_file(path: Path) -> str:
    return file_sha256(path)


def _config(root: Path, *, identity: object | None = None) -> SimpleNamespace:
    init = root / "init.pt"
    init.write_bytes(b"registered initialization")
    init_hash = _hash_file(init)
    if identity is None:
        identity = SimpleNamespace(
            source_metadata_sha256="a" * 64,
            images_metadata_sha256="b" * 64,
            raw_labels_sha256="c" * 64,
            split_sha256="d" * 64,
            metadata_sha256="e" * 64,
            initialization_checkpoint_sha256=init_hash,
            fit_ids_sha256="f" * 64,
            development_ids_sha256="0" * 64,
        )
    return SimpleNamespace(
        condition="F0",
        identity=identity,
        paths=SimpleNamespace(initialization_checkpoint=init, output_root=root / "out"),
        fit_ids=("fit-a", "fit-b"),
        development_ids=("dev-a",),
    )


def _clean_git() -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "tracked_clean": True,
        "tracked_changes": (),
        "untracked_files": (),
    }


class FactorRepairRunnerTest(unittest.TestCase):
    def _build(self, root: Path, **kwargs):
        config = _config(root)
        return build_factor_repair_run(
            config,
            loader_ids=kwargs.pop("loader_ids", config.fit_ids),
            repository_root=root,
            run_dir=kwargs.pop("run_dir", root / "run"),
            git_provenance=kwargs.pop("git_provenance", _clean_git()),
            **kwargs,
        )

    def test_runner_rejects_development_id_in_fit_loader(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "development leakage"):
                self._build(root, loader_ids=("fit-a", "dev-a"))

    def test_dirty_checkout_fails_closed(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "clean Git checkout"):
                self._build(Path(directory), git_provenance={**_clean_git(), "tracked_clean": False, "tracked_changes": (" M x",)})

    def test_missing_or_nonhex_hash_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bad = _config(root, identity=SimpleNamespace(foo="not-a-digest"))
            with self.assertRaisesRegex(ValueError, "64-hex"):
                build_factor_repair_run(bad, loader_ids=bad.fit_ids, repository_root=root, run_dir=root / "run", git_provenance=_clean_git())

    def test_initialization_checkpoint_mismatch_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            config.identity.initialization_checkpoint_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                build_factor_repair_run(config, loader_ids=config.fit_ids, repository_root=root, run_dir=root / "run", git_provenance=_clean_git())

    def test_duplicate_process_lock_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._build(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "duplicate factor-repair process lock"):
                    self._build(root, run_dir=root / "run")
            finally:
                first.release()

    def test_nonfinite_loss_marks_run_failed(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_finite_loss({"box": float("nan")})

    def test_empty_last_or_best_checkpoint_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            last = weights / "last.pt"
            best = weights / "best.pt"
            last.write_bytes(b"")
            best.write_bytes(b"best")
            with self.assertRaisesRegex(ValueError, "empty"):
                verify_checkpoint_artifacts(root, expected={"primary_sha256": "0" * 64, "diagnostic_sha256": _hash_file(best)})

    def test_interrupted_epoch_draw_journal_resumes_exactly_once(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "draws.json"
            path.write_text(json.dumps({"records": [{"epoch": 3, "draw_key": "k", "state": "inflight"}]}), encoding="utf-8")
            journal = EpochDrawJournal.open(path)
            self.assertTrue(journal.resume(3, "k"))
            self.assertFalse(journal.resume(3, "k"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["records"], [{"epoch": 3, "draw_key": "k", "state": "committed"}])

    def test_primary_metrics_use_last_not_best(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            last = weights / "last.pt"
            best = weights / "best.pt"
            last.write_bytes(b"primary")
            best.write_bytes(b"diagnostic")
            seen: list[Path] = []
            output = evaluate_primary_last(root, checkpoint_hash=_hash_file(last), evaluator=lambda path: seen.append(path) or {"ap40": 1.0})
            self.assertEqual(seen, [last])
            self.assertEqual(output.name, "metrics_ap40_primary_last.json")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["ap40"], 1.0)

    def test_missing_empty_or_mismatched_last_hash_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                evaluate_primary_last(root, checkpoint_hash="0" * 64, evaluator=lambda _: {})

    def test_cli_rejects_ad_hoc_training_options(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--config", "x.yaml", "--condition", "F0", "--epochs", "2"])

    def test_formal_run_requires_primary_evaluator_and_real_checkpoint_hashes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._build(root)

            class Trainer:
                loss = 0.25

                def train(self):
                    weights = run.run_dir / "weights"
                    weights.mkdir()
                    (weights / "last.pt").write_bytes(b"formal-primary")
                    (weights / "best.pt").write_bytes(b"formal-diagnostic")

            with self.assertRaisesRegex(ValueError, "evaluator"):
                run_registered_condition(run, trainer_factory=Trainer)
            self.assertEqual(run.store.state, "failed")

            run = self._build(root, run_dir=root / "run-2")
            result = run_registered_condition(
                run,
                trainer_factory=Trainer,
                evaluator=lambda path: {"input": path.name},
            )
            self.assertEqual(result.store.state, "complete")
            metrics = json.loads((result.run_dir / "metrics_ap40_primary_last.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["primary_checkpoint"], "last.pt")
            self.assertEqual(metrics["primary_checkpoint_sha256"], file_sha256(result.run_dir / "weights" / "last.pt"))

    def test_build_releases_lock_when_provenance_write_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.train_factor_repair._atomic_json", side_effect=RuntimeError("write failed")):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    self._build(root)
            self.assertFalse(tuple(root.glob("*.factor_repair.lock")))

    def test_draw_callback_is_bound_to_trainer_and_failure_is_recoverable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._build(root)

            class Trainer:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def train(self):
                    self.kwargs["draw_callback"](1, "draw-1")
                    raise RuntimeError("interrupted")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_registered_condition(run, trainer_factory=Trainer, evaluator=lambda _: {})
            self.assertEqual(run.store.state, "failed")
            self.assertEqual(run.journal.records, [{"epoch": 1, "draw_key": "draw-1", "state": "committed"}])
            self.assertFalse(tuple(root.glob("*.factor_repair.lock")))

    def test_internal_evaluator_typeerror_is_not_retried_without_verified_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._build(root)

            class Trainer:
                def train(self):
                    weights = run.run_dir / "weights"
                    weights.mkdir()
                    (weights / "last.pt").write_bytes(b"last")
                    (weights / "best.pt").write_bytes(b"best")

            def evaluator(_path):
                raise TypeError("internal evaluator failure")

            with self.assertRaisesRegex(TypeError, "internal evaluator failure"):
                run_registered_condition(run, trainer_factory=Trainer, evaluator=evaluator)
            self.assertFalse((run.run_dir / "metrics_ap40_primary_last.json").exists())

    def test_nonfinite_training_failure_records_failed_without_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._build(root)

            class Trainer:
                loss = float("nan")

                def train(self):
                    return {"loss": self.loss}

            with self.assertRaisesRegex(ValueError, "finite"):
                run_registered_condition(run, trainer_factory=Trainer, evaluator=lambda _: {})
            self.assertEqual(run.store.state, "failed")
            self.assertFalse((run.run_dir / "metrics_ap40_primary_last.json").exists())

    def test_empty_best_checkpoint_has_no_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            last = weights / "last.pt"
            best = weights / "best.pt"
            last.write_bytes(b"last")
            best.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                verify_checkpoint_artifacts(root, expected={"primary_sha256": file_sha256(last), "diagnostic_sha256": "0" * 64})
            self.assertFalse((root / "metrics_ap40_primary_last.json").exists())

    def test_last_missing_empty_and_hash_mismatch_have_no_metrics(self):
        for mode in ("missing", "empty", "mismatch"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                weights = root / "weights"
                weights.mkdir()
                best = weights / "best.pt"
                best.write_bytes(b"best")
                last = weights / "last.pt"
                if mode == "empty":
                    last.write_bytes(b"")
                elif mode == "mismatch":
                    last.write_bytes(b"last")
                with self.assertRaises((FileNotFoundError, ValueError)):
                    evaluate_primary_last(root, checkpoint_hash=("0" * 64 if mode != "mismatch" else file_sha256(best)), evaluator=lambda _: {})
                self.assertFalse((root / "metrics_ap40_primary_last.json").exists())


if __name__ == "__main__":
    unittest.main()
