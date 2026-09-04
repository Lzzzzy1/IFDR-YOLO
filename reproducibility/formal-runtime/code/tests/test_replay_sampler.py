from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ifdr_yolo.data.replay_sampler import (
    ReplayDrawJournal,
    build_replay_distribution,
    deterministic_choice,
    mix_m3_probabilities,
    replay_eta,
    sha256_canonical,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _factor_distribution(*, epoch: int = 3, **overrides):
    values = {
        "source_sha256": HASH_A,
        "manifest_sha256": HASH_B,
        "calibration_checkpoint_sha256": HASH_C,
        "metadata_index_sha256": HASH_D,
    }
    values.update(overrides)
    return build_replay_distribution(
        image_ids=("fit_02", "fit_01", "fit_03"),
        cyclist_joint={"fit_01": 0.9, "fit_02": 0.2},
        mode="factor_guided",
        epoch=epoch,
        **values,
    )


class ReplaySamplerTest(unittest.TestCase):
    def test_registered_eta_schedule_boundaries(self) -> None:
        self.assertEqual(replay_eta(1), 0.0)
        self.assertEqual(replay_eta(5), 0.30)
        self.assertEqual(replay_eta(6), 0.30)
        self.assertEqual(replay_eta(40), 0.30)
        self.assertEqual(replay_eta(60), 0.0)

    def test_replay_eta_rejects_epoch_zero_and_61(self) -> None:
        for epoch in (0, 61):
            with self.subTest(epoch=epoch):
                with self.assertRaises(ValueError):
                    replay_eta(epoch)

    def test_m1_is_uniform_over_fit_ids(self) -> None:
        sampler = build_replay_distribution(
            image_ids=("b", "a", "c"), mode="M1", epoch=20
        )
        expected = 1.0 / 3.0
        self.assertEqual(sampler.image_ids, ("a", "b", "c"))
        for image_id in sampler.image_ids:
            self.assertAlmostEqual(sampler.probabilities[image_id], expected)
            self.assertAlmostEqual(sampler.original_probabilities[image_id], expected)

    def test_m2_is_uniform_over_cyclist_pool(self) -> None:
        sampler = build_replay_distribution(
            image_ids=("a", "b", "c", "d"),
            cyclist_joint={"a": 0.1, "c": 0.9},
            mode="M2",
            epoch=20,
        )
        self.assertEqual(sampler.focus_ids, ("a", "c"))
        self.assertAlmostEqual(sampler.focus_probabilities["a"], 0.5)
        self.assertAlmostEqual(sampler.focus_probabilities["c"], 0.5)
        self.assertNotIn("b", sampler.focus_probabilities)
        self.assertNotIn("d", sampler.focus_probabilities)
        self.assertAlmostEqual(sum(sampler.probabilities.values()), 1.0)

    def test_m3_clips_at_fit_p95_and_adds_floor(self) -> None:
        sampler = build_replay_distribution(
            image_ids=("a", "b", "c"),
            cyclist_joint={"a": 0.1, "b": 0.2, "c": 0.9, "heldout": 1.0},
            mode="M3",
            epoch=20,
        )
        # The 95th percentile of fit-only cyclist priorities is 0.83.
        self.assertAlmostEqual(sampler.focus_scores["a"], 0.1)
        self.assertAlmostEqual(sampler.focus_scores["b"], 0.2)
        self.assertAlmostEqual(sampler.focus_scores["c"], 0.83)
        self.assertNotIn("heldout", sampler.focus_probabilities)
        weighted = {"a": 0.15, "b": 0.25, "c": 0.88}
        total = sum(weighted.values())
        for image_id, weight in weighted.items():
            self.assertAlmostEqual(
                sampler.focus_probabilities[image_id], weight / total
            )

    def test_m3_priorities_are_clipped_and_only_cyclist_images_enter_focus(self):
        sampler = build_replay_distribution(
            image_ids=("a", "b", "c"),
            cyclist_joint={"a": 0.9, "b": 0.2},
            mode="M3", epoch=20,
        )
        self.assertEqual(sampler.focus_ids, ("a", "b"))
        self.assertNotIn("c", sampler.focus_probabilities)
        self.assertAlmostEqual(sum(sampler.probabilities.values()), 1.0)
        self.assertGreater(sampler.probabilities["c"], 0.0)

    def test_sampler_draws_with_replacement_for_fit_count(self) -> None:
        distribution = build_replay_distribution(
            image_ids=("a", "b", "c"), mode="M1", epoch=3
        )
        with TemporaryDirectory() as directory:
            journal = ReplayDrawJournal.create(
                Path(directory), seed=17, distribution=distribution
            )
            records = journal.draw_epoch(epoch=3)
        self.assertEqual(len(records), len(distribution.image_ids))
        self.assertTrue(all(record["image_id"] in distribution.image_ids for record in records))

    def test_draw_rejects_epoch_different_from_distribution(self) -> None:
        distribution = build_replay_distribution(
            image_ids=("a", "b", "c"), mode="M1", epoch=3
        )
        with TemporaryDirectory() as directory:
            journal = ReplayDrawJournal.create(
                Path(directory), seed=17, distribution=distribution
            )
            with self.assertRaisesRegex(ValueError, "epoch|distribution"):
                journal.draw(epoch=4, draw_index=0)

    def test_open_rejects_injected_record_with_foreign_epoch(self) -> None:
        distribution = build_replay_distribution(
            image_ids=("a", "b", "c"), mode="M1", epoch=3
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(root, seed=17, distribution=distribution)
            journal.draw(epoch=3, draw_index=0)
            payload = json.loads(journal.journal_path.read_text(encoding="utf-8"))
            foreign_key = (17, 4, 0, distribution.distribution_sha256, None, None, distribution.metadata_index_sha256)
            image_id, probability = deterministic_choice(
                distribution.probabilities, key=foreign_key
            )
            payload["draws"][0].update(
                epoch=4, image_id=image_id, probability=probability
            )
            journal.journal_path.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            journal.journal_path.with_name(journal.journal_path.name + ".bak").write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "epoch|distribution"):
                ReplayDrawJournal.open(root, seed=17, distribution=distribution)

    def test_draw_epoch_rejects_non_fit_budget(self) -> None:
        distribution = build_replay_distribution(
            image_ids=("a", "b", "c"), mode="M1", epoch=3
        )
        with TemporaryDirectory() as directory:
            journal = ReplayDrawJournal.create(
                Path(directory), seed=17, distribution=distribution
            )
            with self.assertRaisesRegex(ValueError, "fit_count|budget|draw"):
                journal.draw_epoch(epoch=3, fit_count=2)

    def test_factor_guided_distribution_has_full_provenance(self) -> None:
        distribution = _factor_distribution()
        self.assertEqual(distribution.mode, "factor_guided")
        self.assertEqual(distribution.manifest_sha256, HASH_B)
        self.assertEqual(distribution.calibration_checkpoint_sha256, HASH_C)
        self.assertEqual(distribution.metadata_index_sha256, HASH_D)
        self.assertEqual(distribution.image_ids, ("fit_01", "fit_02", "fit_03"))
        self.assertEqual(set(distribution.original_probabilities), set(distribution.image_ids))
        self.assertEqual(set(distribution.probabilities), set(distribution.image_ids))
        self.assertRegex(distribution.distribution_sha256, r"^[0-9a-f]{64}$")


class ReplayDrawJournalTest(unittest.TestCase):
    def test_draw_journal_resume_is_exact(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = ReplayDrawJournal.create(root, seed=17, distribution=distribution)
            expected = [first.draw(epoch=3, draw_index=i) for i in range(10)]
            resumed = ReplayDrawJournal.open(root, seed=17, distribution=distribution)
            self.assertEqual(
                [resumed.draw(epoch=3, draw_index=i) for i in range(10)], expected
            )
            self.assertTrue(all("image_id" in record for record in expected))
            self.assertTrue(all("probability" in record for record in expected))
            with self.assertRaisesRegex(ValueError, "scientific identity mismatch"):
                ReplayDrawJournal.open(root, seed=29, distribution=distribution)

    def test_draw_key_changes_sequence(self) -> None:
        probabilities = {"a": 0.2, "b": 0.3, "c": 0.5}
        baseline_key = (17, 3, 0, HASH_A, HASH_B, HASH_C, HASH_D)
        baseline = deterministic_choice(probabilities, key=baseline_key)[0]
        changed_keys = {
            "seed": (19, 3, 0, HASH_A, HASH_B, HASH_C, HASH_D),
            "epoch": (17, 4, 0, HASH_A, HASH_B, HASH_C, HASH_D),
            "draw_index": (17, 3, 1, HASH_A, HASH_B, HASH_C, HASH_D),
            "distribution_sha256": (17, 3, 0, "e" * 64, HASH_B, HASH_C, HASH_D),
        }
        for changed_key, key in changed_keys.items():
            with self.subTest(changed_key=changed_key):
                self.assertNotEqual(deterministic_choice(probabilities, key=key)[0], baseline)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = ReplayDrawJournal.create(
                root / "base", seed=17, distribution=_factor_distribution(epoch=3)
            )
            changed_epoch = ReplayDrawJournal.create(
                root / "epoch", seed=17, distribution=_factor_distribution(epoch=4)
            )
            changed_seed = ReplayDrawJournal.create(
                root / "seed", seed=19, distribution=_factor_distribution(epoch=3)
            )
            base_draws = [baseline.draw(epoch=3, draw_index=i) for i in range(20)]
            epoch_draws = [changed_epoch.draw(epoch=4, draw_index=i) for i in range(20)]
            seed_draws = [changed_seed.draw(epoch=3, draw_index=i) for i in range(20)]
        self.assertNotEqual(
            [record["image_id"] for record in base_draws],
            [record["image_id"] for record in epoch_draws],
        )
        self.assertNotEqual(
            [record["image_id"] for record in base_draws],
            [record["image_id"] for record in seed_draws],
        )

    def test_draw_journal_records_realized_counts(self) -> None:
        distribution = _factor_distribution()
        class_counts = {
            "fit_01": {"Cyclist": 2},
            "fit_02": {"Cyclist": 1},
            "fit_03": {"Car": 3},
        }
        with TemporaryDirectory() as directory:
            journal = ReplayDrawJournal.create(
                Path(directory),
                seed=17,
                distribution=distribution,
                class_counts=class_counts,
            )
            records = journal.draw_epoch(epoch=3)
        self.assertEqual(len(records), len(distribution.image_ids))
        for record in records:
            self.assertEqual(record["realized_image_count"], 1)
            self.assertEqual(
                record["realized_class_counts"], class_counts[record["image_id"]]
            )

    def test_duplicate_draw_content_fails_closed(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            journal = ReplayDrawJournal.create(
                Path(directory), seed=17, distribution=distribution
            )
            original = journal.draw(epoch=3, draw_index=0)
            payload = json.loads(journal.journal_path.read_text(encoding="utf-8"))
            duplicate = dict(original)
            duplicate["image_id"] = "conflicting"
            payload["draws"].append(duplicate)
            journal.journal_path.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            journal.journal_path.with_name(journal.journal_path.name + ".bak").write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                ReplayDrawJournal.open(directory, seed=17, distribution=distribution)

    def test_tampered_realized_counts_fail_closed(self) -> None:
        distribution = _factor_distribution()
        class_counts = {
            "fit_01": {"Cyclist": 2},
            "fit_02": {"Cyclist": 1},
            "fit_03": {"Car": 3},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(
                root, seed=17, distribution=distribution, class_counts=class_counts
            )
            journal.draw(epoch=3, draw_index=0)
            payload = json.loads(journal.journal_path.read_text(encoding="utf-8"))
            selected = payload["draws"][0]["image_id"]
            payload["draws"][0]["realized_class_counts"] = {
                "tampered": class_counts[selected].get("Cyclist", 0) + 1
            }
            journal.journal_path.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            journal.journal_path.with_name(journal.journal_path.name + ".bak").write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "class counts|count conflict"):
                ReplayDrawJournal.open(
                    root, seed=17, distribution=distribution, class_counts=class_counts
                )

    def test_atomic_recovery_uses_last_committed_state(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(root, seed=17, distribution=distribution)
            first = journal.draw(epoch=3, draw_index=0)
            journal.draw(epoch=3, draw_index=1)
            backup = journal.journal_path.with_name(journal.journal_path.name + ".bak")
            self.assertTrue(backup.is_file())
            journal.journal_path.write_bytes(b"{")
            resumed = ReplayDrawJournal.open(root, seed=17, distribution=distribution)
            self.assertEqual(resumed.draw(epoch=3, draw_index=0), first)
            self.assertEqual(len(resumed.records), 2)

    def test_journal_configuration_is_read_only(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            journal = ReplayDrawJournal.create(
                Path(directory), seed=17, distribution=distribution
            )
            for field, value in (
                ("root", Path(directory) / "other"),
                ("seed", 29),
                ("distribution", distribution),
                ("journal_path", Path(directory) / "other.json"),
                ("state_path", Path(directory) / "other.json"),
            ):
                with self.subTest(field=field):
                    with self.assertRaises(AttributeError):
                        setattr(journal, field, value)

    def test_mapping_keys_are_exact_non_empty_strings(self) -> None:
        with self.assertRaises(ValueError):
            mix_m3_probabilities(
                original={1: 0.5, "1": 0.5}, focus={"1": 1.0}, epoch=3
            )
        with self.assertRaises(ValueError):
            mix_m3_probabilities(
                original={" ": 1.0}, focus={" ": 1.0}, epoch=3
            )
        with self.assertRaises(ValueError):
            sha256_canonical({1: "collision"})
        with self.assertRaises(ValueError):
            sha256_canonical({"": "empty"})

    def test_no_matching_replica_identity_never_restores_or_overwrites(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(root, seed=17, distribution=distribution)
            journal.draw(epoch=3, draw_index=0)
            journal.draw(epoch=3, draw_index=1)
            current = json.loads(journal.journal_path.read_text(encoding="utf-8"))
            backup = json.loads(
                journal.journal_path.with_name(journal.journal_path.name + ".bak")
                .read_text(encoding="utf-8")
            )
            current["draws"] = current["draws"][:1]
            current["scientific_identity"]["seed"] = 101
            backup["scientific_identity"]["seed"] = 102
            journal.journal_path.write_text(
                json.dumps(current, sort_keys=True), encoding="utf-8"
            )
            journal.journal_path.with_name(journal.journal_path.name + ".bak").write_text(
                json.dumps(backup, sort_keys=True), encoding="utf-8"
            )
            current_before = journal.journal_path.read_bytes()
            backup_before = journal.journal_path.with_name(
                journal.journal_path.name + ".bak"
            ).read_bytes()
            with self.assertRaisesRegex(ValueError, "scientific identity mismatch"):
                ReplayDrawJournal.open(root, seed=17, distribution=distribution)
            self.assertEqual(journal.journal_path.read_bytes(), current_before)
            self.assertEqual(
                journal.journal_path.with_name(journal.journal_path.name + ".bak").read_bytes(),
                backup_before,
            )

    def test_same_identity_non_superset_replicas_fail_closed_without_restore(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(root, seed=17, distribution=distribution)
            journal.draw(epoch=3, draw_index=0)
            journal.draw(epoch=3, draw_index=1)
            journal.draw(epoch=3, draw_index=2)
            current = json.loads(journal.journal_path.read_text(encoding="utf-8"))
            backup = json.loads(
                journal.journal_path.with_name(journal.journal_path.name + ".bak")
                .read_text(encoding="utf-8")
            )
            current["draws"] = [current["draws"][0]]
            backup["draws"] = [backup["draws"][1], backup["draws"][2]]
            journal.journal_path.write_text(
                json.dumps(current, sort_keys=True), encoding="utf-8"
            )
            journal.journal_path.with_name(journal.journal_path.name + ".bak").write_text(
                json.dumps(backup, sort_keys=True), encoding="utf-8"
            )
            current_before = journal.journal_path.read_bytes()
            backup_before = journal.journal_path.with_name(
                journal.journal_path.name + ".bak"
            ).read_bytes()
            with self.assertRaisesRegex(ValueError, "replicas disagree"):
                ReplayDrawJournal.open(root, seed=17, distribution=distribution)
            self.assertEqual(journal.journal_path.read_bytes(), current_before)
            self.assertEqual(
                journal.journal_path.with_name(journal.journal_path.name + ".bak").read_bytes(),
                backup_before,
            )

    def test_matching_backup_semantic_tamper_fails_before_restore(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(root, seed=17, distribution=distribution)
            for draw_index in range(4):
                journal.draw(epoch=3, draw_index=draw_index)
            current = json.loads(journal.journal_path.read_text(encoding="utf-8"))
            backup = json.loads(
                journal.journal_path.with_name(journal.journal_path.name + ".bak")
                .read_text(encoding="utf-8")
            )
            current["draws"] = current["draws"][:3]
            backup["draws"][0]["image_id"] = "semantic-tamper"
            journal.journal_path.write_text(
                json.dumps(current, sort_keys=True), encoding="utf-8"
            )
            journal.journal_path.with_name(journal.journal_path.name + ".bak").write_text(
                json.dumps(backup, sort_keys=True), encoding="utf-8"
            )
            current_before = journal.journal_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "draw|conflict"):
                ReplayDrawJournal.open(root, seed=17, distribution=distribution)
            self.assertEqual(journal.journal_path.read_bytes(), current_before)

    def test_distribution_hash_changes_with_epoch_and_manifest_identity(self) -> None:
        baseline = _factor_distribution(epoch=3)
        variants = (
            _factor_distribution(epoch=4),
            _factor_distribution(manifest_sha256="e" * 64),
            _factor_distribution(calibration_checkpoint_sha256="e" * 64),
            _factor_distribution(metadata_index_sha256="e" * 64),
        )
        for variant in variants:
            with self.subTest(digest=variant.distribution_sha256):
                self.assertNotEqual(variant.distribution_sha256, baseline.distribution_sha256)

    def test_draw_journal_rejects_missing_provenance_hashes(self) -> None:
        for field in (
            "source_sha256",
            "manifest_sha256",
            "calibration_checkpoint_sha256",
            "metadata_index_sha256",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "provenance|hash"):
                    _factor_distribution(**{field: None})

    def test_legacy_modes_reject_factor_provenance_hashes(self) -> None:
        for mode in ("M1", "M2", "M3"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "legacy|manifest|checkpoint"):
                    build_replay_distribution(
                        image_ids=("a", "b"),
                        cyclist_joint={"a": 0.5},
                        mode=mode,
                        epoch=3,
                        manifest_sha256=HASH_B,
                    )

    def test_class_count_identity_drift_rejects_resume(self) -> None:
        distribution = _factor_distribution()
        class_counts = {
            "fit_01": {"Cyclist": 1},
            "fit_02": {"Cyclist": 2},
            "fit_03": {"Car": 1},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(
                root,
                seed=17,
                distribution=distribution,
                class_counts=class_counts,
            )
            journal.draw(epoch=3, draw_index=0)
            with self.assertRaisesRegex(ValueError, "scientific identity mismatch"):
                ReplayDrawJournal.open(root, seed=17, distribution=distribution)
            with self.assertRaisesRegex(ValueError, "scientific identity mismatch"):
                ReplayDrawJournal.open(
                    root,
                    seed=17,
                    distribution=distribution,
                    class_counts={"fit_01": {"Cyclist": 2}},
                )

    def test_journal_returns_deep_snapshots(self) -> None:
        distribution = _factor_distribution()
        class_counts = {"fit_01": {"Cyclist": 1}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(
                root, seed=17, distribution=distribution, class_counts=class_counts
            )
            record = journal.draw(epoch=3, draw_index=0)
            expected_image_id = record["image_id"]
            expected_counts = dict(record["realized_class_counts"])
            record["realized_class_counts"]["Cyclist"] = 99
            record["image_id"] = "tampered"
            snapshot = journal.records[0]
            snapshot["realized_class_counts"]["Cyclist"] = 88
            snapshot["image_id"] = "tampered-again"
            self.assertEqual(journal.records[0]["image_id"], expected_image_id)
            self.assertEqual(journal.records[0]["realized_class_counts"], expected_counts)
            reopened = ReplayDrawJournal.open(
                root, seed=17, distribution=distribution, class_counts=class_counts
            )
            self.assertEqual(reopened.records[0]["image_id"], expected_image_id)
            self.assertEqual(reopened.records[0]["realized_class_counts"], expected_counts)

    def test_scientific_identity_is_a_deep_snapshot(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = ReplayDrawJournal.create(root, seed=17, distribution=distribution)
            identity = journal.scientific_identity
            identity["seed"] = 99
            identity["distribution"]["mode"] = "M1"
            self.assertEqual(journal.scientific_identity["seed"], 17)
            self.assertEqual(journal.scientific_identity["distribution"]["mode"], "factor_guided")
            reopened = ReplayDrawJournal.open(root, seed=17, distribution=distribution)
            self.assertEqual(reopened.scientific_identity["seed"], 17)

    def test_class_counts_reject_unknown_image_ids(self) -> None:
        distribution = _factor_distribution()
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "class_counts|image ID"):
                ReplayDrawJournal.create(
                    Path(directory),
                    seed=17,
                    distribution=distribution,
                    class_counts={"unknown": {"Car": 1}},
                )


if __name__ == "__main__":
    unittest.main()
