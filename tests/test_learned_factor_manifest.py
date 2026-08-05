from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import inspect
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from ifdr_yolo.data.learned_factor_manifest import (
    PRIMARY_NODE_IDS,
    LearnedFactorManifest,
    LearnedObjectFactor,
    ValidatedMetadataPriorities,
    aggregate_primary_node_factors,
    average_tie_percentile_rank,
    build_learned_factor_manifest,
    build_learned_focus_distribution,
    build_manifest_from_records,
    deterministic_no_augmentation_loader,
    digest_ids,
    evaluate_primary_nodes,
    full_model_state_sha256,
    load_validated_checkpoint,
    manifest_digest,
)
from ifdr_yolo.data.metadata_index import FactorMetadataIndex, FactorObjectRecord
from ifdr_yolo.eval.factor_observer import LetterboxGeometry
from ifdr_yolo.data.replay_sampler import (
    ReplayDrawJournal,
    build_replay_distribution,
    normalize_replay_distribution,
)


class _FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.weight = 7
        self.eval_calls = 0
        self.train_calls: list[bool] = []

    def eval(self):
        self.eval_calls += 1
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.train_calls.append(bool(mode))
        self.training = bool(mode)
        return self

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state, strict: bool = True):
        value = state["weight"]
        self.weight = int(value.item()) if hasattr(value, "item") else int(value)
        return (), ()


class _NestedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
        self.head = nn.BatchNorm1d(2)
        self.register_buffer("ephemeral", torch.tensor([3.0]), persistent=False)

    def forward(self, batch):
        return batch


class _ContextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_images = None

    def forward(self, images):
        self.last_images = images
        return images * self.scale

    def consume_reliability_context(self):
        contexts = {}
        for node in (11, 14, *PRIMARY_NODE_IDS):
            contexts[node] = {
                "factors": torch.full((1, 2, 4, 4), 0.2 + (node - 17) / 100.0),
                "branch_weights": torch.full((1, 2, 4, 4), 0.5),
                "gate_strength": 0.75,
            }
        return contexts


class _BFloatModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        self.register_buffer(
            "ephemeral", torch.tensor([2.0], dtype=torch.bfloat16), persistent=False
        )


class _MappingStateModel:
    def __init__(self) -> None:
        self.weight = 0

    def load_state_dict(self, state, strict: bool = True):
        value = state["weight"]
        self.weight = int(value.item()) if hasattr(value, "item") else int(value)


class _PartiallyLoadingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("running", torch.tensor([2.0]))

    def load_state_dict(self, state, strict: bool = True):
        with torch.no_grad():
            self.weight.copy_(state["weight"])
        raise RuntimeError("missing running buffer")


def _metadata_record(
    image_id: str,
    object_id: str,
    *,
    class_name: str = "Cyclist",
    sampling: float = 0.2,
    visibility: float = 0.3,
    sampling_valid: bool = True,
    visibility_valid: bool = True,
) -> FactorObjectRecord:
    class_id = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}[class_name]
    return FactorObjectRecord(
        image_id=image_id,
        object_id=object_id,
        class_id=class_id,
        class_name=class_name,
        bbox_xyxy=(1.0, 2.0, 21.0, 42.0),
        height=40.0,
        depth_m=20.0,
        occlusion=0,
        truncation=0.0,
        sampling=sampling,
        visibility=visibility,
        joint=1.0 - (1.0 - sampling) * (1.0 - visibility),
        sampling_valid=sampling_valid,
        visibility_valid=visibility_valid,
    )


def _metadata_index() -> FactorMetadataIndex:
    return FactorMetadataIndex(
        by_image={
            "fit-a": (
                _metadata_record("fit-a", "fit-a:cyclist-0", sampling=0.2, visibility=0.3),
                _metadata_record("fit-a", "fit-a:cyclist-1", sampling=0.6, visibility=0.1),
                _metadata_record("fit-a", "fit-a:car-0", class_name="Car"),
            ),
            "fit-b": (
                _metadata_record("fit-b", "fit-b:cyclist-0", sampling=0.1, visibility=0.8),
            ),
        },
        source_sha256="b" * 64,
        split_sha256="a" * 64,
        label_source_sha256="c" * 64,
        sha256="d" * 64,
    )


def _metadata_index_with_empty_cyclist_image() -> FactorMetadataIndex:
    base = _metadata_index()
    return FactorMetadataIndex(
        by_image={**dict(base.by_image), "fit-c": ()},
        source_sha256=base.source_sha256,
        split_sha256=base.split_sha256,
        label_source_sha256=base.label_source_sha256,
        sha256=base.sha256,
    )


def _learned_records() -> tuple[LearnedObjectFactor, ...]:
    return (
        LearnedObjectFactor("fit-a", "fit-a:cyclist-0", 0.1, 0.1, 0.19, True),
        LearnedObjectFactor("fit-a", "fit-a:cyclist-1", 0.1, 0.2, 0.28, True),
        LearnedObjectFactor("fit-b", "fit-b:cyclist-0", 0.7, 0.1, 0.73, True),
        # Non-Cyclist predictions are never allowed into the focus distribution.
        LearnedObjectFactor("fit-a", "fit-a:car-0", 1.0, 1.0, 1.0, False),
    )


def _checkpoint(directory: str) -> tuple[str, str]:
    path = Path(directory) / "calibration-last.pt"
    torch.save({"state_dict": {"weight": torch.tensor(7)}}, path)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _torch_checkpoint(directory: str, model: nn.Module) -> tuple[str, str]:
    path = Path(directory) / "calibration-last-model.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_fixture(directory: str) -> LearnedFactorManifest:
    checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
    return build_manifest_from_records(
        condition="F0",
        checkpoint_path=checkpoint_path,
        checkpoint_role="calibration_last",
        checkpoint_sha256=checkpoint_sha256,
        fit_ids=("fit-a", "fit-b"),
        metadata_index=_metadata_index(),
        records=_learned_records(),
    )


class LearnedFactorManifestTest(unittest.TestCase):
    def test_deterministic_loader_is_streaming_and_checks_coverage_at_end(self) -> None:
        def batches():
            for image_id in ("fit-a", "fit-b"):
                yield {"image_ids": (image_id,)}

        stream = deterministic_no_augmentation_loader(batches(), ("fit-a", "fit-b"))
        self.assertTrue(inspect.isgenerator(stream))
        self.assertEqual(
            tuple(item["image_ids"][0] for item in stream), ("fit-a", "fit-b")
        )

    def test_streaming_loader_keeps_only_one_active_batch(self) -> None:
        class TrackingBatch(dict):
            active = 0
            peak = 0

            def __init__(self, image_id):
                super().__init__(image_ids=(image_id,))
                type(self).active += 1
                type(self).peak = max(type(self).peak, type(self).active)

            def __del__(self):
                type(self).active -= 1

        def batches():
            for image_id in ("fit-a", "fit-b", "fit-c"):
                yield TrackingBatch(image_id)

        for batch in deterministic_no_augmentation_loader(
            batches(), ("fit-a", "fit-b", "fit-c")
        ):
            self.assertLessEqual(TrackingBatch.active, 1)
            del batch
            gc.collect()
        gc.collect()
        self.assertLessEqual(TrackingBatch.peak, 1)

    def test_bfloat16_parameter_and_nonpersistent_buffer_are_hashable(self) -> None:
        model = _BFloatModule()
        before = full_model_state_sha256(model)
        model.ephemeral.add_(1.0)
        self.assertNotEqual(before, full_model_state_sha256(model))

    def test_learned_joint_is_canonical_and_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "learned_joint|joint"):
            LearnedObjectFactor("fit-a", "fit-a:0", 0.2, 0.3, 0.1, True)
        factor = LearnedObjectFactor("fit-a", "fit-a:0", 0.2, 0.3, 0.44, True)
        self.assertAlmostEqual(factor.learned_joint, 0.44)

    def test_digest_ids_rejects_mixed_or_non_string_identity_keys(self) -> None:
        with self.assertRaises(ValueError):
            digest_ids(("fit-a", 1))
        with self.assertRaises(ValueError):
            digest_ids((("fit-a", "obj"), "fit-a"))
        self.assertEqual(len(digest_ids((("fit-a", "obj"),))), 64)

    def test_stable_state_rejects_mapping_key_collisions(self) -> None:
        class NumericKeyModel:
            def state_dict(self):
                return {1: 1, "1": 2}

        with self.assertRaisesRegex(ValueError, "string|key|mapping"):
            full_model_state_sha256(NumericKeyModel())

    def test_ifdr_batch_contract_pools_primary_nodes_for_each_eligible_cyclist(self) -> None:
        model = _ContextModel()
        geometry = LetterboxGeometry(
            original_width=4,
            original_height=4,
            input_size=4,
            scale=1.0,
            resized_width=4,
            resized_height=4,
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
        )
        metadata = _metadata_index()
        batch = {
            "img": torch.zeros(1, 3, 4, 4),
            "image_ids": ("fit-a",),
            "geometries": {"fit-a": geometry},
            "objects": {"fit-a": metadata.by_image["fit-a"]},
        }
        factors = evaluate_primary_nodes(model, batch, PRIMARY_NODE_IDS)
        self.assertEqual(len(factors), 2)
        self.assertIs(model.last_images, batch["img"])
        self.assertTrue(all(item.eligible_cyclist for item in factors))
        self.assertTrue(all(0.0 <= item.learned_joint <= 1.0 for item in factors))

    def test_callable_prediction_fallback_is_rejected_without_ifdr_context_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "context|IFDR|batch"):
            evaluate_primary_nodes(lambda batch: batch, {"image_ids": ("fit-a",)})

    def test_primary_node_macro_average_and_joint(self) -> None:
        factor = aggregate_primary_node_factors(
            {
                17: (0.2, 0.4),
                20: (0.4, 0.6),
                23: (0.6, 0.8),
                26: (0.8, 1.0),
            }
        )
        self.assertEqual(PRIMARY_NODE_IDS, (17, 20, 23, 26))
        self.assertAlmostEqual(factor.sampling, 0.5)
        self.assertAlmostEqual(factor.visibility, 0.7)
        self.assertAlmostEqual(factor.learned_joint, 0.85)
        self.assertTrue(factor.eligible_cyclist)

    def test_primary_node_aggregation_requires_exact_registered_nodes(self) -> None:
        values = {node: (0.1, 0.2) for node in PRIMARY_NODE_IDS}
        for altered in (
            {**values, 17: (0.1, 0.2), 99: (0.1, 0.2)},
            {node: pair for node, pair in values.items() if node != 26},
        ):
            with self.subTest(nodes=sorted(altered)):
                with self.assertRaises(ValueError):
                    aggregate_primary_node_factors(altered)

    def test_average_tie_percentile_rank_keeps_ties_together(self) -> None:
        ranks = average_tie_percentile_rank({"a": 0.5, "b": 0.5, "c": 0.9})
        self.assertEqual(ranks["a"], ranks["b"])
        self.assertAlmostEqual(ranks["a"], 0.25)
        self.assertAlmostEqual(ranks["c"], 1.0)

    def test_manifest_filters_to_eligible_cyclists_and_exact_fit_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest_fixture(directory)
        self.assertEqual(manifest.fit_ids, ("fit-a", "fit-b"))
        self.assertEqual(
            tuple((item.image_id, item.object_id) for item in manifest.objects),
            (
                ("fit-a", "fit-a:cyclist-0"),
                ("fit-a", "fit-a:cyclist-1"),
                ("fit-b", "fit-b:cyclist-0"),
            ),
        )
        self.assertEqual(manifest.primary_node_ids, PRIMARY_NODE_IDS)

    def test_manifest_rejects_development_image_and_bad_checkpoint_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            metadata_index = _metadata_index()
            with self.assertRaisesRegex(ValueError, "fit|coverage"):
                build_manifest_from_records(
                    condition="F0",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="calibration_last",
                    checkpoint_sha256=checkpoint_sha256,
                    fit_ids=("fit-a",),
                    metadata_index=metadata_index,
                    records=_learned_records(),
                )

    def test_only_f0_condition_is_accepted_by_manifest_builders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            metadata_index = _metadata_index()
            with self.assertRaisesRegex(ValueError, "F0|condition"):
                build_manifest_from_records(
                    condition="clean",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="calibration_last",
                    checkpoint_sha256=checkpoint_sha256,
                    fit_ids=("fit-a", "fit-b"),
                    metadata_index=_metadata_index(),
                    records=_learned_records(),
                )
            with self.assertRaisesRegex(ValueError, "calibration_last"):
                build_manifest_from_records(
                    condition="F0",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="development",
                    checkpoint_sha256=checkpoint_sha256,
                    fit_ids=("fit-a", "fit-b"),
                    metadata_index=metadata_index,
                    records=_learned_records(),
                )

    def test_manifest_rejects_credential_bearing_checkpoint_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "credential|secret|path"):
            build_manifest_from_records(
                condition="F0",
                checkpoint_path="https://user:password@example.invalid/calibration.pt",
                checkpoint_role="calibration_last",
                checkpoint_sha256="e" * 64,
                fit_ids=("fit-a", "fit-b"),
                metadata_index=_metadata_index(),
                records=_learned_records(),
            )

    def test_manifest_requires_real_metadata_index_and_checkpoint_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            with self.assertRaisesRegex(ValueError, "metadata|binding"):
                build_manifest_from_records(
                    condition="F0",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="calibration_last",
                    checkpoint_sha256=checkpoint_sha256,
                    fit_ids=("fit-a", "fit-b"),
                    records=_learned_records(),
                )
            with self.assertRaises((FileNotFoundError, ValueError)):
                build_manifest_from_records(
                    condition="F0",
                    checkpoint_path=Path(directory) / "missing.pt",
                    checkpoint_role="calibration_last",
                    checkpoint_sha256=checkpoint_sha256,
                    fit_ids=("fit-a", "fit-b"),
                    metadata_index=_metadata_index(),
                    records=_learned_records(),
                )

    def test_failed_candidate_gate_writes_no_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            output_path = Path(directory) / "manifest.json"
            with self.assertRaisesRegex(ValueError, "candidate|gate"):
                build_manifest_from_records(
                    condition="F0",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="calibration_last",
                    checkpoint_sha256=checkpoint_sha256,
                    fit_ids=("fit-a", "fit-b"),
                    metadata_index=_metadata_index(),
                    records=_learned_records(),
                    candidate_gate=False,
                    output_path=output_path,
                )
            self.assertFalse(output_path.exists())

    def test_manifest_output_is_idempotent_and_refuses_different_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            output_path = Path(directory) / "manifest.json"
            kwargs = {
                "condition": "F0",
                "checkpoint_path": checkpoint_path,
                "checkpoint_role": "calibration_last",
                "checkpoint_sha256": checkpoint_sha256,
                "fit_ids": ("fit-a", "fit-b"),
                "metadata_index": _metadata_index(),
                "records": _learned_records(),
                "output_path": output_path,
            }
            build_manifest_from_records(**kwargs)
            first_bytes = output_path.read_bytes()
            build_manifest_from_records(**kwargs)
            self.assertEqual(first_bytes, output_path.read_bytes())
            changed = list(_learned_records())
            changed[0] = LearnedObjectFactor("fit-a", "fit-a:cyclist-0", 0.2, 0.1, 0.28, True)
            with self.assertRaisesRegex(ValueError, "existing|content|manifest"):
                build_manifest_from_records(**{**kwargs, "records": tuple(changed)})
            self.assertEqual(first_bytes, output_path.read_bytes())

    def test_manifest_link_fallback_handles_concurrent_canonical_creator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            kwargs = {
                "condition": "F0",
                "checkpoint_path": checkpoint_path,
                "checkpoint_role": "calibration_last",
                "checkpoint_sha256": checkpoint_sha256,
                "fit_ids": ("fit-a", "fit-b"),
                "metadata_index": _metadata_index(),
                "records": _learned_records(),
            }
            canonical_path = Path(directory) / "canonical.json"
            build_manifest_from_records(**kwargs, output_path=canonical_path)
            canonical = canonical_path.read_bytes()
            canonical_path.unlink()
            output_path = Path(directory) / "race.json"
            import ifdr_yolo.data.learned_factor_manifest as manifest_module

            real_open = manifest_module.os.open

            def race_open(path, flags, *args):
                if Path(path) == output_path:
                    output_path.write_bytes(canonical)
                    raise FileExistsError(output_path)
                return real_open(path, flags, *args)

            with patch.object(manifest_module.os, "link", side_effect=OSError("link unavailable")):
                with patch.object(manifest_module.os, "open", side_effect=race_open):
                    build_manifest_from_records(**kwargs, output_path=output_path)
            self.assertEqual(output_path.read_bytes(), canonical)

            different_path = Path(directory) / "race-different.json"

            def different_open(path, flags, *args):
                if Path(path) == different_path:
                    different_path.write_bytes(b"different")
                    raise FileExistsError(different_path)
                return real_open(path, flags, *args)

            with patch.object(manifest_module.os, "link", side_effect=OSError("link unavailable")):
                with patch.object(manifest_module.os, "open", side_effect=different_open):
                    with self.assertRaisesRegex(ValueError, "existing|content"):
                        build_manifest_from_records(**kwargs, output_path=different_path)

    def test_manifest_digest_changes_for_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest_fixture(directory)
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            variants = (
                replace(manifest, checkpoint_path=manifest.checkpoint_path + ".drift"),
                replace(manifest, fit_ids=("fit-b",)),
                replace(manifest, metadata_index_sha256="f" * 64),
                replace(manifest, checkpoint_sha256="e" * 64),
                replace(
                    manifest,
                    objects=(replace(manifest.objects[0], sampling=0.2, learned_joint=0.28),)
                    + manifest.objects[1:],
                ),
            )
            self.assertTrue(all(manifest_digest(item) != manifest_digest(manifest) for item in variants))
            # Keep variables used to make it explicit that checkpoint identity is a bound field.
            self.assertTrue(checkpoint_path.endswith("calibration-last.pt"))
            self.assertEqual(len(checkpoint_sha256), 64)

    def test_focus_distribution_uses_half_metadata_half_learned_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest_fixture(directory)
            priorities = ValidatedMetadataPriorities(
                metadata_index_sha256=manifest.metadata_index_sha256,
                values={"fit-a": 0.2, "fit-b": 0.8},
            )
            distribution = build_learned_focus_distribution(
                manifest=manifest,
                metadata_index=_metadata_index(),
                metadata_priorities=priorities,
                epoch=6,
            )
        self.assertEqual(distribution.mode, "factor_guided")
        self.assertEqual(distribution.image_ids, ("fit-a", "fit-b"))
        self.assertAlmostEqual(distribution.focus_scores["fit-a"], 0.1 + 0.5 * 0.0)
        self.assertAlmostEqual(distribution.focus_scores["fit-b"], 0.4 + 0.5 * 1.0)
        normalize_replay_distribution(distribution)

    def test_focus_distribution_uses_max_eligible_cyclist_joint_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest_fixture(directory)
            priorities = ValidatedMetadataPriorities(
                metadata_index_sha256=manifest.metadata_index_sha256,
                values={"fit-a": 0.5, "fit-b": 0.5},
            )
            distribution = build_learned_focus_distribution(
                manifest=manifest,
                metadata_index=_metadata_index(),
                metadata_priorities=priorities,
                epoch=6,
            )
        self.assertGreater(distribution.focus_scores["fit-a"], 0.0)
        self.assertGreater(distribution.focus_scores["fit-b"], 0.0)
        self.assertEqual(set(distribution.focus_scores), {"fit-a", "fit-b"})

    def test_focus_keeps_original_probability_for_fit_image_without_cyclist(self) -> None:
        metadata_index = _metadata_index_with_empty_cyclist_image()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            manifest = build_manifest_from_records(
                condition="F0",
                checkpoint_path=checkpoint_path,
                checkpoint_role="calibration_last",
                checkpoint_sha256=checkpoint_sha256,
                fit_ids=("fit-a", "fit-b", "fit-c"),
                metadata_index=metadata_index,
                records=_learned_records(),
            )
            distribution = build_learned_focus_distribution(
                manifest=manifest,
                metadata_index=metadata_index,
                metadata_priorities=ValidatedMetadataPriorities(
                    metadata_index_sha256=metadata_index.sha256,
                    values={"fit-a": 0.2, "fit-b": 0.8, "fit-c": 1.0},
                ),
                epoch=20,
            )
        self.assertEqual(distribution.image_ids, ("fit-a", "fit-b", "fit-c"))
        self.assertNotIn("fit-c", distribution.focus_scores)
        self.assertGreater(distribution.probabilities["fit-c"], 0.0)
        self.assertEqual(set(distribution.focus_probabilities), {"fit-a", "fit-b"})

    def test_focus_fails_closed_when_no_fit_image_has_an_eligible_cyclist(self) -> None:
        metadata_index = FactorMetadataIndex(
            by_image={"fit-c": ()},
            source_sha256="b" * 64,
            split_sha256="a" * 64,
            label_source_sha256="c" * 64,
            sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            manifest = build_manifest_from_records(
                condition="F0",
                checkpoint_path=checkpoint_path,
                checkpoint_role="calibration_last",
                checkpoint_sha256=checkpoint_sha256,
                fit_ids=("fit-c",),
                metadata_index=metadata_index,
                records=(),
            )
            with self.assertRaisesRegex(ValueError, "eligible|focus"):
                build_learned_focus_distribution(
                    manifest=manifest,
                    metadata_index=metadata_index,
                    metadata_priorities=ValidatedMetadataPriorities(
                        metadata_index_sha256=metadata_index.sha256,
                        values={"fit-c": 1.0},
                    ),
                    epoch=20,
                )

    def test_focus_distribution_rejects_unbound_priorities_or_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest_fixture(directory)
            metadata_index = _metadata_index()
            with self.assertRaisesRegex(ValueError, "metadata|binding|hash|digest"):
                build_learned_focus_distribution(
                    manifest=manifest,
                    metadata_index=metadata_index,
                    metadata_priorities=ValidatedMetadataPriorities(
                        metadata_index_sha256="f" * 64,
                        values={"fit-a": 0.2, "fit-b": 0.8},
                    ),
                    epoch=6,
                )
            with self.assertRaisesRegex(ValueError, "metadata|binding|hash|digest"):
                build_learned_focus_distribution(
                    manifest=replace(manifest, metadata_index_sha256="f" * 64),
                    metadata_index=metadata_index,
                    metadata_priorities=ValidatedMetadataPriorities(
                        metadata_index_sha256="f" * 64,
                        values={"fit-a": 0.2, "fit-b": 0.8},
                    ),
                    epoch=6,
                )

    def test_manifest_generation_restores_model_flags_and_state(self) -> None:
        model = _FakeModel()
        metadata_index = _metadata_index()
        loader = ({"image_ids": ("fit-a", "fit-b")},)
        observed = tuple(_learned_records()[:3])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            with patch(
                "ifdr_yolo.data.learned_factor_manifest.evaluate_primary_nodes",
                return_value=observed,
            ):
                manifest = build_learned_factor_manifest(
                    condition="F0",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="calibration_last",
                    checkpoint_sha256=checkpoint_sha256,
                    model=model,
                    loader=loader,
                    fit_ids=("fit-a", "fit-b"),
                    metadata_index=metadata_index,
                )
        self.assertEqual(manifest.fit_ids, ("fit-a", "fit-b"))
        self.assertTrue(model.training)
        self.assertEqual(model.weight, 7)
        self.assertGreaterEqual(model.eval_calls, 1)
        self.assertIn(True, model.train_calls)

    def test_checkpoint_is_loaded_and_model_without_loader_fails_closed(self) -> None:
        class NoLoadModel:
            training = True

            def eval(self):
                self.training = False
                return self

            def train(self, mode: bool = True):
                self.training = bool(mode)
                return self

            def state_dict(self):
                return {"weight": 0}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            with self.assertRaisesRegex(ValueError, "load|state_dict|checkpoint"):
                build_learned_factor_manifest(
                    condition="F0",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="calibration_last",
                    checkpoint_sha256=checkpoint_sha256,
                    model=NoLoadModel(),
                    loader=({"image_ids": ("fit-a", "fit-b")},),
                    fit_ids=("fit-a", "fit-b"),
                    metadata_index=_metadata_index(),
                )

            model = _FakeModel()
            model.weight = 0
            with patch(
                "ifdr_yolo.data.learned_factor_manifest.evaluate_primary_nodes",
                return_value=tuple(_learned_records()[:3]),
            ):
                build_learned_factor_manifest(
                    condition="F0",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="calibration_last",
                    checkpoint_sha256=checkpoint_sha256,
                    model=model,
                    loader=({"image_ids": ("fit-a", "fit-b")},),
                    fit_ids=("fit-a", "fit-b"),
                    metadata_index=_metadata_index(),
                )
            self.assertEqual(model.weight, 7)
            with self.assertRaisesRegex(ValueError, "SHA256|hash"):
                build_learned_factor_manifest(
                    condition="F0",
                    checkpoint_path=checkpoint_path,
                    checkpoint_role="calibration_last",
                    checkpoint_sha256="e" * 64,
                    model=_FakeModel(),
                    loader=({"image_ids": ("fit-a", "fit-b")},),
                    fit_ids=("fit-a", "fit-b"),
                    metadata_index=_metadata_index(),
                )

    def test_checkpoint_loader_accepts_state_dict_ema_module_and_model_module(self) -> None:
        source = nn.Linear(2, 2)
        target = nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.bias.fill_(4.0)
            target.weight.zero_()
            target.bias.zero_()
        with tempfile.TemporaryDirectory() as directory:
            cases = {
                "state_dict": source.state_dict(),
                "ema": source,
                "model": source,
            }
            for name, payload in cases.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.pt"
                    torch.save(payload if name == "state_dict" else {name: payload}, path)
                    with torch.no_grad():
                        target.weight.zero_()
                        target.bias.zero_()
                    load_validated_checkpoint(target, path, role="calibration_last")
                    self.assertTrue(torch.all(target.weight == 3.0))
                    self.assertTrue(torch.all(target.bias == 4.0))

    def test_checkpoint_strict_load_failure_rolls_back_parameters_and_buffers(self) -> None:
        model = _PartiallyLoadingModel()
        before_hash = full_model_state_sha256(model)
        before_parameters = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        before_buffers = {
            name: value.detach().clone() for name, value in model.named_buffers()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.pt"
            torch.save({"state_dict": {"weight": torch.tensor([9.0])}}, path)
            with self.assertRaisesRegex(ValueError, "state_dict|checkpoint|missing"):
                load_validated_checkpoint(model, path, role="calibration_last")
        self.assertEqual(full_model_state_sha256(model), before_hash)
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, before_parameters[name]))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, before_buffers[name]))

    def test_real_torch_nested_training_flags_and_nonpersistent_state_restore(self) -> None:
        model = _NestedModel()
        model.train(True)
        model.block[0].eval()
        model.head.train(False)
        before_flags = {
            name: module.training for name, module in model.named_modules()
        }
        before_state = full_model_state_sha256(model)
        observed = tuple(_learned_records()[:3])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _torch_checkpoint(directory, model)

            def mutate_state(_model, _batch, _nodes):
                model.ephemeral.add_(1.0)
                return observed

            with patch(
                "ifdr_yolo.data.learned_factor_manifest.evaluate_primary_nodes",
                side_effect=mutate_state,
            ):
                with self.assertRaisesRegex(ValueError, "model state"):
                    build_learned_factor_manifest(
                        condition="F0",
                        checkpoint_path=checkpoint_path,
                        checkpoint_role="calibration_last",
                        checkpoint_sha256=checkpoint_sha256,
                        model=model,
                        loader=({"image_ids": ("fit-a", "fit-b")},),
                        fit_ids=("fit-a", "fit-b"),
                        metadata_index=_metadata_index(),
                    )
        self.assertEqual(full_model_state_sha256(model), before_state)
        self.assertEqual(
            {name: module.training for name, module in model.named_modules()},
            before_flags,
        )

    def test_model_state_and_flags_restore_when_evaluator_raises(self) -> None:
        model = _NestedModel()
        model.block[0].eval()
        before_flags = {name: module.training for name, module in model.named_modules()}
        before_state = full_model_state_sha256(model)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _torch_checkpoint(directory, model)

            def raise_after_mutation(_model, _batch, _nodes):
                model.block[0].running_mean.add_(1.0)
                raise RuntimeError("evaluator failed")

            with patch(
                "ifdr_yolo.data.learned_factor_manifest.evaluate_primary_nodes",
                side_effect=raise_after_mutation,
            ):
                with self.assertRaisesRegex(RuntimeError, "evaluator failed"):
                    build_learned_factor_manifest(
                        condition="F0",
                        checkpoint_path=checkpoint_path,
                        checkpoint_role="calibration_last",
                        checkpoint_sha256=checkpoint_sha256,
                        model=model,
                        loader=({"image_ids": ("fit-a", "fit-b")},),
                        fit_ids=("fit-a", "fit-b"),
                        metadata_index=_metadata_index(),
                    )
        self.assertEqual(full_model_state_sha256(model), before_state)
        self.assertEqual(
            {name: module.training for name, module in model.named_modules()},
            before_flags,
        )

    def test_manifest_generation_rejects_incomplete_image_or_object_coverage(self) -> None:
        model = _FakeModel()
        metadata_index = _metadata_index()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            with patch(
                "ifdr_yolo.data.learned_factor_manifest.evaluate_primary_nodes",
                return_value=tuple(_learned_records()[:2]),
            ):
                with self.assertRaisesRegex(ValueError, "coverage"):
                    build_learned_factor_manifest(
                        condition="F0",
                        checkpoint_path=checkpoint_path,
                        checkpoint_role="calibration_last",
                        checkpoint_sha256=checkpoint_sha256,
                        model=model,
                        loader=({"image_ids": ("fit-a",)},),
                        fit_ids=("fit-a", "fit-b"),
                        metadata_index=metadata_index,
                    )

    def test_manifest_generation_rejects_each_object_coverage_failure_independently(self) -> None:
        metadata_index = _metadata_index()
        expected = tuple(_learned_records()[:3])
        variants = {
            "remove": expected[:-1],
            "development": expected[:-1]
            + (LearnedObjectFactor("fit-a", "dev-1", 0.1, 0.1, 0.19, True),),
            "duplicate": expected + (expected[0],),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path, checkpoint_sha256 = _checkpoint(directory)
            for name, observed in variants.items():
                with self.subTest(name=name):
                    model = _FakeModel()
                    with patch(
                        "ifdr_yolo.data.learned_factor_manifest.evaluate_primary_nodes",
                        return_value=observed,
                    ):
                        with self.assertRaisesRegex(ValueError, "object identity coverage"):
                            build_learned_factor_manifest(
                                condition="F0",
                                checkpoint_path=checkpoint_path,
                                checkpoint_role="calibration_last",
                                checkpoint_sha256=checkpoint_sha256,
                                model=model,
                                loader=({"image_ids": ("fit-a", "fit-b")},),
                                fit_ids=("fit-a", "fit-b"),
                                metadata_index=metadata_index,
                            )

    def test_checkpoint_and_object_hash_mismatch_blocks_resume_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest_fixture(directory)
            priorities = ValidatedMetadataPriorities(
                metadata_index_sha256=manifest.metadata_index_sha256,
                values={"fit-a": 0.2, "fit-b": 0.8},
            )
            distribution = build_learned_focus_distribution(
                manifest=manifest,
                metadata_index=_metadata_index(),
                metadata_priorities=priorities,
                epoch=6,
            )
            with self.assertRaises(ValueError):
                normalize_replay_distribution(
                    replace(distribution, calibration_checkpoint_sha256="e" * 64)
                )
            with self.assertRaises(ValueError):
                normalize_replay_distribution(
                    replace(distribution, manifest_sha256="e" * 64)
                )

    def test_replay_journal_resume_rejects_every_binding_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest_fixture(directory)
            priorities = ValidatedMetadataPriorities(
                metadata_index_sha256=manifest.metadata_index_sha256,
                values={"fit-a": 0.2, "fit-b": 0.8},
            )
            distribution = build_learned_focus_distribution(
                manifest=manifest,
                metadata_index=_metadata_index(),
                metadata_priorities=priorities,
                epoch=6,
            )
            journal_root = Path(directory) / "journal"
            journal = ReplayDrawJournal.create(
                journal_root,
                seed=17,
                distribution=distribution,
            )
            journal.draw_epoch(epoch=6)
            for field in (
                "source_sha256",
                "manifest_sha256",
                "calibration_checkpoint_sha256",
                "metadata_index_sha256",
            ):
                changed = {
                    "source_sha256": "1" * 64,
                    "manifest_sha256": "2" * 64,
                    "calibration_checkpoint_sha256": "3" * 64,
                    "metadata_index_sha256": "4" * 64,
                }
                altered = build_replay_distribution(
                    distribution.image_ids,
                    mode="factor_guided",
                    epoch=distribution.epoch,
                    source_sha256=changed["source_sha256"] if field == "source_sha256" else distribution.source_sha256,
                    manifest_sha256=changed["manifest_sha256"] if field == "manifest_sha256" else distribution.manifest_sha256,
                    calibration_checkpoint_sha256=changed["calibration_checkpoint_sha256"] if field == "calibration_checkpoint_sha256" else distribution.calibration_checkpoint_sha256,
                    metadata_index_sha256=changed["metadata_index_sha256"] if field == "metadata_index_sha256" else distribution.metadata_index_sha256,
                    focus_scores=distribution.focus_scores,
                )
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "identity|distribution|mismatch"):
                        ReplayDrawJournal.open(
                            journal_root,
                            seed=17,
                            distribution=altered,
                        )


if __name__ == "__main__":
    unittest.main()
