from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from collections import Counter
import json
import unittest
from unittest.mock import patch

from ifdr_yolo.eval.natural_factor_audit import (
    NaturalFactorObservation,
    audit_natural_factors,
    average_tie_rank,
    controlled_monotonicity,
    image_cluster_bootstrap,
    intervention_statistics,
    partial_spearman,
    spearman,
)


SEEDS = (17, 29, 41)
NODES = (11, 14, 17, 20, 23, 26)


def _observation(
    *,
    seed: int = 17,
    node_id: int = 11,
    image_id: str = "image-000",
    object_id: int = 0,
    box_height: float = 32.0,
    region_role: str = "target",
    intervention_kind: str = "natural",
    intervention_severity: float = 0.0,
    pair_id: str | None = None,
    natural_sampling: float = 0.3,
    natural_visibility: float = 0.7,
    predicted_sampling: float = 0.3,
    predicted_visibility: float = 0.7,
    class_id: int | None = None,
    class_name: str | None = None,
    branch_weights: tuple[float, float] = (0.6, 0.4),
    intervention_factor: str | None = None,
) -> NaturalFactorObservation:
    if intervention_factor is None:
        if intervention_kind in {"sampling", "visibility"}:
            intervention_factor = intervention_kind
        elif intervention_kind == "clean":
            intervention_factor = "sampling"
    return NaturalFactorObservation(
        seed=seed,
        node_id=node_id,
        image_id=image_id,
        object_id=object_id,
        class_id=object_id % 3 if class_id is None else class_id,
        box_height=box_height,
        region_role=region_role,
        intervention_kind=intervention_kind,
        intervention_severity=intervention_severity,
        pair_id=pair_id,
        natural_sampling=natural_sampling,
        natural_visibility=natural_visibility,
        predicted_sampling=predicted_sampling,
        predicted_visibility=predicted_visibility,
        branch_weights=branch_weights,
        class_name=class_name,
        intervention_factor=intervention_factor,
    )


def _fixture(
    *,
    sampling_sign: float = 1.0,
    visibility_sign: float = 1.0,
    target_boost: float = 0.25,
    image_count: int = 8,
) -> tuple[NaturalFactorObservation, ...]:
    rows: list[NaturalFactorObservation] = []
    sampling_values = (0.08, 0.19, 0.31, 0.38, 0.52, 0.47, 0.69, 0.79)
    visibility_values = (0.18, 0.58, 0.34, 0.42, 0.26, 0.66, 0.50, 0.74)
    for seed in SEEDS:
        for node in NODES:
            for index in range(image_count):
                image_id = f"s{seed}-n{node}-i{index}"
                ns = sampling_values[index % len(sampling_values)]
                nv = visibility_values[index % len(visibility_values)]
                rows.append(
                    _observation(
                        seed=seed,
                        node_id=node,
                        image_id=image_id,
                        object_id=0,
                        box_height=16.0 + index * 6.0,
                        natural_sampling=ns,
                        natural_visibility=nv,
                        predicted_sampling=(
                            0.1 + ns * 0.75
                            if sampling_sign > 0
                            else 0.9 - ns * 0.75
                        ),
                        predicted_visibility=(
                            0.1 + nv * 0.75
                            if visibility_sign > 0
                            else 0.9 - nv * 0.75
                        ),
                    )
                )
                for kind, natural, channel in (
                    ("sampling", ns, "sampling"),
                    ("visibility", nv, "visibility"),
                ):
                    pair = f"pair-{kind}-{seed}-{node}-{index}"
                    for severity in (0.25, 0.50, 0.75, 1.0):
                        for role in ("target", "background"):
                            predicted = 0.25 + severity * (
                                target_boost if role == "target" else 0.05
                            )
                            rows.append(
                                _observation(
                                    seed=seed,
                                    node_id=node,
                                    image_id=image_id,
                                    object_id=0,
                                    region_role=role,
                                    intervention_kind=kind,
                                    intervention_factor=kind,
                                    intervention_severity=severity,
                                    pair_id=pair,
                                    natural_sampling=ns,
                                    natural_visibility=nv,
                                    predicted_sampling=(
                                        predicted if channel == "sampling" else 0.25
                                    ),
                                    predicted_visibility=(
                                        predicted if channel == "visibility" else 0.25
                                    ),
                                )
                            )
                    for role in ("target", "background"):
                        rows.append(
                            _observation(
                                seed=seed,
                                node_id=node,
                                image_id=image_id,
                                object_id=0,
                                region_role=role,
                                intervention_kind="clean",
                                intervention_factor=kind,
                                pair_id=pair,
                                natural_sampling=ns,
                                natural_visibility=nv,
                                predicted_sampling=0.25,
                                predicted_visibility=0.25,
                            )
                        )
    return tuple(rows)


class NaturalFactorObservationTest(unittest.TestCase):
    def test_observation_is_frozen_and_validates_intervention_pair(self) -> None:
        row = _observation()
        with self.assertRaises(FrozenInstanceError):
            row.predicted_sampling = 0.1  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "natural observations must be target"):
            _observation(region_role="background")
        with self.assertRaisesRegex(ValueError, "pair_id"):
            _observation(intervention_kind="sampling", intervention_severity=0.5)
        with self.assertRaisesRegex(ValueError, "intervention_factor"):
            _observation(
                intervention_kind="sampling",
                intervention_severity=0.5,
                pair_id="bad-factor",
                intervention_factor="visibility",
            )
        with self.assertRaisesRegex(ValueError, "intervention_factor"):
            _observation(intervention_factor="sampling")
        for kwargs, message in (
            ({"seed": -1}, "seed"),
            ({"node_id": -1}, "node_id"),
            ({"predicted_sampling": float("nan")}, "finite"),
            ({"region_role": "roi"}, "region_role"),
            ({"intervention_kind": "synthetic"}, "intervention_kind"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    _observation(**kwargs)
        with self.assertRaisesRegex(ValueError, "class_id"):
            _observation(class_id=3)
        with self.assertRaisesRegex(ValueError, "class_name"):
            _observation(class_id=0, class_name="Pedestrian")
        _observation(class_id=0, class_name="Car")
        with self.assertRaisesRegex(ValueError, "sum"):
            _observation(branch_weights=(0.2, 0.2))


class CorrelationTest(unittest.TestCase):
    def test_average_tie_spearman(self) -> None:
        self.assertEqual(average_tie_rank((3.0, 1.0, 1.0, 2.0)), (4.0, 1.5, 1.5, 3.0))
        result = spearman((1.0, 2.0, 2.0, 4.0), (2.0, 4.0, 3.0, 8.0))
        self.assertTrue(result["success"])
        self.assertGreater(result["rho"], 0.0)

    def test_constant_and_insufficient_correlations_are_not_fake_zero(self) -> None:
        self.assertFalse(spearman((1.0, 1.0), (2.0, 3.0))["success"])
        self.assertEqual(spearman((1.0, 1.0), (2.0, 3.0))["status"], "constant")
        self.assertEqual(spearman((1.0,), (2.0,))["status"], "insufficient")

    def test_partial_spearman_controls_ranked_height_and_class(self) -> None:
        result = partial_spearman(
            target=(1.0, 3.0, 2.0, 6.0, 4.0, 5.0),
            prediction=(2.0, 6.0, 4.1, 12.0, 8.2, 10.1),
            box_height=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
            class_ids=(0, 0, 1, 1, 2, 2),
        )
        self.assertTrue(result["success"])
        self.assertGreater(result["rho"], 0.9)

    def test_partial_spearman_constant_residual_is_nonpass_at_small_and_large_n(self) -> None:
        for n in (3, 12):
            with self.subTest(n=n):
                result = partial_spearman(
                    target=tuple(float(index) for index in range(n)),
                    prediction=tuple(float(2 * index) for index in range(n)),
                    box_height=tuple(float(index) for index in range(n)),
                    class_ids=(0,) * n,
                )
                self.assertFalse(result["success"])
                self.assertEqual(result["status"], "constant")
                self.assertIsNone(result["rho"])

    def test_partial_spearman_real_residual_signal_still_passes(self) -> None:
        target = (0.0, 10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0, 11.0)
        prediction = tuple(2.0 * value + (0.01 if index % 2 else -0.01) for index, value in enumerate(target))
        result = partial_spearman(
            target=target,
            prediction=prediction,
            box_height=tuple(float(index) for index in range(len(target))),
            class_ids=(0,) * len(target),
        )
        self.assertTrue(result["success"])
        self.assertGreater(result["rho"], 0.9)

    def test_natural_alignment_result_is_directly_json_serializable(self) -> None:
        rows = _fixture(image_count=2)
        from ifdr_yolo.eval.natural_factor_audit import natural_factor_alignment

        payload = natural_factor_alignment(
            rows, factor="sampling", bootstrap_replicates=5
        )
        json.dumps(payload)
        self.assertIn("17:11", payload["seed_node"])

    def test_intervention_result_is_directly_json_serializable(self) -> None:
        from ifdr_yolo.eval.natural_factor_audit import intervention_statistics

        payload = intervention_statistics(_fixture(image_count=2), factor="sampling")
        json.dumps(payload)
        self.assertIn("17:11", payload["eligible_by_seed_node"])

    def test_duplicate_natural_identity_is_rejected(self) -> None:
        row = _observation()
        with self.assertRaisesRegex(ValueError, "duplicate natural observation"):
            from ifdr_yolo.eval.natural_factor_audit import natural_factor_alignment

            natural_factor_alignment((row, row), factor="sampling", bootstrap_replicates=5)


class FactorAuditTest(unittest.TestCase):
    def test_positive_fixture_passes_both_factors(self) -> None:
        decision = audit_natural_factors(_fixture(), bootstrap_replicates=120)
        self.assertTrue(decision.passed, decision.reasons)
        self.assertTrue(decision.factor_results["sampling"]["passed"])
        self.assertTrue(decision.factor_results["visibility"]["passed"])

    def test_reversed_factor_fails_direction_gate(self) -> None:
        decision = audit_natural_factors(
            _fixture(sampling_sign=-1.0), bootstrap_replicates=80
        )
        self.assertFalse(decision.factor_results["sampling"]["passed"])
        self.assertTrue(any("direction" in reason for reason in decision.reasons))

    def test_missing_required_seed_or_node_cannot_be_hidden_by_pooling(self) -> None:
        rows = tuple(row for row in _fixture() if row.seed != 41)
        decision = audit_natural_factors(rows, bootstrap_replicates=60)
        self.assertFalse(decision.passed)
        self.assertTrue(any("missing" in reason for reason in decision.reasons))

    def test_controlled_monotonicity_and_interventions(self) -> None:
        rows = _fixture()
        monotonic = controlled_monotonicity(rows, factor="sampling", control_factor="visibility")
        self.assertGreaterEqual(monotonic["rate"], 0.8)
        intervention = intervention_statistics(rows, factor="sampling")
        self.assertGreater(intervention["target_mean_response"], intervention["background_mean_response"])
        self.assertGreater(intervention["paired_mean"], 0.0)
        self.assertGreaterEqual(intervention["ordered_pair_rate"], 0.8)

    def test_constant_control_factor_is_insufficient(self) -> None:
        rows = tuple(
            _observation(
                image_id=f"constant-control-{index}",
                natural_sampling=0.1 + 0.1 * index,
                natural_visibility=0.5,
                predicted_sampling=0.2 + 0.1 * index,
                predicted_visibility=0.5,
            )
            for index in range(8)
        )
        result = controlled_monotonicity(rows, factor="sampling", control_factor="visibility")
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "insufficient")
        self.assertIsNone(result["rate"])

    def test_one_eligible_control_bin_is_insufficient(self) -> None:
        # The first control quartile contains both a global lower and upper
        # target tertile; all other bins contain only one tertile.
        target_values = (0.0, 0.95, 0.85, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80)
        rows = tuple(
            _observation(
                image_id=f"one-bin-{index}",
                natural_sampling=target,
                natural_visibility=index / 11.0,
                predicted_sampling=target,
                predicted_visibility=index / 11.0,
            )
            for index, target in enumerate(target_values)
        )
        result = controlled_monotonicity(rows, factor="sampling", control_factor="visibility")
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["eligible"], 1)

    def test_local_tertile_mutant_cannot_replace_global_target_tertiles(self) -> None:
        rows = tuple(
            _observation(
                image_id=f"global-tertile-{index}",
                natural_sampling=index / 23.0,
                natural_visibility=index / 23.0,
                predicted_sampling=index / 23.0,
                predicted_visibility=index / 23.0,
            )
            for index in range(24)
        )
        result = controlled_monotonicity(rows, factor="sampling", control_factor="visibility")
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["eligible"], 0)

    def test_intervention_severity_input_order_does_not_change_result(self) -> None:
        rows = _fixture(image_count=2)
        first = intervention_statistics(rows, factor="visibility")
        second = intervention_statistics(tuple(reversed(rows)), factor="visibility")
        self.assertEqual(first, second)

    def test_missing_middle_severity_is_malformed_even_with_two_remaining(self) -> None:
        rows = list(_fixture(image_count=2))
        removed = False
        filtered: list[NaturalFactorObservation] = []
        for row in rows:
            if (
                not removed
                and row.intervention_kind == "sampling"
                and row.region_role == "target"
                and row.intervention_severity == 0.50
            ):
                removed = True
                continue
            filtered.append(row)
        stats = intervention_statistics(tuple(filtered), factor="sampling")
        self.assertGreater(stats["malformed"], 0)
        self.assertFalse(
            audit_natural_factors(tuple(filtered), bootstrap_replicates=30)
            .factor_results["sampling"]["passed"]
        )

    def test_missing_both_rows_for_registered_severity_is_malformed(self) -> None:
        rows = list(_fixture(image_count=2))
        removed = 0
        filtered: list[NaturalFactorObservation] = []
        for row in rows:
            if (
                row.intervention_kind == "sampling"
                and row.intervention_severity == 0.50
                and removed < 2
            ):
                removed += 1
                continue
            filtered.append(row)
        stats = intervention_statistics(tuple(filtered), factor="sampling")
        self.assertGreater(stats["malformed"], 0)
        self.assertFalse(
            audit_natural_factors(tuple(filtered), bootstrap_replicates=30)
            .factor_results["sampling"]["passed"]
        )

    def test_orphan_factor_pair_without_clean_manifest_is_malformed(self) -> None:
        rows = list(_fixture(image_count=2))
        source = next(
            row
            for row in rows
            if row.intervention_kind == "sampling" and row.intervention_severity == 0.50
        )
        orphan_pair = "orphan-pair"
        rows.append(replace(source, pair_id=orphan_pair))
        stats = intervention_statistics(tuple(rows), factor="sampling")
        self.assertGreater(stats["malformed"], 0)
        self.assertFalse(
            audit_natural_factors(tuple(rows), bootstrap_replicates=30)
            .factor_results["sampling"]["passed"]
        )

    def test_extra_intervention_severity_is_malformed(self) -> None:
        rows = list(_fixture(image_count=2))
        source = next(
            row
            for row in rows
            if row.intervention_kind == "sampling" and row.region_role == "target"
        )
        rows.append(replace(source, intervention_severity=0.90))
        stats = intervention_statistics(tuple(rows), factor="sampling")
        self.assertGreater(stats["malformed"], 0)
        self.assertFalse(
            audit_natural_factors(tuple(rows), bootstrap_replicates=30)
            .factor_results["sampling"]["passed"]
        )

    def test_duplicate_intervention_target_is_malformed(self) -> None:
        rows = list(_fixture(image_count=2))
        duplicate = next(
            row
            for row in rows
            if row.intervention_kind == "sampling"
            and row.region_role == "target"
            and row.intervention_severity == 0.50
        )
        rows.append(duplicate)
        stats = intervention_statistics(tuple(rows), factor="sampling")
        self.assertGreater(stats["malformed"], 0)
        decision = audit_natural_factors(tuple(rows), bootstrap_replicates=30)
        self.assertFalse(decision.factor_results["sampling"]["passed"])

    def test_malformed_pair_does_not_contaminate_response_aggregates(self) -> None:
        rows = list(_fixture(image_count=2))
        duplicate = next(
            row
            for row in rows
            if row.intervention_kind == "sampling"
            and row.region_role == "target"
            and row.intervention_severity == 0.50
        )
        rows.append(replace(duplicate, predicted_sampling=1.0))
        bad_base = (
            duplicate.seed,
            duplicate.node_id,
            duplicate.image_id,
            duplicate.object_id,
            duplicate.pair_id,
        )
        rows_without_bad_group = tuple(
            row
            for row in rows
            if not (
                row.intervention_kind == "sampling"
                and (
                    row.seed,
                    row.node_id,
                    row.image_id,
                    row.object_id,
                    row.pair_id,
                )
                == bad_base
            )
        )
        malformed = intervention_statistics(tuple(rows), factor="sampling")
        clean = intervention_statistics(rows_without_bad_group, factor="sampling")
        self.assertGreater(malformed["malformed"], 0)
        self.assertEqual(malformed["eligible"], clean["eligible"])
        self.assertEqual(malformed["ordered"], clean["ordered"])
        for field in ("target_mean_response", "background_mean_response", "paired_mean"):
            self.assertAlmostEqual(malformed[field], clean[field])

    def test_duplicate_identical_clean_target_is_malformed(self) -> None:
        rows = list(_fixture(image_count=2))
        duplicate = next(
            row
            for row in rows
            if row.intervention_kind == "clean" and row.region_role == "target"
        )
        rows.append(duplicate)
        stats = intervention_statistics(tuple(rows), factor="sampling")
        self.assertGreater(stats["malformed"], 0)
        decision = audit_natural_factors(tuple(rows), bootstrap_replicates=30)
        self.assertFalse(decision.factor_results["sampling"]["passed"])

    def test_missing_clean_target_is_malformed(self) -> None:
        rows = list(_fixture(image_count=2))
        removed = False
        filtered: list[NaturalFactorObservation] = []
        for row in rows:
            if not removed and row.intervention_kind == "clean" and row.region_role == "target":
                removed = True
                continue
            filtered.append(row)
        stats = intervention_statistics(tuple(filtered), factor="sampling")
        self.assertGreater(stats["malformed"], 0)
        decision = audit_natural_factors(tuple(filtered), bootstrap_replicates=30)
        self.assertFalse(decision.factor_results["sampling"]["passed"])

    def test_zero_severity_factor_intervention_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "severity 0"):
            _observation(
                intervention_kind="sampling",
                intervention_severity=0.0,
                pair_id="bad-zero",
            )

    def test_expected_intervention_severities_must_be_registered_and_increasing(self) -> None:
        from ifdr_yolo.eval.natural_factor_audit import intervention_statistics

        for expected in ((), (0.5, 0.5), (0.75, 0.25), (0.0, 0.5), (1.1,)):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    intervention_statistics(
                        (), factor="sampling", expected_intervention_severities=expected
                    )

    def test_target_not_stronger_than_background_fails(self) -> None:
        rows = list(_fixture(target_boost=0.02))
        # Background is deliberately larger than target for all sampling rows.
        for index, row in enumerate(rows):
            if row.intervention_kind == "sampling" and row.region_role == "background":
                rows[index] = _observation(
                    seed=row.seed,
                    node_id=row.node_id,
                    image_id=row.image_id,
                    object_id=row.object_id,
                    box_height=row.box_height,
                    region_role=row.region_role,
                    intervention_kind=row.intervention_kind,
                    intervention_severity=row.intervention_severity,
                    pair_id=row.pair_id,
                    natural_sampling=row.natural_sampling,
                    natural_visibility=row.natural_visibility,
                    predicted_sampling=0.25 + row.intervention_severity * 0.3,
                    predicted_visibility=row.predicted_visibility,
                )
        decision = audit_natural_factors(tuple(rows), bootstrap_replicates=60)
        self.assertFalse(decision.factor_results["sampling"]["passed"])

    def test_cluster_bootstrap_is_deterministic_and_image_based(self) -> None:
        rows = _fixture(image_count=4)
        first = image_cluster_bootstrap(rows, factor="sampling", replicates=40, seed=20260804)
        second = image_cluster_bootstrap(rows, factor="sampling", replicates=40, seed=20260804)
        self.assertEqual(first, second)
        self.assertEqual(first["sampling_unit"], "image_id")
        self.assertEqual(first["unique_image_count"], 4 * len(SEEDS) * len(NODES))

    def test_same_source_image_across_seeds_nodes_and_objects_is_one_cluster(self) -> None:
        import ifdr_yolo.eval.natural_factor_audit as audit_module

        rows: list[NaturalFactorObservation] = []
        for image_id, offset in (("shared", 0.0), ("other", 0.2)):
            for seed in (17, 29):
                for node in (11, 14):
                    for object_id in (0, 1):
                        rows.append(
                            _observation(
                                seed=seed,
                                node_id=node,
                                image_id=image_id,
                                object_id=object_id,
                                natural_sampling=0.2 + offset + object_id * 0.1,
                                natural_visibility=0.4 + offset,
                                predicted_sampling=0.3 + offset + object_id * 0.1,
                                predicted_visibility=0.5 + offset,
                            )
                        )
        captured: list[tuple[tuple[str, ...], tuple[NaturalFactorObservation, ...]]] = []
        original_cluster_rows = audit_module._cluster_rows

        def spy_cluster_rows(
            observations: tuple[NaturalFactorObservation, ...],
            sampled_images: tuple[str, ...],
        ) -> tuple[NaturalFactorObservation, ...]:
            result = original_cluster_rows(observations, sampled_images)
            captured.append((tuple(sampled_images), result))
            return result

        with patch.object(audit_module, "_cluster_rows", side_effect=spy_cluster_rows):
            bootstrap = image_cluster_bootstrap(
                tuple(rows), factor="sampling", replicates=40, seed=20260804, return_samples=True
            )
        self.assertEqual(bootstrap["unique_image_count"], 2)
        self.assertEqual(bootstrap["sampling_unit"], "image_id")
        expected_count = {"shared": 8, "other": 8}
        for image_sample, count_sample in zip(
            bootstrap["sampled_image_ids"], bootstrap["sampled_cluster_sizes"]
        ):
            self.assertEqual(len(image_sample), 2)
            self.assertEqual(
                [expected_count[image_id] for image_id in image_sample], count_sample
            )
        self.assertEqual(len(captured), 40)
        for sampled_images, sampled_rows in captured:
            sampled_counts = Counter(sampled_images)
            returned_counts = Counter(row.image_id for row in sampled_rows)
            self.assertEqual(
                returned_counts,
                Counter(
                    {
                        image_id: expected_count[image_id] * draw_count
                        for image_id, draw_count in sampled_counts.items()
                    }
                ),
            )

    def test_missing_required_seed_node_intervention_evidence_fails(self) -> None:
        rows = tuple(
            row
            for row in _fixture(image_count=2)
            if not (
                row.intervention_kind == "sampling"
                and row.seed == 17
                and row.node_id == 11
            )
        )
        decision = audit_natural_factors(rows, bootstrap_replicates=30)
        self.assertFalse(decision.factor_results["sampling"]["passed"])
        self.assertIn("sampling_missing_intervention_seed_17_node_11", decision.reasons)

    def test_bootstrap_interval_crossing_zero_is_not_a_pass(self) -> None:
        rows = tuple(
            _observation(
                seed=17,
                node_id=11,
                image_id=f"unstable-{index}",
                object_id=0,
                box_height=20.0 + index,
                natural_sampling=natural,
                natural_visibility=0.2 + 0.2 * index,
                predicted_sampling=prediction,
                predicted_visibility=0.2 + 0.2 * index,
            )
            for index, (natural, prediction) in enumerate(
                ((0.1, 0.1), (0.2, 0.9), (0.8, 0.2), (0.9, 0.8))
            )
        )
        bootstrap = image_cluster_bootstrap(rows, factor="sampling", replicates=200, seed=7)
        self.assertLessEqual(bootstrap["ci_lower"], 0.0)
        self.assertGreaterEqual(bootstrap["ci_upper"], 0.0)
        decision = audit_natural_factors(
            rows,
            required_seeds=(17,),
            required_nodes=(11,),
            bootstrap_replicates=60,
        )
        self.assertFalse(decision.factor_results["sampling"]["passed"])
        self.assertIn("sampling_pooled_raw_ci_crosses_zero", decision.reasons)


if __name__ == "__main__":
    unittest.main()
