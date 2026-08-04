from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

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
) -> NaturalFactorObservation:
    return NaturalFactorObservation(
        seed=seed,
        node_id=node_id,
        image_id=image_id,
        object_id=object_id,
        class_id=object_id % 3,
        box_height=box_height,
        region_role=region_role,
        intervention_kind=intervention_kind,
        intervention_severity=intervention_severity,
        pair_id=pair_id,
        natural_sampling=natural_sampling,
        natural_visibility=natural_visibility,
        predicted_sampling=predicted_sampling,
        predicted_visibility=predicted_visibility,
        branch_weights=(0.6, 0.4),
    )


def _fixture(
    *,
    sampling_sign: float = 1.0,
    visibility_sign: float = 1.0,
    target_boost: float = 0.25,
    image_count: int = 8,
) -> tuple[NaturalFactorObservation, ...]:
    rows: list[NaturalFactorObservation] = []
    for seed in SEEDS:
        for node in NODES:
            for index in range(image_count):
                image_id = f"s{seed}-n{node}-i{index}"
                ns = 0.08 + 0.10 * index
                nv = 0.18 + 0.08 * ((index * 3) % image_count)
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
            target=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            prediction=(2.0, 4.0, 6.1, 7.9, 10.2, 12.0),
            box_height=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
            class_ids=(0, 0, 1, 1, 2, 2),
        )
        self.assertTrue(result["success"])
        self.assertGreater(result["rho"], 0.9)


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

    def test_intervention_severity_input_order_does_not_change_result(self) -> None:
        rows = _fixture(image_count=2)
        first = intervention_statistics(rows, factor="visibility")
        second = intervention_statistics(tuple(reversed(rows)), factor="visibility")
        self.assertEqual(first, second)

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
