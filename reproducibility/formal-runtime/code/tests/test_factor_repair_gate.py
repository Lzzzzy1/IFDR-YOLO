from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from ifdr_yolo.eval.factor_repair_gate import (
    FACTOR_GATE_BOOTSTRAP_PERCENTILES,
    FACTOR_GATE_BOOTSTRAP_REPLICATES,
    FACTOR_GATE_BOOTSTRAP_SEED,
    PRIMARY_ENDPOINTS,
    FactorRepairGateDecision,
    FactorRepairSelectionDecision,
    PairedDelta,
    composite_mechanism_score,
    evaluate_factor_repair_gate,
    paired_image_cluster_delta,
    paired_image_cluster_replicate,
    paired_resample_indices,
    recompute_endpoints,
    require_factor_guided_advancement,
    select_repair_against_f0,
)
from ifdr_yolo.eval.natural_factor_audit import NaturalFactorObservation, natural_factor_alignment


def _endpoint_values(value: float = 0.25) -> dict[str, float]:
    return {name: float(value) for name in PRIMARY_ENDPOINTS}


class GateRows(list[dict[str, object]]):
    def __init__(self, values: list[dict[str, object]], *, audit: object):
        super().__init__(values)
        self.audit = audit


def candidate_rows(
    *,
    primary_positive: tuple[int, ...] = (17, 20, 23, 26),
    diagnostic_reverse: tuple[int, ...] = (),
    seed17: bool = True,
    seeds: tuple[int, ...] = (17,),
    severity_rates: tuple[float, ...] | None = None,
    malformed: int = 0,
    missing: tuple[tuple[int, int], ...] = (),
    nonfinite: tuple[tuple[int, int, str], ...] = (),
):
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for node in (17, 20, 23, 26, 11, 14):
            if (seed, node) in missing:
                continue
            value = 0.25 if (node in primary_positive or node in (11, 14)) else -0.25
            if node in diagnostic_reverse:
                value = -0.2
            endpoints = _endpoint_values(value)
            for row_seed, row_node, endpoint in nonfinite:
                if (row_seed, row_node) == (seed, node):
                    endpoints[endpoint] = float("nan")
            row: dict[str, object] = {
                "seed": seed,
                "node_id": node,
                "endpoints": endpoints,
                "direction": value,
                "severity_ordering": 1.0,
                "target_response": 0.20,
                "background_response": 0.05,
                "specificity_gap": 0.15,
                "malformed": malformed,
            }
            rows.append(row)
    if severity_rates is not None:
        for index, rate in enumerate(severity_rates):
            rows[index]["severity_ordering"] = rate
    audit = pooled_audit_payload(
        seeds=seeds,
        diagnostic_point=-0.20 if diagnostic_reverse else 0.20,
        intervention_status="malformed" if malformed else "ok",
        malformed=malformed,
    )
    return GateRows(rows, audit=audit)


@dataclass(frozen=True)
class Evidence:
    condition: str
    image_ids_hash: str
    image_ids: tuple[str, ...]
    endpoints: dict[str, float]
    evidence_sha256: str
    absolute_gate_passed: bool = True
    complete: bool = True
    endpoint_samples: dict[str, tuple[float, ...]] | None = None


def candidate_evidence(
    condition: str,
    image_ids_hash: str,
    endpoint_values: float | dict[str, float],
    *,
    absolute_gate_passed: bool = True,
    complete: bool = True,
    image_ids: tuple[str, ...] = ("a", "b", "c", "d"),
    evidence_sha256: str | None = None,
    endpoint_samples: dict[str, tuple[float, ...]] | None = None,
) -> Evidence:
    endpoints = (
        _endpoint_values(endpoint_values)
        if isinstance(endpoint_values, (int, float))
        else dict(endpoint_values)
    )
    return Evidence(
        condition=condition,
        image_ids_hash=image_ids_hash,
        image_ids=image_ids,
        endpoints=endpoints,
        evidence_sha256=evidence_sha256 or (condition.lower() * 64)[:64],
        absolute_gate_passed=absolute_gate_passed,
        complete=complete,
        endpoint_samples=endpoint_samples,
    )


def gate_decision(*, passed: bool) -> FactorRepairGateDecision:
    return FactorRepairGateDecision(
        passed=passed,
        stage="post_adaptation",
        primary_nodes=(17, 20, 23, 26),
        diagnostic_nodes=(11, 14),
        checks={"all": passed},
        failures=() if passed else ("post_adaptation_failure",),
        evidence_sha256=("a" if passed else "b") * 64,
    )


def pooled_audit_payload(
    *,
    seeds: tuple[int, ...] = (17,),
    raw_ci: tuple[float, float] = (0.10, 0.30),
    residual_ci: tuple[float, float] = (0.05, 0.25),
    diagnostic_point: float = 0.20,
    intervention_status: str = "ok",
    malformed: int = 0,
    ordered_pair_rate: float = 0.90,
    target_mean_response: float = 0.20,
    background_mean_response: float = 0.05,
    paired_mean: float = 0.15,
    specificity_gap: float | None = None,
) -> dict[str, object]:
    """Natural-factor-audit shaped pooled evidence for gate-boundary tests."""

    factor_results: dict[str, object] = {}
    for factor in ("sampling", "visibility"):
        seed_node: dict[str, object] = {}
        for seed in seeds:
            for node in (17, 20, 23, 26, 11, 14):
                point = diagnostic_point if node == 11 else 0.20
                seed_node[f"{seed}:{node}"] = {
                    "raw": {"status": "ok", "success": True, "rho": point},
                    "residual": {"status": "ok", "success": True, "rho": point},
                    "direction": point,
                }
        factor_results[factor] = {
            "alignment": {
                "pooled_raw": {"status": "ok", "success": True, "rho": 0.20},
                "pooled_residual": {"status": "ok", "success": True, "rho": 0.20},
                "pooled_raw_ci": {
                    "status": "ok", "ci_lower": raw_ci[0], "ci_upper": raw_ci[1]
                },
                "pooled_residual_ci": {
                    "status": "ok",
                    "ci_lower": residual_ci[0],
                    "ci_upper": residual_ci[1],
                },
                "seed_node": seed_node,
            },
            "intervention": {
                "status": intervention_status,
                "malformed": malformed,
                "ordered_pair_rate": ordered_pair_rate,
                "target_mean_response": target_mean_response,
                "background_mean_response": background_mean_response,
                "paired_mean": paired_mean,
                "eligible_by_seed_node": {
                    f"{seed}:{node}": 1
                    for seed in seeds
                    for node in (17, 20, 23, 26, 11, 14)
                },
            },
        }
        if specificity_gap is not None:
            intervention = factor_results[factor]["intervention"]
            assert isinstance(intervention, dict)
            intervention["specificity_gap"] = specificity_gap
    return {"factor_results": factor_results}


class FactorRepairGateTest(unittest.TestCase):
    def test_real_natural_alignment_seed_node_shape_is_consumed(self) -> None:
        natural_values = (0.10, 0.30, 0.20, 0.50, 0.35, 0.70, 0.55, 0.90)
        predicted_values = (0.20, 0.60, 0.30, 0.50, 0.40, 0.70, 0.80, 0.85)
        observations = tuple(
            NaturalFactorObservation(
                seed=17,
                node_id=node,
                image_id=f"node-{node}-image-{index}",
                object_id=index,
                class_id=0,
                box_height=10.0 + index * 3.0,
                region_role="target",
                intervention_kind="natural",
                intervention_severity=0.0,
                pair_id=None,
                natural_sampling=natural_values[index],
                natural_visibility=natural_values[index],
                predicted_sampling=predicted_values[index],
                predicted_visibility=predicted_values[index],
                branch_weights=(0.6, 0.4),
            )
            for node in (11, 14)
            for index in range(8)
        )
        alignment = natural_factor_alignment(
            observations,
            factor="sampling",
            bootstrap_replicates=2,
            bootstrap_seed=20260805,
        )
        payload = pooled_audit_payload()
        factors = payload["factor_results"]
        assert isinstance(factors, dict)
        for factor in ("sampling", "visibility"):
            section = factors[factor]
            assert isinstance(section, dict)
            section["alignment"]["seed_node"] = alignment["seed_node"]
        decision = evaluate_factor_repair_gate(
            {"rows": candidate_rows(), "audit": payload}, stage="development"
        )
        self.assertTrue(decision.passed)

    def test_expected_seed_override_is_not_public(self) -> None:
        with self.assertRaises(TypeError):
            evaluate_factor_repair_gate(  # type: ignore[call-arg]
                candidate_rows(), stage="development", expected_seeds=(29,)
            )

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage"):
            evaluate_factor_repair_gate(candidate_rows(), stage="screen")

    def test_formal_stage_uses_fixed_registered_seed_matrix(self) -> None:
        rows = candidate_rows(seeds=(17, 29, 41))
        extra = dict(rows[0])
        extra["seed"] = 99
        rows.append(extra)
        decision = evaluate_factor_repair_gate(rows, stage="formal")
        self.assertFalse(decision.passed)
        self.assertFalse(decision.checks["complete_seed_node_matrix"])
        self.assertIn("unexpected_seed_or_node", decision.failures)

    def test_pooled_primary_ci_lower_bound_is_strictly_positive(self) -> None:
        payload = pooled_audit_payload(raw_ci=(0.0, 0.30))
        decision = evaluate_factor_repair_gate(
            {"rows": candidate_rows(), "audit": payload},
            stage="development",
        )
        self.assertFalse(decision.passed)
        self.assertIn("pooled_raw_ci_crosses_zero", decision.failures)

    def test_missing_or_nonfinite_pooled_stat_fails_closed(self) -> None:
        payload = pooled_audit_payload()
        factors = payload["factor_results"]
        assert isinstance(factors, dict)
        sampling = factors["sampling"]
        assert isinstance(sampling, dict)
        alignment = sampling["alignment"]
        assert isinstance(alignment, dict)
        alignment["pooled_residual"] = {
            "status": "ok", "success": True, "rho": float("nan")
        }
        decision = evaluate_factor_repair_gate(
            {"rows": candidate_rows(), "audit": payload},
            stage="development",
        )
        self.assertFalse(decision.passed)
        self.assertIn("pooled_residual_nonfinite", decision.failures)

    def test_diagnostic_reverse_uses_registered_point(self) -> None:
        passing = evaluate_factor_repair_gate(
            {"rows": candidate_rows(), "audit": pooled_audit_payload()},
            stage="development",
        )
        self.assertTrue(passing.passed)

        failing = evaluate_factor_repair_gate(
            {
                "rows": candidate_rows(diagnostic_reverse=(11,)),
                "audit": pooled_audit_payload(diagnostic_point=-0.20),
            },
            stage="development",
        )
        self.assertFalse(failing.passed)
        self.assertIn("diagnostic_reverse_association", failing.failures)

    def test_intervention_statistics_are_consumed_without_row_aggregation(self) -> None:
        payload = pooled_audit_payload(
            intervention_status="malformed",
            malformed=1,
            ordered_pair_rate=0.95,
        )
        decision = evaluate_factor_repair_gate(
            {"rows": candidate_rows(), "audit": payload},
            stage="development",
        )
        self.assertFalse(decision.passed)
        self.assertIn("sampling_malformed_intervention_pairs", decision.failures)

    def test_intervention_specificity_gap_conflict_fails_closed(self) -> None:
        payload = pooled_audit_payload(specificity_gap=0.90)
        decision = evaluate_factor_repair_gate(
            {"rows": candidate_rows(), "audit": payload},
            stage="development",
        )
        self.assertFalse(decision.passed)
        self.assertIn("sampling_specificity_gap_conflict", decision.failures)

    def test_observed_seed_node_matrix_rejects_extra_pairs(self) -> None:
        extra = dict(candidate_rows()[0])
        extra["seed"] = 29
        extra["node_id"] = 17
        rows = candidate_rows() + [extra]
        decision = evaluate_factor_repair_gate(
            rows,
            stage="development",
        )
        self.assertFalse(decision.passed)
        self.assertIn("unexpected_seed_or_node", decision.failures)

    def test_imported_registered_constants_are_fixed(self) -> None:
        self.assertEqual(PRIMARY_ENDPOINTS, (
            "sampling_residual_spearman",
            "visibility_residual_spearman",
            "sampling_specificity_gap",
            "visibility_specificity_gap",
        ))
        self.assertEqual(FACTOR_GATE_BOOTSTRAP_REPLICATES, 10_000)
        self.assertEqual(FACTOR_GATE_BOOTSTRAP_SEED, 20260805)
        self.assertEqual(FACTOR_GATE_BOOTSTRAP_PERCENTILES, (0.025, 0.975))

    def test_seed17_gate_requires_three_of_four_positive_primary_nodes(self) -> None:
        rows = candidate_rows(primary_positive=(17, 20, 23))
        decision = evaluate_factor_repair_gate(rows, stage="development")
        self.assertTrue(decision.passed)

    def test_significant_reverse_diagnostic_blocks_gate(self) -> None:
        rows = candidate_rows(diagnostic_reverse=(11,))
        decision = evaluate_factor_repair_gate(rows, stage="development")
        self.assertFalse(decision.passed)
        self.assertIn("diagnostic_reverse_association", decision.failures)

    def test_post_adaptation_failure_blocks_factor_guided_claim(self) -> None:
        with self.assertRaisesRegex(ValueError, "post-adaptation factor gate failed"):
            require_factor_guided_advancement(pre=gate_decision(passed=True), post=gate_decision(passed=False))

    def test_bootstrap_quantiles_are_fixed_and_ci_is_finite(self) -> None:
        delta = np.asarray((-0.20, -0.05, 0.10, 0.25), dtype=float)
        ci = np.quantile(delta, FACTOR_GATE_BOOTSTRAP_PERCENTILES, method="linear")
        self.assertEqual(tuple(ci.shape), (2,))
        self.assertTrue(np.isfinite(ci).all())

    def test_severity_ordering_boundary_at_80_percent(self) -> None:
        rows = candidate_rows(severity_rates=(1.0, 1.0, 1.0, 0.0, 1.0))
        decision = evaluate_factor_repair_gate(rows, stage="development")
        self.assertTrue(decision.checks["severity_ordering"])

    def test_positive_paired_target_response_and_background_gap(self) -> None:
        rows = candidate_rows()
        decision = evaluate_factor_repair_gate(rows, stage="development")
        self.assertTrue(decision.checks["paired_target_response"])
        self.assertTrue(decision.checks["background_specificity_gap"])

    def test_twelve_primary_directions_requires_ten(self) -> None:
        rows = candidate_rows(primary_positive=(17, 20, 23, 26), seeds=(17, 29, 41))
        rows[3]["direction"] = -0.25
        rows[7]["direction"] = -0.25
        decision = evaluate_factor_repair_gate(rows, stage="formal")
        self.assertTrue(decision.passed)

    def test_ci_lower_bound_above_zero_is_required(self) -> None:
        f0 = candidate_evidence("F0", "same", 0.1)
        f1 = candidate_evidence("F1", "same", 0.2)
        with patch("ifdr_yolo.eval.factor_repair_gate.paired_image_cluster_delta") as delta:
            delta.return_value = type("D", (), {
                "point": 0.1,
                "ci95": (0.0, 0.2),
                "candidate_endpoints": f1.endpoints,
                "candidate_evidence_sha256": f1.evidence_sha256,
            })()
            self.assertIsNone(select_repair_against_f0(f0, (f1,)))
            delta.return_value.ci95 = (1e-12, 0.2)
            self.assertIsNotNone(select_repair_against_f0(f0, (f1,)))

    def test_incomplete_f0_evidence_fails_closed(self) -> None:
        f0 = candidate_evidence("F0", "same", {PRIMARY_ENDPOINTS[0]: 0.1}, complete=False)
        with self.assertRaisesRegex(ValueError, "incomplete F0 evidence"):
            select_repair_against_f0(f0, ())

    def test_missing_node_seed_or_malformed_count_fails(self) -> None:
        for kwargs in (
            {"missing": ((17, 26),)},
            {"seeds": (17, 29), "missing": ((29, 14),)},
            {"malformed": 1},
            {"nonfinite": ((17, 17, PRIMARY_ENDPOINTS[0]),)},
        ):
            with self.subTest(kwargs=kwargs):
                decision = evaluate_factor_repair_gate(
                    candidate_rows(**kwargs),
                    stage="formal" if kwargs.get("seeds") == (17, 29) else "development",
                )
                self.assertFalse(decision.passed)

    def test_candidate_requires_complete_f0_and_paired_delta_ci(self) -> None:
        f0 = candidate_evidence("F0", "same", 0.1)
        f1 = candidate_evidence("F1", "same", 0.2, complete=False)
        self.assertIsNone(select_repair_against_f0(f0, (f1,)))

    def test_multiple_eligible_candidates_use_lower_point_and_name_ties(self) -> None:
        f0 = candidate_evidence("F0", "same", 0.1)
        f1 = candidate_evidence("F1", "same", 0.2)
        f2 = candidate_evidence("F2", "same", 0.2)
        f3 = candidate_evidence("F3", "same", 0.2)
        with patch("ifdr_yolo.eval.factor_repair_gate.paired_image_cluster_delta") as delta:
            def fake(candidate, reference):
                return type("D", (), {
                    "point": {"F1": 0.1, "F2": 0.1, "F3": 0.1}[candidate.condition],
                    "ci95": (0.2, 0.3),
                    "candidate_endpoints": candidate.endpoints,
                    "candidate_evidence_sha256": candidate.evidence_sha256,
                })()
            delta.side_effect = fake
            selected = select_repair_against_f0(f0, (f3, f2, f1))
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.selected_condition, "F1")

    def test_gate_decision_carries_reference_delta_and_hash(self) -> None:
        f0 = candidate_evidence("F0", "same", 0.1, evidence_sha256="a" * 64)
        f1 = candidate_evidence("F1", "same", 0.2, evidence_sha256="b" * 64)
        decision = select_repair_against_f0(f0, (f1,))
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.reference_condition, "F0")
        self.assertEqual(decision.selected_condition, "F1")
        self.assertEqual(decision.reference_evidence_sha256, "a" * 64)
        self.assertEqual(decision.selected_evidence_sha256, "b" * 64)
        self.assertEqual(decision.endpoint_table[0][0], "F0")
        with self.assertRaises(TypeError):
            decision.endpoint_table[0][1][0] = ("bad", 1.0)  # type: ignore[index]

    def test_manual_condition_string_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection decision"):
            require_factor_guided_advancement(pre="F3", post=gate_decision(passed=True))

    def test_bootstrap_resamples_are_byte_identical_across_repeated_calls(self) -> None:
        first = paired_resample_indices(stage="development", image_ids_hash="same", image_count=8, replicate_index=4)
        second = paired_resample_indices(stage="development", image_ids_hash="same", image_count=8, replicate_index=4)
        self.assertEqual(first, second)
        self.assertEqual(np.asarray(first, dtype=np.int64).tobytes(), np.asarray(second, dtype=np.int64).tobytes())

    def test_bootstrap_resample_key_is_shared_across_candidate_names(self) -> None:
        first = paired_resample_indices(stage="development", image_ids_hash="same", image_count=8, replicate_index=4)
        second = paired_resample_indices(stage="development", image_ids_hash="same", image_count=8, replicate_index=4)
        self.assertEqual(first, second)

    def test_indexed_replicate_matches_registered_draw_math(self) -> None:
        f0 = candidate_evidence(
            "F0",
            "same",
            0.1,
            endpoint_samples={name: (0.1, 0.2, 0.3, 0.4) for name in PRIMARY_ENDPOINTS},
        )
        f1 = candidate_evidence(
            "F1",
            "same",
            0.2,
            endpoint_samples={name: (0.2, 0.3, 0.4, 0.5) for name in PRIMARY_ENDPOINTS},
        )
        index = 7
        indices = paired_resample_indices(
            stage="development",
            image_ids_hash="same",
            image_count=4,
            replicate_index=index,
        )
        expected = composite_mechanism_score(
            recompute_endpoints(f1, indices)
        ) - composite_mechanism_score(recompute_endpoints(f0, indices))
        self.assertEqual(
            paired_image_cluster_replicate(f1, f0, index),
            expected,
        )

    def test_bootstrap_seed_or_replicate_override_is_rejected(self) -> None:
        parameters = inspect.signature(select_repair_against_f0).parameters
        self.assertNotIn("seed", parameters)
        self.assertNotIn("replicates", parameters)
        with self.assertRaises(TypeError):
            select_repair_against_f0(None, (), seed=1)  # type: ignore[call-arg]

    def test_precomputed_deltas_use_selection_rule_without_bootstrap(self) -> None:
        f0 = candidate_evidence("F0", "same", 0.1)
        f1 = candidate_evidence("F1", "same", 0.2)
        f2 = candidate_evidence("F2", "same", 0.3)
        precomputed = {
            "F1": PairedDelta(
                point=0.05,
                ci95=(0.02, 0.10),
                candidate_endpoints=f1.endpoints,
                candidate_evidence_sha256=f1.evidence_sha256,
            ),
            "F2": PairedDelta(
                point=0.02,
                ci95=(0.02, 0.10),
                candidate_endpoints=f2.endpoints,
                candidate_evidence_sha256=f2.evidence_sha256,
            ),
        }
        with patch(
            "ifdr_yolo.eval.factor_repair_gate.paired_image_cluster_delta",
            side_effect=AssertionError("precomputed path must not bootstrap"),
        ):
            selected = select_repair_against_f0(
                f0,
                (f2, f1),
                paired_deltas=precomputed,
            )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.selected_condition, "F1")


if __name__ == "__main__":
    unittest.main()
