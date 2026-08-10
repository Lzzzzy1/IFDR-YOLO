from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ifdr_yolo.eval.factor_repair_gate import (
    FactorRepairGateDecision,
    PRIMARY_ENDPOINTS,
)
from ifdr_yolo.eval.natural_factor_audit import NaturalFactorObservation
from ifdr_yolo.eval.factor_repair_evidence import (
    build_factor_repair_evidence,
    canonical_evidence_payload,
    load_factor_repair_evidence,
)


def _checkpoint_roles(path: Path, digest: str) -> dict[str, object]:
    return {
        "primary_checkpoint": {
            "path": str(path),
            "sha256": digest,
            "role": "primary",
            "checkpoint_role": "calibration_last",
        }
    }


def _gate(passed: bool = True) -> FactorRepairGateDecision:
    return FactorRepairGateDecision(
        passed=passed,
        stage="development",
        primary_nodes=(17, 20, 23, 26),
        diagnostic_nodes=(11, 14),
        checks={"complete": passed},
        failures=() if passed else ("invalid",),
        evidence_sha256=("a" if passed else "b") * 64,
    )


def _natural_rows(*, seed: int = 17, image_ids: tuple[str, ...] = ("a", "b", "c", "d")):
    rows: list[NaturalFactorObservation] = []
    for image_index, image_id in enumerate(image_ids):
        for node in (11, 14, 17, 20, 23, 26):
            rows.append(
                NaturalFactorObservation(
                    seed=seed,
                    node_id=node,
                    image_id=image_id,
                    object_id=0,
                    class_id=0,
                    box_height=10.0,
                    region_role="target",
                    intervention_kind="natural",
                    intervention_severity=0.0,
                    pair_id=None,
                    natural_sampling=0.1 + image_index * 0.1,
                    natural_visibility=0.1 + image_index * 0.1,
                    predicted_sampling=0.2 + image_index * 0.1,
                    predicted_visibility=0.2 + image_index * 0.1,
                    branch_weights=(0.5, 0.5),
                )
            )
    return tuple(rows)


def _complete_rows(*, seed: int = 17, image_ids: tuple[str, ...] = ("a", "b", "c", "d")):
    rows = list(_natural_rows(seed=seed, image_ids=image_ids))
    for image_index, image_id in enumerate(image_ids):
        for node in (11, 14, 17, 20, 23, 26):
            for factor in ("sampling", "visibility"):
                pair_id = f"pair-{image_id}-{node}-{factor}"
                for intervention_kind, severity in (
                    ("clean", 0.0),
                    (factor, 0.25),
                    (factor, 0.5),
                    (factor, 0.75),
                    (factor, 1.0),
                ):
                    for region_role in ("target", "background"):
                        baseline = 0.20 if region_role == "target" else 0.10
                        value = baseline + (severity * 0.7 if region_role == "target" else severity * 0.1)
                        rows.append(
                            NaturalFactorObservation(
                                seed=seed,
                                node_id=node,
                                image_id=image_id,
                                object_id=0,
                                class_id=0,
                                box_height=10.0,
                                region_role=region_role,
                                intervention_kind=intervention_kind,
                                intervention_severity=severity,
                                pair_id=pair_id,
                                natural_sampling=0.1 + image_index * 0.1,
                                natural_visibility=0.1 + image_index * 0.1,
                                predicted_sampling=value if factor == "sampling" else baseline,
                                predicted_visibility=value if factor == "visibility" else baseline,
                                branch_weights=(0.5, 0.5),
                                intervention_factor=factor,
                            )
                        )
    return tuple(rows)


class FactorRepairEvidenceTest(unittest.TestCase):
    def _checkpoint(self):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "last.pt"
        path.write_bytes(b"calibration-last")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return directory, path, digest

    def test_development_rejects_any_seed_other_than_17(self):
        directory, path, digest = self._checkpoint()
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, "seed 17"):
            build_factor_repair_evidence(
                condition="F0",
                stage="development",
                checkpoint=path,
                checkpoint_roles=_checkpoint_roles(path, digest),
                observations=_natural_rows(seed=29),
                audit={"factor_results": {}},
                image_ids=("a", "b", "c", "d"),
            )

    def test_best_checkpoint_or_hash_mismatch_fails_closed(self):
        directory, path, digest = self._checkpoint()
        self.addCleanup(directory.cleanup)
        roles = _checkpoint_roles(path, "0" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            build_factor_repair_evidence(
                condition="F0",
                checkpoint=path,
                checkpoint_roles=roles,
                observations=_complete_rows(),
                audit={"factor_results": {}},
                image_ids=("a", "b", "c", "d"),
            )

    def test_raw_observations_are_required_and_point_values_are_not_samples(self):
        directory, path, digest = self._checkpoint()
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, "observations"):
            build_factor_repair_evidence(
                condition="F0",
                checkpoint=path,
                checkpoint_roles=_checkpoint_roles(path, digest),
                observations=(),
                audit={"factor_results": {}},
                image_ids=("a", "b", "c", "d"),
            )

        gate = _gate()
        with patch(
            "ifdr_yolo.eval.factor_repair_evidence.evaluate_factor_repair_gate",
            return_value=gate,
        ):
            result = build_factor_repair_evidence(
                condition="F0",
                checkpoint=path,
                checkpoint_roles=_checkpoint_roles(path, digest),
                observations=_complete_rows(),
                audit={"factor_results": {}},
                image_ids=("a", "b", "c", "d"),
            )
        self.assertIsNone(result.endpoint_samples)
        self.assertTrue(result.raw_observations)
        self.assertEqual(result.absolute_gate, gate)

    def test_image_ids_and_hash_are_bound_and_digest_is_canonical(self):
        directory, path, digest = self._checkpoint()
        self.addCleanup(directory.cleanup)
        image_ids = ("a", "b", "c", "d")
        image_hash = hashlib.sha256(
            json.dumps(list(image_ids), separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        gate = _gate()
        with patch(
            "ifdr_yolo.eval.factor_repair_evidence.evaluate_factor_repair_gate",
            return_value=gate,
        ):
            result = build_factor_repair_evidence(
                condition="F3",
                checkpoint=path,
                checkpoint_roles=_checkpoint_roles(path, digest),
                observations=_complete_rows(),
                audit={"factor_results": {}},
                image_ids=image_ids,
                image_ids_hash=image_hash,
            )
        self.assertEqual(result.image_ids, image_ids)
        self.assertEqual(result.image_ids_hash, image_hash)
        payload = canonical_evidence_payload(result)
        self.assertEqual(result.evidence_sha256, _sha256(payload))
        persisted_payload = result.to_dict()
        persisted_payload.pop("evidence_sha256")
        self.assertEqual(result.evidence_sha256, _sha256(persisted_payload))
        self.assertTrue(result.verify_digest())
        with self.assertRaisesRegex(ValueError, "image IDs"):
            build_factor_repair_evidence(
                condition="F3",
                checkpoint=path,
                checkpoint_roles=_checkpoint_roles(path, digest),
                observations=_complete_rows(),
                audit={"factor_results": {}},
                image_ids=("a", "b", "c"),
                image_ids_hash=image_hash,
            )

    def test_persisted_bundle_round_trips_without_point_sample_fallback(self):
        directory, path, digest = self._checkpoint()
        self.addCleanup(directory.cleanup)
        gate = _gate()
        with patch(
            "ifdr_yolo.eval.factor_repair_evidence.evaluate_factor_repair_gate",
            return_value=gate,
        ):
            result = build_factor_repair_evidence(
                condition="F1",
                checkpoint=path,
                checkpoint_roles=_checkpoint_roles(path, digest),
                observations=_complete_rows(),
                audit={"factor_results": {}},
                image_ids=("a", "b", "c", "d"),
            )
        output = Path(directory.name) / "evidence.json"
        output.write_text(json.dumps(result.to_dict(), sort_keys=True), encoding="utf-8")
        loaded = load_factor_repair_evidence(output)
        self.assertEqual(loaded.evidence_sha256, result.evidence_sha256)
        self.assertIsNone(loaded.endpoint_samples)
        self.assertTrue(loaded.raw_observations)
        with self.assertRaises(TypeError):
            loaded.raw_observations[0]["image_id"] = "tampered"  # type: ignore[index]

    def test_positional_builder_calls_registered_development_gate(self):
        directory, path, digest = self._checkpoint()
        self.addCleanup(directory.cleanup)
        gate = _gate()
        with patch(
            "ifdr_yolo.eval.factor_repair_evidence.evaluate_factor_repair_gate",
            return_value=gate,
        ) as mocked_gate:
            result = build_factor_repair_evidence(
                "F2",
                "development",
                path,
                _complete_rows(),
                {"factor_results": {}},
                checkpoint_roles=_checkpoint_roles(path, digest),
                image_ids=("a", "b", "c", "d"),
            )
        self.assertIs(result.absolute_gate, gate)
        mocked_gate.assert_called_once()
        self.assertEqual(mocked_gate.call_args.kwargs, {"stage": "development"})

    def test_recompute_keeps_duplicate_image_clusters_as_raw_resamples(self):
        directory, path, digest = self._checkpoint()
        self.addCleanup(directory.cleanup)
        gate = _gate()
        with patch(
            "ifdr_yolo.eval.factor_repair_evidence.evaluate_factor_repair_gate",
            return_value=gate,
        ):
            result = build_factor_repair_evidence(
                "F0",
                "development",
                path,
                _complete_rows(),
                {"factor_results": {}},
                checkpoint_roles=_checkpoint_roles(path, digest),
                image_ids=("a", "b", "c", "d"),
            )
        draw = result.recompute_endpoints((1, 1, 2, 3))
        self.assertEqual(set(draw), set(PRIMARY_ENDPOINTS))

    def test_recompute_parses_and_groups_raw_rows_once_per_bundle(self):
        directory, path, digest = self._checkpoint()
        self.addCleanup(directory.cleanup)
        gate = _gate()
        with patch(
            "ifdr_yolo.eval.factor_repair_evidence.evaluate_factor_repair_gate",
            return_value=gate,
        ):
            result = build_factor_repair_evidence(
                "F0",
                "development",
                path,
                _complete_rows(),
                {"factor_results": {}},
                checkpoint_roles=_checkpoint_roles(path, digest),
                image_ids=("a", "b", "c", "d"),
            )
        with patch(
            "ifdr_yolo.eval.factor_repair_evidence._mapping_observation",
            wraps=__import__(
                "ifdr_yolo.eval.factor_repair_evidence",
                fromlist=["_mapping_observation"],
            )._mapping_observation,
        ) as parser:
            result.recompute_endpoints((0, 1, 2, 3))
            result.recompute_endpoints((1, 1, 2, 3))
            result.recompute_endpoints((3, 2, 1, 0))
        self.assertEqual(parser.call_count, len(result.raw_observations))


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    unittest.main()
