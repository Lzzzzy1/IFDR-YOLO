from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from ifdr_yolo.experiments.config import load_factor_repair_config


ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity": {
            "source_metadata_sha256": HASH,
            "images_metadata_sha256": HASH,
            "raw_labels_sha256": HASH,
            "split_sha256": HASH,
            "metadata_sha256": HASH,
            "initialization_checkpoint_sha256": HASH,
            "fit_ids_sha256": HASH,
            "development_ids_sha256": HASH,
        },
        "development": {"seed": 20260805, "fraction": 0.10},
        "conditions": {
            "M1": {"track": "metadata", "epochs": 60},
            "M2": {"track": "metadata", "epochs": 60},
            "M3": {"track": "metadata", "epochs": 60},
            "F0": {"track": "factor", "epochs": 30},
            "F1": {"track": "factor", "epochs": 30},
            "F2": {"track": "factor", "epochs": 30},
            "F3": {"track": "factor", "epochs": 30},
        },
        "task_adaptation_epochs": 60,
        "max_selected_factor_repairs": 1,
        "early_stopping": False,
        "training": {"imgsz": 640},
        "factor_loss": {
            "natural_gain": 1.0,
            "specificity_gain": 0.5,
            "specificity_margin": 0.05,
            "factor_weights": [1.0, 1.0],
        },
        "model": {
            "nodes": [11, 14, 17, 20, 23, 26],
            "primary_nodes": [17, 20, 23, 26],
        },
        "paths": {
            "metadata_jsonl": "metadata.jsonl",
            "images_jsonl": "images.jsonl",
            "raw_label_dir": "labels",
            "initialization_checkpoint": "init.pt",
            "output_root": "runs/factor-repair",
        },
        "schedule": {
            "replay": {
                "eta_peak": 0.30,
                "ramp_epochs": 5,
                "focus_end_epoch": 40,
                "recovery_start_epoch": 41,
                "total_epochs": 60,
                "priority_clip_quantile": 0.95,
                "eligible_floor": 0.05,
                "replacement": True,
                "draws_per_epoch": "fit_count",
            },
            "factor_calibration": {
                "epochs": 30,
                "views_per_sample": 3,
                "fusion_schedule": 0.0,
                "dcli_schedule": 0.0,
            },
            "task_adaptation": {"epochs": 60},
        },
        "checkpoint_policy": {
            "primary": "last.pt",
            "diagnostic": "best.pt",
            "early_stopping": False,
        },
        "metadata_replay": {
            "M1": "original",
            "M2": "cyclist_uniform",
            "M3": "joint_score",
        },
        "factor_gate": {
            "seed17_min_positive_primary_directions": 3,
            "formal_min_positive_seed_node_directions": 10,
            "formal_total_seed_node_directions": 12,
            "minimum_severity_ordering": 0.8,
            "diagnostic_reverse_abs_rho": 0.1,
            "selection_tie_tolerance": 1.0e-12,
            "require_paired_delta_ci_lower_positive": True,
            "require_zero_malformed": True,
        },
    }


def write_valid_config(
    directory: Path,
    *,
    extra: dict[str, object] | None = None,
) -> Path:
    payload = valid_payload()
    if extra:
        for section, values in extra.items():
            section_mapping = payload[section]
            assert isinstance(section_mapping, dict)
            assert isinstance(values, dict)
            section_mapping.update(values)
    path = directory / "factor-repair.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return path


class FactorRepairConfigTest(unittest.TestCase):
    def test_registered_conditions_have_equal_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_factor_repair_config(
                write_valid_config(Path(directory)),
                repository_root=ROOT,
            )

        self.assertEqual(
            {config.conditions[name].epochs for name in ("M1", "M2", "M3")},
            {60},
        )
        self.assertEqual(
            {config.conditions[name].epochs for name in ("F0", "F1", "F2", "F3")},
            {30},
        )
        self.assertEqual(config.task_adaptation_epochs, 60)
        self.assertEqual(config.max_selected_factor_repairs, 1)
        self.assertFalse(config.early_stopping)
        self.assertEqual(config.development.seed, 20260805)
        self.assertEqual(config.development.fraction, 0.10)
        self.assertEqual(config.factor_loss.natural_gain, 1.0)
        self.assertEqual(config.factor_loss.specificity_gain, 0.5)
        self.assertEqual(config.factor_loss.specificity_margin, 0.05)

    def test_canonical_seed17_yaml_is_strictly_valid(self) -> None:
        config = load_factor_repair_config(
            ROOT / "configs/experiments/kitti_ifdr_factor_repair_dev_s17.yaml",
            repository_root=ROOT,
        )
        self.assertEqual(config.development.seed, 20260805)
        self.assertEqual(config.model.nodes, (11, 14, 17, 20, 23, 26))
        self.assertEqual(config.model.primary_nodes, (17, 20, 23, 26))
        self.assertEqual(config.checkpoint_policy.primary, "last.pt")

    def test_unknown_or_unregistered_threshold_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_valid_config(
                Path(directory), extra={"factor_gate": {"rho_threshold": 0.01}}
            )
            with self.assertRaisesRegex(ValueError, "unknown factor_gate fields"):
                load_factor_repair_config(path, repository_root=ROOT)

    def test_bootstrap_seed_or_replicate_override_fails(self) -> None:
        for extra in (
            {"bootstrap_replicates": 10},
            {"bootstrap_seed": 17},
        ):
            with self.subTest(extra=extra):
                with tempfile.TemporaryDirectory() as directory:
                    path = write_valid_config(
                        Path(directory), extra={"factor_gate": extra}
                    )
                    with self.assertRaisesRegex(
                        ValueError, "unknown factor_gate fields"
                    ):
                        load_factor_repair_config(path, repository_root=ROOT)

    def test_required_hashes_are_real_sha256_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_payload()
            identity = payload["identity"]
            assert isinstance(identity, dict)
            identity["metadata_sha256"] = "not-a-hash"
            path = Path(directory) / "factor-repair.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity.metadata_sha256"):
                load_factor_repair_config(path, repository_root=ROOT)

    def test_missing_top_level_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_payload()
            del payload["factor_loss"]
            path = Path(directory) / "factor-repair.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing top-level fields"):
                load_factor_repair_config(path, repository_root=ROOT)

    def test_unknown_top_level_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_payload()
            payload["unregistered"] = True
            path = Path(directory) / "factor-repair.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown top-level fields"):
                load_factor_repair_config(path, repository_root=ROOT)

    def test_registered_float_offsets_fail_without_tolerance(self) -> None:
        cases = (
            ("development", "fraction", 0.10 + 1.0e-13),
            ("factor_loss", "natural_gain", 1.0 + 5.0e-13),
            ("factor_loss", "specificity_gain", 0.5 + 5.0e-13),
            ("factor_loss", "specificity_margin", 0.05 + 5.0e-13),
        )
        for section, field, value in cases:
            with self.subTest(section=section, field=field):
                with tempfile.TemporaryDirectory() as directory:
                    payload = valid_payload()
                    section_mapping = payload[section]
                    assert isinstance(section_mapping, dict)
                    section_mapping[field] = value
                    path = Path(directory) / "factor-repair.yaml"
                    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "registered"):
                        load_factor_repair_config(path, repository_root=ROOT)

    def test_bool_nan_inf_and_wrong_bool_types_fail_closed(self) -> None:
        cases = (
            ("task_adaptation_epochs", True, "integer"),
            ("max_selected_factor_repairs", True, "integer"),
            ("early_stopping", 0, "boolean"),
            ("development.fraction", float("nan"), "finite"),
            ("factor_loss.natural_gain", float("inf"), "finite"),
            ("factor_gate.minimum_severity_ordering", float("-inf"), "finite"),
            (
                "factor_gate.require_zero_malformed",
                1,
                "boolean",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    payload = valid_payload()
                    if "." in field:
                        section, nested_field = field.split(".", 1)
                        section_mapping = payload[section]
                        assert isinstance(section_mapping, dict)
                        section_mapping[nested_field] = value
                    else:
                        payload[field] = value
                    path = Path(directory) / "factor-repair.yaml"
                    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_factor_repair_config(path, repository_root=ROOT)

    def test_relative_paths_resolve_from_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory)
            config = load_factor_repair_config(
                write_valid_config(repository_root),
                repository_root=repository_root,
            )
        self.assertEqual(
            config.paths.output_root,
            repository_root / "runs/factor-repair",
        )

    def test_config_is_frozen_and_metadata_replay_is_registered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_factor_repair_config(
                write_valid_config(Path(directory)), repository_root=ROOT
            )
        self.assertEqual(config.metadata_replay["M2"], "cyclist_uniform")
        self.assertEqual(config.metadata_replay.M3, "joint_score")
        with self.assertRaises((AttributeError, TypeError)):
            config.task_adaptation_epochs = 30  # type: ignore[misc]
        with self.assertRaises(TypeError):
            config.conditions["M1"] = config.conditions["M1"]  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
