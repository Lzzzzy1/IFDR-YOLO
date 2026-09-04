from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml

import scripts.run_p2_interaction_s0 as stage9_s0

from scripts.run_p2_interaction_s0 import (
    REGISTERED_SEEDS,
    build_screen_identity,
    resume_epoch_from_results,
    sync_mirror,
    _training_args,
    _validate_gradient_diagnostics,
    _train,
    _status,
    _completed_status_epoch,
    primary_checkpoint,
    run_screen,
    variant_components,
)
from ifdr_yolo.experiments.config import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/diagnostics/kitti_ifdr_p2_interaction_s0_s17.yaml"


class P2InteractionS0Test(unittest.TestCase):
    def test_stage9_seed0_candidate_selection_contract_is_narrow_and_not_diagnostic(self) -> None:
        contract = stage9_s0._execution_contract(
            variant="ifdr-p2-interaction-b",
            seed=0,
            mode="full",
            benchmark_context=None,
            execution_purpose="stage9_seed0_candidate_selection",
        )
        self.assertEqual(
            contract,
            {
                "diagnostic_only": False,
                "execution_purpose": "stage9_seed0_candidate_selection",
            },
        )
        for variant, seed, mode, benchmark_context in (
            ("ifdr-p2-interaction-c", 0, "full", None),
            ("ifdr-p2-interaction-b", 1, "full", None),
            ("ifdr-p2-interaction-b", 0, "smoke", None),
            ("ifdr-p2-interaction-b", 0, "full", object()),
        ):
            with self.subTest(variant=variant, seed=seed, mode=mode):
                with self.assertRaisesRegex(ValueError, "Stage9 candidate selection"):
                    stage9_s0._execution_contract(
                        variant=variant,
                        seed=seed,
                        mode=mode,
                        benchmark_context=benchmark_context,
                        execution_purpose="stage9_seed0_candidate_selection",
                    )

    def test_stage9_seed0_identity_is_formal_candidate_selection_not_diagnostic(self) -> None:
        identity = build_screen_identity(
            variant="ifdr-p2-interaction-b",
            seed=0,
            config_sha256="a" * 64,
            code_sha256="b" * 64,
            model_sha256="c" * 64,
            pretrained_sha256="d" * 64,
            fit_ids_sha256="e" * 64,
            development_ids_sha256="f" * 64,
            run_mode="full",
            execution_purpose="stage9_seed0_candidate_selection",
            diagnostic_only=False,
        )
        self.assertEqual(identity["execution_purpose"], "stage9_seed0_candidate_selection")
        self.assertFalse(identity["diagnostic_only"])
        self.assertEqual(identity["seed"], 0)
        with self.assertRaisesRegex(ValueError, "execution purpose"):
            build_screen_identity(
                variant="ifdr-p2-interaction-b",
                seed=0,
                config_sha256="a" * 64,
                code_sha256="b" * 64,
                model_sha256="c" * 64,
                pretrained_sha256="d" * 64,
                fit_ids_sha256="e" * 64,
                development_ids_sha256="f" * 64,
                run_mode="full",
                execution_purpose="stage9_seed0_candidate_selection",
                diagnostic_only=True,
            )

    def test_cli_forwards_explicit_stage9_candidate_selection_purpose(self) -> None:
        with patch("scripts.run_p2_interaction_s0.run_screen", return_value=None) as runner:
            result = stage9_s0.main(
                [
                    "--config",
                    "config.yaml",
                    "--fit-ids",
                    "fit.txt",
                    "--development-ids",
                    "development.txt",
                    "--output-dir",
                    "output",
                    "--mirror-dir",
                    "mirror",
                    "--mode",
                    "full",
                    "--execution-purpose",
                    "stage9_seed0_candidate_selection",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            runner.call_args.kwargs["execution_purpose"],
            "stage9_seed0_candidate_selection",
        )

    def test_completed_status_epoch_uses_benchmark_effective_epochs_only_with_context(self) -> None:
        context = SimpleNamespace(identity={"effective_epochs": 2})
        self.assertEqual(_completed_status_epoch(mode="full", benchmark_context=None), 30)
        self.assertEqual(_completed_status_epoch(mode="smoke", benchmark_context=None), 1)
        self.assertEqual(_completed_status_epoch(mode="full", benchmark_context=context), 2)

    def test_registered_seeds_have_matched_c_b_ab_configs_and_rng(self) -> None:
        self.assertEqual(REGISTERED_SEEDS, (17, 29, 41))
        for seed in REGISTERED_SEEDS:
            names = ("c", "b", "s0") if seed == 17 else ("c", "b", "ab")
            for name in names:
                with self.subTest(seed=seed, name=name):
                    payload = yaml.safe_load(
                        (CONFIG_PATH.parent / f"kitti_ifdr_p2_interaction_{name}_s{seed}.yaml").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(payload["experiment"]["seed"], seed)
                    self.assertEqual(payload["ifdr"]["intervention"]["base_seed"], seed)
                    self.assertEqual(payload["training"]["epochs"], 30)

    def test_config_locks_s0_budget_and_registered_components(self) -> None:
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment"]["variant"], "ifdr-p2-interaction-s0")
        self.assertEqual(payload["experiment"]["seed"], 17)
        self.assertEqual(payload["training"]["epochs"], 30)
        self.assertEqual(payload["ifdr"]["schedule"], {"frozen_epochs": 5, "ramp_epochs": 10})
        self.assertTrue(payload["ifdr"]["components"]["fusion_gate"])
        self.assertTrue(payload["ifdr"]["components"]["dcli"])
        self.assertFalse(payload["ifdr"]["components"]["semantic_protection"])
        self.assertFalse(payload["ifdr"]["components"]["counterfactual_consistency"])
        self.assertEqual(payload["ifdr"]["loss"]["counterfactual_gain"], 0.0)
        self.assertEqual(payload["ifdr"]["p2_path_switches"]["nodes"], [17])
        self.assertTrue(payload["ifdr"]["p2_path_switches"]["fusion_modulation"])
        self.assertTrue(payload["ifdr"]["p2_path_switches"]["dcli_factor_conditioning"])
        self.assertEqual(payload["paths"]["model_sha256"], "0d2cbd9215dc62c4c9920127964ce0ab61b5d10581861d8e83c33fc168aa302a")
        self.assertEqual(payload["initialization"]["pretrained_sha256"], "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5")

    def test_registered_configs_change_only_fusion_and_dcli_flags(self) -> None:
        expected = {
            "c": (False, False),
            "a": (True, False),
            "b": (False, True),
            "s0": (True, True),
        }
        common = None
        for name, flags in expected.items():
            payload = yaml.safe_load(
                (CONFIG_PATH.parent / f"kitti_ifdr_p2_interaction_{name}_s17.yaml").read_text(
                    encoding="utf-8"
                )
            )
            components = payload["ifdr"]["components"]
            self.assertEqual(
                (components["fusion_gate"], components["dcli"]),
                flags,
            )
            self.assertTrue(components["factor_supervision"])
            self.assertTrue(components["interventions"])
            self.assertEqual(payload["training"]["epochs"], 30)
            self.assertEqual(payload["ifdr"]["gradient_diagnostic_interval"], 50)
            self.assertEqual(payload["ifdr"]["schedule"], {"frozen_epochs": 5, "ramp_epochs": 10})
            self.assertEqual(payload["ifdr"]["p2_path_switches"], {"nodes": [17], "fusion_modulation": True, "dcli_factor_conditioning": True})
            comparable = dict(payload)
            comparable["experiment"] = dict(payload["experiment"])
            comparable["experiment"].pop("variant")
            comparable["ifdr"] = dict(payload["ifdr"])
            comparable["ifdr"]["components"] = dict(components)
            comparable["ifdr"]["components"].pop("fusion_gate")
            comparable["ifdr"]["components"].pop("dcli")
            if common is None:
                common = comparable
            else:
                self.assertEqual(comparable, common)

    def test_component_registry_exposes_c_a_b_ab_and_rejects_unknown(self) -> None:
        self.assertEqual(variant_components("ifdr-p2-interaction-c"), (False, False))
        self.assertEqual(variant_components("ifdr-p2-interaction-a"), (True, False))
        self.assertEqual(variant_components("ifdr-p2-interaction-b"), (False, True))
        self.assertEqual(variant_components("ifdr-p2-interaction-ab"), (True, True))
        self.assertEqual(variant_components("ifdr-p2-interaction-s0"), (True, True))
        with self.assertRaises(ValueError):
            variant_components("ifdr-p2-interaction-unknown")

    def test_identity_rejects_variant_component_mismatch(self) -> None:
        identity = build_screen_identity(
            variant="ifdr-p2-interaction-c",
            seed=17,
            config_sha256="a" * 64,
            code_sha256="b" * 64,
            model_sha256="c" * 64,
            pretrained_sha256="d" * 64,
            fit_ids_sha256="e" * 64,
            development_ids_sha256="f" * 64,
            fusion_gate=False,
            dcli=False,
        )
        self.assertFalse(identity["fusion_gate"])
        self.assertFalse(identity["dcli"])
        with self.assertRaises(ValueError):
            build_screen_identity(
                variant="ifdr-p2-interaction-c",
                seed=17,
                config_sha256="a" * 64,
                code_sha256="b" * 64,
                model_sha256="c" * 64,
                pretrained_sha256="d" * 64,
                fit_ids_sha256="e" * 64,
                development_ids_sha256="f" * 64,
                fusion_gate=True,
                dcli=False,
            )

    def test_status_binds_variant_component_flags(self) -> None:
        identity = build_screen_identity(
            variant="ifdr-p2-interaction-a",
            seed=17,
            config_sha256="a" * 64,
            code_sha256="b" * 64,
            model_sha256="c" * 64,
            pretrained_sha256="d" * 64,
            fit_ids_sha256="e" * 64,
            development_ids_sha256="f" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _status(output, identity, "prepared", epoch=0, next_action="run")
            payload = json.loads((output / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["variant"], "ifdr-p2-interaction-a")
        self.assertTrue(payload["fusion_gate"])
        self.assertFalse(payload["dcli"])
        self.assertEqual(payload["execution_purpose"], "diagnostic_screen")
        self.assertTrue(payload["diagnostic_only"])

    def test_run_screen_rejects_unregistered_diagnostic_interval_or_node(self) -> None:
        def fake_config(interval: int, nodes: tuple[int, ...]) -> SimpleNamespace:
            return SimpleNamespace(
                experiment=SimpleNamespace(variant="ifdr-p2-interaction-c", seed=17),
                paths=SimpleNamespace(
                    model=Path("model.yaml"),
                    model_sha256="a" * 64,
                ),
                initialization=SimpleNamespace(
                    pretrained=Path("pretrained.pt"),
                    pretrained_sha256="a" * 64,
                ),
                method=SimpleNamespace(
                    components=SimpleNamespace(
                        fusion_gate=False,
                        dcli=False,
                        factor_supervision=True,
                        interventions=True,
                        semantic_protection=False,
                        counterfactual_consistency=False,
                    ),
                    loss=SimpleNamespace(counterfactual_gain=0.0),
                    schedule=SimpleNamespace(frozen_epochs=5, ramp_epochs=10),
                    gradient_diagnostic_interval=interval,
                    p2_path_switches=SimpleNamespace(
                        nodes=nodes,
                        fusion_modulation=True,
                        dcli_factor_conditioning=True,
                    ),
                ),
                training=SimpleNamespace(epochs=30),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            fit_ids = root / "fit.txt"
            development_ids = root / "development.txt"
            config_path.write_text("{}\n", encoding="utf-8")
            fit_ids.write_text("fit\n", encoding="utf-8")
            development_ids.write_text("dev\n", encoding="utf-8")
            for config in (fake_config(49, (17,)), fake_config(50, (17, 20))):
                with self.subTest(
                    interval=config.method.gradient_diagnostic_interval,
                    nodes=config.method.p2_path_switches.nodes,
                ), self.assertRaises(ValueError):
                    with patch(
                        "scripts.run_p2_interaction_s0.load_ifdr_config",
                        return_value=config,
                    ), patch(
                        "scripts.run_p2_interaction_s0.sha256_file",
                        return_value="a" * 64,
                    ), patch(
                        "scripts.run_p2_interaction_s0._validate_split",
                        return_value=(("fit",), ("dev",)),
                    ), patch(
                        "scripts.run_p2_interaction_s0._materialize_views",
                    ), patch(
                        "scripts.run_p2_interaction_s0._write_resolved_data",
                        return_value=root / "data.yaml",
                    ), patch(
                        "scripts.run_p2_interaction_s0.sync_mirror",
                    ), patch(
                        "scripts.run_p2_interaction_s0._code_sha256",
                        return_value="b" * 64,
                    ), patch(
                        "scripts.run_p2_interaction_s0._git_commit",
                        return_value="c" * 40,
                    ):
                        run_screen(
                            config_path=config_path,
                            fit_ids=fit_ids,
                            development_ids=development_ids,
                            output_dir=root / "output",
                            mirror_dir=root / "mirror",
                            mode="dry-run",
                        )

    def test_identity_binds_split_config_code_and_pretrained(self) -> None:
        identity = build_screen_identity(
            variant="ifdr-p2-interaction-s0",
            seed=17,
            config_sha256="a" * 64,
            code_sha256="b" * 64,
            model_sha256="c" * 64,
            pretrained_sha256="d" * 64,
            fit_ids_sha256="e" * 64,
            development_ids_sha256="f" * 64,
        )
        self.assertEqual(identity["diagnostic_nodes"], [17, 20, 23, 26])
        self.assertEqual(identity["run_mode"], "full")
        self.assertTrue(identity["diagnostic_only"])
        self.assertEqual(identity["epochs"], 30)
        self.assertEqual(identity["frozen_epochs"], 5)
        self.assertEqual(identity["ramp_epochs"], 10)
        self.assertIsNone(identity["eta_seconds"])
        self.assertEqual(identity["eta_source"], "first remote smoke")
        unsigned = {key: value for key, value in identity.items() if key != "identity_sha256"}
        encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(identity["identity_sha256"], hashlib.sha256(encoded).hexdigest())
        smoke_identity = build_screen_identity(
            variant="ifdr-p2-interaction-s0",
            seed=17,
            config_sha256="a" * 64,
            code_sha256="b" * 64,
            model_sha256="c" * 64,
            pretrained_sha256="d" * 64,
            fit_ids_sha256="e" * 64,
            development_ids_sha256="f" * 64,
            run_mode="smoke",
        )
        self.assertNotEqual(identity["identity_sha256"], smoke_identity["identity_sha256"])

    def test_smoke_uses_real_640_geometry_and_formal_batch(self) -> None:
        training = TrainingConfig(
            epochs=30,
            imgsz=640,
            batch=16,
            workers=8,
            device="0",
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            patience=0,
            amp=True,
            deterministic=True,
            cache=False,
        )
        args = _training_args(
            SimpleNamespace(training=training, experiment=SimpleNamespace(seed=17)),
            data_path=Path("data.yaml"),
            output=Path("run"),
            device="0",
            mode="smoke",
        )
        self.assertEqual(args["epochs"], 1)
        self.assertEqual(args["imgsz"], 640)
        self.assertEqual(args["batch"], 16)
        self.assertTrue(args["amp"])
        self.assertEqual(args["save_period"], -1)

    def test_interrupted_results_resume_from_next_epoch_and_mirror_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            results.write_text("epoch,loss\n1,1.0\n2,0.9\n", encoding="utf-8")
            self.assertEqual(resume_epoch_from_results(results), 2)
            uninterrupted = "epoch,loss\n" + "".join(f"{epoch},{1.0 / epoch:.8f}\n" for epoch in range(1, 31))
            results.write_text(uninterrupted.split("\n", 3)[0] + "\n" + "".join(uninterrupted.splitlines()[1:2]) + "\n", encoding="utf-8")
            with results.open("a", encoding="utf-8") as output:
                for epoch in range(2, 31):
                    output.write(f"{epoch},{1.0 / epoch:.8f}\n")
            self.assertEqual(results.read_text(encoding="utf-8"), uninterrupted)
            primary = root / "primary"
            mirror = root / "mirror"
            primary.mkdir()
            (primary / "status.json").write_text('{"epoch": 2}\n', encoding="utf-8")
            (primary / "gradient_diagnostics.jsonl").write_text('{"step": 2}\n', encoding="utf-8")
            sync_mirror(primary, mirror)
            self.assertEqual(
                (mirror / "status.json").read_bytes(),
                (primary / "status.json").read_bytes(),
            )
            self.assertEqual(
                (mirror / "gradient_diagnostics.jsonl").read_bytes(),
                (primary / "gradient_diagnostics.jsonl").read_bytes(),
            )

    def test_sync_mirror_exactly_mirrors_prediction_labels_and_binds_them_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            mirror = root / "mirror"
            labels = primary / "predictions" / "labels"
            labels.mkdir(parents=True)
            expected = {
                "000001.txt": b"0 0.1 0.2 0.3 0.4 0.9\n",
                "000002.txt": b"1 0.5 0.6 0.7 0.8 0.8\n",
                "nested/000003.txt": b"0 0.2 0.3 0.4 0.5 0.7\n",
            }
            for relative, content in expected.items():
                target = labels / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            stale = mirror / "predictions" / "labels" / "obsolete.txt"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"obsolete\n")

            sync_mirror(primary, mirror)

            mirrored = mirror / "predictions" / "labels"
            self.assertEqual(
                sorted(path.relative_to(mirrored).as_posix() for path in mirrored.rglob("*.txt")),
                sorted(expected),
            )
            for relative, content in expected.items():
                self.assertEqual((mirrored / relative).read_bytes(), content)
            manifest = json.loads((mirror / "manifest.json").read_text(encoding="utf-8"))
            records = {record["path"]: record for record in manifest["files"]}
            for relative, content in expected.items():
                path = f"predictions/labels/{relative}"
                self.assertEqual(records[path]["size"], len(content))
                self.assertEqual(records[path]["sha256"], hashlib.sha256(content).hexdigest())

    def test_gradient_diagnostic_gate_requires_all_registered_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "gradient_diagnostics.jsonl").write_text(
                json.dumps({"node_diagnostics": {"nodes": {str(node): {} for node in (17, 20, 23, 26)}}}) + "\n",
                encoding="utf-8",
            )
            _validate_gradient_diagnostics(output)

    def test_primary_ap_checkpoint_is_last_not_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights"
            weights.mkdir()
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
            self.assertEqual(primary_checkpoint(Path(directory)), weights / "last.pt")

    def test_resume_trainer_receives_registered_gradient_diagnostic_interval(self) -> None:
        captured: dict[str, object] = {}

        class RecordingTrainer:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)
                self.output = Path(kwargs["overrides"]["save_dir"])

            def train(self) -> None:
                weights = self.output / "weights"
                weights.mkdir(parents=True, exist_ok=True)
                (weights / "best.pt").write_bytes(b"best")
                (weights / "last.pt").write_bytes(b"last")

        fake_runtime = ModuleType("ifdr_yolo.experiments.ifdr_runtime")
        fake_runtime.IFDRRuntimeAdapter = object
        fake_trainer = ModuleType("ifdr_yolo.experiments.ifdr_trainer")
        fake_trainer.IFDRComponentSwitches = lambda **kwargs: kwargs
        fake_trainer.FusionSchedule = lambda **kwargs: kwargs
        fake_trainer.IFDRDetectionTrainer = RecordingTrainer
        fake_sampler = ModuleType("ifdr_yolo.data.interventions.sampler")
        fake_sampler.SamplingPolicy = lambda **kwargs: kwargs
        method = SimpleNamespace(
            components=SimpleNamespace(
                fusion_gate=True,
                dcli=True,
                factor_supervision=True,
                interventions=True,
                semantic_protection=False,
                counterfactual_consistency=False,
            ),
            intervention=SimpleNamespace(
                base_seed=17,
                identity_probability=0.2,
                sampling_probability=0.4,
                visibility_probability=0.4,
                minimum_strength=0.1,
                maximum_strength=0.8,
            ),
            schedule=SimpleNamespace(frozen_epochs=5, ramp_epochs=10),
            gradient_diagnostic_interval=50,
        )
        config = SimpleNamespace(method=method, training=SimpleNamespace(workers=8))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.dict(
                sys.modules,
                {
                    "ifdr_yolo.experiments.ifdr_runtime": fake_runtime,
                    "ifdr_yolo.experiments.ifdr_trainer": fake_trainer,
                    "ifdr_yolo.data.interventions.sampler": fake_sampler,
                },
            ):
                _train(
                    config,
                    output,
                    Path("data.yaml"),
                    device="0",
                    mode="full",
                    resume=True,
                )
        self.assertEqual(captured["gradient_diagnostic_interval"], 50)


if __name__ == "__main__":
    unittest.main()
