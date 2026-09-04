from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import yaml

from ifdr_yolo.experiments.kitti_seed0_training_benchmark import (
    BenchmarkIdentityError,
    BenchmarkEpochHook,
    BenchmarkEpochResumeHook,
    DATALOADER_EPOCH_SEED_BASE,
    BenchmarkInterrupted,
    BenchmarkRunContext,
    build_benchmark_state,
    build_benchmark_identity,
    canonical_checkpoint_digest,
    capture_rng_state,
    compare_recovery,
    publish_generation,
    prepare_resume_checkpoint,
    reconcile_common_generation,
    load_preflight_pair,
    run_registered_benchmark_stage,
    run_preflight,
    registered_arm_for_config_path,
    restore_rng_state,
    restore_benchmark_training_state,
    verify_ultralytics_callback_contract,
    configure_benchmark_callbacks,
    clean_git_identity,
    optimizer_step_offset_for_epoch,
    run_synthetic_recovery_probe,
)
from ifdr_yolo.experiments.config import load_baseline_config, load_ifdr_config
from scripts.run_p2_interaction_s0 import _training_args
from scripts.run_p2_interaction_s0 import build_screen_identity
from scripts.run_kitti_seed0_training_benchmark import main as benchmark_cli_main


ROOT = Path(__file__).resolve().parents[1]
P3P5 = ROOT / "configs/experiments/selection/kitti_p3p5_control_s0.yaml"
DCLI = ROOT / "configs/experiments/selection/kitti_dcli_s0.yaml"
DCLI_PARENT = ROOT / "configs/experiments/diagnostics/kitti_ifdr_p2_interaction_b_s17.yaml"


class _WorkerRandomnessDataset:
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> tuple[int, int, int, int, int]:
        import numpy as np
        import random
        from torch.utils.data import get_worker_info
        import torch
        return index, get_worker_info().id if get_worker_info() else -1, random.randrange(1 << 20), int(np.random.randint(1 << 20)), int(torch.randint(1 << 20, ()).item())


class _EpochGenerator:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def manual_seed(self, seed: int) -> "_EpochGenerator":
        self.seed = seed
        return self


class _EpochLoader:
    """Small loader model: a fresh resumed iterator otherwise replays epoch 0."""

    def __init__(self, seed: int) -> None:
        self.generator = _EpochGenerator(seed)
        self.iterator_seed = seed
        self.reset_calls = 0

    def reset(self) -> None:
        self.iterator_seed = self.generator.seed
        self.reset_calls += 1

    def __len__(self) -> int:
        return 3

    def epoch_trajectory(self) -> tuple[tuple[int, int, int], ...]:
        seed = self.iterator_seed
        self.iterator_seed += 1
        return tuple((seed + offset, (seed * 17 + offset) % 997, (seed * 31 + offset) % 991) for offset in range(3))


class KittiSeed0TrainingBenchmarkTest(unittest.TestCase):
    def test_preflight_rejects_relevant_dirty_git_before_publication(self) -> None:
        replies = (
            SimpleNamespace(returncode=0, stdout="a" * 40),
            SimpleNamespace(returncode=0, stdout="diff"),
            SimpleNamespace(returncode=0, stdout=" M ifdr_yolo/experiments/config.py\n"),
        )
        with patch("ifdr_yolo.experiments.kitti_seed0_training_benchmark.subprocess.run", side_effect=replies):
            with self.assertRaisesRegex(BenchmarkIdentityError, "clean tracked Git state"):
                clean_git_identity(ROOT, ("ifdr_yolo/experiments/config.py",), P3P5)

    def _write_preflight_pair(self, root: Path, *, arm: str, role: str) -> tuple[Path, Path, dict[str, object]]:
        effective_epochs = 1 if role == "timing_one_epoch" else 2
        identity = build_benchmark_identity(
            arm=arm, execution_role=role, config_sha256="a" * 64,
            code_sha256="b" * 64, fit_ids_sha256="c" * 64,
            development_ids_sha256="d" * 64, model_sha256="e" * 64,
            pretrained_sha256="f" * 64, effective_epochs=effective_epochs,
        )
        payload = {"preflight_state": "PASS", "benchmark_launch_authorized": True,
                   "training_authorized": False, "identity": identity}
        content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        manifest = {"schema_version": 1, "identity_sha256": identity["identity_sha256"],
                    "files": [{"name": "preflight_identity.json", "sha256": hashlib.sha256(content).hexdigest()}]}
        manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        primary, mirror = root / "preflight", root / "preflight-mirror"
        for side in (primary, mirror):
            side.mkdir()
            (side / "preflight_identity.json").write_bytes(content)
            (side / "manifest.json").write_bytes(manifest_bytes)
        return primary, mirror, identity

    def test_real_stage_consumes_paired_preflight_and_never_expands_epoch_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight, preflight_mirror, identity = self._write_preflight_pair(root, arm="DCLI", role="timing_one_epoch")
            self.assertEqual(load_preflight_pair(preflight, preflight_mirror, arm="DCLI", execution_role="timing_one_epoch"), identity)
            with patch("scripts.run_p2_interaction_s0.run_screen", return_value=root / "out" / "metrics_ap40.json") as run:
                result = run_registered_benchmark_stage(
                    arm="DCLI", execution_role="timing_one_epoch", config_path=DCLI,
                    fit_ids=root / "fit", development_ids=root / "dev",
                    repository_root=ROOT, output_dir=root / "out", mirror_dir=root / "mirror",
                    preflight_dir=preflight, preflight_mirror_dir=preflight_mirror,
                    device="0", resume=False,
                )
            self.assertEqual(result, root / "out" / "metrics_ap40.json")
            context = run.call_args.kwargs["benchmark_context"]
            self.assertEqual(context.identity["effective_epochs"], 1)
            self.assertIsNone(context.stop_after_epoch)
            self.assertEqual(run.call_args.kwargs["mode"], "full")

    def test_real_recovery_stop_stage_binds_epoch_one_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight, preflight_mirror, _ = self._write_preflight_pair(root, arm="DCLI", role="recovery_interrupted_two_epoch")
            with patch("scripts.run_p2_interaction_s0.run_screen", return_value=None) as run:
                run_registered_benchmark_stage(
                    arm="DCLI", execution_role="recovery_interrupted_two_epoch", config_path=DCLI,
                    fit_ids=root / "fit", development_ids=root / "dev",
                    repository_root=ROOT, output_dir=root / "out", mirror_dir=root / "mirror",
                    preflight_dir=preflight, preflight_mirror_dir=preflight_mirror,
                    device="0", resume=False, stop_after_epoch=1,
                )
            self.assertEqual(run.call_args.kwargs["benchmark_context"].stop_after_epoch, 1)

    def test_cli_real_timing_dispatches_only_the_registered_one_epoch_role(self) -> None:
        args = [
            "timing", "--arm", "DCLI", "--execution-role", "timing_one_epoch",
            "--config", str(DCLI), "--fit-ids", "fit", "--development-ids", "dev",
            "--output-dir", "out", "--mirror-dir", "mirror",
            "--preflight-dir", "pre", "--preflight-mirror-dir", "pre-mirror",
        ]
        with patch("scripts.run_kitti_seed0_training_benchmark.run_registered_benchmark_stage") as run:
            self.assertEqual(benchmark_cli_main(args), 0)
        self.assertEqual(run.call_args.kwargs["execution_role"], "timing_one_epoch")
        self.assertFalse(run.call_args.kwargs["resume"])
        self.assertIsNone(run.call_args.kwargs["stop_after_epoch"])
    def test_benchmark_context_rejects_invalid_role_epoch_and_stop(self) -> None:
        identity = build_benchmark_identity(arm="DCLI", execution_role="timing_one_epoch", config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=1)
        with self.assertRaisesRegex(BenchmarkIdentityError, "stop"):
            BenchmarkRunContext(identity=identity, primary_root=Path("p"), mirror_root=Path("m"), stop_after_epoch=1)
        context = BenchmarkRunContext(identity=identity, primary_root=Path("p"), mirror_root=Path("m"))
        class Target:
            def __init__(self) -> None: self.calls = []
            def add_callback(self, event: str, callback: object) -> None: self.calls.append((event, callback))
        target = Target()
        configure_benchmark_callbacks(target, context, resume=False)
        self.assertEqual([name for name, _ in target.calls], ["on_train_epoch_start", "on_train_batch_start", "on_train_batch_end", "on_model_save"])

    def test_epoch_loader_policy_prevents_fresh_resume_from_replaying_epoch_zero(self) -> None:
        identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_uninterrupted_two_epoch", config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=2)

        uninterrupted_loader = _EpochLoader(101)
        uninterrupted_loader.epoch_trajectory()
        uninterrupted_epoch_two = uninterrupted_loader.epoch_trajectory()
        fresh_resume_epoch_two = _EpochLoader(101).epoch_trajectory()
        self.assertNotEqual(uninterrupted_epoch_two, fresh_resume_epoch_two)

        class Target:
            def __init__(self) -> None:
                self.calls: list[tuple[str, object]] = []
                self.callbacks = {"on_train_epoch_start": [lambda trainer: None]}

            def add_callback(self, event: str, callback: object) -> None:
                self.calls.append((event, callback))
                self.callbacks.setdefault(event, []).append(callback)

        target = Target()
        context = BenchmarkRunContext(identity=identity, primary_root=Path("p"), mirror_root=Path("m"))
        configure_benchmark_callbacks(target, context, resume=True)
        self.assertEqual([event for event, _ in target.calls], ["on_train_start", "on_train_epoch_start", "on_train_batch_start", "on_train_batch_end", "on_model_save"])
        epoch_reset = target.callbacks["on_train_epoch_start"][-1]
        batch_start = target.callbacks["on_train_batch_start"][-1]
        batch_end = target.callbacks["on_train_batch_end"][-1]

        uninterrupted_loader = _EpochLoader(101)
        trainer = SimpleNamespace(epoch=0, train_loader=uninterrupted_loader, args=SimpleNamespace(warmup_epochs=3.0, nbs=64), batch_size=16)
        epoch_reset(trainer)
        uninterrupted_loader.epoch_trajectory()
        trainer.epoch = 1
        epoch_reset(trainer)
        batch_start(trainer)
        batch_end(trainer)
        uninterrupted_epoch_two = uninterrupted_loader.epoch_trajectory()

        resumed_loader = _EpochLoader(101)
        resumed_trainer = SimpleNamespace(epoch=1, train_loader=resumed_loader, args=SimpleNamespace(warmup_epochs=3.0, nbs=64), batch_size=16)
        epoch_reset(resumed_trainer)
        batch_start(resumed_trainer)
        batch_end(resumed_trainer)
        self.assertEqual(uninterrupted_epoch_two, resumed_loader.epoch_trajectory())
        self.assertEqual((uninterrupted_loader.reset_calls, resumed_loader.reset_calls), (2, 1))

    def test_resume_restores_lossless_live_training_state_not_fp16_ordinary_checkpoint(self) -> None:
        import torch
        from ultralytics.utils.torch_utils import ModelEMA

        class StateHolder:
            def __init__(self, value: float) -> None:
                self.value = value

            def state_dict(self) -> dict[str, float]:
                return {"value": self.value}

            def load_state_dict(self, state: dict[str, float]) -> None:
                self.value = state["value"]

        source_model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            source_model.weight.fill_(1.0001234); source_model.bias.fill_(2.0002345)
        source_ema = ModelEMA(source_model)
        with torch.no_grad():
            source_ema.ema.weight.fill_(3.0003456); source_ema.ema.bias.fill_(4.0004567)
        source_ema.updates = 17
        source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1, momentum=0.9)
        source_model(torch.ones(1, 2)).sum().backward(); source_optimizer.step(); source_optimizer.zero_grad()
        source = SimpleNamespace(
            epoch=0,
            model=source_model,
            ema=source_ema,
            optimizer=source_optimizer,
            scaler=StateHolder(8.0),
            scheduler=StateHolder(3.0),
            args=SimpleNamespace(seed=0, epochs=2),
        )
        state = build_benchmark_state(source)

        target_model = torch.nn.Linear(2, 1).half().float()
        target_ema = ModelEMA(target_model)
        target_ema.updates = 0
        target = SimpleNamespace(
            start_epoch=1,
            model=target_model,
            ema=target_ema,
            optimizer=torch.optim.SGD(target_model.parameters(), lr=9.0, momentum=0.9),
            scaler=StateHolder(1.0),
            scheduler=StateHolder(1.0),
        )
        restore_benchmark_training_state(target, state)
        self.assertEqual(target.ema.updates, 17)
        restored = build_benchmark_state(SimpleNamespace(epoch=0, args=source.args, **vars(target)))
        self.assertEqual(canonical_checkpoint_digest(state), canonical_checkpoint_digest(restored))
        with torch.no_grad():
            source.model.weight.add_(0.01); target.model.weight.add_(0.01)
        source.ema.update(source.model); target.ema.update(target.model)
        for left, right in zip(source.ema.ema.parameters(), target.ema.ema.parameters()):
            self.assertTrue(torch.equal(left, right))
        target.start_epoch = 2
        with self.assertRaisesRegex(BenchmarkIdentityError, "start epoch"):
            restore_benchmark_training_state(target, state)

    def test_epoch_loader_policy_fails_closed_without_generator_or_reset(self) -> None:
        identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_uninterrupted_two_epoch", config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=2)

        class Target:
            def __init__(self) -> None:
                self.callbacks: dict[str, list[object]] = {}

            def add_callback(self, event: str, callback: object) -> None:
                self.callbacks.setdefault(event, []).append(callback)

        target = Target()
        configure_benchmark_callbacks(target, BenchmarkRunContext(identity=identity, primary_root=Path("p"), mirror_root=Path("m")), resume=False)
        epoch_reset = target.callbacks["on_train_epoch_start"][-1]
        with self.assertRaisesRegex(BenchmarkIdentityError, "generator"):
            epoch_reset(SimpleNamespace(epoch=0, train_loader=SimpleNamespace(reset=lambda: None)))
        with self.assertRaisesRegex(BenchmarkIdentityError, "reset"):
            epoch_reset(SimpleNamespace(epoch=0, train_loader=SimpleNamespace(generator=_EpochGenerator(1))))

    def test_resume_accumulation_policy_matches_uninterrupted_epoch_two_first_step(self) -> None:
        nb, warmup_epochs, batch_size, default_nbs = 209, 3.0, 16, 64
        warmup_steps = max(round(warmup_epochs * nb), 100)

        def accumulate(ni: int, nbs: int) -> int:
            return max(1, int(round(1 + ni / warmup_steps * (nbs / batch_size - 1))))

        def epoch_steps(*, epoch: int, prior_last_step: int, hook: BenchmarkEpochResumeHook | None) -> list[int]:
            trainer = SimpleNamespace(epoch=1, train_loader=SimpleNamespace(__len__=lambda: nb), args=SimpleNamespace(warmup_epochs=warmup_epochs, nbs=default_nbs), batch_size=batch_size)
            # Special method lookup ignores instance __len__; use a compact loader class for the real callback.
            trainer.train_loader = type("Loader", (), {"__len__": lambda self: nb, "generator": _EpochGenerator(1), "reset": lambda self: None})()
            trainer.epoch = epoch
            trainer.accumulate = 4
            if hook is not None:
                hook.on_train_epoch_start(trainer)
            last_step, steps = prior_last_step, []
            for index in range(nb):
                if hook is not None:
                    hook.on_train_batch_start(trainer)
                ni = nb * epoch + index
                trainer.accumulate = accumulate(ni, trainer.args.nbs) if ni <= warmup_steps else trainer.accumulate
                if ni - last_step >= trainer.accumulate:
                    steps.append(index)
                    last_step = ni
                if hook is not None:
                    hook.on_train_batch_end(trainer)
            return steps

        epoch_zero_last_step = 208
        self.assertEqual(optimizer_step_offset_for_epoch(epoch=1, batches_per_epoch=nb, warmup_epochs=warmup_epochs, nominal_batch_size=default_nbs, batch_size=batch_size), 1)
        self.assertEqual(optimizer_step_offset_for_epoch(epoch=3, batches_per_epoch=nb, warmup_epochs=warmup_epochs, nominal_batch_size=default_nbs, batch_size=batch_size), 3)
        self.assertEqual(optimizer_step_offset_for_epoch(epoch=4, batches_per_epoch=nb, warmup_epochs=warmup_epochs, nominal_batch_size=default_nbs, batch_size=batch_size), 2)
        self.assertEqual(optimizer_step_offset_for_epoch(epoch=10, batches_per_epoch=nb, warmup_epochs=warmup_epochs, nominal_batch_size=default_nbs, batch_size=batch_size), 0)
        self.assertEqual(optimizer_step_offset_for_epoch(epoch=29, batches_per_epoch=nb, warmup_epochs=warmup_epochs, nominal_batch_size=default_nbs, batch_size=batch_size), 1)
        self.assertEqual(epoch_steps(epoch=1, prior_last_step=epoch_zero_last_step, hook=None)[0], 1)
        self.assertEqual(epoch_steps(epoch=1, prior_last_step=-1, hook=None)[0], 0)

        uninterrupted_hook = BenchmarkEpochResumeHook()
        resumed_hook = BenchmarkEpochResumeHook()
        self.assertEqual(epoch_steps(epoch=1, prior_last_step=epoch_zero_last_step, hook=uninterrupted_hook)[0], 1)
        self.assertEqual(epoch_steps(epoch=1, prior_last_step=-1, hook=resumed_hook)[0], 1)
        self.assertEqual(epoch_steps(epoch=4, prior_last_step=-1, hook=BenchmarkEpochResumeHook())[0], 2)
        self.assertEqual(epoch_steps(epoch=29, prior_last_step=-1, hook=BenchmarkEpochResumeHook())[0], 1)

    def test_resume_accumulation_policy_fails_closed_for_unpaired_or_nonwarmup_first_batch(self) -> None:
        loader = type("Loader", (), {"__len__": lambda self: 209, "generator": _EpochGenerator(1), "reset": lambda self: None})()
        trainer = SimpleNamespace(epoch=1, train_loader=loader, args=SimpleNamespace(warmup_epochs=3.0, nbs=64), batch_size=16)
        hook = BenchmarkEpochResumeHook()
        hook.on_train_epoch_start(trainer)
        hook.on_train_batch_start(trainer)
        with self.assertRaisesRegex(BenchmarkIdentityError, "unpaired"):
            hook.on_train_batch_start(trainer)
        hook.on_train_batch_end(trainer)
        self.assertEqual(trainer.args.nbs, 64)
        nonwarmup = SimpleNamespace(epoch=1, train_loader=loader, args=SimpleNamespace(warmup_epochs=0.0, nbs=64), batch_size=16)
        self.assertEqual(optimizer_step_offset_for_epoch(epoch=1, batches_per_epoch=209, warmup_epochs=0.0, nominal_batch_size=64, batch_size=16), 2)

    def test_real_infinite_loader_epoch_reset_reproduces_order_and_worker_rng(self) -> None:
        import torch
        from ultralytics.data.build import InfiniteDataLoader

        def loader() -> object:
            return InfiniteDataLoader(_WorkerRandomnessDataset(), batch_size=2, shuffle=True, num_workers=2,
                                      generator=torch.Generator().manual_seed(DATALOADER_EPOCH_SEED_BASE))

        def epoch_batches(active_loader: object) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
            return tuple(tuple(tuple(int(value) for value in field) for field in batch) for batch in active_loader)

        uninterrupted = loader()
        untouched_epoch_zero = epoch_batches(uninterrupted)
        uncontrolled_epoch_one = epoch_batches(uninterrupted)
        fresh_resume = loader()
        self.assertNotEqual(uncontrolled_epoch_one, epoch_batches(fresh_resume))
        reset_train_loader = BenchmarkEpochResumeHook()
        trainer = SimpleNamespace(epoch=0, train_loader=uninterrupted, args=SimpleNamespace(warmup_epochs=3.0, nbs=64), batch_size=2)
        reset_train_loader.on_train_epoch_start(trainer)
        self.assertEqual(untouched_epoch_zero, epoch_batches(uninterrupted))
        reset_train_loader.on_train_batch_start(trainer); reset_train_loader.on_train_batch_end(trainer)
        trainer.epoch = 1
        reset_train_loader.on_train_epoch_start(trainer)
        reset_train_loader.on_train_batch_start(trainer); reset_train_loader.on_train_batch_end(trainer)
        controlled_epoch_one = epoch_batches(uninterrupted)
        resumed = loader()
        resumed_trainer = SimpleNamespace(epoch=1, train_loader=resumed, args=SimpleNamespace(warmup_epochs=3.0, nbs=64), batch_size=2)
        resumed_hook = BenchmarkEpochResumeHook()
        resumed_hook.on_train_epoch_start(resumed_trainer)
        resumed_hook.on_train_batch_start(resumed_trainer); resumed_hook.on_train_batch_end(resumed_trainer)
        self.assertEqual(controlled_epoch_one, epoch_batches(resumed))

    def test_dcli_context_uses_effective_epoch_without_changing_formal_geometry(self) -> None:
        config = load_ifdr_config(DCLI, repository_root=ROOT)
        identity = build_benchmark_identity(arm="DCLI", execution_role="timing_one_epoch", config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=1)
        context = BenchmarkRunContext(identity=identity, primary_root=Path("p"), mirror_root=Path("m"))
        args = _training_args(config, data_path=Path("data.yaml"), output=Path("out"), device="0", mode="full", benchmark_context=context)
        self.assertEqual((args["epochs"], args["imgsz"], args["batch"], args["workers"], args["amp"]), (1, 640, 16, 8, True))
        self.assertEqual(args["close_mosaic"], 0)
        self.assertEqual(config.training.epochs, 30)

    def test_screen_identity_changes_when_benchmark_role_changes(self) -> None:
        common = dict(variant="ifdr-p2-interaction-b", seed=0, config_sha256="a" * 64, code_sha256="b" * 64, model_sha256="c" * 64, pretrained_sha256="d" * 64, fit_ids_sha256="e" * 64, development_ids_sha256="f" * 64, fusion_gate=False, dcli=True, benchmark_identity_sha256="1" * 64, benchmark_effective_epochs=2)
        one = build_screen_identity(**common, benchmark_execution_role="recovery_uninterrupted_two_epoch")
        two = build_screen_identity(**common, benchmark_execution_role="recovery_interrupted_two_epoch")
        self.assertNotEqual(one["identity_sha256"], two["identity_sha256"])
    def test_fixed_ultralytics_callback_source_is_bound(self) -> None:
        contract = verify_ultralytics_callback_contract()
        self.assertEqual(contract["version"], "8.4.98")
        self.assertEqual(contract["trainer_sha256"], "d98009b8d9acfc61fde8941e8b029990da53757dfe7ac2a946d771860c754c1d")
    def test_seed0_configs_freeze_registered_protocol_and_only_method_difference(self) -> None:
        p3p5 = yaml.safe_load(P3P5.read_text(encoding="utf-8"))
        dcli = yaml.safe_load(DCLI.read_text(encoding="utf-8"))
        for payload in (p3p5, dcli):
            self.assertEqual(payload["experiment"]["seed"], 0)
            self.assertEqual(payload["training"]["epochs"], 30)
            self.assertEqual(payload["training"]["imgsz"], 640)
            self.assertEqual(payload["training"]["batch"], 16)
            self.assertEqual(payload["training"]["workers"], 8)
            self.assertTrue(payload["training"]["amp"])
            self.assertTrue(payload["training"]["deterministic"])
            self.assertEqual(payload["prediction"], {"conf": 0.001, "iou": 0.7, "max_det": 300, "half": False})
        self.assertEqual(registered_arm_for_config_path(P3P5), "P3P5_CONTROL")
        self.assertEqual(registered_arm_for_config_path(DCLI), "DCLI")
        self.assertNotIn("benchmark", p3p5)
        self.assertNotIn("benchmark", dcli)
        self.assertEqual(p3p5["experiment"]["variant"], "baseline")
        self.assertEqual(dcli["experiment"]["variant"], "ifdr-p2-interaction-b")
        self.assertFalse(dcli["ifdr"]["components"]["fusion_gate"])
        self.assertTrue(dcli["ifdr"]["components"]["dcli"])
        self.assertEqual(dcli["ifdr"]["intervention"]["base_seed"], 0)

    def test_seed0_configs_load_through_the_formal_schema_loaders(self) -> None:
        baseline = load_baseline_config(P3P5, repository_root=ROOT)
        ifdr = load_ifdr_config(DCLI, repository_root=ROOT)
        self.assertEqual(baseline.experiment.seed, 0)
        self.assertEqual(ifdr.experiment.seed, 0)

    def test_dcli_seed0_loader_semantics_differ_from_frozen_parent_only_in_seeds(self) -> None:
        parent = asdict(load_ifdr_config(DCLI_PARENT, repository_root=ROOT))
        candidate = asdict(load_ifdr_config(DCLI, repository_root=ROOT))
        parent.pop("source_path")
        candidate.pop("source_path")

        def leaves(value: object, prefix: str = "") -> dict[str, object]:
            if isinstance(value, dict):
                return {
                    path: leaf
                    for key, item in value.items()
                    for path, leaf in leaves(item, f"{prefix}.{key}" if prefix else key).items()
                }
            return {prefix: value}

        parent_leaves, candidate_leaves = leaves(parent), leaves(candidate)
        changed = {
            path: (parent_leaves.get(path), candidate_leaves.get(path))
            for path in sorted(set(parent_leaves) | set(candidate_leaves))
            if parent_leaves.get(path) != candidate_leaves.get(path)
        }
        self.assertEqual(
            changed,
            {
                "experiment.seed": (17, 0),
                "method.intervention.base_seed": (17, 0),
            },
        )

    def test_identity_fails_closed_on_wrong_arm_seed_or_effective_epochs(self) -> None:
        kwargs = dict(
            arm="DCLI", execution_role="timing_one_epoch", config_sha256="a" * 64,
            code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64,
            model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=1,
        )
        identity = build_benchmark_identity(**kwargs)
        self.assertEqual(identity["seed"], 0)
        for field, value in (("arm", "R"), ("seed", 1), ("effective_epochs", 2)):
            with self.subTest(field=field), self.assertRaises(BenchmarkIdentityError):
                build_benchmark_identity(**{**kwargs, field: value})

    def test_real_torch_checkpoint_digest_and_rng_restore_are_exact_and_nonfinite_fails_closed(self) -> None:
        import random
        import torch

        checkpoint = {
            "completed_epoch": 1, "model": {"w": torch.tensor([1.0])}, "ema": {"w": torch.tensor([2.0])}, "ema_updates": 17,
            "optimizer": {"state": {0: {"momentum": torch.tensor([3.0])}}}, "scaler": {},
            "scheduler": {}, "rng_state": b"rng", "train_args": {"seed": 0},
        }
        digest = canonical_checkpoint_digest(checkpoint)
        self.assertEqual(digest, canonical_checkpoint_digest(checkpoint))
        checkpoint["model"] = {"w": torch.tensor([float("nan")])}
        with self.assertRaisesRegex(BenchmarkIdentityError, "nonfinite"):
            canonical_checkpoint_digest(checkpoint)
        checkpoint["model"] = {"w": torch.tensor([1.0])}
        for invalid_updates in (True, -1, 1.5, "17"):
            with self.subTest(ema_updates=invalid_updates), self.assertRaisesRegex(BenchmarkIdentityError, "EMA update"):
                canonical_checkpoint_digest({**checkpoint, "ema_updates": invalid_updates})
        random.seed(0)
        torch.manual_seed(0)
        state = capture_rng_state()
        expected = (random.random(), torch.rand(1).item())
        random.seed(999)
        torch.manual_seed(999)
        restore_rng_state(state)
        self.assertEqual((random.random(), torch.rand(1).item()), expected)

    def test_benchmark_state_uses_live_trainer_fields_and_completed_epoch(self) -> None:
        import torch
        class EmaWrapper:
            def __init__(self) -> None:
                self.ema = torch.nn.Linear(1, 1)
                self.updates = 0
        class Trainer:
            epoch = 0
            model = torch.nn.Linear(1, 1)
            ema = EmaWrapper()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            args = {"seed": 0}
        state = build_benchmark_state(Trainer())
        self.assertEqual(state["completed_epoch"], 1)
        self.assertEqual(set(state), {"completed_epoch", "model", "ema", "ema_updates", "optimizer", "scaler", "scheduler", "rng_state", "train_args"})
        self.assertEqual(canonical_checkpoint_digest(state), canonical_checkpoint_digest(state))

    def test_benchmark_state_reads_model_ema_wrapper_inner_module(self) -> None:
        import torch
        class EmaWrapper:
            def __init__(self) -> None:
                self.ema = torch.nn.Linear(1, 1)
                self.updates = 9
        class Trainer:
            epoch = 0; model = torch.nn.Linear(1, 1); ema = EmaWrapper()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            args = {"seed": 0}
        state = build_benchmark_state(Trainer())
        self.assertIn("weight", state["ema"])

    def test_publish_generation_is_manifest_last_and_mirror_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            identity = build_benchmark_identity(
                arm="DCLI", execution_role="recovery_uninterrupted_two_epoch",
                config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64,
                development_ids_sha256="d" * 64, model_sha256="e" * 64,
                pretrained_sha256="f" * 64, effective_epochs=2,
            )
            checkpoint = {"completed_epoch": 1, "model": {"w": b"one"}, "ema": {"w": b"one"}, "ema_updates": 1, "optimizer": {"step": 1}, "scaler": {}, "scheduler": {}, "rng_state": b"rng", "train_args": {"seed": 0}}
            import torch
            archive_buffer = __import__("io").BytesIO()
            torch.save({"epoch": 0, "model": {}, "ema": None, "optimizer": {}, "scaler": {}, "train_args": {}}, archive_buffer)
            archive = archive_buffer.getvalue()
            with self.assertRaisesRegex(OSError, "mirror"):
                publish_generation(primary, mirror, identity, epoch=1, checkpoint_archive=archive, benchmark_state=checkpoint, results_csv="epoch,loss,time\n1,1.0,9\n", diagnostics={"epoch": 1}, fail_mirror=True)
            self.assertFalse((primary / "generations" / "1" / "manifest.json").exists())
            published = publish_generation(primary, mirror, identity, epoch=1, checkpoint_archive=archive, benchmark_state=checkpoint, results_csv="epoch,loss,time\n1,1.0,9\n", diagnostics={"epoch": 1})
            self.assertTrue((primary / "generations" / "1" / "manifest.json").is_file())
            self.assertTrue((mirror / "generations" / "1" / "manifest.json").is_file())
            self.assertEqual((primary / "generations" / "1" / "last.pt").read_bytes(), archive)
            self.assertEqual(torch.load(primary / "generations" / "1" / "last.pt", map_location="cpu", weights_only=False)["epoch"], 0)
            self.assertEqual(published["checkpoint_digest"], canonical_checkpoint_digest(checkpoint))
            wrong_identity = {**identity, "identity_sha256": "0" * 64}
            with self.assertRaisesRegex(BenchmarkIdentityError, "output reuse"):
                publish_generation(primary, mirror, wrong_identity, epoch=1, checkpoint_archive=archive, benchmark_state=checkpoint, results_csv="epoch,loss,time\n1,1.0,9\n", diagnostics={"epoch": 1})

    def test_model_save_hook_publishes_both_manifests_before_stop(self) -> None:
        from ultralytics.utils.torch_utils import ModelEMA

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_interrupted_two_epoch", config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=2)
            class Trainer:
                epoch = 0
                save_dir = root / "run"
                stop = False
            trainer = Trainer()
            (trainer.save_dir / "weights").mkdir(parents=True)
            import torch
            trainer.model = torch.nn.Linear(1, 1); trainer.ema = ModelEMA(trainer.model)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
            trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
            trainer.args = {"seed": 0}
            torch.save({"epoch": 0, "model": trainer.model, "ema": trainer.ema.ema, "optimizer": trainer.optimizer.state_dict(), "scaler": trainer.scaler.state_dict(), "train_args": trainer.args}, trainer.save_dir / "weights" / "last.pt")
            (trainer.save_dir / "results.csv").write_text("epoch,loss,time\n1,1,1\n", encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkInterrupted, "durable epoch 1"):
                BenchmarkEpochHook(root / "primary", root / "mirror", identity, stop_after_epoch=1).on_model_save(trainer)
            self.assertTrue((root / "primary" / "generations" / "1" / "manifest.json").is_file())
            self.assertTrue((root / "mirror" / "generations" / "1" / "manifest.json").is_file())
            self.assertFalse(trainer.stop)

    def test_model_save_hook_mirror_failure_never_requests_stop(self) -> None:
        from ultralytics.utils.torch_utils import ModelEMA

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_interrupted_two_epoch", config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=2)
            class Trainer:
                epoch = 0
                save_dir = root / "run"
                stop = False
            trainer = Trainer()
            (trainer.save_dir / "weights").mkdir(parents=True)
            import torch
            trainer.model = torch.nn.Linear(1, 1); trainer.ema = ModelEMA(trainer.model)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
            trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
            trainer.args = {"seed": 0}
            torch.save({"epoch": 0, "model": trainer.model, "ema": trainer.ema.ema, "optimizer": trainer.optimizer.state_dict(), "scaler": trainer.scaler.state_dict(), "train_args": trainer.args}, trainer.save_dir / "weights" / "last.pt")
            (trainer.save_dir / "results.csv").write_text("epoch,loss,time\n1,1,1\n", encoding="utf-8")
            def fail_mirror(*_: object, **__: object) -> object:
                raise OSError("mirror")
            hook = BenchmarkEpochHook(root / "primary", root / "mirror", identity, stop_after_epoch=1, publisher=fail_mirror)
            with self.assertRaisesRegex(OSError, "mirror"):
                hook.on_model_save(trainer)
            self.assertFalse(trainer.stop)

    def test_resume_requires_common_untampered_generation_and_restores_ambient_999_rng(self) -> None:
        import random
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_uninterrupted_two_epoch", config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=2)
            random.seed(0); torch.manual_seed(0)
            checkpoint = {"completed_epoch": 1, "model": {}, "ema": {}, "ema_updates": 0, "optimizer": {}, "scaler": {}, "scheduler": {}, "rng_state": capture_rng_state(), "train_args": {}}
            archive_buffer = __import__("io").BytesIO(); torch.save({"epoch": 0, "model": {}, "ema": None, "optimizer": {}, "scaler": {}, "train_args": {}}, archive_buffer)
            publish_generation(root / "primary", root / "mirror", identity, epoch=1, checkpoint_archive=archive_buffer.getvalue(), benchmark_state=checkpoint, results_csv="epoch,loss,time\n1,1,1\n", diagnostics={"epoch": 1})
            random.seed(999); torch.manual_seed(999)
            restored = prepare_resume_checkpoint(root / "primary", root / "mirror", identity, root / "scratch" / "last.pt", ambient_seed=999)
            self.assertEqual(restored.name, "last.pt")
            self.assertEqual(random.random(), random.Random(999).random())
            (root / "mirror" / "generations" / "1" / "last.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(BenchmarkIdentityError, "common|committed manifest"):
                prepare_resume_checkpoint(root / "primary", root / "mirror", identity, root / "scratch" / "last.pt", ambient_seed=999)

    def test_resume_rejects_byte_identical_state_tampering_against_committed_manifest(self) -> None:
        import io
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_uninterrupted_two_epoch", config_sha256="a"*64, code_sha256="b"*64, fit_ids_sha256="c"*64, development_ids_sha256="d"*64, model_sha256="e"*64, pretrained_sha256="f"*64, effective_epochs=2)
            state = {"completed_epoch": 1, "model": {}, "ema": {}, "ema_updates": 0, "optimizer": {}, "scaler": {}, "scheduler": {}, "rng_state": capture_rng_state(), "train_args": {}}
            archive = io.BytesIO(); torch.save({"epoch": 0, "model": {}, "ema": None}, archive)
            publish_generation(root/"p", root/"m", identity, epoch=1, checkpoint_archive=archive.getvalue(), benchmark_state=state, results_csv="epoch\n1\n", diagnostics={})
            tampered = {**state, "ema_updates": 1}
            content = io.BytesIO(); torch.save(tampered, content)
            for side in (root/"p", root/"m"):
                (side/"generations"/"1"/"benchmark_state.pt").write_bytes(content.getvalue())
            with self.assertRaisesRegex(BenchmarkIdentityError, "committed manifest"):
                prepare_resume_checkpoint(root/"p", root/"m", identity, root/"resume.pt", ambient_seed=999)
            class Target:
                def __init__(self) -> None:
                    self.callbacks: dict[str, list[object]] = {}
                def add_callback(self, event: str, callback: object) -> None:
                    self.callbacks.setdefault(event, []).append(callback)
            target = Target()
            configure_benchmark_callbacks(target, BenchmarkRunContext(identity=identity, primary_root=root/"p", mirror_root=root/"m"), resume=True)
            with self.assertRaisesRegex(BenchmarkIdentityError, "committed manifest"):
                target.callbacks["on_train_start"][-1](SimpleNamespace())

    def test_resume_copies_the_checkpoint_bytes_that_were_validated(self) -> None:
        import io
        import torch
        import ifdr_yolo.experiments.kitti_seed0_training_benchmark as benchmark

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_uninterrupted_two_epoch", config_sha256="a"*64, code_sha256="b"*64, fit_ids_sha256="c"*64, development_ids_sha256="d"*64, model_sha256="e"*64, pretrained_sha256="f"*64, effective_epochs=2)
            state = {"completed_epoch": 1, "model": {}, "ema": {}, "ema_updates": 0, "optimizer": {}, "scaler": {}, "scheduler": {}, "rng_state": capture_rng_state(), "train_args": {}}
            archive = io.BytesIO(); torch.save({"epoch": 0, "model": {}, "ema": None}, archive)
            expected = archive.getvalue()
            publish_generation(root/"p", root/"m", identity, epoch=1, checkpoint_archive=expected, benchmark_state=state, results_csv="epoch\n1\n", diagnostics={})
            real_loader = benchmark._load_committed_benchmark_state
            def race_after_validation(*args: object, **kwargs: object) -> object:
                result = real_loader(*args, **kwargs)
                for side in (root/"p", root/"m"):
                    (side/"generations"/"1"/"last.pt").write_bytes(b"RACED-CHECKPOINT")
                return result
            with patch.object(benchmark, "_load_committed_benchmark_state", side_effect=race_after_validation):
                target = prepare_resume_checkpoint(root/"p", root/"m", identity, root/"resume.pt", ambient_seed=999)
            self.assertEqual(target.read_bytes(), expected)

    def test_reconcile_completes_only_a_byte_identical_single_manifest(self) -> None:
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); identity = build_benchmark_identity(arm="DCLI", execution_role="recovery_uninterrupted_two_epoch", config_sha256="a"*64, code_sha256="b"*64, fit_ids_sha256="c"*64, development_ids_sha256="d"*64, model_sha256="e"*64, pretrained_sha256="f"*64, effective_epochs=2)
            state = {"completed_epoch": 1, "model": {}, "ema": {}, "ema_updates": 0, "optimizer": {}, "scaler": {}, "scheduler": {}, "rng_state": capture_rng_state(), "train_args": {}}
            stream = __import__("io").BytesIO(); torch.save({"epoch": 0, "model": {}, "ema": None}, stream)
            publish_generation(root/"p", root/"m", identity, epoch=1, checkpoint_archive=stream.getvalue(), benchmark_state=state, results_csv="epoch\n1\n", diagnostics={})
            (root/"m"/"generations"/"1"/"manifest.json").unlink()
            reconcile_common_generation(root/"p", root/"m", identity)
            self.assertEqual((root/"p"/"generations"/"1"/"manifest.json").read_bytes(), (root/"m"/"generations"/"1"/"manifest.json").read_bytes())
            (root/"m"/"generations"/"1"/"last.pt").write_bytes(b"tamper")
            (root/"m"/"generations"/"1"/"manifest.json").unlink()
            with self.assertRaisesRegex(BenchmarkIdentityError, "reconcilable"):
                reconcile_common_generation(root/"p", root/"m", identity)

    def test_synthetic_stop_after_durable_epoch_one_resumes_under_ambient_999_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = run_synthetic_recovery_probe(root / "clean", root / "clean-mirror", stop_after_epoch=None)
            with self.assertRaisesRegex(RuntimeError, "after durable epoch 1"):
                run_synthetic_recovery_probe(root / "stopped", root / "stopped-mirror", stop_after_epoch=1)
            resumed = run_synthetic_recovery_probe(root / "stopped", root / "stopped-mirror", stop_after_epoch=None, resume=True, ambient_seed=999)
            self.assertEqual(resumed["completed_epochs"], [1, 2])
            self.assertEqual(compare_recovery(uninterrupted, resumed)["decision"], "GO_SEED0_30_EPOCH_PREFLIGHT")

    def test_recovery_comparison_rejects_unregistered_difference(self) -> None:
        left = {"completed_epochs": [1, 2], "results_csv": "epoch,loss,time\n1,1,1\n2,1,2\n", "checkpoint_digest": "a" * 64, "diagnostics": {"stable": True}, "prediction_bytes": b"p", "evaluator": {"metric": 1.0}}
        right = {**left, "diagnostics": {"stable": False}}
        with self.assertRaisesRegex(BenchmarkIdentityError, "diagnostics"):
            compare_recovery(left, right)

    def test_cli_exposes_only_explicit_benchmark_stages(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/run_kitti_seed0_training_benchmark.py", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for stage in ("preflight", "timing", "recovery-uninterrupted", "recovery-stop1", "recovery-resume", "compare-recovery"):
            self.assertIn(stage, completed.stdout)

    def test_preflight_rejects_nonempty_or_overlapping_roots_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, mirror = root / "output", root / "mirror"
            output.mkdir(); (output / "old").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkIdentityError, "fresh output"):
                run_preflight(arm="P3P5_CONTROL", execution_role="timing_one_epoch", config_path=P3P5, fit_ids=ROOT / "configs/splits/kitti_train.txt", development_ids=ROOT / "configs/splits/kitti_val.txt", resolved_data=root / "missing.yaml", raw_label_dir=root / "missing-labels", repository_root=ROOT, output_dir=output, mirror_dir=mirror, device="0")
            with self.assertRaisesRegex(BenchmarkIdentityError, "disjoint"):
                run_preflight(arm="P3P5_CONTROL", execution_role="timing_one_epoch", config_path=P3P5, fit_ids=ROOT / "configs/splits/kitti_train.txt", development_ids=ROOT / "configs/splits/kitti_val.txt", resolved_data=root / "missing.yaml", raw_label_dir=root / "missing-labels", repository_root=ROOT, output_dir=root / "same", mirror_dir=root / "same", device="0")

    def test_preflight_success_is_atomic_identity_bound_and_never_calls_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "fresh-primary-parent" / "out"
            mirror = root / "fresh-mirror-parent" / "out"
            config = SimpleNamespace(experiment=SimpleNamespace(seed=0), training=SimpleNamespace(epochs=30, imgsz=640, batch=16, workers=8, amp=True, deterministic=True), prediction=SimpleNamespace(conf=0.001, iou=0.7, max_det=300, half=False), paths=SimpleNamespace(model=root / "model", initialization=None, generated_data=root / "generated", train_ids=root / "splits" / "train.txt"), initialization=None)
            def fake_sha(path: Path) -> str:
                path = Path(path)
                if path.name == "fit":
                    return "50a1f8d72b747d1e2f460a4e0a355469484d33dbf1e9928b64ba6594c1873362"
                if path.name == "dev":
                    return "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
                if path.name == "preflight_identity.json":
                    return hashlib.sha256(path.read_bytes()).hexdigest()
                return "a" * 64
            import torch
            git_replies = [SimpleNamespace(returncode=0, stdout="a" * 40), SimpleNamespace(returncode=0, stdout=""), SimpleNamespace(returncode=0, stdout="")]
            with patch("ifdr_yolo.experiments.config.load_baseline_config", return_value=config), patch("ifdr_yolo.experiments.kitti_seed0_training_benchmark._only_allowed_parent_difference", return_value=True), patch("ifdr_yolo.data.splits.load_ids", side_effect=[tuple(f"{i:06d}" for i in range(3341)), tuple(f"{i:06d}" for i in range(3341, 3712))]), patch("ifdr_yolo.experiments.p2_candidate_survival_audit._fit_image_manifest_sha256", return_value="15d326c539153c2a54c78f9af196038639e82be0de0af600808d25e67de23df3"), patch("ifdr_yolo.experiments.p2_candidate_survival_audit._directory_sha256", return_value="72e50ec65d019a8da17393c9e6d3e592c8eea52561bbb136173831b7325259d9"), patch("ifdr_yolo.experiments.provenance.verify_dataset", return_value={"image_count": 7481}), patch("ifdr_yolo.experiments.kitti_seed0_training_benchmark._sha_file", side_effect=fake_sha), patch("ifdr_yolo.experiments.kitti_seed0_training_benchmark.verify_ultralytics_callback_contract", return_value={"version": "8.4.98", "trainer_sha256": "b" * 64}), patch("ifdr_yolo.experiments.kitti_seed0_training_benchmark.subprocess.run", side_effect=git_replies), patch("ifdr_yolo.experiments.ultralytics_runtime.UltralyticsAdapter.train", side_effect=AssertionError("training must not run")), patch("ifdr_yolo.experiments.ifdr_runtime.IFDRRuntimeAdapter.train", side_effect=AssertionError("training must not run")), patch("ifdr_yolo.experiments.p2_fit_reference.run_p2_fit_reference", side_effect=AssertionError("training must not run")), patch("scripts.run_p2_interaction_s0.run_screen", side_effect=AssertionError("training must not run")), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device_count", return_value=1), patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="fake", total_memory=1)):
                result = run_preflight(arm="P3P5_CONTROL", execution_role="timing_one_epoch", config_path=P3P5, fit_ids=root / "fit", development_ids=root / "dev", resolved_data=root / "data", raw_label_dir=root / "labels", repository_root=ROOT, output_dir=output, mirror_dir=mirror, device="0")
            self.assertEqual(result["preflight_state"], "PASS")
            self.assertTrue(result["benchmark_launch_authorized"])
            self.assertFalse(result["training_authorized"])
            for name in ("preflight_identity.json", "manifest.json"):
                self.assertEqual((output / name).read_bytes(), (mirror / name).read_bytes())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["files"][0]["sha256"], hashlib.sha256((output / "preflight_identity.json").read_bytes()).hexdigest())

    def test_preflight_facts_change_the_canonical_benchmark_identity(self) -> None:
        common = dict(arm="P3P5_CONTROL", execution_role="timing_one_epoch", config_sha256="a" * 64, code_sha256="b" * 64, fit_ids_sha256="c" * 64, development_ids_sha256="d" * 64, model_sha256="e" * 64, pretrained_sha256="f" * 64, effective_epochs=1)
        facts = {"fit_image_manifest_sha256": "1" * 64, "raw_label_dir_sha256": "2" * 64, "resolved_data_sha256": "3" * 64, "git_head": "a" * 40, "tracked_diff_sha256": "4" * 64, "relevant_status_sha256": "5" * 64, "runtime": {"version": "8.4.98", "trainer_sha256": "6" * 64, "device": "0"}}
        baseline = build_benchmark_identity(**common, preflight_facts=facts)
        for path, replacement in (("fit_image_manifest_sha256", "7" * 64), ("resolved_data_sha256", "8" * 64), ("tracked_diff_sha256", "9" * 64), ("relevant_status_sha256", "0" * 64)):
            changed = {**facts, path: replacement}
            self.assertNotEqual(baseline["identity_sha256"], build_benchmark_identity(**common, preflight_facts=changed)["identity_sha256"])
        runtime_changed = {**facts, "runtime": {**facts["runtime"], "device": "1"}}
        self.assertNotEqual(baseline["identity_sha256"], build_benchmark_identity(**common, preflight_facts=runtime_changed)["identity_sha256"])

    def test_preflight_code_identity_includes_content_hash_implementation(self) -> None:
        source = (ROOT / "ifdr_yolo/experiments/kitti_seed0_training_benchmark.py").read_text(encoding="utf-8")
        self.assertIn('"ifdr_yolo/experiments/p2_candidate_survival_audit.py"', source)
        self.assertIn('"ifdr_yolo/models/ifdr_model.py"', source)
        self.assertIn('"ifdr_yolo/models/gated_fusion.py"', source)
        self.assertIn('"ifdr_yolo/experiments/gradient_diagnostics.py"', source)
        self.assertIn('"ifdr_yolo/losses/ifdr_detection.py"', source)

    def test_preflight_verifies_and_binds_the_generated_dataset(self) -> None:
        source = (ROOT / "ifdr_yolo/experiments/kitti_seed0_training_benchmark.py").read_text(encoding="utf-8")
        self.assertIn("verify_dataset(config, verify_all_hashes=False)", source)
        self.assertIn('"generated_dataset"', source)

    def test_preflight_rejects_split_content_with_registered_counts_but_wrong_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = SimpleNamespace(experiment=SimpleNamespace(seed=0), training=SimpleNamespace(epochs=30, imgsz=640, batch=16, workers=8, amp=True, deterministic=True), prediction=SimpleNamespace(conf=0.001, iou=0.7, max_det=300, half=False), paths=SimpleNamespace(model=root / "model"), initialization=None)
            with patch("ifdr_yolo.experiments.config.load_baseline_config", return_value=config), patch("ifdr_yolo.experiments.kitti_seed0_training_benchmark._only_allowed_parent_difference", return_value=True), patch("ifdr_yolo.data.splits.load_ids", side_effect=[tuple(f"{i:06d}" for i in range(3341)), tuple(f"{i:06d}" for i in range(3341, 3712))]), patch("ifdr_yolo.experiments.p2_candidate_survival_audit._fit_image_manifest_sha256", return_value="15d326c539153c2a54c78f9af196038639e82be0de0af600808d25e67de23df3"), patch("ifdr_yolo.experiments.p2_candidate_survival_audit._directory_sha256", return_value="72e50ec65d019a8da17393c9e6d3e592c8eea52561bbb136173831b7325259d9"), patch("ifdr_yolo.experiments.kitti_seed0_training_benchmark._sha_file", return_value="a" * 64):
                with self.assertRaisesRegex(BenchmarkIdentityError, "split"):
                    run_preflight(arm="P3P5_CONTROL", execution_role="timing_one_epoch", config_path=P3P5, fit_ids=root / "fit", development_ids=root / "dev", resolved_data=root / "data", raw_label_dir=root / "labels", repository_root=ROOT, output_dir=root / "out", mirror_dir=root / "mirror", device="0")


if __name__ == "__main__":
    unittest.main()
