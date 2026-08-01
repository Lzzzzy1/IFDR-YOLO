from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ifdr_yolo.data.interventions.sampler import SamplingPolicy
from ifdr_yolo.experiments.config import IFDRConfig, InitializationConfig
from ifdr_yolo.experiments.ultralytics_runtime import (
    EXPECTED_ULTRALYTICS_VERSION,
    PreparedModel,
    UltralyticsAdapter,
)
from ifdr_yolo.models.initialization import (
    apply_semantic_prefix_initialization,
)


@dataclass(frozen=True)
class IFDRPreparedHandle:
    model: Any
    model_path: Path


def _default_model_factory(**kwargs):
    from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

    return IFDRDetectionModel(**kwargs)


def _default_yolo_factory(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def _default_seed_initializer(seed: int, deterministic: bool) -> None:
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(seed, deterministic=deterministic)


class IFDRRuntimeAdapter:
    """Runtime boundary for project-owned model, trainer and checkpoint flow."""

    def __init__(
        self,
        config: IFDRConfig,
        *,
        model_factory: Callable[..., Any] = _default_model_factory,
        yolo_factory: Callable[[str], Any] = _default_yolo_factory,
        seed_initializer: Callable[[int, bool], None] = (
            _default_seed_initializer
        ),
        model_initializer: Callable[..., Any] = (
            apply_semantic_prefix_initialization
        ),
        trainer_factory: Callable[..., Any] | None = None,
        prediction_adapter: UltralyticsAdapter | None = None,
    ) -> None:
        if not isinstance(config, IFDRConfig):
            raise ValueError("config must be an IFDRConfig")
        self.config = config
        self._model_factory = model_factory
        self._yolo_factory = yolo_factory
        self._seed_initializer = seed_initializer
        self._model_initializer = model_initializer
        self._trainer_factory = trainer_factory
        self._prediction_adapter = prediction_adapter

    def _trainer(self) -> Callable[..., Any]:
        if self._trainer_factory is None:
            from ifdr_yolo.experiments.ifdr_trainer import (
                IFDRDetectionTrainer,
            )

            self._trainer_factory = IFDRDetectionTrainer
        return self._trainer_factory

    def runtime_info(self) -> dict[str, object]:
        return UltralyticsAdapter().runtime_info()

    def prepare_model(
        self,
        *,
        model_path: Path,
        model_sha256: str,
        initialization: InitializationConfig | None,
        seed: int,
        deterministic: bool,
    ) -> PreparedModel:
        if initialization is None:
            raise ValueError("IFDR requires semantic-prefix initialization")
        self._seed_initializer(seed, deterministic)
        method = self.config.method
        loss = method.loss
        model = self._model_factory(
            cfg=str(model_path),
            verbose=False,
            reliability_channels=method.reliability_channels,
            dcli_beta=loss.dcli_beta,
            uncertainty_calibration_gain=(
                loss.uncertainty_calibration_gain
            ),
            uncertainty_factor_weights=loss.factor_weights,
            dfl_entropy_weight=loss.dfl_entropy_weight,
            factor_supervision_gain=loss.factor_supervision_gain,
            semantic_protection=method.components.semantic_protection,
            counterfactual_gain=(
                loss.counterfactual_gain
                if method.components.counterfactual_consistency
                else 0.0
            ),
        )
        source_handle = self._yolo_factory(str(initialization.pretrained))
        source_model = getattr(source_handle, "model", None)
        if source_model is None:
            raise RuntimeError("pretrained handle does not expose a model")
        report = self._model_initializer(
            model,
            source_model,
            max_layer=initialization.max_layer,
            expected_items=initialization.expected_items,
        )
        payload = report.to_payload()
        payload.update(
            {
                "architecture": str(model_path),
                "architecture_sha256": model_sha256,
                "pretrained": str(initialization.pretrained),
                "pretrained_sha256": initialization.pretrained_sha256,
                "ultralytics": EXPECTED_ULTRALYTICS_VERSION,
                "seed": seed,
                "deterministic": deterministic,
                "fusion_nodes": [11, 14, 17, 20, 23, 26],
                "shared_factor_estimator": True,
                "components": {
                    "fusion_gate": method.components.fusion_gate,
                    "dcli": method.components.dcli,
                    "factor_supervision": (
                        method.components.factor_supervision
                    ),
                    "interventions": method.components.interventions,
                    "semantic_protection": (
                        method.components.semantic_protection
                    ),
                    "counterfactual_consistency": (
                        method.components.counterfactual_consistency
                    ),
                },
            }
        )
        return PreparedModel(
            handle=IFDRPreparedHandle(
                model=model,
                model_path=model_path,
            ),
            initialization=payload,
        )

    def train(
        self,
        *,
        prepared_model: PreparedModel,
        data_path: Path,
        run_dir: Path,
        args: Mapping[str, object],
    ) -> Path:
        handle = prepared_model.handle
        if not isinstance(handle, IFDRPreparedHandle):
            raise ValueError("prepared model is not an IFDR handle")
        method = self.config.method
        intervention = method.intervention
        from ifdr_yolo.experiments.ifdr_trainer import (
            IFDRComponentSwitches,
            FusionSchedule,
        )

        trainer = self._trainer()(
            overrides={
                **dict(args),
                "model": str(handle.model_path),
                "data": str(data_path),
                "project": str(run_dir.parent),
                "name": run_dir.name,
                "exist_ok": True,
            },
            fusion_schedule=FusionSchedule(
                frozen_epochs=method.schedule.frozen_epochs,
                ramp_epochs=method.schedule.ramp_epochs,
            ),
            component_switches=IFDRComponentSwitches(
                fusion_gate=method.components.fusion_gate,
                dcli=method.components.dcli,
                factor_supervision=(
                    method.components.factor_supervision
                ),
                interventions=method.components.interventions,
                semantic_protection=(
                    method.components.semantic_protection
                ),
                counterfactual_consistency=(
                    method.components.counterfactual_consistency
                ),
            ),
            intervention_seed=intervention.base_seed,
            intervention_policy=SamplingPolicy(
                identity_probability=intervention.identity_probability,
                sampling_probability=intervention.sampling_probability,
                visibility_probability=intervention.visibility_probability,
                minimum_strength=intervention.minimum_strength,
                maximum_strength=intervention.maximum_strength,
            ),
        )
        trainer.model = handle.model
        trainer.train()
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(
                f"IFDR training did not create best weight: {best}"
            )
        return best

    def predict(
        self,
        *,
        weights: Path,
        image_paths: tuple[Path, ...],
        output_dir: Path,
        args: Mapping[str, object],
    ) -> Path:
        if self._prediction_adapter is None:
            self._prediction_adapter = UltralyticsAdapter()
        return self._prediction_adapter.predict(
            weights=weights,
            image_paths=image_paths,
            output_dir=output_dir,
            args=args,
        )
