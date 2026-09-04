from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ifdr_yolo.data.interventions.schema import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
)
from ifdr_yolo.data.interventions.targets import factor_target_for_spec
from ifdr_yolo.data.interventions.transforms import apply_intervention


@dataclass(frozen=True)
class RobustnessView:
    image_dir: Path
    manifest_path: Path
    kind: str
    strength: float
    seed: int
    image_count: int
    source_sha256: str
    view_sha256: str


def _condition_seed(
    *,
    seed: int,
    image_id: str,
    kind: InterventionKind,
    strength: float,
) -> int:
    payload = f"{seed}:{image_id}:{kind.value}:{strength:.12f}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _update_digest(
    digest: hashlib._Hash,
    image_id: str,
    path: Path,
) -> None:
    digest.update(image_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")


def build_robustness_view(
    *,
    source_image_dir: Path,
    image_ids: tuple[str, ...],
    output_dir: Path,
    kind: InterventionKind,
    strength: float,
    seed: int,
) -> RobustnessView:
    if kind not in (
        InterventionKind.SAMPLING,
        InterventionKind.VISIBILITY,
    ):
        raise ValueError("robustness kind must be sampling or visibility")
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 <= float(strength) <= 1.0
    ):
        raise ValueError("strength must be finite and within [0, 1]")
    strength = float(strength)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not image_ids:
        raise ValueError("image_ids must not be empty")
    if len(set(image_ids)) != len(image_ids) or any(
        not isinstance(image_id, str) or not image_id
        for image_id in image_ids
    ):
        raise ValueError("image_ids must be unique non-empty strings")
    if not source_image_dir.is_dir():
        raise FileNotFoundError(
            f"source image directory does not exist: {source_image_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"robustness output directory is non-empty: {output_dir}"
        )

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    view_digest = hashlib.sha256()
    for image_id in image_ids:
        source_path = source_image_dir / f"{image_id}.png"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"robustness source image does not exist: {source_path}"
            )
        image = cv2.imdecode(
            np.frombuffer(source_path.read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError(f"failed to decode image: {source_path}")
        condition_seed = _condition_seed(
            seed=seed,
            image_id=image_id,
            kind=kind,
            strength=strength,
        )
        spec = InterventionSpec(
            image_id=image_id,
            kind=kind,
            role=InterventionRole.GLOBAL,
            strength=strength,
            seed=condition_seed,
        )
        target = factor_target_for_spec(spec)
        transformed = apply_intervention(image, spec, target)
        output_path = image_dir / f"{image_id}.png"
        encoded, encoded_image = cv2.imencode(
            ".png",
            transformed.image,
        )
        if not encoded:
            raise OSError(f"failed to write robustness image: {output_path}")
        output_path.write_bytes(encoded_image.tobytes())
        _update_digest(source_digest, image_id, source_path)
        _update_digest(view_digest, image_id, output_path)

    view = RobustnessView(
        image_dir=image_dir,
        manifest_path=output_dir / "manifest.json",
        kind=kind.value,
        strength=strength,
        seed=seed,
        image_count=len(image_ids),
        source_sha256=source_digest.hexdigest(),
        view_sha256=view_digest.hexdigest(),
    )
    payload = asdict(view)
    payload["image_dir"] = str(view.image_dir)
    payload["manifest_path"] = str(view.manifest_path)
    payload["seed_derivation"] = (
        "sha256(base_seed:image_id:kind:strength)[:8]-big-endian"
    )
    view.manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return view
