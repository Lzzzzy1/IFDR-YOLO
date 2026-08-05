"""Deterministic, leakage-free development split construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import floor, isfinite
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any


REGISTERED_SEED = 20260805
REGISTERED_FRACTION = 0.10
_STRATUM_NAMES = ("no_cyclist", "cyclist_lower", "cyclist_middle", "cyclist_upper")


@dataclass(frozen=True)
class DevelopmentSplit:
    """Immutable IDs, strata and digest for a development split."""

    seed: int
    fit_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    strata: Mapping[str, tuple[str, ...]]
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_ids", tuple(self.fit_ids))
        object.__setattr__(self, "development_ids", tuple(self.development_ids))
        frozen_strata = {
            str(name): tuple(image_ids)
            for name, image_ids in self.strata.items()
        }
        object.__setattr__(self, "strata", MappingProxyType(frozen_strata))


def _validate_parameters(*, seed: int, fraction: float) -> None:
    if seed != REGISTERED_SEED:
        raise ValueError(
            "development split requires seed=20260805 and fraction=0.10"
        )
    if fraction != REGISTERED_FRACTION:
        raise ValueError(
            "development split requires seed=20260805 and fraction=0.10"
        )


def _validate_unique_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[str, bool, float]]:
    if not rows:
        raise ValueError("development split requires at least one row")

    normalized: list[tuple[str, bool, float]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")

        raw_image_id = row.get("image_id")
        if not isinstance(raw_image_id, str):
            raise ValueError(f"row {index} image_id must be a string")
        image_id = raw_image_id.strip()
        if not image_id:
            raise ValueError("image_id must be non-empty")
        if image_id in seen:
            raise ValueError(f"duplicate image_id: {image_id}")
        seen.add(image_id)

        cyclist = row.get("cyclist")
        if not isinstance(cyclist, bool):
            raise ValueError(f"row {index} cyclist must be a boolean")
        if "cyclist_joint" not in row:
            raise ValueError(f"row {index} cyclist_joint is required")
        try:
            cyclist_joint = float(row["cyclist_joint"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"row {index} cyclist_joint must be finite"
            ) from error
        if not isfinite(cyclist_joint):
            raise ValueError(f"row {index} cyclist_joint must be finite")
        if cyclist_joint < 0.0 or cyclist_joint > 1.0:
            raise ValueError(
                f"row {index} cyclist_joint must be between 0 and 1"
            )
        if not cyclist and cyclist_joint != 0.0:
            raise ValueError(
                f"row {index} cyclist_joint must be 0.0 without Cyclist"
            )
        normalized.append((image_id, cyclist, cyclist_joint))
    return normalized


def _build_strata(
    normalized: Sequence[tuple[str, bool, float]],
) -> dict[str, tuple[str, ...]]:
    no_cyclist = sorted(
        image_id for image_id, cyclist, _ in normalized if not cyclist
    )
    cyclist_rows = sorted(
        ((cyclist_joint, image_id) for image_id, cyclist, cyclist_joint in normalized if cyclist),
        key=lambda item: (item[0], item[1]),
    )

    base_size, remainder = divmod(len(cyclist_rows), 3)
    cursor = 0
    cyclist_strata: dict[str, tuple[str, ...]] = {}
    for offset, name in enumerate(_STRATUM_NAMES[1:]):
        size = base_size + (1 if offset < remainder else 0)
        group = cyclist_rows[cursor : cursor + size]
        cyclist_strata[name] = tuple(image_id for _, image_id in group)
        cursor += size

    return {
        "no_cyclist": tuple(no_cyclist),
        **cyclist_strata,
    }


def _round_half_up(value: float) -> int:
    return int(floor(value + 0.5))


def _allocate_development_counts(
    strata: Mapping[str, tuple[str, ...]],
    *,
    total_count: int,
    development_count: int,
    fraction: float,
) -> dict[str, int]:
    if development_count < 0 or development_count > total_count:
        raise ValueError("quota constraints: invalid development count")

    non_empty = [(name, image_ids) for name, image_ids in strata.items() if image_ids]
    minimum = {
        name: 1 if len(image_ids) >= 2 else 0
        for name, image_ids in non_empty
    }
    maximum = {
        name: len(image_ids) - (1 if len(image_ids) >= 2 else 0)
        for name, image_ids in non_empty
    }
    minimum_total = sum(minimum.values())
    maximum_total = sum(maximum.values())
    if development_count < minimum_total or development_count > maximum_total:
        raise ValueError("quota constraints: development seats cannot be allocated")

    counts: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for name, image_ids in non_empty:
        raw_quota = fraction * len(image_ids)
        lower = floor(raw_quota)
        counts[name] = min(max(lower, minimum[name]), maximum[name])
        remainders[name] = raw_quota - lower

    allocated = sum(counts.values())
    if allocated > development_count:
        candidates = sorted(
            (
                name
                for name in counts
                if counts[name] > minimum[name]
            ),
            key=lambda name: (remainders[name], name),
        )
        for name in candidates:
            if allocated <= development_count:
                break
            counts[name] -= 1
            allocated -= 1
    elif allocated < development_count:
        candidates = sorted(
            (
                name
                for name in counts
                if counts[name] < maximum[name]
            ),
            key=lambda name: (-remainders[name], name),
        )
        while allocated < development_count:
            changed = False
            for name in candidates:
                if counts[name] >= maximum[name]:
                    continue
                counts[name] += 1
                allocated += 1
                changed = True
                if allocated == development_count:
                    break
            if not changed:
                raise ValueError(
                    "quota constraints: development seats cannot be allocated"
                )

    if allocated != development_count:
        raise ValueError("quota constraints: development seats cannot be allocated")
    return counts


def _stable_stratum_order(image_ids: Sequence[str], *, seed: int) -> list[str]:
    def key(image_id: str) -> tuple[str, str]:
        token = f"{seed}\0{image_id}".encode("utf-8")
        return sha256(token).hexdigest(), image_id

    return sorted(image_ids, key=key)


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_development_split(
    rows: Sequence[Mapping[str, object]], *, seed: int, fraction: float
) -> DevelopmentSplit:
    """Build the registered deterministic 90/10 fit/development split."""
    _validate_parameters(seed=seed, fraction=fraction)
    normalized = _validate_unique_rows(rows)
    strata = _build_strata(normalized)
    development_count = _round_half_up(fraction * len(normalized))
    allocations = _allocate_development_counts(
        strata,
        total_count=len(normalized),
        development_count=development_count,
        fraction=fraction,
    )

    development: set[str] = set()
    for name, image_ids in strata.items():
        ordered = _stable_stratum_order(image_ids, seed=seed)
        development.update(ordered[: allocations.get(name, 0)])

    all_ids = tuple(sorted(image_id for image_id, _, _ in normalized))
    development_ids = tuple(sorted(development))
    fit_ids = tuple(image_id for image_id in all_ids if image_id not in development)
    immutable_strata = {
        name: tuple(image_ids) for name, image_ids in strata.items()
    }
    payload = {
        "seed": seed,
        "fit_ids": fit_ids,
        "development_ids": development_ids,
    }
    return DevelopmentSplit(
        seed=seed,
        fit_ids=fit_ids,
        development_ids=development_ids,
        strata=immutable_strata,
        sha256=_digest(payload),
    )


def _manifest(split: DevelopmentSplit) -> dict[str, object]:
    return {
        "seed": split.seed,
        "fraction": REGISTERED_FRACTION,
        "fit_ids": list(split.fit_ids),
        "development_ids": list(split.development_ids),
        "strata": {
            name: list(image_ids) for name, image_ids in split.strata.items()
        },
        "sha256": split.sha256,
    }


def _split_output_bytes(split: DevelopmentSplit) -> dict[str, bytes]:
    manifest = json.dumps(
        _manifest(split),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    fit = "\n".join(split.fit_ids)
    development = "\n".join(split.development_ids)
    return {
        "fit_ids.txt": (fit + "\n" if fit else "").encode("utf-8"),
        "development_ids.txt": (
            development + "\n" if development else ""
        ).encode("utf-8"),
        "development_split.json": manifest,
    }


def write_split_outputs(split: DevelopmentSplit, output_dir: Path) -> None:
    """Write split files atomically, idempotently and fail-closed."""
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output directory is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = _split_output_bytes(split)

    existing: set[str] = set()
    for name, content in expected.items():
        path = output_dir / name
        if not path.exists():
            continue
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"existing output is not identical: {path}")
        existing.add(name)

    if len(existing) == len(expected):
        return

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".development-split-", dir=str(output_dir))
    )
    try:
        for name, content in expected.items():
            (temporary_dir / name).write_bytes(content)
        for name in expected:
            path = output_dir / name
            if name not in existing:
                os.replace(temporary_dir / name, path)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
