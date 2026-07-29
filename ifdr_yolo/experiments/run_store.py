from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


ALLOWED_TRANSITIONS = {
    "prepared": {"running", "failed"},
    "running": {"trained", "failed"},
    "trained": {"evaluating", "failed"},
    "evaluating": {"complete", "failed"},
    "complete": set(),
    "failed": set(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id(
    *,
    timestamp: datetime,
    dataset: str,
    model: str,
    variant: str,
    seed: int,
    git_sha: str,
) -> str:
    if len(git_sha) < 7:
        raise ValueError("git_sha must contain at least 7 characters")
    utc = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{utc}-{dataset}-{model}-{variant}-s{seed}-{git_sha[:7]}"


def atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


@dataclass
class RunStore:
    root: Path
    state: str

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @classmethod
    def create(cls, root: Path) -> "RunStore":
        if root.exists():
            raise FileExistsError(f"run directory already exists: {root}")
        root.mkdir(parents=True, exist_ok=False)
        store = cls(root=root, state="prepared")
        store._write_status({"state": "prepared"})
        return store

    def transition(self, state: str) -> None:
        if state not in ALLOWED_TRANSITIONS:
            raise ValueError(f"unknown run state: {state}")
        allowed = ALLOWED_TRANSITIONS[self.state]
        if state not in allowed:
            raise ValueError(f"illegal run state transition: {self.state} -> {state}")
        self.state = state
        self._write_status({"state": state})

    def fail(self, *, stage: str, error: BaseException) -> None:
        if "failed" not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"cannot fail terminal run state: {self.state} -> failed"
            )
        self.state = "failed"
        self._write_status(
            {
                "state": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

    def _write_status(self, payload: dict[str, object]) -> None:
        atomic_write_json(
            self.status_path,
            {
                **payload,
                "updated_at_utc": _utc_now(),
            },
        )
