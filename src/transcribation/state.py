"""Per-video task state: work/<video_id>/state.json + an exclusive file lock."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

STAGES = ("download", "transcribe", "diarize", "markdown")


class TaskBusyError(RuntimeError):
    """Another process is already working on this video."""


@dataclass
class Stage:
    name: str
    status: str = "pending"  # pending | done | skipped | error
    at: float | None = None
    detail: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {"status": self.status, "at": self.at, "detail": self.detail, "error": self.error}

    @classmethod
    def from_dict(cls, name: str, d: dict) -> Stage:
        return cls(
            name=name,
            status=d.get("status", "pending"),
            at=d.get("at"),
            detail=d.get("detail") or {},
            error=d.get("error"),
        )


class Task:
    """Task state: work dir + state.json."""

    def __init__(self, work_root: Path, video_id: str, url: str, *, reset: bool = False):
        self.video_id = video_id
        self.url = url
        self.work_dir = Path(work_root) / video_id
        if reset and self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.work_dir / "state.json"
        self.stages: dict[str, Stage] = {name: Stage(name) for name in STAGES}
        self._lock_fd: int | None = None
        if self.state_file.exists():
            try:
                raw = json.loads(self.state_file.read_text(encoding="utf-8"))
                for name, d in raw.get("stages", {}).items():
                    if name in self.stages:
                        self.stages[name] = Stage.from_dict(name, d)
            except (json.JSONDecodeError, OSError):
                pass  # corrupted state — start over

    # --- locking -----------------------------------------------------------
    def lock(self) -> None:
        if fcntl is None:
            return
        fd = os.open(self.work_dir / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise TaskBusyError(f"{self.video_id} is already being processed by another process") from None
        self._lock_fd = fd

    def unlock(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self) -> Task:
        self.lock()
        return self

    def __exit__(self, *exc) -> None:
        self.unlock()

    # --- persistence -------------------------------------------------------
    @property
    def state_dict(self) -> dict:
        return {
            "url": self.url,
            "video_id": self.video_id,
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
        }

    def save(self) -> None:
        tmp = self.state_file.with_name("state.json.tmp")
        tmp.write_text(json.dumps(self.state_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_file)

    def mark_done(self, name: str, **detail: Any) -> None:
        stage = self.stages[name]
        stage.status = "done"
        stage.at = time.time()
        stage.error = None
        stage.detail.update(detail)
        self.save()

    def mark_skipped(self, name: str, reason: str) -> None:
        stage = self.stages[name]
        stage.status = "skipped"
        stage.at = time.time()
        stage.error = None
        stage.detail = {"reason": reason}
        self.save()

    def mark_error(self, name: str, err: str) -> None:
        stage = self.stages[name]
        stage.status = "error"
        stage.at = time.time()
        stage.error = err[:500]
        self.save()

    def invalidate(self, name: str) -> None:
        self.stages[name] = Stage(name)
        self.save()

    def is_done(self, name: str) -> bool:
        return self.stages[name].status == "done"

    def validate_done(self, name: str) -> bool:
        """Stage is done and its artifact (detail["artifact"]) is actually present."""
        stage = self.stages[name]
        if stage.status != "done":
            return False
        artifact = stage.detail.get("artifact")
        if artifact and not (self.work_dir / artifact).exists():
            return False
        return True


def read_state(work_root: Path, video_id: str) -> dict | None:
    """state.json of a task without creating anything (for status queries)."""
    f = Path(work_root) / video_id / "state.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
