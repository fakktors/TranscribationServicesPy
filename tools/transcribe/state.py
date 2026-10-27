from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

STAGES = ("download", "transcribe", "diarize", "markdown")


@dataclass
class Stage:
    name: str
    status: str = "pending"  # pending | done | skipped | error
    at: Optional[float] = None
    detail: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "at": self.at,
            "detail": self.detail,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Stage":
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
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.work_dir / "state.json"
        self.stages: dict[str, Stage] = {name: Stage(name) for name in STAGES}
        if reset:
            self.state_file.unlink(missing_ok=True)
        elif self.state_file.exists():
            try:
                raw = json.loads(self.state_file.read_text(encoding="utf-8"))
                for name, d in raw.get("stages", {}).items():
                    if name in self.stages:
                        self.stages[name] = Stage.from_dict(name, d)
            except (json.JSONDecodeError, OSError):
                pass  # corrupted state — start over
        self.save()

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
        stage.detail["reason"] = reason
        self.save()

    def mark_error(self, name: str, err: str) -> None:
        stage = self.stages[name]
        stage.status = "error"
        stage.at = time.time()
        stage.error = err[:500]
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
