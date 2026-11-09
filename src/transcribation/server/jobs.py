"""In-process job queue: one worker thread, jobs keyed by video id.

ASR and diarization saturate the CPU/GPU, so jobs run strictly one at a time.
Finished results live on disk (work/ + transcripts/), so they survive restarts;
the in-memory table only tracks queued/running jobs of the current process.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Settings
from ..pipeline import Options, PipelineError, Result, is_up_to_date, run_pipeline
from ..state import STAGES, read_state

log = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"


@dataclass
class Job:
    id: str
    url: str
    options: Options
    status: str = QUEUED
    stage: str | None = None
    error: str | None = None
    error_stage: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    event: threading.Event = field(default_factory=threading.Event, repr=False)

    def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = time.time()
        self.event.set()


class JobManager:
    def __init__(self, settings: Settings, runner: Callable[[Options, Settings], Result] | None = None):
        self.settings = settings
        self._runner = runner or (lambda opts, s, on_stage: run_pipeline(opts, s, on_stage=on_stage))
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="transcribe-worker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5)
            self._thread = None

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self._jobs.get(job_id)
            if job is None or job.status != QUEUED:
                continue
            job.status, job.started_at = RUNNING, time.time()
            log.info("job %s: started (%s)", job.id, job.url)

            def on_stage(name: str, job: Job = job) -> None:
                job.stage = name

            try:
                self._runner(job.options, self.settings, on_stage)
            except PipelineError as e:
                job.error, job.error_stage = e.message, e.stage
                job.finish(ERROR)
                log.warning("job %s: failed at %s: %s", job.id, e.stage, e.message)
            except Exception as e:  # report anything to the client
                job.error, job.error_stage = str(e), job.stage
                job.finish(ERROR)
                log.exception("job %s: crashed", job.id)
            else:
                job.stage = None
                job.finish(DONE)
                log.info("job %s: done in %.0f s", job.id, job.finished_at - job.started_at)

    # --- API ---------------------------------------------------------------
    def submit(self, opts: Options, *, force: bool = False) -> Job:
        with self._lock:
            existing = self._jobs.get(opts.video_id)
            if existing and existing.status in (QUEUED, RUNNING):
                return existing  # same video already in flight
            job = Job(id=opts.video_id, url=opts.url, options=opts)
            self._jobs[job.id] = job
            if not force and not opts.reset and is_up_to_date(opts, self.settings):
                job.started_at = job.created_at
                job.finish(DONE)  # served from cache, nothing to run
                return job
            if force:
                opts.reset = True
            self._queue.put(job.id)
            return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> int | None:
        if job.status != QUEUED:
            return None
        queued = sorted((j for j in self._jobs.values() if j.status == QUEUED), key=lambda j: j.created_at)
        return next((i + 1 for i, j in enumerate(queued) if j.id == job.id), None)

    def queue_size(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in (QUEUED, RUNNING))

    def describe(self, job_id: str) -> dict | None:
        """Job status from memory, falling back to the state on disk."""
        job = self._jobs.get(job_id)
        state = read_state(self.settings.work_dir, job_id)
        if job is None and state is None:
            return None
        stages = (state or {}).get("stages", {})
        md_path = self.settings.transcripts_dir / f"{job_id}.md"

        if job is not None:
            status, stage, error = job.status, job.stage, job.error
            if job.status == ERROR:
                stage = job.error_stage
            url = job.url
        else:  # not seen by this process: infer from disk
            url = state.get("url")
            failed = next((n for n in STAGES if stages.get(n, {}).get("status") == "error"), None)
            if stages.get("markdown", {}).get("status") == "done" and md_path.exists():
                status, stage, error = DONE, None, None
            elif failed:
                status, stage, error = ERROR, failed, stages[failed].get("error")
            else:
                status, stage, error = "incomplete", None, "interrupted; submit again to resume"

        out = {
            "id": job_id,
            "url": url,
            "status": status,
            "stage": stage,
            "error": error,
            "queue_position": self.queue_position(job) if job else None,
            "stages": {n: {"status": stages.get(n, {}).get("status", "pending")} for n in STAGES},
        }
        if job is not None:
            out.update(created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at)
        info = stages.get("download", {}).get("detail", {})
        if info.get("title"):
            out["title"] = info["title"]
        if info.get("duration"):
            out["duration"] = info["duration"]
        diar = stages.get("diarize", {})
        if diar.get("status") == "skipped":
            out["stages"]["diarize"]["reason"] = diar.get("detail", {}).get("reason")
        return out

    def list_transcripts(self, limit: int = 50) -> list[dict]:
        files = sorted(self.settings.transcripts_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for f in files[:limit]:
            title = f.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()
            out.append({"id": f.stem, "title": title, "updated_at": f.stat().st_mtime})
        return out
