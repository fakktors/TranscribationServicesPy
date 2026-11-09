"""Pipeline orchestration shared by the CLI and the HTTP/MCP server.

    url -> download -> transcribe -> diarize -> markdown

Each stage persists its result in work/<video_id>/; a re-run only does the work
whose inputs changed:

* transcribe re-runs when the backend/model/language/prompt differ from the
  cached result (and then diarize re-runs too);
* diarize re-runs when it was skipped earlier and a token/pyannote is now available;
* download runs only if a later stage needs audio that is not on disk;
* markdown (cheap) is always re-rendered.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .asr import asr_fingerprint, run_asr
from .config import Settings, hf_token
from .diarize import diarization_available, run_diarize
from .download import audio_ready, extract_video_id, normalize_url, run_download
from .markdown import run_markdown
from .state import STAGES, Task

log = logging.getLogger(__name__)


@dataclass
class Options:
    url: str
    language: str | None = None
    prompt: str | None = None
    diarize: bool = True
    allow_long: bool = False
    uncertain: float = 0.5
    stop: str = "markdown"
    reset: bool = False

    def __post_init__(self) -> None:
        self.url = normalize_url(self.url)
        if self.language:
            self.language = self.language.strip().lower() or None
        if self.prompt is not None and not self.prompt.strip():
            self.prompt = None
        if self.stop not in STAGES:
            raise ValueError(f"stop must be one of {STAGES}")

    @property
    def video_id(self) -> str:
        return extract_video_id(self.url)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    video_id: str
    stopped_after: str
    markdown: str | None = None
    json: str | None = None
    stages: dict = field(default_factory=dict)


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message


def _plan(task: Task, opts: Options, settings: Settings) -> dict[str, bool]:
    """Which stages must actually run for these options."""
    tr = task.stages["transcribe"]
    cached_cfg = tr.detail.get("config")
    need_transcribe = not task.validate_done("transcribe") or (
        cached_cfg is not None and cached_cfg != asr_fingerprint(settings, opts.language, opts.prompt)
    )

    need_diarize = False
    if opts.diarize:
        st = task.stages["diarize"]
        if need_transcribe or st.status in ("pending", "error"):
            need_diarize = True
        elif st.status == "done":
            need_diarize = (
                not task.validate_done("diarize")
                or st.detail.get("config", {}).get("model", settings.diarize_model) != settings.diarize_model
            )
        elif st.status == "skipped":
            # retry only if whatever was missing is available now
            need_diarize = diarization_available() and hf_token() is not None

    need_download = (need_transcribe or need_diarize) and not audio_ready(task.work_dir)
    return {"download": need_download, "transcribe": need_transcribe, "diarize": need_diarize, "markdown": True}


def _markdown_config(opts: Options) -> dict:
    return {"speakers": opts.diarize, "uncertain": opts.uncertain}


def is_up_to_date(opts: Options, settings: Settings) -> bool:
    """The final transcript for these options already exists — nothing to run."""
    work_dir = settings.work_dir / opts.video_id
    if not (work_dir / "state.json").exists():
        return False
    task = Task(settings.work_dir, opts.video_id, opts.url)
    md = task.stages["markdown"]
    if md.status != "done" or md.detail.get("config") != _markdown_config(opts):
        return False
    out = md.detail.get("out")
    if not out or not (settings.transcripts_dir / f"{opts.video_id}.md").exists():
        return False
    plan = _plan(task, opts, settings)
    return not (plan["transcribe"] or plan["diarize"])


def run_pipeline(
    opts: Options,
    settings: Settings,
    *,
    on_stage: Callable[[str], None] | None = None,
) -> Result:
    task = Task(settings.work_dir, opts.video_id, opts.url, reset=opts.reset)
    with task:
        task.url = opts.url
        task.save()
        plan = _plan(task, opts, settings)
        log.info("Task %s (work dir %s)", task.video_id, task.work_dir)

        runners: dict[str, Callable[[], object]] = {
            "download": lambda: run_download(task, settings, allow_long=opts.allow_long),
            "transcribe": lambda: run_asr(task, settings, language=opts.language, prompt=opts.prompt),
            "diarize": lambda: run_diarize(task, settings),
            "markdown": lambda: run_markdown(
                task, settings.transcripts_dir, use_speakers=opts.diarize, uncertain=opts.uncertain
            ),
        }

        for name in STAGES:
            stage = task.stages[name]
            if name == "diarize" and not opts.diarize:
                log.info("  diarize: disabled for this run")
                if stage.status in ("pending", "error"):
                    task.mark_skipped("diarize", reason="disabled (diarize=false)")
            elif not plan[name]:
                reason = stage.detail.get("reason", "") if stage.status == "skipped" else ""
                log.info("  %s: %s", name, f"skipped ({reason})" if reason else "up to date")
            else:
                log.info("  %s:", name)
                if on_stage:
                    on_stage(name)
                if name == "transcribe" and stage.status == "done":
                    task.invalidate("diarize")  # speakers were aligned to the old segments
                try:
                    runners[name]()
                except Exception as e:
                    task.mark_error(name, str(e))
                    raise PipelineError(name, str(e)) from e
                if name == "markdown":
                    task.mark_done("markdown", config=_markdown_config(opts))

            if name == opts.stop:
                break

        if opts.stop == "markdown" and not settings.keep_audio:
            (task.work_dir / "audio.wav").unlink(missing_ok=True)

        md = task.stages["markdown"]
        done = opts.stop == "markdown" and md.status == "done"
        return Result(
            video_id=task.video_id,
            stopped_after=opts.stop,
            markdown=md.detail.get("out") if done else None,
            json=md.detail.get("json") if done else None,
            stages={n: s.to_dict() for n, s in task.stages.items()},
        )
