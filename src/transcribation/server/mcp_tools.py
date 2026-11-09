"""MCP tools (Streamable HTTP at /mcp) for AI agents such as Hermes Agent.

Designed for LLM callers:
* one call does the whole job when the video is short (it waits for the result);
* long jobs return a status line instead of blocking past the client's timeout —
  the agent calls ``get_transcript`` again, which also waits a little;
* long transcripts are paginated by characters (``offset`` / ``max_chars``) so a
  single tool result never blows up the agent's context.
"""

from __future__ import annotations

import asyncio
import json
import re

from mcp.server.mcpserver import MCPServer

from ..download import is_url_allowed, normalize_url
from ..pipeline import Options
from .jobs import DONE, ERROR, JobManager

INSTRUCTIONS = """\
Transcribes YouTube videos (speech-to-text with timestamps and speaker labels).
Call transcribe_video with the URL. If the result says the job is still running,
call get_transcript with the returned id until status is done. Long transcripts
are returned in pages: pass next_offset as offset to read further.
Markdown format: paragraphs per speaker ("**Speaker 1:**"), [mm:ss] timestamps,
(?word?) marks low-confidence words, (en)/(ru) marks a language switch."""

DEFAULT_PAGE = 40_000


def _status_text(desc: dict) -> str:
    lines = [f"status: {desc['status']}", f"id: {desc['id']}"]
    if desc.get("title"):
        lines.append(f"title: {desc['title']}")
    if desc.get("stage"):
        lines.append(f"stage: {desc['stage']}")
    if desc.get("queue_position"):
        lines.append(f"queue_position: {desc['queue_position']}")
    if desc.get("error"):
        lines.append(f"error: {desc['error']}")
    if desc["status"] in ("queued", "running"):
        lines.append("hint: still working — call get_transcript with this id to wait for the result")
    return "\n".join(lines)


def _page(text: str, offset: int, max_chars: int) -> tuple[str, str]:
    offset = max(0, offset)
    max_chars = max(1000, min(max_chars, 200_000))
    chunk = text[offset : offset + max_chars]
    end = offset + len(chunk)
    header = f"chars: {offset}-{end} of {len(text)}"
    if end < len(text):
        header += f" (more: call again with offset={end})"
    return header, chunk


def build_mcp(manager: JobManager) -> MCPServer:
    settings = manager.settings
    mcp = MCPServer(name="transcribation", instructions=INSTRUCTIONS)

    async def _wait(job_id: str, seconds: int) -> None:
        job = manager.get(job_id)
        if job is not None and seconds > 0:
            await asyncio.to_thread(job.event.wait, min(seconds, settings.max_wait))

    def _render(job_id: str, fmt: str, offset: int, max_chars: int) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id or ""):
            return f"status: invalid_id\nid: {job_id!r}"
        desc = manager.describe(job_id)
        if desc is None:
            return f"status: not_found\nid: {job_id}\nhint: call transcribe_video first"
        if desc["status"] != DONE:
            return _status_text(desc)
        ext = "json" if fmt == "json" else "md"
        path = settings.transcripts_dir / f"{job_id}.{ext}"
        text = path.read_text(encoding="utf-8")
        header, chunk = _page(text, offset, max_chars)
        diar = desc["stages"]["diarize"]
        note = f"\nspeaker diarization: {diar['status']}" + (f" ({diar['reason']})" if diar.get("reason") else "")
        return f"{_status_text(desc)}{note}\n{header}\n\n{chunk}"

    @mcp.tool()
    async def transcribe_video(
        url: str,
        language: str | None = None,
        prompt: str | None = None,
        diarize: bool = True,
        allow_long: bool = False,
        wait_seconds: int = 90,
        max_chars: int = DEFAULT_PAGE,
    ) -> str:
        """Transcribe a YouTube video and return the Markdown transcript.

        Args:
            url: YouTube video URL or 11-character video id.
            language: ISO code such as "ru" or "en"; omit to auto-detect.
            prompt: comma-separated domain terms/names that appear in the audio (spelling hint).
            diarize: label speakers ("Speaker 1", "Speaker 2", ...).
            allow_long: allow videos longer than the server's limit (1 hour by default).
            wait_seconds: how long to wait for the result before returning a "running" status.
            max_chars: maximum characters of transcript to return in this call.

        Results are cached: calling again for the same video is instant.
        """
        url = normalize_url(url)
        if not is_url_allowed(url, settings.allowed_hosts):
            return f"status: rejected\nerror: URL host not allowed (allowed: {', '.join(settings.allowed_hosts)})"
        try:
            opts = Options(url=url, language=language, prompt=prompt, diarize=diarize, allow_long=allow_long)
        except ValueError as e:
            return f"status: rejected\nerror: {e}"
        job = manager.submit(opts)
        await _wait(job.id, wait_seconds)
        return _render(job.id, "markdown", 0, max_chars)

    @mcp.tool()
    async def get_transcript(
        id: str,
        format: str = "markdown",
        offset: int = 0,
        max_chars: int = DEFAULT_PAGE,
        wait_seconds: int = 60,
    ) -> str:
        """Get a transcript (or the job status) by video id.

        Args:
            id: video id returned by transcribe_video.
            format: "markdown" (readable, with timestamps and speakers) or "json"
                (segments with start/end seconds, speaker and text).
            offset: character offset for paging through long transcripts.
            max_chars: page size in characters.
            wait_seconds: if the job is still running, wait up to this long for it.
        """
        job = manager.get(id)
        if job is not None and job.status not in (DONE, ERROR):
            await _wait(id, wait_seconds)
        return _render(id, format, offset, max_chars)

    @mcp.tool()
    async def list_transcripts(limit: int = 20) -> str:
        """List already transcribed videos, newest first (id and title)."""
        return json.dumps(manager.list_transcripts(limit=max(1, min(limit, 200))), ensure_ascii=False, indent=1)

    return mcp
