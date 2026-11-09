"""Download stage: yt-dlp metadata + audio, converted to 16 kHz mono WAV."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from .config import Settings

log = logging.getLogger(__name__)

AUDIO_EXTS = (".m4a", ".webm", ".mp3", ".opus", ".aac", ".flac", ".mp4", ".ogg")
MIN_AUDIO_BYTES = 10_000

_VIDEO_ID_RE = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/|/v/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")


def extract_video_id(url: str) -> str:
    """Video id from the URL (no network); other links get a stable hash."""
    m = _VIDEO_ID_RE.search(url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url  # bare video id
    return "video-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def normalize_url(url: str) -> str:
    """Bare YouTube ids become full URLs; everything else is passed through."""
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return f"https://www.youtube.com/watch?v={url}"
    return url


def is_url_allowed(url: str, allowed_hosts: tuple[str, ...]) -> bool:
    """http(s) URL whose host is one of ``allowed_hosts`` (or a subdomain); "*" allows any."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if "*" in allowed_hosts:
        return True
    host = parsed.hostname.lower()
    return any(host == h or host.endswith("." + h) for h in allowed_hosts)


def _yt_dlp(settings: Settings) -> list[str]:
    exe = shutil.which("yt-dlp")
    cmd = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    cmd += ["--no-playlist", "--no-progress"]
    if settings.ytdlp_cookies:
        cmd += ["--cookies", settings.ytdlp_cookies]
    cmd += list(settings.ytdlp_extra_args)
    return cmd


def _too_long(duration, settings: Settings, allow_long: bool) -> bool:
    return bool(duration) and settings.max_duration > 0 and duration > settings.max_duration and not allow_long


def _too_long_error(duration) -> RuntimeError:
    return RuntimeError(
        f"video is {int(duration) // 60} min long — over the duration limit; pass allow_long to continue"
    )


def fetch_metadata(url: str, info_file: Path, settings: Settings, allow_long: bool = False) -> dict:
    """Video metadata: title, duration, chapters."""
    result = subprocess.run(_yt_dlp(settings) + ["--skip-download", "-J", url], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("yt-dlp failed to fetch video info: " + (result.stderr or "").strip()[-400:])
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"yt-dlp returned invalid JSON: {e}") from e
    if _too_long(info.get("duration"), settings, allow_long):
        raise _too_long_error(info["duration"])
    info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _find_downloaded_audio(work_dir: Path) -> Path | None:
    for f in sorted(work_dir.glob("audio.*")):
        if f.suffix.lower() in AUDIO_EXTS and f.stat().st_size > MIN_AUDIO_BYTES:
            return f
    return None


def fetch_audio(url: str, work_dir: Path, settings: Settings) -> Path:
    """Downloads audio and converts it to 16 kHz mono WAV (input for Whisper and pyannote)."""
    wav = work_dir / "audio.wav"
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found — install it (macOS: brew install ffmpeg, Debian: apt install ffmpeg)")

    src = _find_downloaded_audio(work_dir)
    if src is None:
        cmd = _yt_dlp(settings) + ["--no-mtime", "-f", "bestaudio/best", "-o", str(work_dir / "audio.%(ext)s"), url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("yt-dlp failed to download audio: " + (result.stderr or "").strip()[-400:])
        src = _find_downloaded_audio(work_dir)
        if src is None:
            raise RuntimeError("yt-dlp finished, but no audio file was found")

    tmp = wav.with_name("audio.wav.tmp")
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            "-f",
            "wav",
            str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed to convert audio: " + (result.stderr or "").strip()[-400:])
    tmp.replace(wav)
    src.unlink(missing_ok=True)  # the WAV is all later stages need
    return wav


def audio_ready(work_dir: Path) -> bool:
    wav = work_dir / "audio.wav"
    return wav.exists() and wav.stat().st_size > MIN_AUDIO_BYTES


def run_download(task, settings: Settings, *, allow_long: bool = False) -> dict:
    info_file = task.work_dir / "info.json"

    if info_file.exists():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if _too_long(info.get("duration"), settings, allow_long):
            raise _too_long_error(info["duration"])
        log.info("    metadata: already fetched")
    else:
        info = fetch_metadata(task.url, info_file, settings, allow_long=allow_long)

    if audio_ready(task.work_dir):
        log.info("    audio: already downloaded")
    else:
        log.info("    downloading audio...")
        fetch_audio(task.url, task.work_dir, settings)
        log.info("    audio: ready (16 kHz mono WAV)")

    task.mark_done("download", artifact="audio.wav", duration=info.get("duration"), title=info.get("title"))
    return info
