from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

MAX_DURATION = 3600  # первая версия принимает видео длиной до часа

AUDIO_EXTS = (".m4a", ".webm", ".mp3", ".opus", ".aac", ".flac", ".mp4")


def _yt_dlp() -> list[str]:
    exe = shutil.which("yt-dlp")
    return [exe] if exe else [sys.executable, "-m", "yt_dlp"]


def extract_video_id(url: str) -> str:
    """ID видео из ссылки (без сети); для не-YouTube ссылок — хэш."""
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    return "video-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def fetch_metadata(url: str, info_file: Path, allow_long: bool = False) -> dict:
    """Метаданные видео: название, длительность, главы (chapters)."""
    cmd = _yt_dlp() + ["--no-playlist", "--skip-download", "-J", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "yt-dlp не смог получить информацию о видео: " + (result.stderr or "").strip()[-400:]
        )
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"yt-dlp вернул некорректный JSON: {e}") from e
    duration = info.get("duration")
    if duration and duration > MAX_DURATION and not allow_long:
        raise RuntimeError(
            f"Видео длится {int(duration) // 60} мин — дольше часа. "
            "Первая версия поддерживает видео до 1 часа; продолжите с флагом --allow-long"
        )
    info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _find_downloaded_audio(work_dir: Path) -> Path | None:
    for f in sorted(work_dir.glob("audio.*")):
        if f.suffix.lower() in AUDIO_EXTS and f.stat().st_size > 100_000:
            return f
    return None


def fetch_audio(url: str, work_dir: Path) -> Path:
    """Скачивает аудио и превращает в 16 кГц моно WAV (формат для Whisper и pyannote)."""
    wav = work_dir / "audio.wav"
    if wav.exists() and wav.stat().st_size > 100_000:
        return wav  # этап уже был завершён (возобновление после сбоя)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Не найден ffmpeg — установите: brew install ffmpeg")

    src = _find_downloaded_audio(work_dir)
    if src is None:
        out_tpl = str(work_dir / "audio.%(ext)s")
        cmd = _yt_dlp() + [
            "--no-playlist",
            "--no-mtime",
            "-f", "bestaudio/best",
            "-o", out_tpl,
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "yt-dlp не смог скачать аудио: " + (result.stderr or "").strip()[-400:]
            )
        src = _find_downloaded_audio(work_dir)
        if src is None:
            raise RuntimeError("yt-dlp завершился, но аудиофайл не найден")

    tmp = wav.with_name("audio.wav.tmp")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-f", "wav", str(tmp)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ffmpeg не смог сконвертировать аудио: " + (result.stderr or "").strip()[-400:])
    tmp.replace(wav)
    return wav


def run_download(task, *, allow_long: bool = False) -> dict:
    info_file = task.work_dir / "info.json"
    wav = task.work_dir / "audio.wav"

    if info_file.exists():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        duration = info.get("duration") or 0
        if duration > MAX_DURATION and not allow_long:
            raise RuntimeError(
                f"Видео длится {int(duration) // 60} мин — дольше часа; продолжите с --allow-long"
            )
        print("    метаданы: уже получены, пропускаю")
    else:
        info = fetch_metadata(task.url, info_file, allow_long=allow_long)

    if wav.exists() and wav.stat().st_size > 100_000:
        print("    аудио: уже готово, пропускаю")
    else:
        print("    скачиваю аудио с YouTube...")
        fetch_audio(task.url, task.work_dir)
        print("    аудио: готово (16 кГц моно WAV)")

    task.mark_done("download", artifact="audio.wav", duration=info.get("duration"))
    return info
