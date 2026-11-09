"""Runtime settings, read from environment variables (prefix ``TRANSCRIBE_``).

All settings have defaults, so the CLI works with zero configuration. The
server and the Docker image are configured exclusively through the environment
(see ``.env.example``).
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_ALLOWED_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"TRANSCRIBE_{name}")
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


@dataclass(frozen=True)
class Settings:
    # Storage: <data_dir>/work/<video_id>/ (intermediate) and <data_dir>/transcripts/
    data_dir: Path = Path(".")

    # ASR
    asr_backend: str = "auto"  # auto | mlx | faster-whisper
    model: str | None = None  # None -> backend default
    device: str = "auto"  # faster-whisper: auto | cpu | cuda
    compute_type: str = "default"  # faster-whisper: default | int8 | float16 | ...
    cpu_threads: int = 0  # faster-whisper: 0 = library default
    vad_filter: bool = True  # faster-whisper: Silero VAD (fewer hallucinations on silence)

    # Diarization
    diarize_model: str = DEFAULT_DIARIZE_MODEL

    # Download
    max_duration: int = 3600  # seconds; longer videos need allow_long
    ytdlp_cookies: str | None = None  # path to a Netscape cookies.txt for yt-dlp
    ytdlp_extra_args: tuple[str, ...] = ()
    keep_audio: bool = True  # False: delete audio.wav once the transcript is written

    # Server
    api_token: str | None = None  # Bearer token for /v1 and /mcp; None = no auth
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS  # ("*",) = any URL
    max_wait: int = 600  # upper bound for the ?wait= / wait_seconds parameters

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    def with_overrides(self, **kwargs) -> Settings:
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    @classmethod
    def from_env(cls) -> Settings:
        hosts = _env("ALLOWED_HOSTS")
        return cls(
            data_dir=Path(_env("DATA_DIR", ".")).expanduser(),
            asr_backend=_env("ASR_BACKEND", "auto").lower(),
            model=_env("MODEL"),
            device=_env("DEVICE", "auto"),
            compute_type=_env("COMPUTE_TYPE", "default"),
            cpu_threads=_env_int("CPU_THREADS", 0),
            vad_filter=_env_bool("VAD_FILTER", True),
            diarize_model=_env("DIARIZE_MODEL", DEFAULT_DIARIZE_MODEL),
            max_duration=_env_int("MAX_DURATION", 3600),
            ytdlp_cookies=_env("YTDLP_COOKIES"),
            ytdlp_extra_args=tuple(shlex.split(_env("YTDLP_ARGS", "") or "")),
            keep_audio=_env_bool("KEEP_AUDIO", True),
            api_token=_env("API_TOKEN"),
            allowed_hosts=tuple(h.strip().lower() for h in hosts.split(",") if h.strip())
            if hosts
            else DEFAULT_ALLOWED_HOSTS,
            max_wait=_env_int("MAX_WAIT", 600),
        )


def hf_token() -> str | None:
    """Hugging Face token: env (HF_TOKEN / HUGGING_FACE_HUB_TOKEN) or `hf auth login` cache."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token

        return get_token() or None
    except Exception:
        return None
