"""Speech recognition stage.

Two interchangeable backends produce the same ``segments.json``:

* ``mlx``            — mlx-whisper, Apple Silicon GPU (macOS arm64 only);
* ``faster-whisper`` — CTranslate2, CPU or NVIDIA GPU (Linux / Docker / anywhere).

``auto`` picks MLX when it is importable on an Apple Silicon Mac, otherwise
faster-whisper.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
import wave
from typing import Protocol

import numpy as np

from ..config import Settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

DEFAULT_MODELS = {
    "mlx": "mlx-community/whisper-large-v3-mlx",
    "faster-whisper": "large-v3-turbo",
}


class ASRBackend(Protocol):
    name: str
    model_name: str

    def transcribe(self, audio: np.ndarray, *, language: str | None, initial_prompt: str | None) -> tuple[str, list]:
        """Full-file recognition -> (detected language, raw segments).

        Raw segment: {start, end, text, avg_logprob, no_speech_prob,
        words: [{word, start, end, probability}]}
        """

    def redecode_if_foreign(self, audio: np.ndarray, global_lang: str, threshold: float) -> tuple[str, str] | None:
        """Detect the language of a short (<=30 s) clip; if it confidently differs
        from ``global_lang``, decode it again in that language -> (lang, text)."""


def read_wav(path) -> np.ndarray:
    """16 kHz mono s16 WAV -> float32 array (stdlib only, no scipy)."""
    with wave.open(str(path), "rb") as w:
        if not (w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2):
            raise RuntimeError(f"{path}: expected 16 kHz mono s16 WAV (re-run with reset to regenerate)")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def _rms(audio: np.ndarray, start: float, end: float) -> float:
    a = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(a))))


def _mlx_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend_name(settings: Settings) -> str:
    name = settings.asr_backend
    if name == "auto":
        return "mlx" if _mlx_available() else "faster-whisper"
    if name not in DEFAULT_MODELS:
        raise ValueError(f"unknown ASR backend {name!r} (expected auto | mlx | faster-whisper)")
    return name


def resolve_model_name(settings: Settings) -> str:
    return settings.model or DEFAULT_MODELS[resolve_backend_name(settings)]


_BACKENDS: dict[tuple, ASRBackend] = {}


def get_backend(settings: Settings) -> ASRBackend:
    """Backend instance, cached per process (the server keeps the model in memory)."""
    name = resolve_backend_name(settings)
    model = resolve_model_name(settings)
    key = (name, model, settings.device, settings.compute_type, settings.cpu_threads, settings.vad_filter)
    if key not in _BACKENDS:
        if name == "mlx":
            from .mlx_backend import MLXBackend

            _BACKENDS[key] = MLXBackend(model)
        else:
            from .faster_whisper_backend import FasterWhisperBackend

            _BACKENDS[key] = FasterWhisperBackend(
                model,
                device=settings.device,
                compute_type=settings.compute_type,
                cpu_threads=settings.cpu_threads,
                vad_filter=settings.vad_filter,
            )
    return _BACKENDS[key]


def asr_fingerprint(settings: Settings, language: str | None, prompt: str | None) -> dict:
    """Inputs that change the transcribe output; a mismatch invalidates the stage."""
    return {
        "backend": resolve_backend_name(settings),
        "model": resolve_model_name(settings),
        "language": language,
        "prompt": prompt,
    }


def _fix_language_switches(
    backend: ASRBackend, audio: np.ndarray, segments: list, global_lang: str, threshold: float
) -> int:
    """Whisper detects the language once per file (from the first 30 s). For
    videos that switch between languages, check each sufficiently long segment
    and re-decode it if it is confidently in another language."""
    n_switch = 0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        # too short, silence, or longer than Whisper's 30-second window — skip
        if dur < 3.0 or dur > 30.0 or (seg.get("no_speech_prob") or 0.0) > 0.5:
            continue
        clip = audio[int(seg["start"] * SAMPLE_RATE) : int(seg["end"] * SAMPLE_RATE)]
        res = backend.redecode_if_foreign(clip, global_lang, threshold)
        if res is None:
            continue
        lang, text = res
        if text:
            seg["text"] = " " + text
            seg["lang"] = lang
            seg["switched"] = True
            seg["words"] = []  # word-level confidence is unknown after re-decoding
            n_switch += 1
    return n_switch


def run_asr(
    task, settings: Settings, *, language: str | None, prompt: str | None, lang_switch_threshold: float = 0.6
) -> dict:
    wav = task.work_dir / "audio.wav"
    backend = get_backend(settings)

    audio = read_wav(wav)
    duration = len(audio) / SAMPLE_RATE
    log.info("    recognizing: %.1f min of audio, %s / %s...", duration / 60, backend.name, backend.model_name)
    t0 = time.time()
    global_lang, raw_segments = backend.transcribe(audio, language=language, initial_prompt=prompt)
    elapsed = time.time() - t0
    log.info("    recognition finished: %.0f s (x%.1f of realtime)", elapsed, duration / elapsed if elapsed > 0 else 0)

    segments = []
    for s in raw_segments:
        start, end = float(s.get("start") or 0.0), float(s.get("end") or 0.0)
        segments.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": s.get("text") or "",
                "lang": global_lang,
                "switched": False,
                "avg_logprob": s.get("avg_logprob"),
                "no_speech_prob": s.get("no_speech_prob"),
                "rms": round(_rms(audio, start, end), 5),
                "words": [
                    {"w": w.get("word", ""), "s": w.get("start"), "e": w.get("end"), "p": w.get("probability")}
                    for w in s.get("words") or []
                ],
            }
        )

    if language is None:
        n_switch = _fix_language_switches(backend, audio, segments, global_lang, lang_switch_threshold)
        if n_switch:
            log.info("    language switches: segments re-decoded: %d", n_switch)

    out = {
        "backend": backend.name,
        "model": backend.model_name,
        "language": global_lang,
        "lexicon": prompt,
        "duration": round(duration, 1),
        "elapsed": round(elapsed, 1),
        "segments": segments,
    }
    (task.work_dir / "segments.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    task.mark_done(
        "transcribe",
        artifact="segments.json",
        config=asr_fingerprint(settings, language, prompt),
        language=global_lang,
        n_segments=len(segments),
        elapsed=round(elapsed, 1),
    )
    return out
