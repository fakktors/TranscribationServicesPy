from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcribation.config import Settings

VIDEO_ID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.fixture
def settings(tmp_path: Path, monkeypatch) -> Settings:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("transcribation.pipeline.hf_token", lambda: None)
    return Settings(data_dir=tmp_path, asr_backend="faster-whisper", model="tiny")


def fake_segments() -> dict:
    return {
        "backend": "faster-whisper",
        "model": "tiny",
        "language": "en",
        "duration": 12.0,
        "segments": [
            {
                "start": 0.0,
                "end": 4.0,
                "text": " Hello there.",
                "lang": "en",
                "switched": False,
                "words": [
                    {"w": " Hello", "s": 0.0, "e": 0.5, "p": 0.9},
                    {"w": " there.", "s": 0.5, "e": 1.0, "p": 0.3},
                ],
            },
            {"start": 4.0, "end": 8.0, "text": " Привет.", "lang": "ru", "switched": True, "words": []},
            {"start": 8.0, "end": 12.0, "text": " Bye.", "lang": "en", "switched": False, "words": []},
        ],
    }


@pytest.fixture
def fake_stages(monkeypatch):
    """Replace the heavy stages with fakes that write plausible artifacts; returns a call log."""
    calls: list[str] = []

    def fake_download(task, settings, allow_long=False):
        calls.append("download")
        (task.work_dir / "info.json").write_text(json.dumps({"title": "Test video", "duration": 12}))
        (task.work_dir / "audio.wav").write_bytes(b"\0" * 20_000)
        task.mark_done("download", artifact="audio.wav", duration=12, title="Test video")

    def fake_asr(task, settings, language=None, prompt=None):
        from transcribation.asr import asr_fingerprint

        calls.append(f"transcribe:{settings.model}")
        (task.work_dir / "segments.json").write_text(json.dumps(fake_segments()))
        task.mark_done("transcribe", artifact="segments.json", config=asr_fingerprint(settings, language, prompt))

    def fake_diarize(task, settings):
        calls.append("diarize")
        task.mark_skipped("diarize", reason="no HF token")

    monkeypatch.setattr("transcribation.pipeline.run_download", fake_download)
    monkeypatch.setattr("transcribation.pipeline.run_asr", fake_asr)
    monkeypatch.setattr("transcribation.pipeline.run_diarize", fake_diarize)
    return calls
