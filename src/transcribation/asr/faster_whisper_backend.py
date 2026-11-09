"""faster-whisper (CTranslate2) backend: CPU or NVIDIA GPU, any OS."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        compute_type: str = "default",
        cpu_threads: int = 0,
        vad_filter: bool = True,
    ):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "int8"
        self.model_name = model_name
        self.vad_filter = vad_filter
        log.info("    loading faster-whisper %s (%s, %s)...", model_name, device, compute_type)
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)

    def transcribe(self, audio: np.ndarray, *, language: str | None, initial_prompt: str | None) -> tuple[str, list]:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            beam_size=5,
        )
        out = []
        for s in segments:  # generator: decoding happens while iterating
            out.append(
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                        for w in (s.words or [])
                    ],
                }
            )
        return info.language, out

    def redecode_if_foreign(self, audio: np.ndarray, global_lang: str, threshold: float) -> tuple[str, str] | None:
        lang, p, _ = self._model.detect_language(audio=audio)
        if lang == global_lang or p < threshold:
            return None
        segments, _ = self._model.transcribe(
            audio, language=lang, beam_size=5, vad_filter=False, condition_on_previous_text=False
        )
        return lang, " ".join(s.text.strip() for s in segments).strip()
