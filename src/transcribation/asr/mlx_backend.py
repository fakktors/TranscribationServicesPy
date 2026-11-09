"""mlx-whisper backend (Apple Silicon only)."""

from __future__ import annotations

import numpy as np


class MLXBackend:
    name = "mlx"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio: np.ndarray, *, language: str | None, initial_prompt: str | None) -> tuple[str, list]:
        from mlx_whisper import transcribe

        result = transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            verbose=False,  # tqdm progress bar, no per-segment output
        )
        return result["language"], result["segments"]

    def _get_model(self):
        """The model mlx-whisper cached during transcribe(), or a fresh load."""
        if self._model is not None:
            return self._model
        from mlx_whisper.transcribe import ModelHolder

        model = getattr(ModelHolder, "model", None)
        if model is None or getattr(ModelHolder, "model_path", None) != self.model_name:
            import mlx.core as mx
            from mlx_whisper.load_models import load_model

            model = load_model(self.model_name, dtype=mx.float16)
        self._model = model
        return model

    def redecode_if_foreign(self, audio: np.ndarray, global_lang: str, threshold: float) -> tuple[str, str] | None:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions

        model = self._get_model()
        mel = log_mel_spectrogram(mx.array(audio.astype(np.float32)), n_mels=model.dims.n_mels)
        mel_seg = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(mel_seg)
        top, p = max(probs.items(), key=lambda kv: kv[1])
        if top == global_lang or p < threshold:
            return None
        result = model.decode(mel_seg, DecodingOptions(task="transcribe", language=top, fp16=True))
        return top, (result.text or "").strip()
