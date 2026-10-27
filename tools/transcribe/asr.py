from __future__ import annotations

import json
import time
import wave

import numpy as np


def _read_wav(path) -> np.ndarray:
    """16 kHz mono s16 WAV -> float32 array (stdlib only, no scipy)."""
    with wave.open(str(path), "rb") as w:
        if not (w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2):
            raise RuntimeError(f"{path}: expected 16 kHz mono s16 WAV (regenerate the audio)")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def _rms(audio: np.ndarray, start: float, end: float, sr: int = 16000) -> float:
    a = audio[int(start * sr):int(end * sr)]
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(a))))


def _get_model(model_name: str):
    """The model used in transcribe() (mlx-whisper cache) or a fresh load."""
    from mlx_whisper.transcribe import ModelHolder

    model = getattr(ModelHolder, "model", None)
    if model is not None and getattr(ModelHolder, "model_path", None) == model_name:
        return model
    import mlx.core as mx

    from mlx_whisper.load_models import load_model

    return load_model(model_name, dtype=mx.float16)


def _fix_language_switches(model, audio: np.ndarray, segments: list, global_lang: str, cfg: dict) -> int:
    """Language switching: segments in a "foreign" language are decoded again.

    Whisper detects the language once per file (from the first 30 s). For videos
    with ru/en switching, check the language of each sufficiently long segment;
    if it confidently differs from the global one — re-decode that fragment.
    """
    import mlx.core as mx

    from mlx_whisper.audio import N_FRAMES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions

    n_mels = model.dims.n_mels  # mel spectrogram size differs across models
    threshold = float(cfg.get("lang_switch_threshold", 0.6))
    n_switch = 0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        # too short, silence, or longer than Whisper's 30-second window
        # (pad_or_trim would cut the tail and the segment text would be lost) — skip it
        if dur < 3.0 or dur > 30.0 or (seg.get("no_speech_prob") or 0.0) > 0.5:
            continue
        a0 = int(seg["start"] * SAMPLE_RATE)
        a1 = int(seg["end"] * SAMPLE_RATE)
        sl = mx.array(audio[a0:a1].astype(np.float32))
        mel = log_mel_spectrogram(sl, n_mels=n_mels)
        mel_seg = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(mel_seg)
        top, p = max(probs.items(), key=lambda kv: kv[1])
        if top == global_lang or p < threshold:
            continue
        result = model.decode(mel_seg, DecodingOptions(task="transcribe", language=top, fp16=True))
        text = (result.text or "").strip()
        if text:
            seg["text"] = " " + text
            seg["lang"] = top
            seg["switched"] = True
            seg["words"] = []  # word-level confidence is unknown after re-decoding
            n_switch += 1
    return n_switch


def _transcribe(wav, cfg: dict, model_name: str, lexicon_prompt: str | None):
    from mlx_whisper import transcribe

    return transcribe(
        _read_wav(wav),
        path_or_hf_repo=model_name,
        language=cfg.get("language"),
        initial_prompt=lexicon_prompt,
        word_timestamps=True,
        verbose=False,
    )


def run_asr(task, cfg: dict) -> dict:
    wav = task.work_dir / "audio.wav"
    model_name = cfg["model"]
    lexicon_prompt = cfg.get("lexicon_prompt")

    audio = _read_wav(wav)
    duration = len(audio) / 16000.0
    print(f"    recognizing: {duration / 60:.1f} min of audio, model {model_name}...")
    t0 = time.time()
    result = _transcribe(wav, cfg, model_name, lexicon_prompt)
    elapsed = time.time() - t0
    speed = duration / elapsed if elapsed > 0 else 0
    print(f"    recognition finished: {elapsed:.0f} s (x{speed:.1f} of realtime)")

    global_lang = result["language"]
    segments = []
    for s in result["segments"]:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
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
                    {
                        "w": w.get("word", ""),
                        "s": w.get("start"),
                        "e": w.get("end"),
                        "p": w.get("probability"),
                    }
                    for w in s.get("words", [])
                ],
            }
        )

    if cfg.get("language_switching") and cfg.get("language") is None:
        model = _get_model(model_name)
        n_switch = _fix_language_switches(model, audio, segments, global_lang, cfg)
        if n_switch:
            print(f"    language switches: segments re-decoded: {n_switch}")

    out = {
        "model": model_name,
        "language": global_lang,
        "lexicon": lexicon_prompt,
        "duration": round(duration, 1),
        "elapsed": round(elapsed, 1),
        "segments": segments,
    }
    seg_file = task.work_dir / "segments.json"
    seg_file.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    task.mark_done(
        "transcribe",
        artifact="segments.json",
        language=global_lang,
        n_segments=len(segments),
        elapsed=round(elapsed, 1),
    )
    return out
