"""Speaker diarization stage (pyannote). Optional: needs the ``diarize`` extra and an HF token."""

from __future__ import annotations

import json
import logging
import time

from .config import Settings, hf_token

log = logging.getLogger(__name__)

_PIPELINES: dict[str, object] = {}


def diarization_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False
    return True


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(turns_by_label: dict[str, list[tuple[float, float]]], segments: list) -> tuple[list, list]:
    """Map ASR segments to speakers by time overlap.

    ``turns_by_label``: raw pyannote label -> [(start, end), ...].
    Speakers are renamed speaker_1, speaker_2, ... in order of first appearance.
    """
    order = sorted(turns_by_label, key=lambda k: min(s[0] for s in turns_by_label[k]))
    raw_to_name = {raw: f"speaker_{i + 1}" for i, raw in enumerate(order)}
    named = {raw_to_name[r]: v for r, v in turns_by_label.items()}

    assigned: list[str | None] = []
    last: str | None = None
    for seg in segments:
        s0, s1 = float(seg["start"]), float(seg["end"])
        dur = max(s1 - s0, 1e-6)
        best, best_overlap = None, 0.0
        for name, t in named.items():
            ov = sum(_overlap(s0, s1, t0, t1) for t0, t1 in t)
            if ov > best_overlap:
                best, best_overlap = name, ov
        if best is not None and best_overlap < 0.1 * dur:
            best = None  # segment is almost entirely silence
        if best is None:
            best = last  # inherit the previous speaker
        else:
            last = best
        assigned.append(best)

    speakers_info = [
        {
            "id": raw_to_name[raw],
            "raw": raw,
            "first_at": round(min(a for a, _ in turns_by_label[raw]), 2),
            "duration": round(sum(b - a for a, b in turns_by_label[raw]), 1),
        }
        for raw in order
    ]
    return assigned, speakers_info


def _get_pipeline(model_name: str, token: str):
    if model_name not in _PIPELINES:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(model_name, token=token)
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception:  # pragma: no cover - best effort
            pass
        _PIPELINES[model_name] = pipeline
    return _PIPELINES[model_name]


def run_diarize(task, settings: Settings) -> dict | None:
    from .asr import SAMPLE_RATE, read_wav

    seg_file = task.work_dir / "segments.json"
    if not seg_file.exists():
        raise RuntimeError("segments.json is missing — run the transcribe stage first")
    segments = json.loads(seg_file.read_text(encoding="utf-8"))["segments"]

    if not diarization_available():
        task.mark_skipped("diarize", reason="pyannote.audio is not installed (install the 'diarize' extra)")
        log.info("    pyannote.audio not installed — speaker diarization skipped")
        return None
    token = hf_token()
    if token is None:
        task.mark_skipped("diarize", reason="no HF token (HF_TOKEN); a later run with a token retries this stage")
        log.info("    HF token not found — speaker diarization skipped")
        return None

    model_name = settings.diarize_model
    t0 = time.time()
    log.info("    loading pyannote (%s)...", model_name)
    try:
        pipeline = _get_pipeline(model_name, token)
    except Exception as e:
        task.mark_skipped("diarize", reason=f"model unavailable: {str(e)[:200]}")
        log.warning("    pyannote model unavailable — speaker diarization skipped: %s", str(e)[:200])
        return None
    log.info("    model ready (%.0f s)", time.time() - t0)

    import torch

    # Pass the waveform in memory: no dependency on torchcodec/ffmpeg decoding inside pyannote.
    waveform = torch.from_numpy(read_wav(task.work_dir / "audio.wav")).unsqueeze(0)
    t0 = time.time()
    log.info("    diarizing (takes roughly as long as the audio on CPU)...")
    output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": task.video_id})
    elapsed = time.time() - t0
    log.info("    diarization finished: %.0f s", elapsed)

    # pyannote 4 returns DiarizeOutput; the "exclusive" variant (no overlapping turns)
    # is meant for aligning with transcripts. pyannote 3.x returns an Annotation.
    annotation = output
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        if getattr(output, attr, None) is not None:
            annotation = getattr(output, attr)
            break
    turns: dict[str, list] = {}
    # itertracks(yield_label=True) yields (segment, track, label) — label is the speaker
    for turn, _track, label in annotation.itertracks(yield_label=True):
        turns.setdefault(str(label), []).append((turn.start, turn.end))

    assigned, speakers_info = assign_speakers(turns, segments)
    out = {
        "model": model_name,
        "speakers": speakers_info,
        "elapsed": round(elapsed, 1),
        "segments": [{"speaker": s} for s in assigned],
    }
    (task.work_dir / "speakers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    task.mark_done("diarize", artifact="speakers.json", n_speakers=len(speakers_info), config={"model": model_name})
    log.info("    speakers: %d", len(speakers_info))
    return out
