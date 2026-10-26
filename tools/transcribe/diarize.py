from __future__ import annotations

import json
import os
import time

DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"


def resolve_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token

        tok = get_token()
    except Exception:
        return None
    return tok or None


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _assign_speakers(annotation, segments: list) -> tuple[list, list]:
    turns: dict[str, list] = {}
    for turn, label, _track in annotation.itertracks(yield_label=True):
        turns.setdefault(str(label), []).append((turn.start, turn.end))

    order = sorted(turns, key=lambda k: min(s[0] for s in turns[k]))
    raw_to_name = {raw: f"speaker_{i + 1}" for i, raw in enumerate(order)}
    named = {raw_to_name[r]: v for r, v in turns.items()}

    assigned = []
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
            best = None  # сегмент почти весь в молчании
        if best is None:
            best = last  # наследуем предыдущего
        else:
            last = best
        assigned.append(best)

    speakers_info = [
        {
            "id": name,
            "raw": next(r for r, n in raw_to_name.items() if n == name),
            "first_at": round(min(a for a, _ in t), 2),
            "duration": round(sum(b - a for a, b in t), 1),
        }
        for name, t in sorted(named.items(), key=lambda kv: min(a for a, _ in kv[1]))
    ]
    return assigned, speakers_info


def run_diarize(task, cfg: dict) -> dict | None:
    wav = task.work_dir / "audio.wav"
    seg_file = task.work_dir / "segments.json"
    if not seg_file.exists():
        raise RuntimeError("нет segments.json — сначала выполните стадию transcribe")
    segments = json.loads(seg_file.read_text(encoding="utf-8"))["segments"]

    model_name = cfg.get("diarize_model", DIARIZE_MODEL)
    token = resolve_token()
    if token is None:
        task.mark_skipped(
            "diarize",
            reason="нет HF-токена (HF_TOKEN); повторный запуск с токеном повторит стадию",
        )
        print("    HF-токен не найден — разделение говорящих пропущено")
        return None

    t0 = time.time()
    print(f"    загрузка pyannote ({model_name})...")
    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except Exception as e:
        task.mark_skipped("diarize", reason=f"модель недоступна: {str(e)[:200]}")
        print(f"    pyannote-модель недоступна — разделение говорящих пропущено: {str(e)[:200]}")
        return None
    print(f"    модель загружена ({time.time() - t0:.0f} с)")

    t0 = time.time()
    print("    разделение говорящих (CPU, время ~ длительности аудио)...")
    output = pipeline({"uri": task.video_id, "audio": str(wav)}, protocol="diarization")
    elapsed = time.time() - t0
    print(f"    разделение завершено: {elapsed:.0f} с")

    assigned, speakers_info = _assign_speakers(output, segments)
    out = {
        "model": model_name,
        "speakers": speakers_info,
        "elapsed": round(elapsed, 1),
        "segments": [{"speaker": s} for s in assigned],
    }
    out_file = task.work_dir / "speakers.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    task.mark_done("diarize", artifact="speakers.json", n_speakers=len(speakers_info))
    print(f"    говорящих: {len(speakers_info)}")
    return out
