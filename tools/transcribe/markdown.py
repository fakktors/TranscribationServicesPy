from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TURN_MAX_SEGMENTS = 30  # разбивка на абзацы, если говорящие не разделены


def _fmt_time(seconds: float) -> str:
    t = max(0, int(round(seconds)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _segment_text(seg: dict, uncertain: float) -> str:
    """Текст сегмента; слова с вероятностью ниже порога помечаются (?слово?)."""
    words = seg.get("words") or []
    if not words:
        return (seg.get("text") or "").strip()
    parts = []
    for w in words:
        core = (w.get("w") or "").strip()
        if not core:
            continue
        p = w.get("p")
        token = f"(?{core}?)" if p is not None and p < uncertain else core
        parts.append((" " if parts else "") + token)
    return "".join(parts)


def _segment_chunk(seg: dict, base_lang: str | None, uncertain: float) -> str:
    text = _segment_text(seg, uncertain)
    if seg.get("switched") and seg.get("lang") and seg["lang"] != base_lang:
        text = f"({seg['lang']}) {text}"
    if not text:
        return ""
    return f"[{_fmt_time(seg['start'])}] {text}"


def _build_turns(segments: list, speaker_ids: list | None) -> list:
    """Группа сегментов → список абзацев (speaker_id | None, [сегменты]).

    Говорящий-None (пауза, нет diarize) наследует предыдущего, чтобы
    реплика с паузой внутри оставалась одним абзацем.
    """
    if speaker_ids is None:
        speaker_ids = [None] * len(segments)
    resolved, last = [], None
    for s in speaker_ids:
        if s is not None:
            last = s
        resolved.append(last)

    turns = []
    for seg, speaker in zip(segments, resolved):
        if turns and turns[-1][0] == speaker and len(turns[-1][1]) < TURN_MAX_SEGMENTS:
            turns[-1][1].append(seg)
        else:
            turns.append([speaker, [seg]])
    return [(speaker, segs) for speaker, segs in turns if any(_segment_chunk(s, None, 1e9) for s in segs)]


def run_markdown(task, cfg: dict) -> str:
    seg_file = task.work_dir / "segments.json"
    if not seg_file.exists():
        raise RuntimeError("нет segments.json — сначала выполните стадию transcribe")
    info_file = task.work_dir / "info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    seg_data = json.loads(seg_file.read_text(encoding="utf-8"))
    segments = seg_data["segments"]
    base_lang = seg_data.get("language")
    uncertain = float(cfg.get("uncertain_threshold", 0.5))

    speaker_ids = None
    n_speakers = 0
    speakers_file = task.work_dir / "speakers.json"
    if speakers_file.exists() and task.stages["diarize"].status == "done":
        sp = json.loads(speakers_file.read_text(encoding="utf-8"))
        if len(sp.get("segments", [])) == len(segments):
            speaker_ids = [d.get("speaker") for d in sp["segments"]]
            n_speakers = len(sp.get("speakers", []))

    turns = _build_turns(segments, speaker_ids)

    lines = [f"# {info.get('title', task.video_id)}", info.get("webpage_url") or task.url, ""]
    diar = f"{n_speakers} говорящих" if n_speakers else "выкл"
    meta = (
        f"Модель: {seg_data.get('model', '?')} · Язык: {base_lang or '?'} · "
        f"Длительность: {_fmt_time(seg_data.get('duration', 0))} · "
        f"Разделение говорящих: {diar}"
    )
    lines.append(meta)
    lines.append("")
    for speaker, segs in turns:
        prefix = f"**{speaker.replace('speaker_', 'Speaker ')}:** " if speaker else ""
        body = " ".join(chunk for chunk in (_segment_chunk(s, base_lang, uncertain) for s in segs) if chunk)
        if body:
            lines.append(prefix + body)
            lines.append("")

    out_dir = Path(cfg.get("transcripts_dir") or (PROJECT_ROOT / "transcripts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task.video_id}.md"
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    task.mark_done("markdown", out=str(out_path))
    print(f"    markdown: {out_path} ({len(turns)} абзацев)")
    return str(out_path)
