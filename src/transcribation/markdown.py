"""Render stage: segments (+ speakers) -> transcripts/<id>.md and transcripts/<id>.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

TURN_MAX_SEGMENTS = 30  # paragraph splitting when speakers are not diarized


def fmt_time(seconds: float) -> str:
    t = max(0, int(round(seconds)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def segment_text(seg: dict, uncertain: float) -> str:
    """Segment text; words with a probability below the threshold are marked (?word?)."""
    words = seg.get("words") or []
    if not words:
        return (seg.get("text") or "").strip()
    parts = []
    for w in words:
        core = (w.get("w") or "").strip()
        if not core:
            continue
        p = w.get("p")
        parts.append(f"(?{core}?)" if p is not None and p < uncertain else core)
    return " ".join(parts)


def _segment_chunk(seg: dict, base_lang: str | None, uncertain: float) -> str:
    text = segment_text(seg, uncertain)
    if not text:
        return ""
    if seg.get("switched") and seg.get("lang") and seg["lang"] != base_lang:
        text = f"({seg['lang']}) {text}"
    return f"[{fmt_time(seg['start'])}] {text}"


def build_turns(segments: list, speaker_ids: list | None) -> list:
    """Segments -> paragraphs [(speaker_id | None, [segments])].

    A speaker of None (pause, no diarization) inherits the previous speaker so
    a turn with a pause inside stays a single paragraph.
    """
    if speaker_ids is None:
        speaker_ids = [None] * len(segments)
    resolved, last = [], None
    for s in speaker_ids:
        if s is not None:
            last = s
        resolved.append(last)

    turns: list[list] = []
    for seg, speaker in zip(segments, resolved):
        if not segment_text(seg, 0.0):
            continue
        if turns and turns[-1][0] == speaker and len(turns[-1][1]) < TURN_MAX_SEGMENTS:
            turns[-1][1].append(seg)
        else:
            turns.append([speaker, [seg]])
    return [(speaker, segs) for speaker, segs in turns]


def _speaker_label(speaker: str) -> str:
    return speaker.replace("speaker_", "Speaker ")


def run_markdown(task, transcripts_dir: Path, *, use_speakers: bool = True, uncertain: float = 0.5) -> dict:
    seg_file = task.work_dir / "segments.json"
    if not seg_file.exists():
        raise RuntimeError("segments.json is missing — run the transcribe stage first")
    info_file = task.work_dir / "info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    seg_data = json.loads(seg_file.read_text(encoding="utf-8"))
    segments = seg_data["segments"]
    base_lang = seg_data.get("language")

    speaker_ids = None
    speakers: list = []
    speakers_file = task.work_dir / "speakers.json"
    if use_speakers and speakers_file.exists() and task.stages["diarize"].status == "done":
        sp = json.loads(speakers_file.read_text(encoding="utf-8"))
        if len(sp.get("segments", [])) == len(segments):
            speaker_ids = [d.get("speaker") for d in sp["segments"]]
            speakers = sp.get("speakers", [])

    turns = build_turns(segments, speaker_ids)
    title = info.get("title") or task.video_id
    url = info.get("webpage_url") or task.url

    lines = [f"# {title}", "", url, ""]
    diar = f"{len(speakers)} speakers" if speakers else "off"
    lines.append(
        f"Model: {seg_data.get('model', '?')} · Language: {base_lang or '?'} · "
        f"Duration: {fmt_time(seg_data.get('duration', 0))} · Speaker diarization: {diar}"
    )
    lines.append("")
    for speaker, segs in turns:
        prefix = f"**{_speaker_label(speaker)}:** " if speaker else ""
        body = " ".join(c for c in (_segment_chunk(s, base_lang, uncertain) for s in segs) if c)
        lines += [prefix + body, ""]

    transcripts_dir = Path(transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    md_path = transcripts_dir / f"{task.video_id}.md"
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    # Machine-readable twin: what agents and scripts should parse.
    resolved = speaker_ids or [None] * len(segments)
    doc = {
        "video_id": task.video_id,
        "title": title,
        "url": url,
        "channel": info.get("channel") or info.get("uploader"),
        "duration": seg_data.get("duration"),
        "language": base_lang,
        "model": seg_data.get("model"),
        "speakers": speakers,
        "segments": [
            {
                "start": s["start"],
                "end": s["end"],
                "speaker": spk,
                "lang": s.get("lang"),
                "text": (s.get("text") or "").strip(),
            }
            for s, spk in zip(segments, resolved)
            if (s.get("text") or "").strip()
        ],
    }
    json_path = transcripts_dir / f"{task.video_id}.json"
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    task.mark_done("markdown", out=str(md_path), json=str(json_path), n_speakers=len(speakers))
    log.info("    markdown: %s (%d paragraphs)", md_path, len(turns))
    return {"markdown": str(md_path), "json": str(json_path)}
