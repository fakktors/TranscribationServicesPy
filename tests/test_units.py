import json
from pathlib import Path

import pytest

from transcribation.diarize import assign_speakers
from transcribation.download import extract_video_id, is_url_allowed, normalize_url
from transcribation.markdown import build_turns, fmt_time, run_markdown, segment_text
from transcribation.state import Task

from .conftest import fake_segments


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=10", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?si=abc", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id(url, expected):
    assert extract_video_id(url) == expected


def test_extract_video_id_fallback_is_stable_hash():
    a = extract_video_id("https://example.com/video/1")
    assert a.startswith("video-") and a == extract_video_id("https://example.com/video/1")


def test_normalize_url_bare_id():
    assert normalize_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_url_allowlist():
    hosts = ("youtube.com", "youtu.be")
    assert is_url_allowed("https://www.youtube.com/watch?v=x", hosts)
    assert is_url_allowed("https://youtu.be/x", hosts)
    assert not is_url_allowed("https://evil-youtube.com/x", hosts)
    assert not is_url_allowed("file:///etc/passwd", hosts)
    assert not is_url_allowed("http://169.254.169.254/", hosts)
    assert is_url_allowed("https://vimeo.com/1", ("*",))


def test_fmt_time():
    assert fmt_time(5) == "00:05"
    assert fmt_time(3725) == "1:02:05"


def test_segment_text_marks_uncertain_words():
    seg = fake_segments()["segments"][0]
    assert segment_text(seg, 0.5) == "Hello (?there.?)"
    assert segment_text(seg, 0.1) == "Hello there."


def test_build_turns_inherits_speaker_across_pauses():
    segs = fake_segments()["segments"]
    turns = build_turns(segs, ["speaker_1", None, "speaker_2"])
    assert [(s, len(x)) for s, x in turns] == [("speaker_1", 2), ("speaker_2", 1)]


def test_assign_speakers_orders_by_first_appearance():
    segs = [{"start": 0, "end": 2}, {"start": 2, "end": 4}, {"start": 4, "end": 6}]
    turns = {"SPK_B": [(2.0, 4.0)], "SPK_A": [(0.0, 2.0), (4.0, 6.0)]}
    assigned, info = assign_speakers(turns, segs)
    assert assigned == ["speaker_1", "speaker_2", "speaker_1"]
    assert [s["raw"] for s in info] == ["SPK_A", "SPK_B"]


def test_task_state_roundtrip_and_validate(tmp_path):
    t = Task(tmp_path, "vid", "u")
    t.mark_done("download", artifact="audio.wav")
    assert not t.validate_done("download")  # artifact missing
    (t.work_dir / "audio.wav").write_bytes(b"x")
    assert Task(tmp_path, "vid", "u").validate_done("download")
    assert Task(tmp_path, "vid", "u", reset=True).stages["download"].status == "pending"


def test_task_lock_is_exclusive(tmp_path):
    from transcribation.state import TaskBusyError

    with Task(tmp_path, "vid", "u"), pytest.raises(TaskBusyError):
        Task(tmp_path, "vid", "u").lock()


def test_markdown_render(tmp_path):
    t = Task(tmp_path / "work", "vid", "https://youtu.be/vid")
    (t.work_dir / "segments.json").write_text(json.dumps(fake_segments()))
    (t.work_dir / "info.json").write_text(json.dumps({"title": "T", "webpage_url": "https://youtu.be/vid"}))
    out = run_markdown(t, tmp_path / "transcripts")
    md = Path(out["markdown"]).read_text(encoding="utf-8")
    assert md.startswith("# T\n")
    assert "[00:00] Hello (?there.?)" in md
    assert "[00:04] (ru) Привет." in md
    doc = json.loads(Path(out["json"]).read_text(encoding="utf-8"))
    assert [s["text"] for s in doc["segments"]] == ["Hello there.", "Привет.", "Bye."]
