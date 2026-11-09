from dataclasses import replace

from transcribation.pipeline import Options, is_up_to_date, run_pipeline

from .conftest import URL, VIDEO_ID


def test_full_run_then_cached(settings, fake_stages):
    res = run_pipeline(Options(url=URL), settings)
    assert res.markdown and res.markdown.endswith(f"{VIDEO_ID}.md")
    assert fake_stages == ["download", "transcribe:tiny", "diarize"]
    assert is_up_to_date(Options(url=URL), settings)

    fake_stages.clear()
    run_pipeline(Options(url=URL), settings)
    assert fake_stages == []  # skipped diarize is not retried without a token


def test_model_change_invalidates_transcribe_without_redownload(settings, fake_stages):
    run_pipeline(Options(url=URL), settings)
    fake_stages.clear()
    other = replace(settings, model="small")
    assert not is_up_to_date(Options(url=URL), other)
    run_pipeline(Options(url=URL), other)
    assert fake_stages == ["transcribe:small", "diarize"]


def test_skipped_diarize_is_retried_when_token_appears(settings, fake_stages, monkeypatch):
    run_pipeline(Options(url=URL), settings)
    fake_stages.clear()
    monkeypatch.setattr("transcribation.pipeline.hf_token", lambda: "hf_x")
    monkeypatch.setattr("transcribation.pipeline.diarization_available", lambda: True)
    run_pipeline(Options(url=URL), settings)
    assert fake_stages == ["diarize"]


def test_audio_deleted_when_keep_audio_false_and_not_redownloaded(settings, fake_stages):
    s = replace(settings, keep_audio=False)
    run_pipeline(Options(url=URL), s)
    assert not (s.work_dir / VIDEO_ID / "audio.wav").exists()
    fake_stages.clear()
    run_pipeline(Options(url=URL), s)
    assert fake_stages == []


def test_stop_after_download(settings, fake_stages):
    res = run_pipeline(Options(url=URL, stop="download"), settings)
    assert res.markdown is None and fake_stages == ["download"]


def test_no_diarize_option(settings, fake_stages):
    run_pipeline(Options(url=URL, diarize=False), settings)
    assert "diarize" not in fake_stages
