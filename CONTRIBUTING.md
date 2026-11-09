# Contributing

Thanks for helping! Bug reports, docs fixes and pull requests are all welcome.

## Development setup

```bash
git clone https://github.com/fakktors/TranscribationServicesPy.git
cd TranscribationServicesPy
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[server,dev]"          # enough for tests: no models, no torch
pre-commit install                       # optional: ruff on every commit
```

To actually transcribe locally add a backend: `".[mlx]"` on Apple Silicon or
`".[faster-whisper]"` elsewhere, plus `".[diarize]"` for speaker labels. You also
need `ffmpeg` and, for YouTube, `deno` on `PATH`.

## Checks (same as CI)

```bash
ruff check . && ruff format --check . && pytest
docker build --build-arg WITH_DIARIZATION=false -t transcribation:dev .   # if you touched Docker
```

Tests never download models or videos: heavy stages are replaced with fakes
(`tests/conftest.py::fake_stages`). Keep it that way — add a fake rather than
a network call.

## Project layout

```
src/transcribation/
  config.py        settings from TRANSCRIBE_* environment variables
  pipeline.py      stage orchestration, caching/invalidation rules
  state.py         work/<id>/state.json + per-video file lock
  download.py      yt-dlp + ffmpeg -> 16 kHz mono WAV
  asr/             Whisper backends (mlx, faster-whisper) behind one interface
  diarize.py       pyannote speaker diarization
  markdown.py      transcripts/<id>.md + <id>.json
  cli.py           `transcribation` command
  server/          FastAPI app, job queue, MCP tools
integrations/      Hermes Agent skill + config, Claude Code command
```

Adding an ASR backend: implement `transcribe()` and `redecode_if_foreign()`
(see `asr/__init__.py::ASRBackend`) and register it in `get_backend()`.

## Pull requests

- One topic per PR; describe *why* in the description.
- Behaviour changes need a test; user-visible changes need README / `.env.example`
  updates and a line in `CHANGELOG.md` under **Unreleased**.
- The CLI output contract (`OK <path>` / `ERROR <stage>: <msg>`) and the REST/MCP
  shapes are public API — change them only with a clear note in the changelog.

## Releases

Maintainers: bump `__version__` in `src/transcribation/__init__.py`, move the
changelog entries under a version heading, tag `vX.Y.Z` and push the tag — the
`Release image` workflow publishes `ghcr.io/fakktors/transcribationservicespy`.
