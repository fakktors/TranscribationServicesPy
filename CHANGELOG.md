# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-09

### Added
- HTTP service (`transcribation-server`): REST API under `/v1/transcripts`, MCP
  endpoint at `/mcp` (tools `transcribe_video`, `get_transcript`, `list_transcripts`),
  `/healthz`, optional bearer-token auth, host allowlist, single-worker job queue.
- `faster-whisper` ASR backend (CPU / NVIDIA), selected automatically off Apple Silicon.
- Docker image and `compose.yaml` for running as an always-on service; GPU override.
- Machine-readable `transcripts/<id>.json` next to the Markdown.
- Hermes Agent integration (`integrations/hermes/`): MCP config and a skill.
- Packaging (`pyproject.toml`, `transcribation` command), tests, CI, release workflow,
  contribution guide, security policy, MIT license.

### Fixed
- Speakers were grouped by pyannote *track* instead of *label*; pyannote 4
  `DiarizeOutput` is now handled (uses the exclusive, non-overlapping diarization).
- A diarize stage skipped for lack of an HF token was marked for retry but never
  re-run, and the Markdown was never re-rendered with speakers.
- Changing the model, language or prompt silently reused the old transcription.
- `--reset` only dropped `state.json` and re-used old audio and metadata.
- Two runs on the same video could corrupt each other's work dir (now file-locked).
- Downloaded source audio (`audio.m4a`/`.webm`) was kept next to the WAV.

### Changed
- Code moved from `tools/transcribe/` to the `src/transcribation/` package;
  `python tools/fetch_transcript.py` still works.
- `requirements.txt` replaced by extras: `mlx`, `faster-whisper`, `diarize`, `server`, `dev`.
- Logging instead of `print`; `-q/--quiet` CLI flag; `--backend`, `--data-dir` options.
