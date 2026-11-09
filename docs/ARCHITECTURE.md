# Architecture, review and roadmap

## Overview

```
                   ┌──────────── transcribation-server (FastAPI) ────────────┐
Hermes / agents ──►│ /mcp  (MCP tools)   ┐                                   │
scripts, curl  ───►│ /v1/transcripts     ├─► JobManager ─► worker thread ─┐  │
                   │ /healthz            ┘   (dedupe by video id)         │  │
                   └──────────────────────────────────────────────────────┼──┘
CLI `transcribation` ──────────────────────────────────────────────────────┤
                                                                           ▼
                  pipeline.run_pipeline:  download → transcribe → diarize → markdown
                                          yt-dlp     mlx-whisper   pyannote   .md + .json
                                          ffmpeg     faster-whisper
                  storage:  <DATA_DIR>/work/<id>/{state.json,info.json,audio.wav,segments.json,speakers.json}
                            <DATA_DIR>/transcripts/<id>.{md,json}
```

Design decisions:

- **Video id is the job id.** Jobs are idempotent: resubmitting a video returns the
  running job or the cached result. No database — `state.json` is the source of truth,
  so results survive restarts and the CLI and server can share a data dir.
- **One worker.** ASR and diarization saturate CPU/GPU; parallel jobs would only
  thrash memory. Models stay loaded between jobs.
- **Stage cache with fingerprints.** The transcribe stage stores
  `{backend, model, language, prompt}`; a mismatch re-runs it (and diarize), not the
  download. Markdown is always re-rendered (milliseconds).
- **Agent-first API.** MCP tools block for a bounded time (`wait_seconds`) and return
  plain text with a `status:` header, so a single tool call is enough for short videos
  and long ones degrade into a polling loop the agent understands. Transcripts are
  paged by characters to protect the agent's context window.

## Review (2026-11) — findings in v0.1

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | Critical | Only runs on Apple Silicon (MLX); impossible to host on a Linux server or in Docker | Fixed: `faster-whisper` backend, auto-selected |
| 2 | Critical | No programmatic interface for agents — only a CLI with free-form stdout | Fixed: REST + MCP server |
| 3 | High | Diarization grouped speakers by pyannote *track* instead of *label*; pyannote 4 output type not handled | Fixed |
| 4 | High | Skipped diarization was marked "retry" but never executed; Markdown never refreshed | Fixed |
| 5 | High | Changing model/language/prompt silently reused the cached transcription | Fixed: fingerprints |
| 6 | Medium | `--reset` didn't clear artifacts; concurrent runs could clobber a work dir | Fixed: rmtree + flock |
| 7 | Medium | yt-dlp on servers: no JS runtime (required for YouTube since 2025.11), no cookies/proxy options | Fixed: deno in image, `TRANSCRIBE_YTDLP_COOKIES` / `_ARGS` |
| 8 | Medium | Disk growth: source audio and WAVs kept forever | Fixed: source deleted, `TRANSCRIBE_KEEP_AUDIO` |
| 9 | Medium | Accepting arbitrary URLs from the network = SSRF via yt-dlp | Fixed: host allowlist, bearer token |
| 10 | Low | `sys.path` hacks, `print` logging, `__pycache__` committed, README referenced an untracked file | Fixed |
| 11 | Low | No license, tests, CI, contribution docs | Fixed |

## Roadmap (not done yet)

- [ ] **Persistent queue.** Queued (not yet started) jobs are lost on restart; a client
      gets `incomplete` and must resubmit. A tiny SQLite queue would fix it.
- [ ] **Retention.** Automatic cleanup of `work/` and old transcripts (`TRANSCRIBE_RETENTION_DAYS`).
- [ ] **YouTube subtitles fast path.** Use existing manual captions when present
      (seconds instead of minutes), Whisper as a fallback.
- [ ] **Progress inside a stage** (percent of audio decoded) in the job status.
- [ ] **Webhooks** (`callback_url`) so agents don't need to poll long jobs.
- [ ] **pyannote community-1 / pyannoteAI** diarization models as options.
- [ ] **GPU image** tested end-to-end in CI (currently documented as experimental).
- [ ] Rate limiting per token if the service is shared by several users.
