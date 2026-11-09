# transcribation

[![CI](https://github.com/fakktors/TranscribationServicesPy/actions/workflows/ci.yml/badge.svg)](https://github.com/fakktors/TranscribationServicesPy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Self-hosted **YouTube → transcript** service for AI agents and humans.
Whisper speech recognition + pyannote speaker diarization → Markdown with
timestamps and speaker labels (plus a JSON twin for machines).

- **For agents:** MCP endpoint (works with [Hermes Agent](https://hermes-agent.nousresearch.com/), Claude, any MCP client) and a plain REST API.
- **For servers:** one `docker compose up -d`; CPU by default, NVIDIA GPU optional.
- **For laptops:** a CLI that uses the Apple Silicon GPU via MLX.
- **Resumable and cached:** each stage is persisted; the same video is never processed twice.

```
url → download (yt-dlp) → transcribe (Whisper) → diarize (pyannote) → transcripts/<id>.md + .json
```

Example output:

```markdown
# How We Built It — Interview

https://www.youtube.com/watch?v=...

Model: large-v3-turbo · Language: ru · Duration: 42:10 · Speaker diarization: 2 speakers

**Speaker 1:** [00:00] Всем привет, сегодня у нас в гостях... [00:07] (en) Thanks for having me.

**Speaker 2:** [00:12] Спасибо. Начнём с (?Kubernetes?) ...
```

`[mm:ss]` — segment start · `(?word?)` — low-confidence word · `(en)` — segment in another language.

## Quick start: server (Docker)

Requirements: Docker with Compose v2.24+, ~4 GB RAM (8 GB with diarization), x86-64 or arm64.

```bash
git clone https://github.com/fakktors/TranscribationServicesPy.git
cd TranscribationServicesPy
cp .env.example .env
```

Edit `.env`: set `TRANSCRIBE_API_TOKEN` (`openssl rand -hex 32`) and, for speaker
labels, `HF_TOKEN` (accept the terms of
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) first).

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/healthz
```

Try it (the first request also downloads the model, ~1.5 GB):

```bash
curl -X POST "http://127.0.0.1:8000/v1/transcripts?wait=300" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}'
curl "http://127.0.0.1:8000/v1/transcripts/jNQXAC9IVRw.md" -H "Authorization: Bearer $TOKEN"
```

Notes:

- The port is published on `127.0.0.1` only. To reach it from other machines put a
  TLS reverse proxy in front (Caddy/nginx/Traefik) or use Tailscale; or set
  `BIND_ADDR=0.0.0.0` — never without a token.
- Data lives in Docker volumes: `data` (transcripts, work files) and `hf-cache` (models).
- Speed on CPU with `large-v3-turbo`: roughly 1–4× realtime depending on cores;
  diarization adds about 1× audio length. NVIDIA GPU:
  `docker compose -f compose.yaml -f compose.gpu.yaml up -d --build` (experimental).
- Smaller image without torch/diarization: `WITH_DIARIZATION=false` in `.env`.
- YouTube may block datacenter IPs ("Sign in to confirm you're not a bot"): mount a
  `cookies.txt` and set `TRANSCRIBE_YTDLP_COOKIES`, or use `TRANSCRIBE_YTDLP_ARGS=--proxy ...`.
- Update: `git pull && docker compose up -d --build`. yt-dlp changes often — rebuild
  when downloads start failing.

## Use with Hermes Agent

Add the MCP server to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  transcribation:
    url: "http://127.0.0.1:8000/mcp"
    headers:
      Authorization: "Bearer ${TRANSCRIBE_API_TOKEN}"
    timeout: 300
```

put `TRANSCRIBE_API_TOKEN=...` into `~/.hermes/.env`, run `/reload-mcp`, and install
the skill so the agent knows when and how to use it:

```bash
mkdir -p ~/.hermes/skills/youtube-transcribe
cp integrations/hermes/SKILL.md ~/.hermes/skills/youtube-transcribe/
```

Then just ask: *"Summarize this video: https://youtu.be/..."*. Details:
[integrations/hermes/README.md](integrations/hermes/README.md).

### MCP tools

| Tool | What it does |
|---|---|
| `transcribe_video(url, language?, prompt?, diarize=true, allow_long=false, wait_seconds=90, max_chars=40000)` | Queues the video, waits up to `wait_seconds`, returns the transcript (or a `status: running` header) |
| `get_transcript(id, format="markdown"\|"json", offset=0, max_chars=40000, wait_seconds=60)` | Status or a page of the transcript; long transcripts are paged with `offset` |
| `list_transcripts(limit=20)` | Already transcribed videos |

Every result starts with a `status:` line (`done`, `queued`, `running`, `error`, ...).

## REST API

All `/v1` routes require `Authorization: Bearer <TRANSCRIBE_API_TOKEN>` when a token is set.
Interactive docs: `http://<host>:8000/docs`.

| Method | Path | |
|---|---|---|
| `POST` | `/v1/transcripts?wait=N` | Body `{"url", "language"?, "prompt"?, "diarize"?, "allow_long"?, "force"?}`. `200` when done (cached or finished within `wait`), `202` when queued/running |
| `GET` | `/v1/transcripts/{id}?wait=N` | Job status: `status`, `stage`, `queue_position`, per-stage status, `links` |
| `GET` | `/v1/transcripts/{id}.md` | Markdown transcript (`409` if not ready) |
| `GET` | `/v1/transcripts/{id}.json` | Segments with `start`, `end`, `speaker`, `lang`, `text` |
| `GET` | `/v1/transcripts` | List of transcripts, newest first |
| `DELETE` | `/v1/transcripts/{id}` | Remove a transcript and its work files |
| `GET` | `/healthz` | Liveness + backend info (no auth) |

`{id}` is the YouTube video id. Jobs run one at a time; submitting a video that is
already queued or done returns the existing job.

## Local CLI (macOS / Linux)

Requirements: Python 3.10+, `ffmpeg`, `deno` (yt-dlp uses it for YouTube).

```bash
brew install ffmpeg deno                    # Debian/Ubuntu: apt install ffmpeg; deno from deno.com
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx,diarize]"             # Apple Silicon
pip install -e ".[faster-whisper,diarize]"  # Linux / Intel Mac / NVIDIA
export HF_TOKEN=...                          # optional, for speaker labels

transcribation "https://www.youtube.com/watch?v=jNQXAC9IVRw"
transcribation "https://youtu.be/..." ru --prompt "Kubernetes, Istio" --allow-long
```

| Option | |
|---|---|
| `language` (positional) | `ru`, `en`, ... (default: auto-detect, with per-segment language switching) |
| `--backend` | `auto` (MLX on Apple Silicon, else faster-whisper), `mlx`, `faster-whisper` |
| `--model` | Whisper model (default `mlx-community/whisper-large-v3-mlx` / `large-v3-turbo`) |
| `--prompt "a, b"` | Vocabulary hint (names, jargon) |
| `--no-diarize` | No speaker labels |
| `--allow-long` | Allow videos over 1 hour |
| `--stop STAGE` | Stop after `download` / `transcribe` / `diarize` / `markdown` |
| `--reset` | Start over (deletes `work/<id>/`) |
| `--uncertain T` | Confidence threshold for `(?word?)` (default 0.5) |
| `--data-dir DIR` | Where `work/` and `transcripts/` go (default `.`) |
| `-q` | Print only the result line |

Quote the URL (`?`, `&` are special in the shell). The last line is `OK <path>` on
success or `ERROR <stage>: <message>` (exit code 1) — re-run to resume from that stage.
`python tools/fetch_transcript.py ...` still works as an alias. You can also run the
server locally: `pip install -e ".[server,mlx]" && transcribation-server`.

## Configuration

All settings are environment variables; see [.env.example](.env.example) for the full,
commented list. The most important ones:

| Variable | Default | |
|---|---|---|
| `TRANSCRIBE_API_TOKEN` | — | Bearer token for `/v1` and `/mcp` |
| `HF_TOKEN` | — | Enables speaker diarization |
| `TRANSCRIBE_MODEL` | backend default | e.g. `large-v3`, `large-v3-turbo`, `small` |
| `TRANSCRIBE_DEVICE` | `auto` | `cpu` / `cuda` (faster-whisper) |
| `TRANSCRIBE_ALLOWED_HOSTS` | YouTube domains | `*` allows any site yt-dlp supports |
| `TRANSCRIBE_MAX_DURATION` | `3600` | Seconds; longer videos need `allow_long` |
| `TRANSCRIBE_KEEP_AUDIO` | `true` (`false` in `.env.example`) | Keep the WAV after transcription |
| `TRANSCRIBE_DATA_DIR` | `.` (`/data` in Docker) | Storage root |

## How it works

Each video gets `work/<id>/` with `state.json`; every stage writes its artifact there
(`info.json`, `audio.wav`, `segments.json`, `speakers.json`). Re-runs skip finished
stages, re-transcribe only if the model/language/prompt changed, and retry a skipped
diarization once a token is available. More in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Claude Code

`/practices <url>` ([.claude/commands/practices.md](.claude/commands/practices.md)) turns a
video into a best-practices document built from its transcript.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE). Model weights have their own licenses (Whisper: MIT; pyannote
models: gated on Hugging Face, check their terms). Respect YouTube's Terms of Service
and the rights of content owners.
