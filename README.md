# TranscribationServicesPy — YouTube video → transcript

Transcription pipeline: **Whisper via MLX** (Apple Silicon) + **pyannote** speaker diarization → Markdown.

```
url → download → transcribe → diarize → markdown
```

| Stage | What it does | Artifacts in `work/<video_id>/` |
|---|---|---|
| `download` | yt-dlp: audio (16 kHz mono WAV) + video metadata | `audio.wav`, `info.json` |
| `transcribe` | Whisper (MLX), auto language detection, ru/en switching | `segments.json` |
| `diarize` | pyannote/speaker-diarization-3.1 (needs HF token; **skipped** without one) | `speakers.json` |
| `markdown` | final transcript | `transcripts/<video_id>.md` |

Task state lives in `work/<video_id>/state.json`: each stage is persisted, so a re-run **resumes from the last completed stage** instead of starting over.

## Setup

Apple Silicon recommended (MLX uses the GPU). ffmpeg is required:

```bash
brew install ffmpeg

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional, for speaker diarization:
export HF_TOKEN=...   # plus one-time access agreement for pyannote/speaker-diarization-3.1 on Hugging Face
```

Without an HF token the pipeline still works — the `diarize` stage is skipped (marked `skipped`) and will run on the next launch with a token.

## Usage

```bash
python tools/fetch_transcript.py "<youtube_url>" [language] [options]
```

- `language` — `ru` / `en` (default: auto-detect)
- `--model NAME` — Whisper model (default `mlx-community/whisper-large-v3-mlx`)
- `--prompt "term1, term2"` — vocabulary hint for the model (a hint, not a command: a term appears in the text only if it is actually in the audio)
- `--no-diarize` — skip speaker diarization
- `--allow-long` — allow videos longer than 1 hour (the limit applies by default)
- `--stop STAGE` — stop after: `download | transcribe | diarize | markdown`
- `--reset` — start over (clears `work/<video_id>/`)
- `--uncertain T` — confidence threshold for `(?word?)` marks (default `0.5`)

```bash
python tools/fetch_transcript.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python tools/fetch_transcript.py "https://youtu.be/..." ru --allow-long
```

Quote the URL — otherwise `?`, `=`, `&` are interpreted by the shell.

On success the script prints `OK <path to transcripts/<video_id>.md>`; on failure `ERROR <stage>: <message>` with exit code 1 — re-run to resume from that stage.

## Transcript format

Markdown: title, link, metadata (model, language, duration, speaker count), then paragraphs — one per speaker turn (without diarization: blocks of 30 segments). Each segment starts with an `[mm:ss]` timestamp; low-confidence words are marked `(?word?)`; a segment re-decoded in a different language is marked `(en)` / `(ru)`.

## Claude Code

The `/practices <url>` command (`.claude/commands/practices.md`) turns a video into a best-practices document from its transcript.
