---
name: youtube-transcribe
description: Transcribe YouTube videos into timestamped, speaker-labelled Markdown via the self-hosted transcribation service. Use when the user shares a YouTube link and wants a transcript, summary, quotes, notes, or answers about what was said in the video.
version: 1.0.0
metadata:
  hermes:
    tags: [youtube, transcription, video, summarization]
---

# YouTube transcription

A self-hosted service turns a YouTube URL into a transcript. It is reachable as
the MCP server `transcribation` (tools below). Results are cached per video, so
asking again for the same video is instant.

## When to use

- The user sends a YouTube link (youtube.com, youtu.be, shorts, live) and asks
  what is said in it, wants a summary, notes, quotes, a translation, or Q&A.
- Do NOT use it for videos the user only wants to watch or for non-YouTube links
  (the server rejects other hosts unless configured otherwise).

## Workflow

1. Call `transcribe_video(url=..., wait_seconds=90)`.
   - Pass `language` ("ru", "en", ...) only if the user states it; otherwise auto-detect.
   - Pass `prompt` with names/jargon you already know appear in the video
     (comma-separated) — it improves spelling of terms.
   - Videos longer than 1 hour are rejected unless `allow_long=true`; ask the user
     first, because transcription of long videos takes a while.
2. If the result starts with `status: queued` or `status: running`, tell the user it
   is in progress, then call `get_transcript(id=..., wait_seconds=60)` until
   `status: done`. CPU transcription runs at roughly 1-4x realtime.
3. If the header shows `(more: call again with offset=N)`, fetch the remaining
   pages with `get_transcript(id=..., offset=N)` before summarizing the whole video.
4. On `status: error`, report the error text verbatim (e.g. YouTube bot check,
   private video). Do not invent content.

## Reading the transcript

- `**Speaker 1:**` — paragraphs per speaker (when diarization is enabled on the server).
- `[mm:ss]` / `[h:mm:ss]` — segment start time; cite these when quoting.
- `(?word?)` — low-confidence word; treat it as uncertain, don't quote it as fact.
- `(en)` / `(ru)` — the segment was spoken in another language.
- `get_transcript(id, format="json")` returns segments with `start`, `end`,
  `speaker`, `text` — use it when you need exact timings.

Base every summary or answer strictly on the transcript text.

## Without MCP (REST fallback)

```bash
curl -s -X POST "$TRANSCRIBE_URL/v1/transcripts?wait=120" \
  -H "Authorization: Bearer $TRANSCRIBE_API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"url": "https://youtu.be/VIDEO_ID"}'
curl -s "$TRANSCRIBE_URL/v1/transcripts/VIDEO_ID?wait=60" -H "Authorization: Bearer $TRANSCRIBE_API_TOKEN"
curl -s "$TRANSCRIBE_URL/v1/transcripts/VIDEO_ID.md" -H "Authorization: Bearer $TRANSCRIBE_API_TOKEN"
```
