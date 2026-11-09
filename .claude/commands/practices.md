---
description: YouTube URL -> transcript -> best practices in .md
argument-hint: <youtube url> [ru,en]
allowed-tools: Bash(python tools/fetch_transcript.py:*), Read, Write
---

Create a best-practices document from the video.

1. Run `python tools/fetch_transcript.py $ARGUMENTS`.
   - If the last line is `ERROR <stage>: ...`, stop and show me the error;
     do not invent anything.
2. Read the transcript file (the path is in the script output). If it is
   long, read it in parts, keep a running list of practices, then merge.
3. Build the document strictly from the transcript text, with no assumptions
   or general knowledge "from yourself". Auto-captions contain recognition
   errors: fix terms only when the meaning is obvious, mark doubtful ones "(?)".
4. Document structure:
   - **The point**: 3 sentences on what the video is about and who it is for.
   - **Best practices**: numbered list. For each: the rule,
     why it matters, how to apply it in practice, time code `[mm:ss]`.
   - **Checklist**: short "done / not done" items.
   - **Anti-patterns**: what the author advises against.
   - **Debatable / to verify**: claims without justification or
     outdated advice.
5. Save the result to `output/<video_id>.md` (first line: the video
   title and the URL).
6. In your reply, show only the file path and the 3–5 most valuable practices.
