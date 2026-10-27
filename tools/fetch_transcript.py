#!/usr/bin/env python3
"""YouTube video -> transcript: download -> transcribe -> diarize -> markdown.

Re-running resumes from the last completed stage (work/<video_id>/state.json).

Usage:
    python tools/fetch_transcript.py <youtube_url> [language] [options]

    language   ru / en (default: auto-detect)

Options:
    --model NAME        Whisper model (default mlx-community/whisper-large-v3-mlx)
    --prompt "term1, term2"   vocabulary hint for the model
    --no-diarize        skip speaker diarization
    --allow-long        allow videos longer than 1 hour
    --stop STAGE        stop after stage: download | transcribe | diarize | markdown
    --reset             start over (clears work/<video_id>/)
    --uncertain T       confidence threshold for (?word?) marks (default 0.5)

Output:
    Success: OK <path to transcripts/<video_id>.md>
    Failure: ERROR <stage>: <message>, exit code 1
"""

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from transcribe.asr import run_asr
from transcribe.diarize import DIARIZE_MODEL, resolve_token, run_diarize
from transcribe.download import extract_video_id, run_download
from transcribe.markdown import run_markdown
from transcribe.state import STAGES, Task

ROOT = TOOLS_DIR.parent
DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="YouTube video -> transcript (Whisper MLX + pyannote)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )
    ap.add_argument("url", help="YouTube video URL")
    ap.add_argument("language", nargs="?", default=None, help="language: ru / en (default: auto)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model (default {DEFAULT_MODEL})")
    ap.add_argument("--prompt", default=None, help="vocabulary terms, comma-separated (model hint)")
    ap.add_argument("--no-diarize", action="store_true", help="skip speaker diarization")
    ap.add_argument("--allow-long", action="store_true", help="allow videos longer than 1 hour")
    ap.add_argument("--stop", choices=STAGES, default="markdown", help="stop after this stage")
    ap.add_argument("--reset", action="store_true", help="start the task over")
    ap.add_argument("--uncertain", type=float, default=0.5, help="confidence threshold for (?word?) marks")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    video_id = extract_video_id(args.url)
    task = Task(ROOT / "work", video_id, args.url, reset=args.reset)
    print(f"Task: {video_id} (work dir {task.work_dir.relative_to(ROOT)}/)")

    cfg = {
        "model": args.model,
        "language": args.language or None,
        "language_switching": True,
        "lang_switch_threshold": 0.6,
        "lexicon_prompt": args.prompt,
        "diarize_model": DIARIZE_MODEL,
        "uncertain_threshold": args.uncertain,
    }

    if args.no_diarize and task.stages["diarize"].status in ("pending", "error"):
        task.mark_skipped("diarize", reason="--no-diarize")

    runners = {
        "download": lambda: run_download(task, allow_long=args.allow_long),
        "transcribe": lambda: run_asr(task, cfg),
        "diarize": lambda: run_diarize(task, cfg),
        "markdown": lambda: run_markdown(task, cfg),
    }

    stop_at = STAGES.index(args.stop)
    for i, name in enumerate(STAGES):
        stage = task.stages[name]

        if task.validate_done(name):
            print(f"  {name}: already done, skipping")
        elif stage.status == "skipped":
            if name == "diarize" and resolve_token():
                print(f"  {name}: was skipped, HF token is now available — retrying")
                stage.status = "pending"
                task.save()
            else:
                print(f"  {name}: skipped ({stage.detail.get('reason', '')})")
        else:
            print(f"  {name}:")
            try:
                runners[name]()
            except Exception as e:
                task.mark_error(name, str(e))
                print(f"ERROR {name}: {e}")
                return 1

        if i == stop_at:
            if name == "markdown" and task.stages["markdown"].status == "done":
                print(f"OK {task.stages['markdown'].detail.get('out')}")
            else:
                print(f"Stopped after stage {name} (--stop)")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
