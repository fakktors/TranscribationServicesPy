"""Command line: ``transcribation <url> [language] [options]``.

Output contract (stable, used by agents and scripts):
    success: last line ``OK <path to transcripts/<video_id>.md>``
    failure: last line ``ERROR <stage>: <message>``, exit code 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings
from .pipeline import Options, PipelineError, run_pipeline
from .state import STAGES, TaskBusyError


def setup_logging(quiet: bool = False) -> None:
    """Our progress lines at INFO; third-party libraries only from WARNING (they are noisy)."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(lambda r: r.name.startswith("transcribation") or r.levelno >= logging.WARNING)
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, handlers=[handler], force=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="transcribation",
        description="YouTube video -> Markdown transcript (Whisper + pyannote). "
        "Re-running resumes from the last completed stage.",
    )
    ap.add_argument("url", help="YouTube video URL (or a bare 11-character video id)")
    ap.add_argument("language", nargs="?", default=None, help="language code, e.g. ru / en (default: auto-detect)")
    ap.add_argument("--backend", choices=("auto", "mlx", "faster-whisper"), help="ASR backend (default: auto)")
    ap.add_argument("--model", help="Whisper model (default depends on the backend)")
    ap.add_argument("--prompt", help="vocabulary hint, comma-separated terms")
    ap.add_argument("--no-diarize", action="store_true", help="skip speaker diarization")
    ap.add_argument("--allow-long", action="store_true", help="allow videos longer than the duration limit (1 h)")
    ap.add_argument("--stop", choices=STAGES, default="markdown", help="stop after this stage")
    ap.add_argument("--reset", action="store_true", help="start over (deletes work/<video_id>/)")
    ap.add_argument("--uncertain", type=float, default=0.5, help="confidence threshold for (?word?) marks")
    ap.add_argument("--data-dir", help="where work/ and transcripts/ live (default: $TRANSCRIBE_DATA_DIR or .)")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only the final OK/ERROR line")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(quiet=args.quiet)

    settings = Settings.from_env().with_overrides(
        asr_backend=args.backend,
        model=args.model,
        data_dir=Path(args.data_dir).expanduser() if args.data_dir else None,
    )
    try:
        opts = Options(
            url=args.url,
            language=args.language,
            prompt=args.prompt,
            diarize=not args.no_diarize,
            allow_long=args.allow_long,
            uncertain=args.uncertain,
            stop=args.stop,
            reset=args.reset,
        )
        result = run_pipeline(opts, settings)
    except PipelineError as e:
        print(f"ERROR {e.stage}: {e.message}")
        return 1
    except (TaskBusyError, ValueError) as e:
        print(f"ERROR setup: {e}")
        return 1

    if result.markdown:
        print(f"OK {result.markdown}")
    else:
        print(f"Stopped after stage {result.stopped_after} (--stop)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
