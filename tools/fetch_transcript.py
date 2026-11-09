#!/usr/bin/env python3
"""Backwards-compatible entry point: ``python tools/fetch_transcript.py <url> [language] [options]``.

Prefer the installed command ``transcribation`` (``pip install -e .``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcribation.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
