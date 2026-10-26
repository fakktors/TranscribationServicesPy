#!/usr/bin/env python3
"""Транскрипт YouTube-видео: конвейер download → transcribe → diarize → markdown.

Стадии (каждая сохраняется, повторный запуск продолжает с последней
завершённой; рабочая директория — work/<video_id>/):
    download   yt-dlp: аудио 16 кГц моно + метаданные видео
    transcribe Whisper через MLX (mlx-whisper), переключения ru/en
    diarize    pyannote/speaker-diarization-3.1 (нужен HF-токен;
               без токена стадия пропускается)
    markdown   transcripts/<video_id>.md

Использование:
    python tools/fetch_transcript.py <youtube_url> [language] [options]

    language         ru / en (по умолчанию — автоопределение)

Варианты:
    --model NAME         Whisper-модель (по умолчанию mlx-community/whisper-large-v3-turbo)
    --prompt "терм1, терм2"  словарь: передаётся модели как подсказка
    --no-diarize         не делать разделение говорящих
    --allow-long         разрешить видео длиннее часа
    --stop STAGE         остановиться после стадии: download | transcribe | diarize | markdown
    --reset              начать задачу заново (очистить work/<video_id>/)
    --uncertain T        порог уверенности для пометки (?слово?) (по умолчанию 0.5)

Вывод:
    Успех:  OK <путь к transcripts/<video_id>.md>
    Ошибка: ERROR <стадия>: <сообщение>, код выхода 1
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
        description="YouTube → транскрипт (Whisper MLX + pyannote)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Использование:")[-1],
    )
    ap.add_argument("url", help="ссылка на YouTube-видео")
    ap.add_argument("language", nargs="?", default=None, help="язык: ru / en (иначе авто)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper-модель (по умолчанию {DEFAULT_MODEL})")
    ap.add_argument("--prompt", default=None, help="словарь терминов через запятую (подсказка модели)")
    ap.add_argument("--no-diarize", action="store_true", help="пропустить разделение говорящих")
    ap.add_argument("--allow-long", action="store_true", help="разрешить видео длиннее часа")
    ap.add_argument("--stop", choices=STAGES, default="markdown", help="остановиться после этой стадии")
    ap.add_argument("--reset", action="store_true", help="начать задачу заново")
    ap.add_argument("--uncertain", type=float, default=0.5, help="порог пометки (?слово?)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    video_id = extract_video_id(args.url)
    task = Task(ROOT / "work", video_id, args.url, reset=args.reset)
    print(f"Задача: {video_id} (рабочая директория {task.work_dir.relative_to(ROOT)}/)")

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
            print(f"  {name}: уже выполнено, пропускаю")
        elif stage.status == "skipped":
            if name == "diarize" and resolve_token():
                print(f"  {name}: была пропущена, HF-токен появился — пробую снова")
                stage.status = "pending"
                task.save()
            else:
                print(f"  {name}: пропущена ({stage.detail.get('reason', '')})")
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
                print(f"Остановлено после стадии {name} (--stop)")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
