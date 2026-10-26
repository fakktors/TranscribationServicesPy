# yotubeAgent — YouTube-видео → транскрипт

Конвейер транскрибации YouTube-видео: Whisper через MLX (Apple Silicon) +
разделение говорящих (pyannote) → Markdown.

```
url → download → transcribe → diarize → markdown
```

| Стадия | Что делает | Артефакты в `work/<video_id>/` |
|---|---|---|
| `download` | yt-dlp: аудио (16 кГц моно WAV) + метаданные видео | `audio.wav`, `info.json` |
| `transcribe` | Whisper (MLX), автоопределение языка, переключения ru/en | `segments.json` |
| `diarize` | pyannote/speaker-diarization-3.1 (нужен HF-токен; без токена стадия **пропускается**) | `speakers.json` |
| `markdown` | финальный транскрипт | `transcripts/<video_id>.md` |

Состояние задачи — `work/<video_id>/state.json`. Каждая стадия сохраняется на
диск, поэтому **повторный запуск продолжает с последнего завершённого этапа**,
а не начинает всё заново.

## Установка

Рекомендуется Apple Silicon (MLX использует GPU). ffmpeg обязателен:

```bash
brew install ffmpeg

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Опционально: для разделения говорящих
export HF_TOKEN=...   # и разовое согласование доступа к pyannote/speaker-diarization-3.1
```

Без HF-токена конвейер работает, а стадия `diarize` просто пропускается
(маркируется `skipped`); при повторном запуске с токеном стадия выполнится.

## Использование

```bash
python tools/fetch_transcript.py "<youtube_url>" [language] [варианты]
```

- `language` — `ru` / `en` (без аргумента — автоопределение)

Варианты:

- `--model NAME` — модель Whisper (по умолчанию `mlx-community/whisper-large-v3-turbo`)
- `--prompt "терм1, терм2"` — словарь терминов как подсказка модели (подсказка,
  а не команда: термин попадёт в текст только если он есть в аудио)
- `--no-diarize` — не делать разделение говорящих
- `--allow-long` — разрешить видео длиннее часа (по умолчанию ограничено)
- `--stop STAGE` — остановиться после стадии: `download | transcribe | diarize | markdown`
- `--reset` — начать задачу заново (очистить `work/<video_id>/`)
- `--uncertain T` — порог уверенности для пометки `(?слово?)` (по умолчанию `0.5`)

Примеры:

```bash
python tools/fetch_transcript.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python tools/fetch_transcript.py "https://www.youtube.com/watch?v=..." ru
python tools/fetch_transcript.py "https://youtu.be/..." --model mlx-community/whisper-large-v3
```

URL в командной строке заключайте в кавычки — символы `?`, `=`, `&`
иначе интерпретируются оболочкой.

## Вывод

- Успех: `OK <путь к transcripts/<video_id>.md>`
- Ошибка: `ERROR <стадия>: <сообщение>`, код выхода `1` (повторный запуск
  продолжит с этой стадии)

## 📄 Формат транскрипта

Markdown: название, ссылка, метаданные (модель, язык, длительность, число
говорящих). Далее — абзацы:

- абзац = реплика одного говорящего (без diarize — блоки по 30 сегментов),
  каждый сегмент начинается с тайм-кода `[мм:сс]`
- слова с низкой уверенностью распознавания помечаются `(?слово?)`
- сегмент, передекодированный в другом языке, помечается `(en)` / `(ru)`

## Claude Code

Команда `/practices <url>` (`.claude/commands/practices.md`) превращает видео
в документ best practices по тексту транскрипта.
