# Auto Caption Videos

Add word-level highlighted subtitles to any video using Whisper transcription.

![Python](https://img.shields.io/badge/python-3.9+-blue)

## How it works

1. **Transcribe** — Extracts audio and runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper) to get word-level timestamps. Outputs an editable JSON file.
2. **Render** — Reads the timestamps JSON and burns subtitles onto the video frame-by-frame with per-word highlight tracking.

The two-step pipeline lets you manually fix transcription errors in the JSON before rendering.

## Setup

```bash
pip install -r requirements.txt
```

Requires `ffmpeg` and `ffprobe` on your PATH.

## Usage

```bash
# Step 1: Transcribe
python transcribe.py video.mp4

# (Optional) Edit video_timestamps.json to fix any errors

# Step 2: Render subtitles
python render.py video.mp4 video_timestamps.json
```

### Options

```
transcribe.py video.mp4 [-s subtitles.srt] [-l en] [-c config.yaml] [-o output.json] [--skip-whisper]
render.py video.mp4 timestamps.json [-c config.yaml] [-o output.mp4] [--cutoff 15]
```

- `-s` / `--srt` — Use existing SRT for segment boundaries (Whisper still provides word timing)
- `--skip-whisper` — With `--srt`, skip Whisper entirely and use evenly-spaced word timing
- `--cutoff` — Only render the first N seconds (useful for previewing style changes)
- `-l` / `--language` — Override language detection (e.g. `en`, `nl`, `fr`)

## Configuration

All styling is controlled via `config.yaml`:

| Section | Key options |
|---------|------------|
| `text` | `max_chars_per_screen`, `max_lines` |
| `font` | `path`, `bold`, `stroke_width`, `stroke_color` |
| `colors` | `text` (unhighlighted), `highlight` (active word) |
| `position` | `vertical` (top/center/bottom), margins |
| `background` | `enabled`, `color`, `padding`, `border_radius` |
| `whisper` | `model`, `language`, `device` |
| `output` | `codec`, `preset`, `crf` |
| `word_replacements` | Fix common Whisper mistranscriptions |

Colors support hex with alpha (e.g. `#FFFFFFB3` for 70% white).

## Fonts

Drop `.ttf`/`.otf` files into `fonts/` and set `font.path` in `config.yaml`. Several fonts are included.
