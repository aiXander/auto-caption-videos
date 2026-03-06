#!/usr/bin/env python3
"""
Step 1: Run Whisper transcription and output word-level timestamps to an editable JSON file.

Usage:
    python transcribe.py video.mp4 [-s subtitles.srt] [-c config.yaml] [-o timestamps.json]

The output JSON file can be manually edited before passing to render.py.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml


def load_config(path):
    default = Path(__file__).parent / "config.yaml"
    cfg_path = Path(path) if path else default
    if not cfg_path.exists():
        print(f"Warning: config not found at {cfg_path}, using defaults")
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get(cfg, *keys, default=None):
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


def parse_srt(srt_path):
    import pysrt
    subs = pysrt.open(srt_path)
    segments = []
    for sub in subs:
        start = sub.start.ordinal / 1000.0
        end = sub.end.ordinal / 1000.0
        text = sub.text.replace("\n", " ").strip()
        segments.append({"text": text, "start": start, "end": end, "words": []})
    return segments


def extract_audio(video_path, out_path):
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", out_path],
        check=True, capture_output=True,
    )


def run_whisper(audio_path, cfg, language_override=None):
    from faster_whisper import WhisperModel

    model_size = get(cfg, "whisper", "model", default="base")
    language = language_override or get(cfg, "whisper", "language", default="en")
    device = get(cfg, "whisper", "device", default="auto")

    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "cpu"
        else:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper model '{model_size}' on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("Transcribing for word-level timestamps...")
    whisper_segments, info = model.transcribe(
        audio_path, language=language, word_timestamps=True
    )

    segments = []
    words = []
    for seg in whisper_segments:
        seg_words = []
        if seg.words:
            for w in seg.words:
                word_entry = {"text": w.word.strip(), "start": w.start, "end": w.end}
                words.append(word_entry)
                seg_words.append(word_entry)
        segments.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end,
            "words": seg_words,
        })
    print(f"  Got {len(segments)} segments, {len(words)} words from Whisper")
    return segments


def _apply_replacements(text, replacements):
    import re
    for wrong, correct in replacements.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text


def align_words_to_segments(segments, words):
    wi = 0
    for seg in segments:
        seg["words"] = []
        while wi < len(words):
            w = words[wi]
            mid = (w["start"] + w["end"]) / 2
            if mid < seg["start"] - 0.3:
                wi += 1
                continue
            if mid > seg["end"] + 0.3:
                break
            seg["words"].append(w)
            wi += 1

    for seg in segments:
        if not seg["words"]:
            raw_words = seg["text"].split()
            n = len(raw_words)
            if n == 0:
                continue
            dur = seg["end"] - seg["start"]
            for i, w in enumerate(raw_words):
                ws = seg["start"] + (i / n) * dur
                we = seg["start"] + ((i + 1) / n) * dur
                seg["words"].append({"text": w, "start": round(ws, 3), "end": round(we, 3)})


def main():
    parser = argparse.ArgumentParser(description="Step 1: Transcribe video and output editable timestamps")
    parser.add_argument("video", help="Input video file (e.g. video.mp4)")
    parser.add_argument("-s", "--srt", default=None, help="Optional input subtitle file (e.g. subtitles.srt)")
    parser.add_argument("-c", "--config", default=None, help="Path to config.yaml (for whisper settings)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path (default: <video>_timestamps.json)")
    parser.add_argument("-l", "--language", default=None, help="Language code (e.g. 'en', 'nl', 'fr'). Overrides config.")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip Whisper; use evenly-spaced word timing from SRT (requires --srt)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_path = args.output
    if not output_path:
        stem = Path(args.video).stem
        output_path = str(Path(args.video).parent / f"{stem}_timestamps.json")

    if args.srt:
        # SRT provided: use it for segment boundaries, Whisper for word timestamps
        print("Parsing SRT...")
        segments = parse_srt(args.srt)
        print(f"  {len(segments)} subtitle segments")

        if args.skip_whisper:
            print("Skipping Whisper, using evenly-spaced word timing from SRT...")
            align_words_to_segments(segments, [])
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio = tmp.name
            try:
                print("Extracting audio...")
                extract_audio(args.video, tmp_audio)
                words = run_whisper(tmp_audio, cfg, language_override=args.language)
                # run_whisper now returns segments, but we only need the words
                all_words = []
                for seg in words:
                    all_words.extend(seg["words"])
                align_words_to_segments(segments, all_words)
            finally:
                os.unlink(tmp_audio)
    else:
        # No SRT: generate everything from Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio = tmp.name
        try:
            print("Extracting audio...")
            extract_audio(args.video, tmp_audio)
            print("Running Whisper (generating segments + word timestamps)...")
            segments = run_whisper(tmp_audio, cfg, language_override=args.language)
        finally:
            os.unlink(tmp_audio)

    # Apply word replacements from config
    replacements = get(cfg, "word_replacements", default={}) or {}
    if replacements:
        print(f"Applying {len(replacements)} word replacement(s)...")
        for seg in segments:
            seg["text"] = _apply_replacements(seg["text"], replacements)
            for w in seg["words"]:
                for wrong, correct in replacements.items():
                    if w["text"].lower() == wrong.lower():
                        w["text"] = correct

    # Round timestamps for readability
    for seg in segments:
        seg["start"] = round(seg["start"], 3)
        seg["end"] = round(seg["end"], 3)
        for w in seg["words"]:
            w["start"] = round(w["start"], 3)
            w["end"] = round(w["end"], 3)

    # Split segments into screen-sized chunks based on max_chars_per_screen
    max_chars = get(cfg, "text", "max_chars_per_screen", default=84)
    chunked_segments = []
    for seg in segments:
        if not seg["words"]:
            chunked_segments.append(seg)
            continue
        chunks = []
        current_chunk = []
        current_len = 0
        for w in seg["words"]:
            addition = len(w["text"]) + (1 if current_chunk else 0)
            if current_chunk and current_len + addition > max_chars:
                # Look back for a punctuation-based split point
                best = None
                for i in range(len(current_chunk) - 1, 0, -1):
                    if current_chunk[i]["text"].rstrip()[-1:] in ".,;:!?":
                        best = i + 1
                        break
                if best and best < len(current_chunk):
                    chunks.append(current_chunk[:best])
                    leftover = current_chunk[best:]
                    current_chunk = leftover + [w]
                    current_len = sum(len(x["text"]) for x in current_chunk) + len(current_chunk) - 1
                else:
                    chunks.append(current_chunk)
                    current_chunk = [w]
                    current_len = len(w["text"])
            else:
                current_chunk.append(w)
                current_len += addition
        if current_chunk:
            chunks.append(current_chunk)
        # Merge tiny trailing chunk back into the previous one to avoid
        # orphaned words like a single "light" getting its own segment.
        min_orphan_words = 3
        if len(chunks) > 1 and len(chunks[-1]) < min_orphan_words:
            chunks[-2].extend(chunks[-1])
            chunks.pop()
        for chunk_words in chunks:
            chunked_segments.append({
                "text": " ".join(w["text"] for w in chunk_words),
                "start": chunk_words[0]["start"],
                "end": chunk_words[-1]["end"],
                "words": chunk_words,
            })
    segments = chunked_segments
    print(f"  Split into {len(segments)} screen chunks (max {max_chars} chars each)")

    # 4. Write output
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(segments)} segments to: {output_path}")
    print("You can now edit this file, then run:")
    print(f"  python render.py {args.video} {output_path}")


if __name__ == "__main__":
    main()
