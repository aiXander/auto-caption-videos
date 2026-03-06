#!/usr/bin/env python3
"""
Step 2: Render subtitles onto video using a timestamps JSON file.

Usage:
    python render.py video.mp4 video_timestamps.json --cutoff 10  [-c config.yaml] [-o output.mp4]

The timestamps.json file is produced by transcribe.py and can be manually edited.
"""

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


# -- Data structures ----------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class SubtitleSegment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


# -- Config loading -----------------------------------------------------------

def load_config(path: Optional[str]) -> dict:
    default = Path(__file__).parent / "config.yaml"
    cfg_path = Path(path) if path else default
    if not cfg_path.exists():
        print(f"Warning: config not found at {cfg_path}, using defaults")
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get(cfg: dict, *keys, default=None):
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


def parse_color(c: str) -> tuple:
    if c is None:
        return None
    c = c.strip().lstrip("#")
    if len(c) == 6:
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255)
    elif len(c) == 8:
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(c[6:8], 16))
    raise ValueError(f"Invalid color: #{c}")


# -- Load timestamps JSON ----------------------------------------------------

def load_timestamps(json_path: str) -> list[SubtitleSegment]:
    with open(json_path) as f:
        data = json.load(f)

    segments = []
    for seg in data:
        words = [Word(text=w["text"], start=w["start"], end=w["end"]) for w in seg["words"]]
        segments.append(SubtitleSegment(
            text=seg["text"],
            start=seg["start"],
            end=seg["end"],
            words=words,
        ))
    return segments


# -- Rendering ----------------------------------------------------------------

class SubtitleRenderer:
    def __init__(self, cfg: dict, width: int, height: int):
        self.cfg = cfg
        self.width = width
        self.height = height

        self.font_path = get(cfg, "font", "path")
        self.bold = get(cfg, "font", "bold", default=False)
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}

        self.highlight_scale = get(cfg, "highlight", "scale", default=1.0)

        self.text_color = parse_color(get(cfg, "colors", "text", default="#FFFFFF"))
        self.highlight_color = parse_color(get(cfg, "colors", "highlight", default="#FFD700"))
        self.stroke_color = parse_color(get(cfg, "font", "stroke_color", default="#000000"))
        self.stroke_width = get(cfg, "font", "stroke_width", default=2)

        self.bg_enabled = get(cfg, "background", "enabled", default=True)
        self.bg_color = parse_color(get(cfg, "background", "color", default="#00000099"))
        self.bg_pad_x = get(cfg, "background", "padding_x", default=20)
        self.bg_pad_y = get(cfg, "background", "padding_y", default=12)
        self.bg_radius = get(cfg, "background", "border_radius", default=10)

        self.v_pos = get(cfg, "position", "vertical", default="bottom")
        self.margin_bottom = int(get(cfg, "position", "margin_bottom", default=0.06) * height)
        self.margin_top = int(get(cfg, "position", "margin_top", default=0.06) * height)
        self.margin_side = int(get(cfg, "position", "margin_side", default=0.03) * width)

        self.max_lines = get(cfg, "text", "max_lines", default=1)

        # Fonts are set by auto_size_font() — initialized to None
        self.font = None
        self.highlight_font = None

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size in self._font_cache:
            return self._font_cache[size]
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size)
            else:
                for name in ["Arial Bold.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf",
                             "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc",
                             "/System/Library/Fonts/SFNSMono.ttf"]:
                    try:
                        font = ImageFont.truetype(name, size)
                        break
                    except (OSError, IOError):
                        continue
                else:
                    font = ImageFont.load_default()
            if self.bold:
                try:
                    font.set_variation_by_axes([700])
                except Exception:
                    pass
        except Exception:
            font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    def auto_size_font(self, segments: list[SubtitleSegment]):
        """Find the largest font size where all segments fit in max_lines."""
        max_width = self.width - 2 * self.margin_side - 2 * self.bg_pad_x
        avail_height = self.height - self.margin_top - self.margin_bottom - 2 * self.bg_pad_y

        tmp_img = Image.new("RGBA", (self.width, self.height))
        draw = ImageDraw.Draw(tmp_img)

        def fits(size):
            font = self._load_font(size)
            bbox = font.getbbox("Ay")
            line_height = bbox[3] - bbox[1]
            line_spacing = int(line_height * 0.35)
            total_h = self.max_lines * line_height + max(0, self.max_lines - 1) * line_spacing
            if total_h > avail_height:
                return False
            space_w = draw.textlength(" ", font=font)
            for seg in segments:
                if not seg.words:
                    continue
                num_lines = 1
                cur_w = 0.0
                for word in seg.words:
                    w_w = draw.textlength(word.text, font=font)
                    test_w = cur_w + (space_w if cur_w > 0 else 0) + w_w
                    if cur_w > 0 and test_w > max_width:
                        num_lines += 1
                        cur_w = w_w
                        if num_lines > self.max_lines:
                            return False
                    else:
                        cur_w = test_w
            return True

        lo, hi = 8, min(300, self.height // 2)
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        self.font = self._load_font(best)
        if self.highlight_scale != 1.0:
            self.highlight_font = self._load_font(int(best * self.highlight_scale))
        else:
            self.highlight_font = self.font
        print(f"  Auto-sized font to {best}px (max_lines={self.max_lines})")

    def _get_active_segment(self, segments: list[SubtitleSegment], t: float):
        for seg in segments:
            if seg.start <= t <= seg.end:
                return seg
        return None

    def _get_active_word_index(self, seg: SubtitleSegment, t: float) -> int:
        for i, w in enumerate(seg.words):
            if w.start <= t <= w.end:
                return i
        for i, w in enumerate(seg.words):
            if t < w.start:
                return max(0, i - 1)
        return len(seg.words) - 1

    def _word_wrap(self, words: list[Word], draw: ImageDraw.ImageDraw) -> list[list[int]]:
        max_width = self.width - 2 * self.margin_side - 2 * self.bg_pad_x
        lines = []
        current_line = []
        current_width = 0
        space_w = draw.textlength(" ", font=self.font)

        for i, word in enumerate(words):
            w_width = draw.textlength(word.text, font=self.font)
            test_width = current_width + (space_w if current_line else 0) + w_width
            if current_line and test_width > max_width:
                lines.append(current_line)
                current_line = [i]
                current_width = w_width
            else:
                current_line.append(i)
                current_width = test_width

        if current_line:
            lines.append(current_line)

        # Balance two lines so the second isn't much shorter than the first.
        # Move words from end of line 1 to start of line 2 while it improves balance.
        if len(lines) == 2:
            def _line_width(indices):
                w = 0
                for j, idx in enumerate(indices):
                    w += draw.textlength(words[idx].text, font=self.font)
                    if j < len(indices) - 1:
                        w += space_w
                return w

            while len(lines[0]) > 1:
                w1 = _line_width(lines[0])
                w2 = _line_width(lines[1])
                # Try moving last word of line 1 to start of line 2
                candidate_l1 = lines[0][:-1]
                candidate_l2 = [lines[0][-1]] + lines[1]
                new_w1 = _line_width(candidate_l1)
                new_w2 = _line_width(candidate_l2)
                if new_w2 > max_width:
                    break
                # Stop if it makes balance worse
                if abs(new_w1 - new_w2) >= abs(w1 - w2):
                    break
                lines[0] = candidate_l1
                lines[1] = candidate_l2

        return lines

    def render_frame(self, frame: np.ndarray, segments: list[SubtitleSegment], t: float) -> np.ndarray:
        seg = self._get_active_segment(segments, t)
        if seg is None or not seg.words:
            return frame

        active_idx = self._get_active_word_index(seg, t)

        img = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Wrap all words in the segment (JSON chunks are the source of truth, show all lines)
        lines = self._word_wrap(seg.words, draw)
        visible_words = seg.words

        space_w = draw.textlength(" ", font=self.font)
        bbox_test = self.font.getbbox("Ay")
        line_height = bbox_test[3] - bbox_test[1]
        baseline_offset = bbox_test[1]  # top bearing of normal font
        line_spacing = int(line_height * 0.35)
        total_text_height = len(lines) * line_height + (len(lines) - 1) * line_spacing

        # Compute line widths accounting for highlight scale
        hl_space_w = draw.textlength(" ", font=self.highlight_font)
        line_widths = []
        for line_indices in lines:
            lw = 0
            for j, i in enumerate(line_indices):
                font = self.highlight_font if i == active_idx else self.font
                sw = hl_space_w if i == active_idx else space_w
                lw += draw.textlength(visible_words[i].text, font=font)
                if j < len(line_indices) - 1:
                    lw += sw
            line_widths.append(lw)

        max_line_w = max(line_widths) if line_widths else 0

        box_w = max_line_w + 2 * self.bg_pad_x
        box_h = total_text_height + 2 * self.bg_pad_y
        box_x = (self.width - box_w) / 2

        if self.v_pos == "bottom":
            box_y = self.height - self.margin_bottom - box_h
        elif self.v_pos == "top":
            box_y = self.margin_top
        else:
            box_y = (self.height - box_h) / 2

        if self.bg_enabled and self.bg_color:
            draw.rounded_rectangle(
                [box_x, box_y, box_x + box_w, box_y + box_h],
                radius=self.bg_radius,
                fill=self.bg_color,
            )

        hl_bbox = self.highlight_font.getbbox("Ay")
        hl_baseline_offset = hl_bbox[1]

        y_cursor = box_y + self.bg_pad_y
        for line_indices, lw in zip(lines, line_widths):
            x_cursor = (self.width - lw) / 2

            for idx in line_indices:
                word = visible_words[idx]
                is_active = idx == active_idx
                color = self.highlight_color if is_active else self.text_color
                font = self.highlight_font if is_active else self.font

                # Align baselines: shift scaled word up so baselines match
                y_draw = y_cursor
                if is_active and self.highlight_scale != 1.0:
                    y_draw = y_cursor + (baseline_offset - hl_baseline_offset)

                if self.stroke_color and self.stroke_width > 0:
                    draw.text(
                        (x_cursor, y_draw), word.text, font=font,
                        fill=color, stroke_width=self.stroke_width,
                        stroke_fill=self.stroke_color[:3],
                    )
                else:
                    draw.text((x_cursor, y_draw), word.text, font=font, fill=color)

                x_cursor += draw.textlength(word.text, font=font)
                x_cursor += hl_space_w if is_active else space_w

            y_cursor += line_height + line_spacing

        result = Image.alpha_composite(img, overlay)
        return np.array(result.convert("RGB"))


# -- Video processing ---------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", video_path],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            w = int(stream["width"])
            h = int(stream["height"])
            r = stream.get("r_frame_rate", "30/1")
            num, den = map(int, r.split("/"))
            fps = num / den if den else 30.0
            nb_frames = stream.get("nb_frames")
            duration = float(info["format"].get("duration", 0))
            return {
                "width": w, "height": h, "fps": fps,
                "duration": duration,
                "nb_frames": int(nb_frames) if nb_frames else int(duration * fps),
            }
    raise RuntimeError("No video stream found")


def process_video(video_path: str, segments: list[SubtitleSegment],
                  cfg: dict, output_path: str, cutoff: float = None):
    info = get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]
    total_frames = info["nb_frames"]

    if cutoff is not None:
        total_frames = min(total_frames, int(cutoff * fps))
        print(f"Video: {w}x{h} @ {fps:.2f} fps, cutoff {cutoff}s (~{total_frames} frames)")
    else:
        print(f"Video: {w}x{h} @ {fps:.2f} fps, ~{total_frames} frames")

    renderer = SubtitleRenderer(cfg, w, h)
    renderer.auto_size_font(segments)

    codec = get(cfg, "output", "codec", default="libx264")
    audio_codec = get(cfg, "output", "audio_codec", default="aac")
    preset = get(cfg, "output", "preset", default="medium")
    crf = get(cfg, "output", "crf", default=18)

    reader_cmd = ["ffmpeg", "-i", video_path]
    if cutoff is not None:
        reader_cmd += ["-t", str(cutoff)]
    reader_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "quiet", "-"]

    reader = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE)

    writer_cmd = ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "pipe:0",
         "-i", video_path,
         "-map", "0:v", "-map", "1:a?",
         "-c:v", codec, "-preset", preset, "-crf", str(crf),
         "-c:a", audio_codec,
         "-pix_fmt", "yuv420p",
         "-shortest"]
    if cutoff is not None:
        writer_cmd += ["-t", str(cutoff)]
    writer_cmd.append(output_path)

    writer = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)

    frame_size = w * h * 3
    frame_num = 0

    try:
        while True:
            raw = reader.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            t = frame_num / fps
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            frame = renderer.render_frame(frame, segments, t)

            writer.stdin.write(frame.tobytes())
            frame_num += 1

            if frame_num % int(fps) == 0:
                pct = min(100, frame_num / total_frames * 100)
                print(f"\r  Rendering: {pct:5.1f}% ({frame_num}/{total_frames})", end="", flush=True)
    finally:
        reader.stdout.close()
        writer.stdin.close()
        reader.wait()
        writer.wait()

    print(f"\n  Done! Wrote {frame_num} frames to {output_path}")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2: Render subtitles onto video from timestamps JSON")
    parser.add_argument("video", help="Input video file (e.g. video.mp4)")
    parser.add_argument("timestamps", help="Timestamps JSON file from transcribe.py")
    parser.add_argument("-c", "--config", default=None, help="Path to config.yaml")
    parser.add_argument("-o", "--output", default=None, help="Output video path")
    parser.add_argument("--cutoff", type=float, default=None,
                        help="Only render the first N seconds (e.g. --cutoff 15)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_path = args.output
    if not output_path:
        stem = Path(args.video).stem
        output_path = str(Path(args.video).parent / f"{stem}_subtitled.mp4")

    # 1. Load timestamps
    print("Loading timestamps...")
    segments = load_timestamps(args.timestamps)
    print(f"  {len(segments)} subtitle segments")

    # 2. Process video
    print("Processing video...")
    process_video(args.video, segments, cfg, output_path, cutoff=args.cutoff)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
