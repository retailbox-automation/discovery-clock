"""Stamp the video time on every frame before the recording goes to the model.

Video models are good at reading text in a frame and weak at keeping track of elapsed time over minutes of footage
(in a live test without stamps Muse Spark 1.3 placed a claim at 1124 s in a 222 s video). So the code burns
"t=NNN s" into the top-left corner of each frame and the prompt tells the model to read times from it.
The player still shows the original recording; the stamped copy is only uploaded.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

STAMP_VERSION = "t-seconds-v1"
FONT_CANDIDATES = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf"]


def stamp_text(second: int) -> str:
    return f"t={second} s"


def _font(size: int):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_stamps(n_seconds: int, out_dir: Path, height: int = 720) -> None:
    """One PNG per second of video: 00000.png = "t=0 s", 00001.png = "t=1 s", ..."""
    from PIL import Image, ImageDraw
    size = max(18, int(height * 0.045))
    font = _font(size)
    w, h = int(size * 5.4), int(size * 1.5)
    for s in range(n_seconds + 1):
        img = Image.new("RGB", (w, h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((int(size * 0.3), int(size * 0.2)), stamp_text(s), fill=(255, 230, 0), font=font)
        img.save(out_dir / f"{s:05d}.png")


def _encoder() -> list[str]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in out:  # hardware encoder on macOS: keeps the CPU free
        return ["-c:v", "h264_videotoolbox", "-b:v", "1000k"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "2"]


def make_stamped(video: str | Path, out: str | Path, duration_s: float, max_height: int = 720) -> Path:
    video, out = Path(video), Path(out)
    with tempfile.TemporaryDirectory() as td:
        render_stamps(int(duration_s) + 1, Path(td), height=max_height)
        vf = (f"[0:v]scale=-2:'min({max_height},ih)'[v];[1:v]format=rgba[s];"
              f"[v][s]overlay=x=12:y=12:shortest=0:eof_action=repeat[o]")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-framerate", "1", "-i", str(Path(td) / "%05d.png"),
               "-filter_complex", vf, "-map", "[o]", "-map", "0:a?", *_encoder(), "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-t", f"{duration_s:.3f}", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
    return out


def available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None
