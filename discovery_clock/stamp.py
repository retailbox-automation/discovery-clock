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
PIECE_KEYINT_S = 2.0  # key frame every 2 s in the stamped copy of a long recording (cut_piece)
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


class StampCancelled(RuntimeError):
    pass


def make_stamped(video: str | Path, out: str | Path, duration_s: float, max_height: int = 720,
                 keyint_s: float | None = None, cancel=None) -> Path:
    """keyint_s: force a key frame every keyint_s seconds, so the stamped copy can later be cut into pieces
    without re-encoding (cut_piece) at points no more than keyint_s before the planned start.
    cancel: a threading.Event; when it is set, ffmpeg is stopped, the partial file removed and StampCancelled raised
    (a long recording is stamped in the background, and a failed run must not leave ffmpeg running)."""
    video, out = Path(video), Path(out)
    with tempfile.TemporaryDirectory() as td:
        render_stamps(int(duration_s) + 1, Path(td), height=max_height)
        vf = (f"[0:v]scale=-2:'min({max_height},ih)'[v];[1:v]format=rgba[s];"
              f"[v][s]overlay=x=12:y=12:shortest=0:eof_action=repeat[o]")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-framerate", "1", "-i", str(Path(td) / "%05d.png"),
               "-filter_complex", vf, "-map", "[o]", "-map", "0:a?", *_encoder(), "-pix_fmt", "yuv420p",
               *(["-force_key_frames", f"expr:gte(t,n_forced*{keyint_s:g})"] if keyint_s else []),
               "-c:a", "copy", "-t", f"{duration_s:.3f}", "-movflags", "+faststart", str(out)]
        if cancel is None:
            subprocess.run(cmd, check=True)
            return out
        proc = subprocess.Popen(cmd)
        try:
            while proc.poll() is None:
                if cancel.wait(0.5):
                    proc.terminate()
                    try:
                        proc.wait(10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    out.unlink(missing_ok=True)
                    raise StampCancelled("stamping stopped: the run failed")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    return out


def available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None


def cut_piece(src: str | Path, out: str | Path, start_s: float, end_s: float) -> Path:
    """Cut [start_s, end_s] out of the stamped copy without re-encoding (stream copy). The cut starts at the key frame
    at or before start_s, so the piece can begin up to one key-frame interval early; the burned-in stamp still shows
    the time of the full recording, so the model reads absolute seconds either way."""
    src, out = Path(src), Path(out)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{end_s - start_s:.3f}",
           "-map", "0:v", "-map", "0:a?", "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out
