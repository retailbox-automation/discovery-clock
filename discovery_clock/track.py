"""SAM 3.1 on a short window of the recording.

- track(): follow one person the lawyer clicked on (object prompt: a point on one frame, tracked both ways).
- faces(): find every face in a window (concept prompt "face") so bystanders can be blurred.
- export_clip(): cut the window for a claim and blur every face except the followed person.

Only a window around the claim is sent (default 10 frames per second), because SAM video is billed per frame.
Boxes come back in the window's pixels and frame numbers; we convert them to the original video's pixels and seconds.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from .meta import SAM_MODEL, sam_cost

CLIP_FPS = 10
FACE_MARGIN = 0.25  # grow face boxes by 25 % on each side before blurring
CLIP_PAD_S = 2.0
CLIP_MAX_S = 30.0


def clip_window(claim: dict, duration: float) -> tuple[float, float]:
    """Window exported for a claim: its timecode plus 2 s on each side, at most 30 s."""
    a = max(0.0, claim["start_s"] - CLIP_PAD_S)
    b = min(duration, max(claim["end_s"], claim["start_s"]) + CLIP_PAD_S)
    return round(a, 2), round(min(b, a + CLIP_MAX_S), 2)

_FRAME = re.compile(r"<(\d+)f>")
_REC = re.compile(r"(\d+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);w=(\d+);h=(\d+)(?:;c=([0-9.]+))?\|>")


def parse_sam(text: str) -> list[dict]:
    """SAM 3.1 output text -> [{"frame": N, "id": k, "box": [x1, y1, x2, y2], "w": W, "h": H, "c": conf|None}].
    Lines can arrive in any frame order and skip frames; masks are ignored."""
    out = []
    pieces = _FRAME.split(text or "")
    # pieces = [before, N1, rest1, N2, rest2, ...]
    for i in range(1, len(pieces) - 1, 2):
        frame = int(pieces[i])
        for m in _REC.finditer(pieces[i + 1]):
            out.append({"frame": frame, "id": int(m.group(1)), "box": [int(m.group(k)) for k in (2, 3, 4, 5)],
                        "w": int(m.group(6)), "h": int(m.group(7)), "c": float(m.group(8)) if m.group(8) else None})
    out.sort(key=lambda r: (r["frame"], r["id"]))
    return out


def window(t_center: float, duration: float, before: float, after: float) -> tuple[float, float]:
    a = max(0.0, t_center - before)
    b = min(duration, t_center + after)
    return round(a, 2), round(max(b, a + 1.0 / CLIP_FPS), 2)


def video_size(video: str | Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0:s=x", str(video)], capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split("\n")[0].split("x")[:2]
    return int(w), int(h)


def cut_window(video: str | Path, a: float, b: float, out: Path, fps: int = CLIP_FPS) -> int:
    """Cut [a, b] at a fixed frame rate, no audio. Returns the number of frames (= billed frames)."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-i", str(video), "-t", f"{b - a:.3f}",
                    "-vf", f"fps={fps}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True).stdout
    return int(n.strip().split("\n")[0])


def to_video(records: list[dict], a: float, src_w: int, src_h: int, fps: int = CLIP_FPS) -> list[dict]:
    """Window frame/pixels -> original video seconds/pixels."""
    out = []
    for r in records:
        sx, sy = src_w / r["w"], src_h / r["h"]
        x1, y1, x2, y2 = r["box"]
        out.append({"t": round(a + r["frame"] / fps, 3), "id": r["id"],
                    "box": [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)], "c": r["c"]})
    return out


def _sam_call(client, file_id: str, prompt: str, *, step: str, frames: int, summary: dict, metadata: dict) -> list[dict]:
    body = {"model": SAM_MODEL, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": file_id}]}],
        "stream": False, "store": False, "metadata": {"mask_encoding": "one_bit", **metadata}}
    text, raw = client.responses(body, step=step, summary=summary, estimate=sam_cost(frames),
                                 cost_fn=lambda r: sam_cost(frames), usage_extra={"frames": frames})
    return parse_sam(text)


def _upload_window(client, video: Path, media_sha: str, a: float, b: float, td: Path) -> tuple[str, int]:
    clip = td / "window.mp4"
    frames = cut_window(video, a, b, clip) if not client.replay else int(round((b - a) * CLIP_FPS))
    fid = client.upload(clip, media_key=f"{media_sha}:window:{a:.2f}-{b:.2f}@{CLIP_FPS}fps", step=f"upload-window-{a:.0f}-{b:.0f}")
    return fid, frames


def track(client, video: str | Path, media_sha: str, duration: float, t: float, x: int, y: int,
          before: float = 4.0, after: float = 8.0) -> dict:
    """Follow the person at pixel (x, y) of the original video at second t. Returns boxes in video seconds/pixels."""
    video = Path(video)
    src_w, src_h = video_size(video) if video.exists() else (1280, 720)
    a, b = window(t, duration, before, after)
    with tempfile.TemporaryDirectory() as td:
        fid, frames = _upload_window(client, video, media_sha, a, b, Path(td))
        n = int(round((t - a) * CLIP_FPS))
        prompt = f"<{n}f>1<|point;x={int(x)};y={int(y)};w={src_w};h={src_h}|>"
        recs = _sam_call(client, fid, prompt, step=f"sam-track-{t:.1f}-{x}-{y}", frames=frames,
                         summary={"task": "follow one person through the window", "prompt": prompt,
                                  "video": f"window {a:.1f}-{b:.1f} s at {CLIP_FPS} fps ({frames} frames)",
                                  "propagation_direction": "both"},
                         metadata={"propagation_direction": "both"})
    boxes = [r for r in to_video(recs, a, src_w, src_h) if r["id"] == 1]
    return {"t": t, "x": x, "y": y, "window": [a, b], "fps": CLIP_FPS, "size": [src_w, src_h], "boxes": boxes,
            "model": SAM_MODEL}


def faces(client, video: str | Path, media_sha: str, duration: float, a: float, b: float) -> dict:
    video = Path(video)
    src_w, src_h = video_size(video) if video.exists() else (1280, 720)
    with tempfile.TemporaryDirectory() as td:
        fid, frames = _upload_window(client, video, media_sha, a, b, Path(td))
        recs = _sam_call(client, fid, "face", step=f"sam-faces-{a:.1f}-{b:.1f}", frames=frames,
                         summary={"task": "find faces to blur", "prompt": "face",
                                  "video": f"window {a:.1f}-{b:.1f} s at {CLIP_FPS} fps ({frames} frames)"},
                         metadata={})
    return {"window": [a, b], "fps": CLIP_FPS, "size": [src_w, src_h], "boxes": to_video(recs, a, src_w, src_h), "model": SAM_MODEL}


def _inside(face: list[int], person: list[int]) -> bool:
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def boxes_at(boxes: list[dict], t: float, fps: int = CLIP_FPS) -> list[dict]:
    """Boxes of the SAM frame nearest to t (within one SAM frame)."""
    if not boxes:
        return []
    best = min(boxes, key=lambda r: abs(r["t"] - t))
    if abs(best["t"] - t) > 1.0 / fps + 1e-6:
        return []
    return [r for r in boxes if abs(r["t"] - best["t"]) < 1e-6]


def blur_plan(face_boxes: list[dict], keep_boxes: list[dict] | None, t: float, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Face rectangles to blur at time t: every face except one inside the followed person's box, grown by a margin."""
    keep = [k["box"] for k in boxes_at(keep_boxes or [], t)]
    out = []
    for f in boxes_at(face_boxes, t):
        if any(_inside(f["box"], k) for k in keep):
            continue
        x1, y1, x2, y2 = f["box"]
        mx, my = (x2 - x1) * FACE_MARGIN, (y2 - y1) * FACE_MARGIN
        out.append((max(0, int(x1 - mx)), max(0, int(y1 - my)), min(w, int(x2 + mx)), min(h, int(y2 + my))))
    return out


def export_clip(video: str | Path, a: float, b: float, face_boxes: list[dict], out: str | Path,
                keep_boxes: list[dict] | None = None) -> Path:
    """Re-encode [a, b] with faces blurred frame by frame (Pillow); keeps the original frame rate and audio."""
    from PIL import Image, ImageFilter
    video, out = Path(video), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = video_size(video)
    fps_s = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                            "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout.strip()
    num, den = (fps_s.split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-ss", f"{a:.3f}", "-i", str(video), "-t", f"{b - a:.3f}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    with tempfile.TemporaryDirectory() as td:
        silent = Path(td) / "v.mp4"
        enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                                "-threads", "2", "-pix_fmt", "yuv420p", str(silent)], stdin=subprocess.PIPE)
        size = w * h * 3
        i = 0
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            t = a + i / fps
            rects = blur_plan(face_boxes, keep_boxes, t, w, h)
            if rects:
                img = Image.frombytes("RGB", (w, h), buf)
                for r in rects:
                    if r[2] > r[0] and r[3] > r[1]:
                        region = img.crop(r)
                        radius = max(6, (r[2] - r[0]) // 4)
                        img.paste(region.filter(ImageFilter.GaussianBlur(radius)), r[:2])
                buf = img.tobytes()
            enc.stdin.write(buf)
            i += 1
        enc.stdin.close()
        enc.wait()
        dec.wait()
        # add the original audio of the window back
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
                        "-i", str(video), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest",
                        "-movflags", "+faststart", str(out)], check=True)
    return out


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
