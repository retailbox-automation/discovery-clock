"""Download the published body-camera videos listed in data/sources.json (or given ids) to data/media/<id>.mp4.

    python scripts/fetch_media.py                 # every id in data/sources.json
    python scripts/fetch_media.py fy6dRGapw_4     # one id

Needs yt-dlp and ffmpeg on PATH. Video is capped at 720p (the camera clock overlay stays readable, uploads stay small).
The videos are public records released by the Volusia Sheriff's Office; they are not stored in this repository.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "data" / "media"


def ids_from_sources() -> list[str]:
    p = ROOT / "data" / "sources.json"
    return [s["id"] for s in json.loads(p.read_text())] if p.exists() else []


def fetch(vid: str) -> Path:
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / f"{vid}.mp4"
    if out.exists() and out.stat().st_size > 0:
        print(f"have {out.relative_to(ROOT)}")
        return out
    cmd = ["yt-dlp", "--no-playlist", "--no-progress", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]",
           "--merge-output-format", "mp4", "-o", str(MEDIA / f"{vid}.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"]
    subprocess.run(cmd, check=True)
    print(f"saved {out.relative_to(ROOT)}")
    return out


def main(argv: list[str]) -> int:
    if not shutil.which("yt-dlp"):
        print("yt-dlp is not installed (pip install yt-dlp)", file=sys.stderr)
        return 2
    ids = argv or ids_from_sources()
    if not ids:
        print("no ids given and data/sources.json is empty", file=sys.stderr)
        return 2
    failed = []
    for vid in ids:
        try:
            fetch(vid)
        except subprocess.CalledProcessError as e:
            failed.append(vid)
            print(f"failed {vid}: {e}", file=sys.stderr)
    print(f"done: {len(ids) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
