"""Download the published body-camera videos listed in data/sources.json (or given ids) to data/media/<id>.mp4.

    python scripts/fetch_media.py                 # every id in data/sources.json
    python scripts/fetch_media.py fy6dRGapw_4     # one id

First tries the byte-identical copy attached to this repository's GitHub release "eval-media" and checks its sha256 against
data/sources.json, so replay matches the recorded runs exactly. Falls back to YouTube via yt-dlp (needs yt-dlp and ffmpeg on PATH;
capped at 720p); a YouTube download may not be byte-identical, in which case replay still works but marks rows "request changed".
The videos are public records released by the Volusia Sheriff's Office; they are not stored in this repository.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "data" / "media"
RELEASE = "https://github.com/retailbox-automation/discovery-clock/releases/download/eval-media/{vid}.mp4"


def expected_sha(vid: str) -> str | None:
    p = ROOT / "data" / "sources.json"
    for s in (json.loads(p.read_text()) if p.exists() else []):
        if s["id"] == vid:
            return s.get("sha256")
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def from_release(vid: str, out: Path) -> bool:
    want = expected_sha(vid)
    if not want:
        return False
    tmp = out.with_suffix(".part")
    try:
        with urllib.request.urlopen(RELEASE.format(vid=vid), timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except OSError as e:
        print(f"release copy of {vid} not available ({e}); trying YouTube", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    got = sha256(tmp)
    if got != want:
        print(f"release copy of {vid}: sha256 {got[:12]} != {want[:12]}; trying YouTube", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(out)
    return True


def ids_from_sources() -> list[str]:
    p = ROOT / "data" / "sources.json"
    # entries without a url (the long-input packet) are built locally by scripts/make_packet.py
    return [s["id"] for s in json.loads(p.read_text()) if s.get("url")] if p.exists() else []


def fetch(vid: str) -> Path:
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / f"{vid}.mp4"
    if out.exists() and out.stat().st_size > 0:
        print(f"have {out.relative_to(ROOT)}")
        return out
    if from_release(vid, out):
        print(f"saved {out.relative_to(ROOT)} (release copy, sha256 ok)")
        return out
    if not shutil.which("yt-dlp"):
        raise subprocess.CalledProcessError(2, "yt-dlp", "yt-dlp is not installed (pip install yt-dlp)")
    cmd = ["yt-dlp", "--no-playlist", "--no-progress", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]",
           "--merge-output-format", "mp4", "-o", str(MEDIA / f"{vid}.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"]
    subprocess.run(cmd, check=True)
    print(f"saved {out.relative_to(ROOT)}")
    return out


def main(argv: list[str]) -> int:
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
