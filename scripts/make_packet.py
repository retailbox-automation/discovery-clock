"""Build the long-input test file: public Volusia Sheriff's Office clips joined in a fixed order into one mp4.

    python scripts/fetch_media.py            # the clips, sha256 checked against data/sources.json
    python scripts/make_packet.py            # -> data/media/packet-volusia.mp4 (+ data/packets/volusia.json offsets)
    python scripts/make_packet.py --check    # only check the inputs and print the planned offsets

Every clip is re-encoded to 1280x720, 30 fps, stereo 44.1 kHz and cut (or padded by its last frame / silence) to
exactly its own duration, so clip k starts at the sum of the durations before it. The offsets are written to
data/packets/volusia.json before the build; the evaluation key uses them to check that each claim is found inside
its own clip. A different machine or ffmpeg build gives a file with another sha256; replay then still works and marks
the requests as changed since recording.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "data" / "media"
MANIFEST = ROOT / "data" / "packets" / "volusia.json"
PACKET_ID = "packet-volusia"

# fixed order: the five evaluation clips, then the clips added for the long-input test
PARTS = [
    ("fy6dRGapw_4", "fisherman-fy6dRGapw_4"),
    ("phHtxd9Ln3Q", "housefire-phHtxd9Ln3Q"),
    ("iqxWGjr1PRk", "i4driver-iqxWGjr1PRk"),
    ("kCu9d3LMN6A", "dog-kCu9d3LMN6A"),
    ("r-ub7DZN8BA", "motorcycle-r-ub7DZN8BA"),
    ("qLXQ1PO2aME", "speeder-qLXQ1PO2aME"),
    ("cPrCvm5R7DI", "gasthieves-cPrCvm5R7DI"),
    ("inKtyjwcU7E", "water-inKtyjwcU7E"),
    ("0WemYQrrIyM", "ocean-0WemYQrrIyM"),
    ("cR5y4V4eGbI", "lakedias-cR5y4V4eGbI"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(p)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def plan(sources: dict) -> list[dict]:
    """Offsets from the clips on disk. A clip whose sha256 differs from data/sources.json (a YouTube download muxed by
    another ffmpeg) is used with a warning; its length must still match the recorded offsets (checked in main)."""
    parts, t = [], 0.0
    for vid, case in PARTS:
        p = MEDIA / f"{vid}.mp4"
        if not p.exists():
            raise SystemExit(f"missing {p.relative_to(ROOT)} (python scripts/fetch_media.py {vid})")
        want = (sources.get(vid) or {}).get("sha256")
        got = sha256(p)
        if want != got:
            print(f"WARNING {vid}: sha256 {got[:12]} != data/sources.json {str(want)[:12]} (not the file of the recorded run; "
                  "replay will mark the requests as changed)", file=sys.stderr)
        d = round(duration(p), 3)
        parts.append({"id": vid, "case": case, "title": sources[vid]["title"], "start_s": round(t, 3),
                      "end_s": round(t + d, 3), "duration_s": d, "sha256": got})
        t += d
    return parts


def encoder() -> list[str]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in out:
        return ["-c:v", "h264_videotoolbox", "-b:v", "2500k"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-threads", "2"]


def build(parts: list[dict], out: Path) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in parts:
        cmd += ["-i", str(MEDIA / f"{p['id']}.mp4")]
    chains, labels = [], []
    for i, p in enumerate(parts):
        d = p["duration_s"]
        chains.append(f"[{i}:v]fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
                      f"setsar=1,format=yuv420p,tpad=stop_mode=clone:stop_duration=2,trim=duration={d:.3f},setpts=PTS-STARTPTS[v{i}]")
        chains.append(f"[{i}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,"
                      f"atrim=duration={d:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    graph = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(parts)}:v=1:a=1[v][a]"
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]", *encoder(), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="check the inputs and print the plan, do not encode")
    a = ap.parse_args(argv)
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")
    sources = {s["id"]: s for s in json.loads((ROOT / "data" / "sources.json").read_text())}
    parts = plan(sources)
    total = parts[-1]["end_s"]
    for p in parts:
        print(f"{p['start_s']:8.1f} - {p['end_s']:8.1f} s  {p['case']}")
    print(f"total {total:.1f} s ({total / 60:.1f} min), {len(parts)} clips")
    manifest = {"id": PACKET_ID, "file": f"data/media/{PACKET_ID}.mp4", "duration_s": round(total, 3), "parts": parts,
                "note": "public Volusia Sheriff's Office clips joined in a fixed order to test long input; not one incident"}
    if MANIFEST.exists():  # the sealed evaluation key uses these offsets: a clip of another length breaks them
        rec = {p["id"]: p for p in json.loads(MANIFEST.read_text())["parts"]}
        bad = [p["id"] for p in parts if p["id"] not in rec or abs(rec[p["id"]]["start_s"] - p["start_s"]) > 0.05
               or abs(rec[p["id"]]["duration_s"] - p["duration_s"]) > 0.05]
        if bad:
            raise SystemExit(f"clip lengths differ from {MANIFEST.relative_to(ROOT)} for {', '.join(bad)}: the offsets of "
                             "the sealed key would not match")
    if a.check:
        return 0
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    out = MEDIA / f"{PACKET_ID}.mp4"
    build(parts, out)
    got = duration(out)
    print(f"wrote {out.relative_to(ROOT)}: {got:.2f} s (planned {total:.2f} s), sha256 {sha256(out)}")
    if abs(got - total) > 1.0:
        print("WARNING: the packet length differs from the planned offsets by more than 1 s", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
