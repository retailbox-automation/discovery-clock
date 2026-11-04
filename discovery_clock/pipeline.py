"""video + report -> runs/<case>/result.json. Steps: claims, transcribe, upload, review, verify."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import RECORDINGS, ROOT, RUNS, claims as claims_mod, review as review_mod, stamp as stamp_mod, verify as verify_mod
from .meta import SPARK_MODEL, VOICE_MODEL, MetaClient, sha256_file, utc_now

STEPS = ["claims", "transcribe", "upload", "review", "verify"]
STEP_LABELS = {"claims": "Split the report into claims", "transcribe": "Transcribe speech with speakers",
               "upload": "Stamp video time on each frame, upload", "review": "Place each claim on the recording",
               "verify": "Code checks the model's answer"}
RUN_LOCK = threading.Lock()


def probe_duration(video: str | Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def source_for(video: Path) -> dict | None:
    p = ROOT / "data" / "sources.json"
    if not p.exists():
        return None
    try:
        for s in json.loads(p.read_text()):
            if s.get("id") and s["id"] in video.stem:
                return s
    except (json.JSONDecodeError, TypeError):
        return None
    return None


class Status:
    def __init__(self, case_dir: Path):
        self.path = case_dir / "status.json"
        self.data = {"state": "running", "started_utc": utc_now(), "steps": {s: {"state": "pending"} for s in STEPS}}
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)

    def step(self, name: str, state: str, **extra):
        self.data["steps"][name] = {"state": state, "utc": utc_now(), **extra}
        self.save()

    def finish(self, state: str, **extra):
        self.data.update({"state": state, "finished_utc": utc_now(), **extra})
        self.save()


def piece_media_key(media_sha: str, a: float, b: float) -> str:
    return f"{media_sha}:{stamp_mod.STAMP_VERSION}:piece-{a:.3f}-{b:.3f}"


def run(case_id: str, video: str | Path, report_text: str, *, claims_mode: str = "extract", use_open: bool = True,
        client: MetaClient | None = None, runs_dir: Path | None = None, duration_s: float | None = None,
        long_over_s: float | None = None, piece_max_s: float | None = None) -> dict:
    video = Path(video).resolve()
    runs = Path(runs_dir or RUNS)
    case_dir = runs / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    client = client or MetaClient(case_id, runs_dir=runs)
    st = Status(case_dir)
    t_start = time.time()
    timings: dict[str, float] = {}
    current = "claims"
    bg = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"stamp-{case_id}")
    stop_stamp = threading.Event()
    stamp_job = None
    stamped = case_dir / "upload-stamped.mp4"
    try:
        duration = duration_s if duration_s is not None else probe_duration(video)
        media_sha = sha256_file(video)
        src = source_for(video)
        pieces = review_mod.plan_pieces(duration, piece_max_s) if review_mod.needs_pieces(duration, long_over_s) else None
        case = {"case": case_id, "created_utc": utc_now(), "video_path": str(video), "video_name": video.name,
                "video_sha256": media_sha, "duration_s": round(duration, 3), "claims_mode": claims_mode,
                "report": report_text, "source": src, "replay": client.replay}
        if pieces:
            case["review_pieces"] = [list(p) for p in pieces]
        (case_dir / "case.json").write_text(json.dumps(case, indent=1, ensure_ascii=False))

        # a long recording: stamp the whole video in the background while the speech is transcribed
        stamped = case_dir / "upload-stamped.mp4"
        if pieces and not client.replay:
            stamp_job = bg.submit(stamp_mod.make_stamped, video, stamped, duration, 720, stamp_mod.PIECE_KEYINT_S, stop_stamp)

        t0 = time.time()
        st.step("claims", "running")
        cl = claims_mod.extract(client, report_text, claims_mode, use_open=use_open)
        st.step("claims", "done", count=len(cl["claims"]), model=cl["model"])
        timings["claims"] = round(time.time() - t0, 1)

        current = "transcribe"
        t0 = time.time()
        st.step("transcribe", "running")
        tr = client.transcribe(video, duration, media_sha=media_sha)
        st.step("transcribe", "done", turns=len(tr["turns"]), pieces=tr["pieces"])
        timings["transcribe"] = round(time.time() - t0, 1)

        current = "upload"
        t0 = time.time()
        st.step("upload", "running", **({"pieces": len(pieces)} if pieces else {}))
        if pieces:
            keys = [piece_media_key(media_sha, a, b) for a, b in pieces]
            if client.replay:
                fids = [client.upload(video, step=f"upload-p{i}", media_key=k) for i, k in enumerate(keys)]
            else:
                stamp_job.result()
                timings["stamp_done_after_s"] = round(time.time() - t_start, 1)

                def up(i):
                    a, b = pieces[i]
                    part = stamp_mod.cut_piece(stamped, case_dir / f"upload-p{i}.mp4", a, b)
                    try:
                        return client.upload(part, step=f"upload-p{i}", media_key=keys[i])
                    finally:
                        part.unlink(missing_ok=True)
                with ThreadPoolExecutor(max_workers=3) as ex:
                    fids = list(ex.map(up, range(len(pieces))))
        else:
            media_key = f"{media_sha}:{stamp_mod.STAMP_VERSION}"
            if client.replay:  # the recorded upload is found by media_key; no need to rebuild the stamped copy
                fid = client.upload(video, media_key=media_key)
            else:
                stamp_mod.make_stamped(video, stamped, duration)
                fid = client.upload(stamped, media_key=media_key)
        st.step("upload", "done", **({"pieces": len(pieces)} if pieces else {}))
        timings["upload"] = round(time.time() - t0, 1)

        current = "review"
        t0 = time.time()
        merge_info = None
        st.step("review", "running", **({"pieces": len(pieces)} if pieces else {}))
        if pieces:
            outs = review_mod.review_pieces(client, fids, tr["turns"], cl["claims"], pieces)
            answer, merge_info = review_mod.merge_answers([o[0] for o in outs], pieces, [c["id"] for c in cl["claims"]])
            usage = {k: sum((o[1].get("usage") or {}).get(k, 0) or 0 for o in outs) for k in ("input_tokens", "output_tokens")}
            review_raw = {"answer": answer, "usage": usage, "status": [o[1].get("status") for o in outs], "merge": merge_info,
                          "pieces": [{"index": i + 1, "window": list(pieces[i]), "answer": o[0], "usage": o[1].get("usage"),
                                      "status": o[1].get("status")} for i, o in enumerate(outs)]}
        else:
            answer, raw = review_mod.review(client, fid, tr["turns"], cl["claims"], duration)
            usage = raw.get("usage")
            review_raw = {"answer": answer, "usage": usage, "status": raw.get("status")}
        (case_dir / "review-raw.json").write_text(json.dumps(review_raw, indent=1, ensure_ascii=False))
        st.step("review", "done", usage=usage, **({"pieces": len(pieces)} if pieces else {}))
        timings["review"] = round(time.time() - t0, 1)

        current = "verify"
        t0 = time.time()
        st.step("verify", "running")
        checked = verify_mod.verify(answer, cl["claims"], tr["turns"], duration)
        checked = review_mod.apply_merge(checked, merge_info)
        st.step("verify", "done", changed=checked["summary"]["changed_by_code"])
        timings["verify"] = round(time.time() - t0, 1)

        result = {"case": case_id, "created_utc": utc_now(), "replay": client.replay,
                  "video": {"name": video.name, "sha256": media_sha, "duration_s": round(duration, 3)},
                  "source": src, "report": report_text, "claims_mode": claims_mode,
                  "models": {"claims": cl["model"], "transcribe": VOICE_MODEL, "review": SPARK_MODEL, "verify": "code"},
                  "claims_notes": cl["notes"], "turns": tr["turns"], "camera_clock": checked["camera_clock"],
                  "claims": checked["claims"], "summary": checked["summary"], "verify_notes": checked["notes"],
                  "verify_version": checked["verify_version"],
                  "spend_usd": round(client.case_spend, 5), "seconds": round(time.time() - t_start, 1)}
        if pieces:
            result["review_pieces"] = merge_info["pieces"]
            result["merge_rule"] = merge_info["rule"]
            result["timings_s"] = timings
        (case_dir / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
        if client.record:
            rec_dir = client.recordings / case_id
            rec_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(case_dir / "result.json", rec_dir / "result.json")
            if pieces:  # the merged answer, so the evaluation can re-check a long case from recordings/ alone
                shutil.copy(case_dir / "review-raw.json", rec_dir / "review-raw.json")
            (rec_dir / "case.json").write_text(json.dumps({k: v for k, v in case.items() if k != "video_path"},
                                                          indent=1, ensure_ascii=False))
        st.finish("done", summary=result["summary"], spend_usd=result["spend_usd"])
        return result
    except Exception as e:  # the UI shows the failure; the CLI re-raises it
        st.step(current, "failed", error=str(e)[:500])
        st.finish("failed", error=f"{type(e).__name__}: {str(e)[:500]}", trace=traceback.format_exc()[-1500:])
        raise
    finally:
        # stop a background stamping job of a failed run and remove the stamped copy and any cut piece
        stop_stamp.set()
        if stamp_job is not None:
            try:
                stamp_job.result(timeout=60)
            except Exception:  # noqa: BLE001 - cancelled or failed; the run's own error is what matters
                pass
        bg.shutdown(wait=False)
        stamped.unlink(missing_ok=True)
        for part in case_dir.glob("upload-p*.mp4"):
            part.unlink(missing_ok=True)


def recorded_case(case_id: str, recordings_dir: Path | None = None) -> dict:
    p = Path(recordings_dir or RECORDINGS) / case_id / "case.json"
    if not p.exists():
        raise FileNotFoundError(f"no recorded case {case_id} ({p})")
    return json.loads(p.read_text())


def reverify(case_id: str, runs_dir: Path | None = None, recordings_dir: Path | None = None) -> dict:
    """Run the current code checks again over the saved model answer of a finished case (no model calls).
    The previous result is kept once as result-v<old version>.json."""
    d = Path(runs_dir or RUNS) / case_id
    status = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    if status.get("state") in ("running", "queued"):
        raise RuntimeError(f"{case_id} is still running")
    result = json.loads((d / "result.json").read_text())
    raw = json.loads((d / "review-raw.json").read_text())
    old_v = result.get("verify_version", 1)
    base = [{k: c.get(k) for k in ("id", "text", "kind", "line")} for c in result["claims"]]
    answer, merge = review_mod.remerge(raw, [c["id"] for c in base])
    checked = verify_mod.verify(answer, base, result["turns"], result["video"]["duration_s"])
    checked = review_mod.apply_merge(checked, merge)
    if old_v != checked["verify_version"]:
        keep = d / f"result-v{old_v}.json"
        if not keep.exists():
            shutil.copy(d / "result.json", keep)
    result.update({"claims": checked["claims"], "summary": checked["summary"], "camera_clock": checked["camera_clock"],
                   "verify_notes": checked["notes"], "verify_version": checked["verify_version"],
                   "reverified_utc": utc_now()})
    (d / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    rec = Path(recordings_dir or RECORDINGS) / case_id
    if (rec / "result.json").exists():
        if old_v != checked["verify_version"] and not (rec / f"result-v{old_v}.json").exists() and (d / f"result-v{old_v}.json").exists():
            shutil.copy(d / f"result-v{old_v}.json", rec / f"result-v{old_v}.json")
        shutil.copy(d / "result.json", rec / "result.json")
    return result


_SAM_TRACK = re.compile(r"^sam-track-([0-9.]+)-(\d+)-(\d+)-[0-9a-f]{16}\.json$")
_SAM_FACES = re.compile(r"^sam-faces-([0-9.]+)-([0-9.]+)-[0-9a-f]{16}\.json$")


def replay_sam(case_id: str, video: str | Path, runs_dir: Path | None = None, recordings_dir: Path | None = None) -> list[str]:
    """Rebuild the recorded SAM 3.1 results of a case (followed person, clips with faces blurred) from recordings/,
    without network, so the web page shows them in replay."""
    from . import track as track_mod
    d = Path(runs_dir or RUNS) / case_id
    rec = Path(recordings_dir or RECORDINGS) / case_id
    result = json.loads((d / "result.json").read_text())
    sha, dur = result["video"]["sha256"], result["video"]["duration_s"]
    client = MetaClient(case_id, runs_dir=Path(runs_dir or RUNS), recordings_dir=Path(recordings_dir or RECORDINGS), replay=True)
    done = []
    for p in sorted(rec.glob("sam-*.json")):
        m = _SAM_TRACK.match(p.name)
        if m:
            t, x, y = float(m.group(1)), int(m.group(2)), int(m.group(3))
            out = track_mod.track(client, video, sha, dur, t, x, y)
            track_mod.save_json(out, d / "tracks" / f"{t:.1f}-{x}-{y}.json")
            (d / "tracks" / "latest.json").write_text(json.dumps({"state": "done", **out}))
            done.append(f"follow {t:.1f}s ({x},{y}): {len(out['boxes'])} boxes")
            continue
        m = _SAM_FACES.match(p.name)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            claim = next((c for c in result["claims"] if c.get("start_s") is not None
                          and track_mod.clip_window(c, dur) == (round(a, 2), round(b, 2))), None)
            if claim is None:
                continue
            fc = track_mod.faces(client, video, sha, dur, a, b)
            track_mod.export_clip(video, a, b, fc["boxes"], d / "clips" / f"{claim['id']}.mp4")
            (d / "clips" / f"{claim['id']}.json").write_text(json.dumps({
                "state": "done", "url": f"/clips/{case_id}/{claim['id']}.mp4", "window": [a, b],
                "face_boxes": len({(r["t"], r["id"]) for r in fc["boxes"]}), "kept_followed_person": False}))
            done.append(f"clip {claim['id']} {a:.1f}-{b:.1f}s: {len(fc['boxes'])} face boxes")
    return done
